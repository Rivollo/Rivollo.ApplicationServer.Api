"""Restoring an account must undo the deletion — no more, no less.

Two failure modes matter, and they are not symmetric:

  * restoring too little is an inconvenience — support can fix it;
  * restoring too much is a privacy incident. A product the user deleted
    themselves, brought back and republished, puts content online that someone
    deliberately took down.

So the fingerprint test is the important one: restore only clears deleted_at on
products stamped with the account deletion's exact instant. Anything else stays
deleted, including products deleted a second earlier.

The other half is proving restore is not a way in. A deleted account is still a
real account with real published data, so restore re-authenticates properly
rather than accepting a typed confirmation the way deletion does.

The database is stubbed; what is under test is which statements we emit, against
which rows, with which values.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.sql import Delete, Update

from app.services.account_service import AccountService

PASSWORD = "correct-horse-battery-staple"
GOOGLE_SUB = "104729384756102938475"


class _Result:
    def __init__(self, first_row=None, scalar=None, rows=None):
        self._first = first_row
        self._scalar = scalar
        self._rows = rows if rows is not None else []

    def first(self):
        return self._first

    def scalar_one_or_none(self):
        return self._scalar

    def fetchall(self):
        return self._rows


class _DB:
    """Replays queued results in order and records every statement."""

    def __init__(self, results):
        self._results = list(results)
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, statement, *_args, **_kwargs):
        self.statements.append(statement)
        return self._results.pop(0) if self._results else _Result()

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def _user(*, deleted_at, purge_after, password_hash="argon2-hash", is_active=False):
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="seller@example.com",
        password_hash=password_hash,
        deleted_at=deleted_at,
        purge_after=purge_after,
        is_active=is_active,
    )


def _pending_user(*, days_ago=3, **kwargs):
    deleted_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return _user(
        deleted_at=deleted_at,
        purge_after=deleted_at + timedelta(days=30),
        **kwargs,
    )


def _updates(db):
    return [s for s in db.statements if isinstance(s, Update)]


def _values_of(statement):
    return {
        col.name: getattr(val, "value", val)
        for col, val in statement._values.items()
    }


def _target_table(statement):
    return statement.table.name


def _lookup(user):
    """Result for UserRepository.get_for_restore."""
    return _Result(scalar=user)


def _user_update_ok(user):
    return _Result(first_row=(user.id,))


def _products_restored(n):
    return _Result(rows=[(uuid.uuid4(),) for _ in range(n)])


@pytest.fixture(autouse=True)
def _accept_password(monkeypatch):
    monkeypatch.setattr(
        "app.services.account_service.verify_password",
        lambda password, hashed: password == PASSWORD,
    )


# ---------------------------------------------------------------------------
# successful restore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_restore():
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(4)])

    result = await AccountService.restore_account(
        db=db, email=user.email, password=PASSWORD
    )

    assert result.user_id == user.id
    assert result.products_restored == 4
    assert db.commits == 1


@pytest.mark.asyncio
async def test_restore_well_inside_the_window_succeeds():
    """Day 29 of 30 is still restorable."""
    user = _pending_user(days_ago=29)
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(1)])

    result = await AccountService.restore_account(
        db=db, email=user.email, password=PASSWORD
    )

    assert result.products_restored == 1


@pytest.mark.asyncio
async def test_restore_clears_all_three_lifecycle_fields():
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(0)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    values = _values_of(
        next(u for u in _updates(db) if _target_table(u) == "tbl_users")
    )
    assert values["is_active"] is True
    assert values["deleted_at"] is None
    assert values["purge_after"] is None


# ---------------------------------------------------------------------------
# the recovery window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restore_after_purge_after_is_refused():
    """Past the window the purge may already have erased blobs."""
    deleted_at = datetime.now(timezone.utc) - timedelta(days=31)
    user = _user(deleted_at=deleted_at, purge_after=deleted_at + timedelta(days=30))
    db = _DB([_lookup(user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email=user.email, password=PASSWORD
        )

    assert exc.value.status_code == 410
    assert _updates(db) == []
    assert db.commits == 0


@pytest.mark.asyncio
async def test_restore_refused_when_purge_after_is_null():
    """Legacy rows deleted before purge_after existed are not restorable."""
    user = _user(
        deleted_at=datetime.now(timezone.utc) - timedelta(days=2), purge_after=None
    )
    db = _DB([_lookup(user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email=user.email, password=PASSWORD
        )

    assert exc.value.status_code == 410
    assert db.commits == 0


@pytest.mark.asyncio
async def test_restore_of_a_live_account_is_refused():
    user = _user(deleted_at=None, purge_after=None, is_active=True)
    db = _DB([_lookup(user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email=user.email, password=PASSWORD
        )

    assert exc.value.status_code == 409
    assert db.commits == 0


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_password_is_refused_and_writes_nothing():
    user = _pending_user()
    db = _DB([_lookup(user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email=user.email, password="not-the-password"
        )

    assert exc.value.status_code == 401
    assert _updates(db) == []
    assert db.commits == 0


@pytest.mark.asyncio
async def test_missing_credential_is_refused():
    user = _pending_user()
    db = _DB([_lookup(user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(db=db, email=user.email, password=None)

    assert exc.value.status_code == 401
    assert db.commits == 0


@pytest.mark.asyncio
async def test_unknown_email_is_refused_without_disclosing_anything():
    """Same 401 as bad credentials, so deleted addresses cannot be probed."""
    db = _DB([_lookup(None)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email="nobody@example.com", password=PASSWORD
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid credentials."
    assert db.commits == 0


@pytest.mark.asyncio
async def test_google_user_restores_with_a_matching_identity():
    """The retained AuthIdentity row is the OAuth restore credential."""
    user = _pending_user(password_hash=None)
    db = _DB([
        _lookup(user),
        _Result(first_row=(uuid.uuid4(),)),  # AuthIdentity match
        _user_update_ok(user),
        _products_restored(2),
    ])

    result = await AccountService.restore_account(
        db=db, email=user.email, google_sub=GOOGLE_SUB
    )

    assert result.products_restored == 2


@pytest.mark.asyncio
async def test_google_user_without_matching_identity_is_refused():
    user = _pending_user(password_hash=None)
    db = _DB([_lookup(user), _Result(first_row=None)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email=user.email, google_sub="someone-elses-sub"
        )

    assert exc.value.status_code == 401
    assert db.commits == 0


@pytest.mark.asyncio
async def test_google_user_cannot_restore_with_a_password():
    user = _pending_user(password_hash=None)
    db = _DB([_lookup(user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email=user.email, password=PASSWORD
        )

    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# the product fingerprint — the part that must not be wrong
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_products_matching_the_deletion_timestamp_are_restored():
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(3)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    product_update = next(
        u for u in _updates(db) if _target_table(u) == "tbl_products"
    )
    where_sql = str(product_update.whereclause)
    # Equality against the captured fingerprint, not a blanket IS NOT NULL.
    assert "deleted_at = " in where_sql
    assert "IS NOT NULL" not in where_sql


@pytest.mark.asyncio
async def test_fingerprint_is_the_users_original_deleted_at():
    user = _pending_user()
    original = user.deleted_at
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(0)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    product_update = next(
        u for u in _updates(db) if _target_table(u) == "tbl_products"
    )
    compiled = product_update.compile()
    assert original in compiled.params.values()


@pytest.mark.asyncio
async def test_products_deleted_at_a_different_time_are_left_alone():
    """A product the user deleted themselves must stay deleted.

    The stub reports zero rows matched, which is what the database returns when
    the only soft-deleted products carry a different timestamp.
    """
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(0)])

    result = await AccountService.restore_account(
        db=db, email=user.email, password=PASSWORD
    )

    assert result.products_restored == 0
    assert db.commits == 1


@pytest.mark.asyncio
async def test_restore_only_clears_deleted_at_on_products():
    """status must survive: an archived product comes back archived."""
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(1)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    product_update = next(
        u for u in _updates(db) if _target_table(u) == "tbl_products"
    )
    values = _values_of(product_update)
    assert values == {"deleted_at": None}
    assert "status" not in values
    assert "updated_date" not in values


@pytest.mark.asyncio
async def test_restore_is_scoped_to_the_owner():
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(1)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    product_update = next(
        u for u in _updates(db) if _target_table(u) == "tbl_products"
    )
    assert "created_by" in str(product_update.whereclause)


# ---------------------------------------------------------------------------
# concurrency and atomicity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_takes_a_row_lock():
    """FOR UPDATE serialises a restore racing a second restore or a delete."""
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(0)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    lookup = db.statements[0]
    assert lookup._for_update_arg is not None


@pytest.mark.asyncio
async def test_lookup_does_not_filter_out_soft_deleted_users():
    """Every other user lookup hides exactly the rows restore needs."""
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(0)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    where_sql = str(db.statements[0].whereclause)
    assert "deleted_at" not in where_sql


@pytest.mark.asyncio
async def test_losing_a_concurrent_restore_race_writes_no_products():
    """The guarded UPDATE matched nothing: another request already restored."""
    user = _pending_user()
    db = _DB([_lookup(user), _Result(first_row=None)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_account(
            db=db, email=user.email, password=PASSWORD
        )

    assert exc.value.status_code == 409
    assert not any(_target_table(u) == "tbl_products" for u in _updates(db))
    assert db.commits == 0


@pytest.mark.asyncio
async def test_user_update_is_guarded_against_double_restore():
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(0)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    user_update = next(
        u for u in _updates(db) if _target_table(u) == "tbl_users"
    )
    where_sql = str(user_update.whereclause)
    assert "deleted_at IS NOT NULL" in where_sql
    assert "purge_after" in where_sql


@pytest.mark.asyncio
async def test_everything_commits_once():
    """One transaction: a failure part-way leaves the account fully deleted."""
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(5)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    assert db.commits == 1
    # user restored before products, both inside the one transaction
    tables = [_target_table(u) for u in _updates(db)]
    assert tables == ["tbl_users", "tbl_products"]


@pytest.mark.asyncio
async def test_a_failure_after_the_user_update_commits_nothing():
    """The product UPDATE raising must not leave a half-restored account."""
    user = _pending_user()

    class _Boom(_DB):
        async def execute(self, statement, *a, **k):
            self.statements.append(statement)
            if isinstance(statement, Update) and _target_table(statement) == "tbl_products":
                raise RuntimeError("database went away")
            return self._results.pop(0) if self._results else _Result()

    db = _Boom([_lookup(user), _user_update_ok(user)])

    with pytest.raises(RuntimeError):
        await AccountService.restore_account(
            db=db, email=user.email, password=PASSWORD
        )

    assert db.commits == 0


# ---------------------------------------------------------------------------
# AuthIdentity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restore_never_writes_auth_identities():
    user = _pending_user()
    db = _DB([_lookup(user), _user_update_ok(user), _products_restored(0)])

    await AccountService.restore_account(db=db, email=user.email, password=PASSWORD)

    assert not any(isinstance(s, Delete) for s in db.statements)
    assert "tbl_auth_identities" not in {
        _target_table(u) for u in _updates(db)
    }
