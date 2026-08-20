"""Deleting an account must schedule erasure, never perform it.

The whole point of the 30-day window is that the account can be put back, and an
account can only be put back if everything needed to rebuild it still exists.
So these tests pin two things:

  * the three lifecycle columns are written together — is_active false (access
    stops), deleted_at set (deletion requested), purge_after ~30 days out (when
    erasure becomes due);
  * nothing is destroyed. In particular AuthIdentity survives, because it is the
    row mapping a Google account to a user: delete it and a Google user can never
    be restored, no matter what the rest of the row says.

The repeat-request test guards the window itself. The guard lives in the UPDATE's
WHERE clause rather than in a Python pre-check, because two concurrent requests
carrying the same still-valid token would both pass a pre-check and the second
would push purge_after another 30 days into the future.

The database is stubbed. What is under test is which statements we emit and what
values they carry, not SQLAlchemy's ability to run them.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.sql import Delete, Update

from app.services.account_service import (
    ACCOUNT_RETENTION_DAYS,
    AccountService,
)

PASSWORD = "correct-horse-battery-staple"


class _Result:
    """Stands in for a SQLAlchemy Result."""

    def __init__(self, first_row=None):
        self._first = first_row

    def first(self):
        return self._first


class _DB:
    """Records every statement executed, and replays queued results in order."""

    def __init__(self, results):
        self._results = list(results)
        self.statements = []
        self.commits = 0
        self.deleted_objects = []

    async def execute(self, statement, *_args, **_kwargs):
        self.statements.append(statement)
        return self._results.pop(0) if self._results else _Result()

    async def commit(self):
        self.commits += 1

    async def delete(self, obj):  # pragma: no cover - must never be called
        self.deleted_objects.append(obj)


def _user(*, password_hash="argon2-hash"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="seller@example.com",
        password_hash=password_hash,
        deleted_at=None,
        purge_after=None,
        is_active=True,
    )


def _updates(db):
    return [s for s in db.statements if isinstance(s, Update)]


def _values_of(statement):
    """The column -> value mapping an UPDATE will actually write.

    SET values arrive wrapped as BindParameter, so unwrap to the literal the
    database would receive.
    """
    return {
        col.name: getattr(val, "value", val)
        for col, val in statement._values.items()
    }


def _target_table(statement):
    return statement.table.name


@pytest.fixture(autouse=True)
def _accept_password(monkeypatch):
    """Identity verification is covered by its own cases; default to a match."""
    monkeypatch.setattr(
        "app.services.account_service.verify_password",
        lambda password, hashed: password == PASSWORD,
    )


@pytest.fixture(autouse=True)
def _no_paid_subscription(monkeypatch):
    """Default every user here to the free plan.

    These tests are about what deletion writes to the database. The paid-
    subscription block has its own file
    (test_account_deletion_subscription_guard.py); leaving the real query live
    here would consume the stub results queued for the deletion statements.
    """

    async def _none(db, user_id):
        return None

    monkeypatch.setattr(
        "app.services.account_service.get_blocking_subscription", _none
    )


# ---------------------------------------------------------------------------
# (a) an active user requests deletion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_user_can_request_deletion():
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    result = await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    assert result.already_pending is False
    assert db.commits == 1


# ---------------------------------------------------------------------------
# (b) (c) (d) the three lifecycle columns are written together
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deletion_writes_all_three_lifecycle_columns():
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    before = datetime.now(timezone.utc)
    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )
    after = datetime.now(timezone.utc)

    user_update = next(u for u in _updates(db) if _target_table(u) == "tbl_users")
    values = _values_of(user_update)

    # (b) access is revoked
    assert values["is_active"] is False

    # (c) deletion is timestamped, and with an aware UTC value
    assert isinstance(values["deleted_at"], datetime)
    assert values["deleted_at"].tzinfo is not None
    assert before <= values["deleted_at"] <= after

    # (d) erasure falls due ~30 days out
    window = values["purge_after"] - values["deleted_at"]
    assert window == timedelta(days=ACCOUNT_RETENTION_DAYS)
    assert values["purge_after"] > datetime.now(timezone.utc) + timedelta(days=29)


@pytest.mark.asyncio
async def test_purge_after_is_thirty_days_not_some_other_window():
    """Pins the number itself — a silent change to 3 or 300 days is a real bug."""
    assert ACCOUNT_RETENTION_DAYS == 30


# ---------------------------------------------------------------------------
# (e) AuthIdentity survives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_identity_is_never_deleted():
    """A Google account cannot be restored once its identity row is gone."""
    user = _user(password_hash=None)
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=None, confirmation="DELETE MY ACCOUNT"
    )

    tables_touched = {
        _target_table(s) for s in db.statements if isinstance(s, Update)
    }
    assert "tbl_auth_identities" not in tables_touched
    assert not any(isinstance(s, Delete) for s in db.statements)


# ---------------------------------------------------------------------------
# (f) a repeat request must not extend the window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeat_request_does_not_extend_purge_after():
    """The guarded UPDATE matches no row, so the original dates stand."""
    user = _user()
    original_deleted_at = datetime.now(timezone.utc) - timedelta(days=10)
    original_purge_after = original_deleted_at + timedelta(days=ACCOUNT_RETENTION_DAYS)

    db = _DB([
        _Result(None),                                        # guarded UPDATE: no match
        _Result((original_deleted_at, original_purge_after)),  # read-back of existing
    ])

    result = await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    assert result.already_pending is True
    assert result.deleted_at == original_deleted_at
    assert result.purge_after == original_purge_after
    # Still ~20 days away, not pushed back out to 30.
    assert result.purge_after < datetime.now(timezone.utc) + timedelta(days=21)


@pytest.mark.asyncio
async def test_repeat_request_does_not_resoft_delete_products():
    """Products were already soft-deleted; re-stamping them would rewrite history."""
    user = _user()
    prior = datetime.now(timezone.utc) - timedelta(days=10)
    db = _DB([
        _Result(None),
        _Result((prior, prior + timedelta(days=ACCOUNT_RETENTION_DAYS))),
    ])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    assert not any(_target_table(u) == "tbl_products" for u in _updates(db))


@pytest.mark.asyncio
async def test_idempotency_guard_is_in_the_where_clause():
    """A Python pre-check would let two concurrent requests both write."""
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    user_update = next(u for u in _updates(db) if _target_table(u) == "tbl_users")
    where_sql = str(user_update.whereclause)
    assert "deleted_at IS NULL" in where_sql


# ---------------------------------------------------------------------------
# (g) nothing is permanently deleted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_row_is_ever_hard_deleted():
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    assert not any(isinstance(s, Delete) for s in db.statements)
    assert db.deleted_objects == []


@pytest.mark.asyncio
async def test_products_are_soft_deleted_not_removed():
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    product_update = next(
        u for u in _updates(db) if _target_table(u) == "tbl_products"
    )
    values = _values_of(product_update)
    assert isinstance(values["deleted_at"], datetime)


@pytest.mark.asyncio
async def test_cascade_does_not_touch_product_update_metadata():
    """Overwriting updated_date/updated_by would be unrecoverable on restore.

    deleted_at already records when the cascade ran. Stamping updated_date too
    destroys the product's real edit history and flattens the list ordering
    (products_repo sorts on it), with nothing left to restore it from.
    """
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    product_update = next(
        u for u in _updates(db) if _target_table(u) == "tbl_products"
    )
    assert set(_values_of(product_update)) == {"deleted_at"}


@pytest.mark.asyncio
async def test_user_and_products_share_one_deletion_timestamp():
    """The fingerprint restore relies on: both UPDATEs bind the same instant.

    If a future edit recomputes `now` for the product cascade, the timestamps
    diverge and restore silently brings back nothing.
    """
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    user_stamp = _values_of(
        next(u for u in _updates(db) if _target_table(u) == "tbl_users")
    )["deleted_at"]
    product_stamp = _values_of(
        next(u for u in _updates(db) if _target_table(u) == "tbl_products")
    )["deleted_at"]

    assert user_stamp == product_stamp


@pytest.mark.asyncio
async def test_only_users_and_products_are_written():
    """Subscriptions, licenses, payments and assets must all survive intact."""
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    written = {_target_table(u) for u in _updates(db)}
    assert written == {"tbl_users", "tbl_products"}


@pytest.mark.asyncio
async def test_everything_lands_in_a_single_transaction():
    """One commit, so a mid-flight failure leaves the account fully intact."""
    user = _user()
    now = datetime.now(timezone.utc)
    db = _DB([_Result((now, now + timedelta(days=ACCOUNT_RETENTION_DAYS)))])

    await AccountService.delete_account(
        db=db, user=user, password=PASSWORD, confirmation=None
    )

    assert db.commits == 1


# ---------------------------------------------------------------------------
# identity verification still gates the whole flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_password_writes_nothing():
    from fastapi import HTTPException

    user = _user()
    db = _DB([])

    with pytest.raises(HTTPException) as exc:
        await AccountService.delete_account(
            db=db, user=user, password="not-the-password", confirmation=None
        )

    assert exc.value.status_code == 400
    assert db.statements == []
    assert db.commits == 0


@pytest.mark.asyncio
async def test_google_user_without_confirmation_writes_nothing():
    from fastapi import HTTPException

    user = _user(password_hash=None)
    db = _DB([])

    with pytest.raises(HTTPException) as exc:
        await AccountService.delete_account(
            db=db, user=user, password=None, confirmation=None
        )

    assert exc.value.status_code == 400
    assert db.statements == []
    assert db.commits == 0


  














    