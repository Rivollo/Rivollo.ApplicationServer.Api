"""Signing in inside the recovery window is how an account comes back.

POST /auth/account/restore still exists, but nobody has to call it. Presenting
the right credential IS the proof restore asks for, so requiring a second,
separate step would only strand people who deleted an account they turned out
to still want.

restore_on_sign_in shares its writes with restore_account via _apply_restore, so
the fingerprint and locking guarantees are already covered by
test_account_restore.py and are not re-litigated here. What these tests pin is
what is *different* about the sign-in entry point:

  * it performs no identity check of its own — the caller already did, and a
    second one here would be a second bcrypt on every restore;
  * losing a concurrent race is success, not a 409. A sign-in only needs the
    account live, and the request that won left it exactly that way;
  * it must clear is_active. Deletion sets it false, so a restore that only
    cleared deleted_at would sail past the deletion check and then be refused
    with "contact support" — the account restored, the sign-in still broken;
  * an ordinary sign-in must be untouched: no lock, no writes, no commit, on
    every successful login in the product.

The database is stubbed. What is under test is which statements we emit.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.sql import Update

from app.services.account_service import AccountService


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

    async def execute(self, statement, *_args, **_kwargs):
        self.statements.append(statement)
        return self._results.pop(0) if self._results else _Result()

    async def commit(self):
        self.commits += 1


def _user(*, deleted_at, purge_after, is_active=False):
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="seller@example.com",
        password_hash="argon2-hash",
        deleted_at=deleted_at,
        purge_after=purge_after,
        is_active=is_active,
    )


def _pending_user(*, days_ago=3, window_days=30):
    deleted_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return _user(
        deleted_at=deleted_at,
        purge_after=deleted_at + timedelta(days=window_days),
    )


def _live_user():
    return _user(deleted_at=None, purge_after=None, is_active=True)


def _updates(db, table=None):
    found = [s for s in db.statements if isinstance(s, Update)]
    if table is None:
        return found
    return [s for s in found if s.table.name == table]


def _values_of(statement):
    return {
        col.name: getattr(val, "value", val)
        for col, val in statement._values.items()
    }


def _restore_succeeds(user, product_count=0):
    """Queued results for lookup -> user update -> product update."""
    return [
        _Result(scalar=user),
        _Result(first_row=(user.id,)),
        _Result(rows=[(uuid.uuid4(),) for _ in range(product_count)]),
    ]


# ---------------------------------------------------------------------------
# the ordinary sign-in must pay nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_live_account_is_left_completely_alone():
    """Every successful login in the product takes this path."""
    db = _DB([])

    assert await AccountService.restore_on_sign_in(db, _live_user()) is None
    assert db.statements == []
    assert db.commits == 0


# ---------------------------------------------------------------------------
# inside the window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signing_in_inside_the_window_restores_the_account():
    user = _pending_user()
    db = _DB(_restore_succeeds(user, product_count=2))

    result = await AccountService.restore_on_sign_in(db, user)

    assert result is not None
    assert result.user_id == user.id
    assert result.products_restored == 2
    assert db.commits == 1


@pytest.mark.asyncio
async def test_restore_clears_is_active():
    """Deletion sets is_active false; clearing deleted_at alone is not enough.

    Miss this and the account is restored but the very next check in the login
    handler answers 403 "contact support" — a self-service action turned into a
    support ticket, for a user who did everything right.
    """
    user = _pending_user()
    db = _DB(_restore_succeeds(user))

    await AccountService.restore_on_sign_in(db, user)

    values = _values_of(_updates(db, "tbl_users")[0])
    assert values["is_active"] is True
    assert values["deleted_at"] is None
    assert values["purge_after"] is None


@pytest.mark.asyncio
async def test_restore_leaves_the_account_deletable_again():
    """The round trip has to close: deleted, signed back in, deletable again.

    delete_account's guard is `deleted_at IS NULL`, so a cleared deleted_at is
    exactly what lets a restored user delete a second time and get a fresh
    window. Asserted rather than assumed, because it is the half of the cycle a
    reader is most likely to take on trust.
    """
    user = _pending_user()
    db = _DB(_restore_succeeds(user))

    await AccountService.restore_on_sign_in(db, user)

    assert _values_of(_updates(db, "tbl_users")[0])["deleted_at"] is None


@pytest.mark.asyncio
async def test_the_last_hours_of_the_window_still_work():
    user = _pending_user(days_ago=30)
    user.purge_after = datetime.now(timezone.utc) + timedelta(hours=1)
    db = _DB(_restore_succeeds(user))

    assert await AccountService.restore_on_sign_in(db, user) is not None


@pytest.mark.asyncio
async def test_sign_in_does_not_re_verify_identity():
    """The caller already proved ownership; a second bcrypt here is waste.

    Emitting only the lookup and the two updates is what proves it: no
    AuthIdentity SELECT, no extra round trip.
    """
    user = _pending_user()
    db = _DB(_restore_succeeds(user))

    await AccountService.restore_on_sign_in(db, user)

    assert len(db.statements) == 3


# ---------------------------------------------------------------------------
# past the window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signing_in_after_the_window_is_refused():
    """The purge job may already have erased data behind this account."""
    user = _pending_user(days_ago=40)
    db = _DB([_Result(scalar=user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_on_sign_in(db, user)

    assert exc.value.status_code == 403
    assert _updates(db) == []
    assert db.commits == 0


@pytest.mark.asyncio
async def test_a_null_purge_after_is_refused_rather_than_guessed():
    user = _user(
        deleted_at=datetime.now(timezone.utc) - timedelta(days=1),
        purge_after=None,
    )
    db = _DB([_Result(scalar=user)])

    with pytest.raises(HTTPException) as exc:
        await AccountService.restore_on_sign_in(db, user)

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_an_extended_purge_after_keeps_an_old_account_restorable():
    """purge_after is the authority, not deleted_at + 30 days.

    That is what makes a legal hold or a support extension possible: write a
    future date to the one row, no code change.
    """
    user = _user(
        deleted_at=datetime.now(timezone.utc) - timedelta(days=120),
        purge_after=datetime.now(timezone.utc) + timedelta(days=30),
    )
    db = _DB(_restore_succeeds(user))

    assert await AccountService.restore_on_sign_in(db, user) is not None


# ---------------------------------------------------------------------------
# concurrency: losing is not an error here
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_locked_read_decides_not_the_object_passed_in():
    """The caller's object can be stale; the FOR UPDATE re-read cannot."""
    stale = _pending_user()
    already_live = _live_user()
    already_live.id = stale.id
    db = _DB([_Result(scalar=already_live)])

    assert await AccountService.restore_on_sign_in(db, stale) is None
    assert _updates(db) == []
    assert db.commits == 0


@pytest.mark.asyncio
async def test_a_vanished_row_signs_nobody_in_and_writes_nothing():
    user = _pending_user()
    db = _DB([_Result(scalar=None)])

    assert await AccountService.restore_on_sign_in(db, user) is None
    assert _updates(db) == []


@pytest.mark.asyncio
async def test_losing_the_guarded_update_writes_no_products():
    """Unlike restore_account this is not a 409 — the account is live already."""
    user = _pending_user()
    db = _DB([_Result(scalar=user), _Result(first_row=None)])

    assert await AccountService.restore_on_sign_in(db, user) is None
    assert _updates(db, "tbl_products") == []
    assert db.commits == 0


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_deleted_at_is_written_on_products():
    """status is untouched, so an archived product does not silently republish."""
    user = _pending_user()
    db = _DB(_restore_succeeds(user, product_count=1))

    await AccountService.restore_on_sign_in(db, user)

    assert _values_of(_updates(db, "tbl_products")[0]) == {"deleted_at": None}


@pytest.mark.asyncio
async def test_products_are_matched_on_the_deletion_fingerprint():
    user = _pending_user()
    db = _DB(_restore_succeeds(user, product_count=1))

    await AccountService.restore_on_sign_in(db, user)

    where = str(
        _updates(db, "tbl_products")[0].whereclause.compile(
            compile_kwargs={"literal_binds": True}
        )
    )
    assert str(user.deleted_at) in where
    assert user.id.hex in where  # literal_binds renders a UUID without dashes
