"""A soft-deleted account must never authenticate, and must say why.

get_current_user deliberately loads the user WITHOUT the `deleted_at IS NULL`
filter every other lookup applies, so that a self-deleted account can be told
apart from an unknown one and answered with the restore path instead of a
generic 401. That makes this function the one place where a soft-deleted User
row is in scope during authentication, and therefore the one place where a
reordered guard would hand a caller an account that is pending erasure.

These tests pin both halves of that trade:

  * the distinction is actually made — pending deletion and deactivated are
    separate messages, not one shared "contact support";
  * removing the filter did not weaken anything — no combination of column
    values gets a deleted account past the guards.

The last case is the one that matters. `deleted_at` set with `is_active` still
true is not hypothetical: three accounts in dev are in exactly that state,
left by the pre-30-day implementation. An `is_active`-only check would let
every one of them straight in.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api import deps


class _Result:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _DB:
    def __init__(self, row):
        self._row = row

    async def execute(self, *_args, **_kwargs):
        return _Result(self._row)


def _creds():
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="a-token")


def _user(*, deleted_at=None, is_active=True):
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="seller@example.com",
        deleted_at=deleted_at,
        is_active=is_active,
    )


@pytest.fixture(autouse=True)
def _valid_token(monkeypatch):
    """Token decoding is not what these tests are about; always return a subject."""
    monkeypatch.setattr(
        deps, "decode_access_token", lambda _t: {"sub": str(uuid.uuid4())}
    )


@pytest.mark.asyncio
async def test_active_user_is_returned():
    user = _user()
    got = await deps.get_current_user(_creds(), _DB(user))
    assert got is user


@pytest.mark.asyncio
async def test_pending_deletion_is_refused_with_the_restore_message():
    user = _user(deleted_at=datetime.now(timezone.utc), is_active=False)

    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_creds(), _DB(user))

    assert exc.value.status_code == 403
    assert exc.value.detail == deps.ACCOUNT_PENDING_DELETION_DETAIL


@pytest.mark.asyncio
async def test_pending_deletion_is_not_told_to_contact_support():
    """The owner put the account here and can undo it; support cannot help."""
    user = _user(deleted_at=datetime.now(timezone.utc), is_active=False)

    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_creds(), _DB(user))

    assert exc.value.detail != deps.ACCOUNT_DEACTIVATED_DETAIL
    assert "support" not in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_deactivated_account_still_points_at_support():
    """is_active false with no deleted_at is our doing, not the owner's."""
    user = _user(deleted_at=None, is_active=False)

    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_creds(), _DB(user))

    assert exc.value.status_code == 403
    assert exc.value.detail == deps.ACCOUNT_DEACTIVATED_DETAIL


@pytest.mark.asyncio
async def test_deleted_account_is_refused_even_while_still_marked_active():
    """The regression this file exists for.

    Three dev accounts carry deleted_at with is_active left true, from the
    implementation that predated the recovery window. deleted_at must be checked
    on its own, never inferred from is_active.
    """
    user = _user(deleted_at=datetime.now(timezone.utc) - timedelta(days=400), is_active=True)

    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_creds(), _DB(user))

    assert exc.value.status_code == 403
    assert exc.value.detail == deps.ACCOUNT_PENDING_DELETION_DETAIL


@pytest.mark.asyncio
async def test_unknown_user_is_a_401_not_a_403():
    """No row means no account to describe; do not leak which case it was."""
    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_creds(), _DB(None))

    assert exc.value.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deleted_at",
    [
        datetime.now(timezone.utc),
        datetime.now(timezone.utc) - timedelta(days=400),
        datetime(1970, 1, 1, tzinfo=timezone.utc),
    ],
    ids=["just-now", "long-ago", "epoch"],
)
@pytest.mark.parametrize("is_active", [True, False], ids=["active", "inactive"])
async def test_no_column_combination_lets_a_deleted_account_through(deleted_at, is_active):
    """Whatever else the row says, a non-null deleted_at is disqualifying."""
    user = _user(deleted_at=deleted_at, is_active=is_active)

    with pytest.raises(HTTPException) as exc:
        await deps.get_current_user(_creds(), _DB(user))

    assert exc.value.status_code == 403
