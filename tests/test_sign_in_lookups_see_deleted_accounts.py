"""The lookups on the sign-in path must SEE accounts pending deletion.

Restoring on sign-in only works if sign-in can find the account at all, and
every lookup it goes through used to filter `deleted_at IS NULL`. That filter is
right almost everywhere in this codebase and wrong on exactly these paths:

  * AuthService.authenticate_email — filtered, so a correct password on a
    deleted account came back as "invalid email or password";
  * get_or_create_google_user, identity branch — filtered, so it fell through to
    the create branch and hit tbl_users' UNIQUE(email): a 500;
  * get_or_create_google_user, email branch — the only branch that can catch an
    account deleted BEFORE deletion started keeping AuthIdentity. Those rows have
    no identity to match on, and f61a03d7b8e4 backfilled purge_after precisely to
    give them a window they could not otherwise use.

These tests exist because re-adding any of those filters breaks the feature
*silently*: every restore test would still pass, because restore would simply
never be reached.

The last test is the other half of the rule — get_user_by_email must KEEP its
filter, or signup, the OTP gate and forgot-password start acting on accounts
that are on their way out.
"""

import uuid
from types import SimpleNamespace

import pytest

from app.models.models import AuthIdentity
from app.services.auth_service import AuthService


class _Result:
    def __init__(self, scalar=None):
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar


class _DB:
    """Replays queued results in order and records the SQL emitted."""

    def __init__(self, *scalars):
        self._scalars = list(scalars)
        self.sql = []
        self.added = []
        self.commits = 0

    async def execute(self, statement, *_args, **_kwargs):
        self.sql.append(str(statement))
        return _Result(self._scalars.pop(0) if self._scalars else None)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


def _deleted_user():
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="seller@example.com",
        password_hash="argon2-hash",
        deleted_at="2026-08-01T00:00:00+00:00",
        purge_after="2026-08-31T00:00:00+00:00",
        is_active=False,
        name="Seller",
        avatar_url=None,
    )


# ---------------------------------------------------------------------------
# email / password
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_email_returns_an_account_pending_deletion(monkeypatch):
    monkeypatch.setattr("app.services.auth_service.verify_password", lambda *_: True)
    user = _deleted_user()
    db = _DB(user)

    found = await AuthService.authenticate_email(db, "seller@example.com", "pw")

    assert found is user, "a correct password on a deleted account must not be a 401"


@pytest.mark.asyncio
async def test_authenticate_email_does_not_filter_deleted_at(monkeypatch):
    monkeypatch.setattr("app.services.auth_service.verify_password", lambda *_: True)
    db = _DB(_deleted_user())

    await AuthService.authenticate_email(db, "seller@example.com", "pw")

    assert "deleted_at IS NULL" not in db.sql[0]


@pytest.mark.asyncio
async def test_a_wrong_password_is_still_refused(monkeypatch):
    """Unfiltering the lookup must not weaken the credential check."""
    monkeypatch.setattr("app.services.auth_service.verify_password", lambda *_: False)
    db = _DB(_deleted_user())

    assert await AuthService.authenticate_email(db, "seller@example.com", "no") is None


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_identity_branch_returns_a_deleted_user_instead_of_creating():
    """Deletion keeps AuthIdentity so this branch can match. It must be allowed to."""
    user = _deleted_user()
    db = _DB(SimpleNamespace(user_id=user.id), user)

    found, is_new = await AuthService.get_or_create_google_user(
        db, google_id="google-sub-1", email=user.email, name="Seller"
    )

    assert found is user
    assert is_new is False
    assert db.added == [], "must not create a second user for the same address"


@pytest.mark.asyncio
async def test_a_legacy_google_user_with_no_identity_row_is_matched_by_email():
    """Accounts deleted before AuthIdentity was retained.

    Their identity row was hard-deleted, so only the email branch can find them.
    Google has already verified the caller owns the address, and the link step
    re-creates the identity row the old deletion removed.
    """
    user = _deleted_user()
    db = _DB(None, user)  # no identity, then the user found by email

    found, is_new = await AuthService.get_or_create_google_user(
        db, google_id="google-sub-1", email=user.email, name="Seller"
    )

    assert found is user
    assert is_new is False
    assert len(db.added) == 1
    assert isinstance(db.added[0], AuthIdentity)
    assert db.added[0].user_id == user.id


# ---------------------------------------------------------------------------
# the filter that must stay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_user_by_email_still_hides_deleted_accounts():
    """Signup, the OTP gate and forgot-password all depend on this filter."""
    db = _DB(None)

    await AuthService.get_user_by_email(db, "seller@example.com")

    assert "deleted_at IS NULL" in db.sql[0]
