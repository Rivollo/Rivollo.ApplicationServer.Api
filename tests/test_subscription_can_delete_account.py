"""GET /subscriptions/me tells the frontend whether to offer Delete Account.

The flag is a hint, not a gate. AccountService.delete_account runs the same check
and returns 409 whatever any client believes, so a wrong value here costs a
confusing click, never an unguarded deletion.

What it must not do is disagree with that guard. The two would drift the moment
someone re-expressed "active paid subscription" a second time, so the route calls
has_active_paid_subscription directly — the same function the guard calls — and
these tests pin that wiring alongside the five states the lead specified.

The value is derived through the real predicate against stubbed subscription
rows, so a change to what counts as "paid and running" shows up here too.
"""

import inspect
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.api.routes import subscriptions as routes
from app.models.subscription_enums import SubscriptionStatus
from app.schemas.subscriptions import QuotaInfo, SubscriptionMe, TrialInfo
from app.services import subscription_guard as guard


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: self._rows)


class _DB:
    """Answers the guard's single subscription query.

    Applies the same two conditions the real WHERE clause does — gateway-backed,
    and in a billable status — so a free-plan row is filtered out here exactly as
    Postgres would filter it. Without that the fake would hand every row to the
    Python expiry check and report free users as blocked, which is a property of
    the stub and not of the code under test.

    Expiry is deliberately NOT simulated: that half runs in Python and belongs to
    the real predicate.
    """

    def __init__(self, rows):
        self._rows = [
            r
            for r in rows
            if r.razorpay_subscription_id is not None
            and r.status in guard.PAID_ACTIVE_STATUSES
        ]

    async def execute(self, *_a, **_k):
        return _ScalarResult(self._rows)


def _sub(*, status=SubscriptionStatus.ACTIVE, rz_id="sub_ABC123", period_end_days=12):
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        razorpay_subscription_id=rz_id,
        status=status,
        current_period_end=(
            None
            if period_end_days is None
            else datetime.now(timezone.utc) + timedelta(days=period_end_days)
        ),
    )


def _subscription_me(plan="free"):
    return SubscriptionMe(
        plan=plan,
        trial=TrialInfo(active=False, daysRemaining=0, startedAt=None),
        quotas={"products": QuotaInfo(used=0, limit=2).model_dump()},
    )


async def _call_endpoint(monkeypatch, rows, plan="free"):
    """Drive the route with a stubbed service and real guard predicate."""

    async def _fake_service(db, user_id):
        return _subscription_me(plan)

    monkeypatch.setattr(
        routes.SubscriptionService, "get_user_subscription", _fake_service
    )
    user = SimpleNamespace(id=uuid.uuid4())
    return await routes.get_my_subscription(current_user=user, db=_DB(rows))


# ---------------------------------------------------------------------------
# the five states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_paid_subscription_cannot_delete(monkeypatch):
    body = await _call_endpoint(monkeypatch, [_sub()], plan="pro")
    assert body["data"]["canDeleteAccount"] is False


@pytest.mark.asyncio
async def test_past_due_paid_subscription_cannot_delete(monkeypatch):
    """The mandate is live and Razorpay is still retrying the charge."""
    body = await _call_endpoint(
        monkeypatch, [_sub(status=SubscriptionStatus.PAST_DUE)], plan="pro"
    )
    assert body["data"]["canDeleteAccount"] is False


@pytest.mark.asyncio
async def test_expired_paid_subscription_can_delete(monkeypatch):
    body = await _call_endpoint(monkeypatch, [_sub(period_end_days=-1)], plan="pro")
    assert body["data"]["canDeleteAccount"] is True


@pytest.mark.asyncio
async def test_free_user_can_delete(monkeypatch):
    """Free-plan rows carry no gateway id, so they never block."""
    body = await _call_endpoint(monkeypatch, [_sub(rz_id=None)])
    assert body["data"]["canDeleteAccount"] is True


@pytest.mark.asyncio
async def test_user_with_no_subscription_can_delete(monkeypatch):
    body = await _call_endpoint(monkeypatch, [])
    assert body["data"]["canDeleteAccount"] is True


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_survives_exclude_none(monkeypatch):
    """The route dumps with exclude_none=True.

    A bool survives it; typing the field Optional[bool] = None would have made it
    vanish for exactly the users it matters to, and the frontend would read the
    absent key as falsy.
    """
    body = await _call_endpoint(monkeypatch, [_sub()], plan="pro")
    assert "canDeleteAccount" in body["data"]


@pytest.mark.asyncio
async def test_flag_is_camel_case_like_its_neighbours(monkeypatch):
    body = await _call_endpoint(monkeypatch, [])
    assert "canDeleteAccount" in body["data"]
    assert "can_delete_account" not in body["data"]


def test_schema_default_leaves_deletion_available():
    """An unset value must not trap someone in an account they cannot delete."""
    assert _subscription_me().can_delete_account is True


# ---------------------------------------------------------------------------
# it cannot drift from the deletion guard
# ---------------------------------------------------------------------------


def test_route_uses_the_same_predicate_as_the_deletion_guard():
    source = inspect.getsource(routes.get_my_subscription)
    assert "has_active_paid_subscription" in source


def test_route_does_not_reimplement_the_rule():
    """No second definition of 'active paid subscription' in the route."""
    source = inspect.getsource(routes.get_my_subscription)
    for reinvented in ("PAST_DUE", "razorpay_subscription_id", "_is_expired"):
        assert reinvented not in source


@pytest.mark.asyncio
async def test_flag_matches_the_guard_for_every_state(monkeypatch):
    """Whatever the guard says, the endpoint reports its negation. No exceptions."""
    states = [
        [],
        [_sub()],
        [_sub(status=SubscriptionStatus.PAST_DUE)],
        [_sub(period_end_days=-1)],
        [_sub(rz_id=None)],
        [_sub(period_end_days=None)],
    ]
    for rows in states:
        blocked = await guard.has_active_paid_subscription(_DB(rows), uuid.uuid4())
        body = await _call_endpoint(monkeypatch, rows)
        assert body["data"]["canDeleteAccount"] is (not blocked)
