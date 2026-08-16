"""Self-serve cancellation must not double-charge the customer's trust.

cancel_subscription() deliberately leaves a cycle-end cancellation in status
ACTIVE, because the customer has paid through the end of the period and only
the subscription.cancelled webhook confirms the end. That intermediate state is
the whole reason `cancel_at_period_end` exists, and these tests pin the two
things that make it correct:

  * the flag is set on every successful cancellation, so the UI can stop
    promising a renewal that will not happen;
  * a second cycle-end cancel does not reach Razorpay, because cancelling an
    already-cancelled subscription is a 400 from them — surfaced as a failure to
    a customer whose cancellation actually worked.

The Razorpay call is stubbed. What is under test is our state machine around
it, not httpx.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.models.subscription_enums import SubscriptionStatus
from app.services import razorpay_subscription_service as svc


class _Result:
    def __init__(self, one=None):
        self._one = one

    def scalar_one_or_none(self):
        return self._one


class _DB:
    def __init__(self, subscription):
        self._subscription = subscription
        self.committed = False

    async def execute(self, *_args, **_kwargs):
        return _Result(self._subscription)

    async def commit(self):
        self.committed = True


def _subscription(*, cancel_at_period_end=False, period_end=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        razorpay_subscription_id="sub_ABC123",
        status=SubscriptionStatus.ACTIVE,
        current_period_end=period_end
        or datetime.now(timezone.utc) + timedelta(days=12),
        cancel_at_period_end=cancel_at_period_end,
        updated_date=None,
    )


@pytest.fixture
def razorpay_ok(monkeypatch):
    """Stub the Razorpay cancel call and record whether it was made."""
    calls = []

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, **kwargs):
            calls.append((url, kwargs.get("json")))
            return _Response()

    monkeypatch.setattr(svc, "_check_credentials", lambda: None)
    monkeypatch.setattr(svc.httpx, "AsyncClient", _Client)
    return calls


@pytest.mark.asyncio
async def test_cycle_end_cancel_sets_the_flag_and_keeps_access(razorpay_ok):
    """The row stays ACTIVE — the flag is the only thing marking the change."""
    subscription = _subscription()
    db = _DB(subscription)

    result = await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=True
    )

    assert subscription.cancel_at_period_end is True
    assert subscription.status == SubscriptionStatus.ACTIVE
    assert result["accessUntil"] == subscription.current_period_end
    assert db.committed
    assert len(razorpay_ok) == 1


@pytest.mark.asyncio
async def test_immediate_cancel_also_sets_the_flag(razorpay_ok):
    """Otherwise the flag would mean 'cancelled at cycle end', not 'cancelled'."""
    subscription = _subscription()
    db = _DB(subscription)

    await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=False
    )

    assert subscription.cancel_at_period_end is True
    assert subscription.status == SubscriptionStatus.CANCELED


@pytest.mark.asyncio
async def test_second_cycle_end_cancel_does_not_call_razorpay(razorpay_ok):
    """A retry or a second tab reports the existing state instead of erroring."""
    period_end = datetime.now(timezone.utc) + timedelta(days=5)
    subscription = _subscription(cancel_at_period_end=True, period_end=period_end)
    db = _DB(subscription)

    result = await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=True
    )

    assert razorpay_ok == []
    assert result["cancelled"] is True
    assert result["accessUntil"] == period_end
    assert "already scheduled" in result["message"]


@pytest.mark.asyncio
async def test_immediate_cancel_after_scheduling_still_reaches_razorpay(razorpay_ok):
    """Escalating from 'cancel later' to 'cancel now' is a real change."""
    subscription = _subscription(cancel_at_period_end=True)
    db = _DB(subscription)

    await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=False
    )

    assert len(razorpay_ok) == 1
    assert razorpay_ok[0][1] == {"cancel_at_cycle_end": 0}
    assert subscription.status == SubscriptionStatus.CANCELED
