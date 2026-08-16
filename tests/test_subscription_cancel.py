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
from app.services import subscription_webhook_service as webhook_svc


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


def _subscription(*, cancel_at_period_end=False, period_end=None, start_at=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        razorpay_subscription_id="sub_ABC123",
        status=SubscriptionStatus.ACTIVE,
        current_period_end=period_end
        or datetime.now(timezone.utc) + timedelta(days=12),
        # None means a cycle is already running at the gateway, which is the
        # ordinary INR case. A future value is the USD intro-price flow.
        start_at=start_at,
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


@pytest.mark.asyncio
async def test_running_cycle_is_cancelled_at_cycle_end(razorpay_ok):
    """The ordinary case: a cycle is underway, so Razorpay gets the 1."""
    subscription = _subscription(start_at=None)
    db = _DB(subscription)

    await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=True
    )

    assert razorpay_ok[0][1] == {"cancel_at_cycle_end": 1}


@pytest.mark.asyncio
async def test_future_start_is_cancelled_immediately_at_the_gateway(razorpay_ok):
    """Razorpay refuses cycle-end on a subscription with no cycle running.

    "Subscription cannot be cancelled since no billing cycle is going on" is a
    hard refusal, not a warning, so asking for cycle-end here cancels nothing
    at all — in the exact window where the USD intro-price customer is most
    likely to cancel.
    """
    start_at = datetime.now(timezone.utc) + timedelta(days=30)
    subscription = _subscription(start_at=start_at, period_end=start_at)
    db = _DB(subscription)

    result = await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=True
    )

    # Immediate at the gateway: there is no cycle end to schedule against.
    assert razorpay_ok[0][1] == {"cancel_at_cycle_end": 0}

    # ...but the customer keeps every day they paid for.
    assert subscription.status == SubscriptionStatus.ACTIVE
    assert subscription.current_period_end == start_at
    assert result["accessUntil"] == start_at


@pytest.mark.asyncio
async def test_past_start_counts_as_a_running_cycle(razorpay_ok):
    """start_at in the past means the first charge already happened."""
    subscription = _subscription(
        start_at=datetime.now(timezone.utc) - timedelta(days=3)
    )
    db = _DB(subscription)

    await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=True
    )

    assert razorpay_ok[0][1] == {"cancel_at_cycle_end": 1}


@pytest.mark.asyncio
async def test_naive_start_at_does_not_crash_the_cancellation(razorpay_ok):
    """A naive timestamp must not raise mid-cancel comparing against utcnow."""
    subscription = _subscription(
        start_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=30)
    )
    db = _DB(subscription)

    await svc.cancel_subscription(
        db, user_id=uuid.uuid4(), cancel_at_cycle_end=True
    )

    assert razorpay_ok[0][1] == {"cancel_at_cycle_end": 0}


# ─────────────────────────────────────────────────────────────────────────────
# subscription.cancelled webhook
# ─────────────────────────────────────────────────────────────────────────────
#
# The gateway cancellation and the customer's access are no longer the same
# event. Cancelling a not-yet-started subscription has to happen immediately at
# Razorpay, so this webhook now arrives seconds after the customer clicks
# Cancel rather than a month later — while the period they paid for is still
# running. Revoking on arrival would take back access that was promised in
# writing on the confirm dialog.


@pytest.fixture
def webhook_stubs(monkeypatch):
    """Stub the subscription lookup and record license revocations."""
    revoked = []

    async def _revoke(_db, subscription_id, user_id):
        revoked.append((subscription_id, user_id))

    monkeypatch.setattr(webhook_svc, "_revoke_license", _revoke)
    return revoked


def _install_lookup(monkeypatch, subscription):
    async def _lookup(_db, _rz_id):
        return subscription

    monkeypatch.setattr(webhook_svc, "_get_subscription_by_rz_id", _lookup)


@pytest.mark.asyncio
async def test_cancelled_webhook_keeps_access_the_customer_paid_for(
    monkeypatch, webhook_stubs
):
    subscription = _subscription(
        cancel_at_period_end=True,
        period_end=datetime.now(timezone.utc) + timedelta(days=28),
    )
    _install_lookup(monkeypatch, subscription)

    await webhook_svc._handle_subscription_cancelled(None, "sub_ABC123", {})

    assert subscription.status == SubscriptionStatus.ACTIVE
    assert webhook_stubs == []


@pytest.mark.asyncio
async def test_cancelled_webhook_revokes_once_the_period_has_passed(
    monkeypatch, webhook_stubs
):
    subscription = _subscription(
        cancel_at_period_end=True,
        period_end=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    _install_lookup(monkeypatch, subscription)

    await webhook_svc._handle_subscription_cancelled(None, "sub_ABC123", {})

    assert subscription.status == SubscriptionStatus.CANCELED
    assert len(webhook_stubs) == 1


@pytest.mark.asyncio
async def test_gateway_initiated_cancellation_still_revokes_immediately(
    monkeypatch, webhook_stubs
):
    """Exhausted retries or a dashboard cancellation must not be deferred.

    cancel_at_period_end is written only by cancel_subscription(), so a false
    value means the cancellation did not come from the customer choosing to
    stop at the end of a period they had paid for.
    """
    subscription = _subscription(
        cancel_at_period_end=False,
        period_end=datetime.now(timezone.utc) + timedelta(days=28),
    )
    _install_lookup(monkeypatch, subscription)

    await webhook_svc._handle_subscription_cancelled(None, "sub_ABC123", {})

    assert subscription.status == SubscriptionStatus.CANCELED
    assert len(webhook_stubs) == 1
