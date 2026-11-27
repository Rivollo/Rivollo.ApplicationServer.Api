"""A plan counts only while its subscription is ACTIVE and inside its period.

Starting a checkout writes the new plan onto the user's subscription row while
it is still PENDING. LicensingService.get_user_plan_code read the plan through
the licence without looking at the subscription's status, so an unpaid or
abandoned checkout was treated as Pro (Pro-only 3D models, USDZ conversion).
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.subscription_enums import SubscriptionStatus
from app.services.licensing_service import LicensingService


async def _plan_code(row):
    db = MagicMock()
    result = MagicMock()
    result.first.return_value = row
    db.execute = AsyncMock(return_value=result)
    licence = SimpleNamespace(subscription_id=uuid.uuid4())
    with patch.object(
        LicensingService, "get_active_license", AsyncMock(return_value=licence)
    ):
        return await LicensingService.get_user_plan_code(db, uuid.uuid4())


FUTURE = datetime.now(timezone.utc) + timedelta(days=10)
PAST = datetime.now(timezone.utc) - timedelta(days=1)


async def test_active_subscription_gives_its_plan():
    assert await _plan_code(("pro", SubscriptionStatus.ACTIVE, FUTURE)) == "pro"


async def test_no_period_end_does_not_expire():
    assert await _plan_code(("pro", SubscriptionStatus.ACTIVE, None)) == "pro"


@pytest.mark.parametrize(
    "status",
    [SubscriptionStatus.PENDING, SubscriptionStatus.CANCELED, SubscriptionStatus.PAST_DUE],
)
async def test_unpaid_or_ended_subscription_is_free(status):
    assert await _plan_code(("pro", status, FUTURE)) == "free"


async def test_expired_period_is_free():
    assert await _plan_code(("pro", SubscriptionStatus.ACTIVE, PAST)) == "free"
    assert await _plan_code(("pro", SubscriptionStatus.ACTIVE, PAST.replace(tzinfo=None))) == "free"


async def test_missing_subscription_is_free():
    assert await _plan_code(None) == "free"


def test_both_checkouts_refuse_a_second_subscription():
    import inspect

    from app.services import razorpay_subscription_service, usd_subscription_service

    for fn, gateway_call in (
        (razorpay_subscription_service.create_subscription, "client.post("),
        (usd_subscription_service.create_usd_subscription, "_call_razorpay("),
    ):
        source = inspect.getsource(fn)
        assert "HTTP_409_CONFLICT" in source
        # refused before anything is created at Razorpay
        assert source.index("get_blocking_subscription(db") < source.index(gateway_call)
