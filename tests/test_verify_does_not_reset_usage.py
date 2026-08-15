"""Regression: verifying a subscription must never reset usage counters.

verify_subscription is driven by the three checkout-callback values the
customer's browser receives. They are static, the customer keeps them, and the
signature over "{payment_id}|{subscription_id}" stays valid indefinitely — so
the endpoint can be replayed at will. When it reset quotas, any customer could
refill their own AI credits on demand, mid-period, as often as they liked.

Resetting a genuine new billing period is the webhook's job
(_upsert_license(reset_usage=True) on subscription.activated / .charged).

These tests use stubs rather than a database because the behaviour under test is
"which attributes get written", not any query.
"""

from types import SimpleNamespace

import pytest

from app.models.subscription_enums import LicenseStatus
from app.services import razorpay_subscription_service as svc

PLAN_CREDITS = 2000
LIMITS = {"max_products": 50, "max_public_views": 25000, "max_galleries": 10}


class _Result:
    """Stands in for a SQLAlchemy Result."""

    def __init__(self, scalars_list=None, one=None):
        self._scalars = scalars_list or []
        self._one = one

    def scalars(self):
        return iter(self._scalars)

    def scalar_one_or_none(self):
        return self._one


class _FakeDB:
    """Returns queued results in order; records anything added."""

    def __init__(self, results):
        self._results = list(results)
        self.added = []

    async def execute(self, *_args, **_kwargs):
        return self._results.pop(0)

    def add(self, obj):
        self.added.append(obj)


def _subscription():
    return SimpleNamespace(
        id="sub-uuid",
        plan=SimpleNamespace(code="pro"),
        billing_interval="monthly",
    )


@pytest.fixture(autouse=True)
def _stub_plan_lookup(monkeypatch):
    """Bypass the plan/price query — irrelevant to what these tests assert."""

    async def fake_lookup(_db, _plan_code, _interval):
        return (
            SimpleNamespace(code="pro"),
            SimpleNamespace(ai_credit_limit=PLAN_CREDITS),
            LIMITS,
        )

    monkeypatch.setattr(svc, "_get_plan_with_features", fake_lookup)


async def test_verify_does_not_reset_usage_on_an_existing_licence():
    """The replay exploit: consumed credits must survive a repeated verify."""
    licence = SimpleNamespace(
        status=LicenseStatus.REVOKED,
        limit_max_products=0,
        limit_max_ai_credits=0,
        limit_max_public_views=0,
        limit_max_galleries=0,
        usage_ai_credits=1750,   # already spent this period
        usage_public_views=9000,
    )
    db = _FakeDB([_Result(scalars_list=[]), _Result(one=licence)])

    await svc._sync_subscription_license(
        db, subscription=_subscription(), user_id="user-uuid"
    )

    assert licence.usage_ai_credits == 1750, "verify reset AI credit usage"
    assert licence.usage_public_views == 9000, "verify reset public-view usage"

    # It must still do its actual job: activate and refresh the limits.
    assert licence.status == LicenseStatus.ACTIVE
    assert licence.limit_max_ai_credits == PLAN_CREDITS
    assert licence.limit_max_products == LIMITS["max_products"]


async def test_a_brand_new_licence_still_starts_at_zero():
    """Removing the reset must not leave a first-time subscriber's quota unset."""
    db = _FakeDB([_Result(scalars_list=[]), _Result(one=None)])

    await svc._sync_subscription_license(
        db, subscription=_subscription(), user_id="user-uuid"
    )

    assert len(db.added) == 1
    licence = db.added[0]
    assert licence.usage_ai_credits == 0
    assert licence.usage_public_views == 0
    assert licence.usage_products == 0
    assert licence.limit_max_ai_credits == PLAN_CREDITS


async def test_other_active_licences_are_still_revoked():
    """A user must not end up holding two active licences."""
    stale = SimpleNamespace(status=LicenseStatus.ACTIVE)
    db = _FakeDB([_Result(scalars_list=[stale]), _Result(one=None)])

    await svc._sync_subscription_license(
        db, subscription=_subscription(), user_id="user-uuid"
    )

    assert stale.status == LicenseStatus.REVOKED
