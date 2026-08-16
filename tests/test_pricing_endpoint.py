"""GET /pricing, driven through the real route.

This exercises the actual FastAPI app: routing, the geo resolver, the pricing
service, currency selection, money formatting and response serialisation. Only
the database is replaced — there is no Postgres in this environment, so the
fake session answers by looking at which table each statement targets.

What it therefore does NOT prove: that the SQL runs against a real schema. That
is the check to run after deploying and before seeding, and it is one curl:

    curl -s -H "cf-ipcountry: US" $API/pricing | jq .data.currency

Everything between the HTTP request and that SQL is covered here.
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db
from app.database.subscription_repo import SubscriptionRepository
from app.main import app

PRO_ID = uuid.uuid4()
FREE_ID = uuid.uuid4()


# ── Fixtures standing in for seeded rows ────────────────────────────────────


def _price(plan_id, interval, currency, amount, credits, rz_plan_id):
    return SimpleNamespace(
        plan_id=plan_id,
        billing_interval=interval,
        currency=currency,
        price_inr=amount,          # whole units of `currency`
        ai_credit_limit=credits,
        razorpay_plan_id=rz_plan_id,
        isactive=True,
    )


# Copied from the dev database, not invented. The INR rows are what
# tbl_plan_prices actually holds; the USD rows are what sql/seed_usd_pricing.sql
# adds beside them.
#
# Two details here are real and load-bearing. INR annual is 23999 -- 12x monthly
# with the two months taken off separately by a Razorpay Offer -- so it produces
# no annual saving of its own. And the Free tier has no INR price row at all,
# which is why the USD seed inserts its rows explicitly rather than mirroring
# zero-priced ones.
ALL_PRICES = [
    _price(PRO_ID, "monthly", "INR", 1999, 2000, "plan_SQbKKGjy3J4cac"),
    _price(PRO_ID, "yearly", "INR", 23999, 24000, "plan_STQ5ouNx8Q9r4X"),
    _price(PRO_ID, "monthly", "USD", 20, 2000, "plan_TQ2m22UBRutnZu"),
    _price(PRO_ID, "yearly", "USD", 200, 24000, "plan_TQ8SZe3nf6a0d3"),
    _price(FREE_ID, "monthly", "USD", 0, 100, None),
    _price(FREE_ID, "yearly", "USD", 0, 100, None),
    _price(FREE_ID, "monthly", "INR", 0, 100, None),
    _price(FREE_ID, "yearly", "INR", 0, 100, None),
]

USDINTRO50 = SimpleNamespace(
    code="USDINTRO50",
    discount_type="percentage",
    discount_value=50,
    billing_interval="monthly",
    plan_code="pro",
    currency="USD",
    max_usage=None,
    used_count=0,
    is_active=True,
    is_public=True,
    valid_from=None,
    valid_to=None,
    created_date=None,
)


def _plans():
    return [
        SimpleNamespace(
            id=PRO_ID, code="pro", name="Pro", description="For growing businesses",
            is_featured=True, plan_features=[],
            plan_prices=[p for p in ALL_PRICES if p.plan_id == PRO_ID],
        ),
        SimpleNamespace(
            id=FREE_ID, code="free", name="Free", description="Getting started",
            is_featured=False, plan_features=[],
            plan_prices=[p for p in ALL_PRICES if p.plan_id == FREE_ID],
        ),
    ]


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _FakeDB:
    """Answers by table, honouring the WHERE clause the service actually emits.

    Filtering on both plan and currency is what makes this a real test: a
    service that dropped either filter would get rows back it did not ask for,
    exactly as Postgres would hand it, and the assertions below would fail.
    """

    async def execute(self, statement, *_a, **_k):
        sql = str(statement)
        if "tbl_plan_prices" in sql:
            where = str(statement.compile(compile_kwargs={"literal_binds": True}))
            return _Result([
                p for p in ALL_PRICES
                # UUIDs render without dashes in compiled SQL, hence .hex.
                if (p.plan_id.hex in where or str(p.plan_id) in where)
                and f"'{p.currency}'" in where
            ])
        if "tbl_promo_codes" in sql:
            return _Result([USDINTRO50])
        return _Result([])


@pytest.fixture
def client(monkeypatch):
    async def fake_get_all_plans(_db):
        return _plans()

    monkeypatch.setattr(SubscriptionRepository, "get_all_plans", fake_get_all_plans)

    async def _db_override():
        yield _FakeDB()

    app.dependency_overrides[get_db] = _db_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _pro(payload):
    return next(t for t in payload["data"]["tiers"] if t["code"] == "pro")


def _period(tier, interval):
    return next(p for p in tier["periods"] if p["interval"] == interval)


# ── The USD path ────────────────────────────────────────────────────────────


def test_us_visitor_is_quoted_in_dollars(client):
    body = client.get("/pricing", headers={"cf-ipcountry": "US"}).json()

    assert body["data"]["currency"] == "USD"
    assert body["data"]["currencySymbol"] == "$"
    assert body["data"]["country"] == "US"


def test_usd_amounts_are_twenty_and_two_hundred(client):
    pro = _pro(client.get("/pricing", headers={"cf-ipcountry": "US"}).json())

    monthly, yearly = _period(pro, "monthly"), _period(pro, "yearly")

    # price_inr holds whole dollars; the API speaks minor units.
    assert monthly["amountMinor"] == 2000
    assert monthly["formatted"] == "$20.00"
    assert yearly["amountMinor"] == 20000
    assert yearly["formatted"] == "$200.00"
    assert monthly["available"] is True and yearly["available"] is True


def test_the_advertised_first_month_is_ten_dollars(client):
    pro = _pro(client.get("/pricing", headers={"cf-ipcountry": "US"}).json())

    promo = pro["promo"]
    assert promo["code"] == "USDINTRO50"
    assert promo["firstAmountMinor"] == 1000
    assert promo["firstFormatted"] == "$10.00"
    # The full price and the date it starts must both be in the copy — "charged
    # more than displayed" is a strong cardholder case.
    assert "$20.00" in promo["detail"]


def test_annual_saving_is_derived_not_stored(client):
    pro = _pro(client.get("/pricing", headers={"cf-ipcountry": "US"}).json())

    saving = pro["annualSaving"]
    # 12 x $20 = $240 against a $200 list price.
    assert saving["amountMinor"] == 4000
    assert saving["formatted"] == "$40.00"
    assert saving["percent"] == 17


# ── The INR path, which must not have moved ─────────────────────────────────


def test_indian_visitor_still_sees_rupees(client):
    body = client.get("/pricing", headers={"cf-ipcountry": "IN"}).json()
    pro = _pro(body)

    assert body["data"]["currency"] == "INR"
    assert _period(pro, "monthly")["formatted"] == "₹1,999"
    assert _period(pro, "monthly")["amountMinor"] == 199900
    assert _period(pro, "yearly")["formatted"] == "₹23,999"
    assert pro.get("promo") is None, "the USD intro promo was advertised in INR"


def test_inr_shows_no_annual_saving_because_there_is_none(client):
    """23999 is more than 12 x 1999, so there is nothing to advertise.

    The saving is computed from the two list prices rather than stored, which is
    what stops a "save 17%" badge outliving the prices that made it true. INR
    takes its two months off through a Razorpay Offer instead, which this
    endpoint does not render.
    """
    pro = _pro(client.get("/pricing", headers={"cf-ipcountry": "IN"}).json())

    assert pro.get("annualSaving") is None


def test_the_free_tier_is_offered_in_usd(client):
    """Without its seeded rows the frontend drops the tier and the USD pricing
    page loses its free column entirely."""
    tiers = {t["code"] for t in
             client.get("/pricing", headers={"cf-ipcountry": "US"}).json()["data"]["tiers"]}

    assert "free" in tiers, f"free tier missing from USD pricing: {tiers}"


def test_unknown_country_falls_back_to_usd(client):
    for header in ({}, {"cf-ipcountry": "XX"}, {"cf-ipcountry": "T1"}):
        body = client.get("/pricing", headers=header).json()
        assert body["data"]["currency"] == "USD", header


# ── Invariants ──────────────────────────────────────────────────────────────


def test_no_plan_id_ever_reaches_the_client(client):
    for country in ("US", "IN", "GB"):
        raw = client.get("/pricing", headers={"cf-ipcountry": country}).text
        assert "plan_" not in raw, f"a Razorpay plan ID leaked for {country}"


def test_response_is_never_shared_cache_eligible(client):
    resp = client.get("/pricing", headers={"cf-ipcountry": "US"})

    assert "no-store" in resp.headers["cache-control"]
    assert "CF-IPCountry" in resp.headers["vary"]


# ── /subscriptions/plans — the Portal's INR list ────────────────────────────
#
# Seeding Free's INR rows changes what this endpoint returns, so it is pinned
# here. It is an INR path: the point of these two tests is that adding a free
# price row gives Free a zero price marked unavailable, and leaves Pro alone.


def _plan(payload, code):
    return next(p for p in payload["data"] if p["code"] == code)


def test_plans_list_offers_free_at_zero_and_not_purchasable(client):
    body = client.get("/subscriptions/plans").json()

    free = _plan(body, "free")
    assert free["priceINR"] == 0
    assert free["priceINRYearly"] == 0
    assert all(p["available"] is False for p in free["pricing"]), (
        "the free tier is being offered as purchasable — it has no gateway plan"
    )
    assert len(free["pricing"]) == 2, f"duplicate currency rows: {free['pricing']}"


def test_plans_list_still_prices_pro_in_rupees(client):
    """The regression that matters: Pro must not pick up a USD row."""
    pro = _plan(client.get("/subscriptions/plans").json(), "pro")

    assert pro["priceINR"] == 1999
    assert pro["priceINRYearly"] == 23999
    assert all(p["available"] for p in pro["pricing"])

    # Count, not a set of intervals: without the currency filter this returns
    # four rows — monthly and yearly in both currencies — and a set of intervals
    # collapses the duplicates back to two, hiding exactly the bug being tested.
    assert len(pro["pricing"]) == 2, (
        f"expected one row per interval, got {pro['pricing']}"
    )
