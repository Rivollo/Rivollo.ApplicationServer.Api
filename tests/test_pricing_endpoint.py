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

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db
from app.database.subscription_repo import SubscriptionRepository
from app.main import app
from app.services import pricing_service

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
# tbl_plan_prices actually holds; the USD rows are the ones specified in
# USD_ROLLOUT_TODO.md, sitting beside them.
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

    def __init__(self):
        self.executed: list[str] = []

    async def execute(self, statement, *_a, **_k):
        sql = str(statement)
        self.executed.append(sql)
        if "tbl_plan_prices" in sql:
            where = str(statement.compile(compile_kwargs={"literal_binds": True}))
            # The pricing page fetches every USD row in one query, so no plan id
            # appears in that statement. Other callers still ask for a single
            # plan. Detecting which shape this is keeps both honest: a query
            # naming a plan must not receive another plan's rows.
            # UUIDs render without dashes in compiled SQL, hence .hex.
            names_a_plan = any(
                p.plan_id.hex in where or str(p.plan_id) in where for p in ALL_PRICES
            )
            return _Result([
                p for p in ALL_PRICES
                if (
                    not names_a_plan
                    or p.plan_id.hex in where
                    or str(p.plan_id) in where
                )
                and f"'{p.currency}'" in where
            ])
        if "tbl_promo_codes" in sql:
            return _Result([USDINTRO50])
        return _Result([])


@pytest.fixture
def db():
    """One fake session for the whole test, so its query log can be inspected."""
    return _FakeDB()


@pytest.fixture
def client(monkeypatch, db):
    async def fake_get_all_plans(_db):
        return _plans()

    monkeypatch.setattr(SubscriptionRepository, "get_all_plans", fake_get_all_plans)

    async def _db_override():
        yield db

    # The tier cache is module state and outlives a single test. Without this,
    # the first test to run decides what every later one sees.
    pricing_service.clear_pricing_cache()

    app.dependency_overrides[get_db] = _db_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    pricing_service.clear_pricing_cache()


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


# ── The tier cache ──────────────────────────────────────────────────────────


def _price_queries(db):
    return [s for s in db.executed if "tbl_plan_prices" in s]


def test_the_page_costs_one_price_query_not_one_per_plan(client, db):
    """The N+1 this cache was added to remove.

    Asserting on the query count rather than on elapsed time: a timing
    assertion passes on a fast machine no matter how many round trips it makes,
    which is exactly the regression that would go unnoticed.
    """
    client.get("/pricing", headers={"cf-ipcountry": "US"})

    assert len(_price_queries(db)) == 1, (
        f"expected one price query for the whole page, got {len(_price_queries(db))}"
    )


def test_a_second_visitor_hits_the_cache(client, db):
    client.get("/pricing", headers={"cf-ipcountry": "US"})
    after_first = len(db.executed)

    for _ in range(5):
        client.get("/pricing", headers={"cf-ipcountry": "US"})

    assert len(db.executed) == after_first, (
        "the cache is not being used — every request re-queried"
    )


def test_currencies_are_cached_separately(client, db):
    """The bug a single-slot cache would introduce: rupees served to Americans."""
    client.get("/pricing", headers={"cf-ipcountry": "US"})
    body = client.get("/pricing", headers={"cf-ipcountry": "IN"}).json()

    assert body["data"]["currency"] == "INR"
    assert _period(_pro(body), "monthly")["formatted"] == "₹1,999"

    usd = client.get("/pricing", headers={"cf-ipcountry": "US"}).json()
    assert _period(_pro(usd), "monthly")["formatted"] == "$20.00"


def test_currency_is_part_of_the_cache_key_in_its_own_right(client, db):
    """Two currencies sharing one promo state must not share one cache entry.

    test_currencies_are_cached_separately above does not actually prove this.
    Every INR visitor is promo-ineligible, so INR and USD differ in the promo
    half of the key as well — and a key built from the promo state alone still
    separates them, which lets a missing currency term pass unnoticed.

    The case that breaks is a *returning* USD customer: promo-ineligible, like
    every INR visitor. Drop currency from the key and those two collide, and one
    of them is served the other's prices. This drives the layer directly so the
    promo state is held equal and only the currency varies.
    """
    async def both():
        usd = await pricing_service._cached_tiers(db, currency="USD", show_promo=False)
        inr = await pricing_service._cached_tiers(db, currency="INR", show_promo=False)
        return usd, inr

    usd, inr = asyncio.run(both())

    def monthly(tiers):
        pro = next(t for t in tiers if t.code == "pro")
        return next(p.formatted for p in pro.periods if p.interval == "monthly")

    assert monthly(usd) == "$20.00"
    assert monthly(inr) == "₹1,999", (
        "an INR visitor was served the cached USD tiers — currency is not in the key"
    )


def test_promo_eligibility_is_part_of_the_cache_key(client, db):
    """A returning customer must not be served the new-customer tiers.

    Both calls are USD, so only the promo half of the key differs. If it were
    dropped, the second call would return the first call's cached list and quote
    "$10.00 today" to someone checkout will charge $20.00 — the displayed-price
    versus charged-price gap this module exists to close.
    """
    async def both():
        eligible = await pricing_service._cached_tiers(
            db, currency="USD", show_promo=True
        )
        returning = await pricing_service._cached_tiers(
            db, currency="USD", show_promo=False
        )
        return eligible, returning

    eligible, returning = asyncio.run(both())

    def pro(tiers):
        return next(t for t in tiers if t.code == "pro")

    assert pro(eligible).promo is not None, "the intro promo was not advertised at all"
    assert pro(returning).promo is None, (
        "a returning customer was shown the new-customer intro promo"
    )


def test_the_cache_expires(client, db):
    """A price edited in the database must appear, not be pinned forever."""
    client.get("/pricing", headers={"cf-ipcountry": "US"})
    after_first = len(db.executed)

    client.get("/pricing", headers={"cf-ipcountry": "US"})
    assert len(db.executed) == after_first, "precondition: the second call should hit"

    # Age the entry past its deadline rather than patching the clock, which
    # TestClient's own event loop also reads.
    key = ("USD", True)
    expires_at, tiers = pricing_service._tier_cache[key]
    pricing_service._tier_cache[key] = (
        expires_at - pricing_service.PRICING_CACHE_TTL_SECONDS - 1,
        tiers,
    )

    client.get("/pricing", headers={"cf-ipcountry": "US"})
    assert len(db.executed) > after_first, "an expired entry was still served"


def test_clearing_the_cache_forces_a_rebuild(client, db):
    client.get("/pricing", headers={"cf-ipcountry": "US"})
    before = len(db.executed)

    pricing_service.clear_pricing_cache()
    client.get("/pricing", headers={"cf-ipcountry": "US"})

    assert len(db.executed) > before, "clear_pricing_cache() did not evict anything"


def test_an_empty_result_is_never_cached(client, db, monkeypatch):
    """A database that answers nothing must not blank the page for a full TTL."""
    async def no_plans(_db):
        return []

    monkeypatch.setattr(SubscriptionRepository, "get_all_plans", no_plans)
    assert client.get("/pricing", headers={"cf-ipcountry": "US"}).json()["data"]["tiers"] == []

    monkeypatch.setattr(SubscriptionRepository, "get_all_plans", lambda _db=None: None)

    async def plans_again(_db):
        return _plans()

    monkeypatch.setattr(SubscriptionRepository, "get_all_plans", plans_again)
    tiers = client.get("/pricing", headers={"cf-ipcountry": "US"}).json()["data"]["tiers"]

    assert tiers, "an empty answer was cached and survived the database recovering"


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
