"""USD billing logic — calendar arithmetic, discount maths and guard rails.

These cover the parts that are pure functions and therefore cheap to pin down:
billing-date clamping, the discount computation, the guard rails, country
resolution, and the structural guarantee that the pricing contract carries no
gateway plan IDs.

Not covered here, because they need a live database and a Razorpay sandbox:
subscription creation end to end, the webhook currency branch, entitlement
grants on subscription.authenticated, and webhook replay idempotency. Those are
listed in the handover notes as the manual test plan.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.core.geo import (
    currency_for_country,
    is_india,
    resolve_checkout_country,
    resolve_display_country,
)
from app.services.usd_promo_service import DISCOUNT_FIXED, DISCOUNT_PERCENTAGE
from app.schemas.pricing import PricingResponse
from app.services import usd_promo_service as promo
from app.utils.billing_dates import add_calendar_months, next_period_start, to_razorpay_start_at
from app.utils.money import format_money, to_minor_units

UTC = timezone.utc

# Confirmed USD price list, in cents.
MONTHLY = 2000   # $20.00
ANNUAL = 20000   # $200.00 — 10x monthly, i.e. two months free, permanently


class FakeRequest:
    """Minimal stand-in carrying only headers."""

    def __init__(self, **headers):
        self.headers = {k.lower().replace("_", "-"): v for k, v in headers.items()}


def percent_promo(value: int):
    return SimpleNamespace(discount_type=DISCOUNT_PERCENTAGE, discount_value=value, code="P")


def flat_promo(cents: int):
    return SimpleNamespace(discount_type=DISCOUNT_FIXED, discount_value=cents, code="F")


# ─── start_at: calendar arithmetic, not day counts ──────────────────────────


@pytest.mark.parametrize(
    "signup, interval, expected",
    [
        (datetime(2026, 3, 15, tzinfo=UTC), "monthly", "2026-04-15"),
        # Month-end clamping: there is no 31 February.
        (datetime(2026, 1, 31, tzinfo=UTC), "monthly", "2026-02-28"),
        (datetime(2028, 1, 31, tzinfo=UTC), "monthly", "2028-02-29"),
        (datetime(2026, 3, 31, tzinfo=UTC), "monthly", "2026-04-30"),
        (datetime(2026, 3, 15, tzinfo=UTC), "yearly", "2027-03-15"),
        # Leap day does not survive a year.
        (datetime(2028, 2, 29, tzinfo=UTC), "yearly", "2029-02-28"),
    ],
)
def test_next_period_start_clamps_to_month_end(signup, interval, expected):
    assert next_period_start(signup, interval).date().isoformat() == expected


def test_add_calendar_months_rolls_the_year():
    assert add_calendar_months(datetime(2026, 12, 31, tzinfo=UTC), 1).date().isoformat() == "2027-01-31"


def test_start_at_is_never_in_the_past():
    """Razorpay rejects a past start_at; clock skew must not be able to cause one."""
    stale = to_razorpay_start_at(datetime(2020, 1, 1, tzinfo=UTC))
    assert stale > int(datetime.now(UTC).timestamp())


# ─── Money formatting ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "minor, currency, expected",
    [
        (MONTHLY, "USD", "$20.00"),
        (ANNUAL, "USD", "$200.00"),
        (1000, "USD", "$10.00"),
        (4000, "USD", "$40.00"),
        (to_minor_units(1999, "INR"), "INR", "₹1,999"),
        # Indian digit grouping, not thousands grouping.
        (to_minor_units(1999000, "INR"), "INR", "₹19,99,000"),
    ],
)
def test_format_money(minor, currency, expected):
    assert format_money(minor, currency) == expected


# ─── Discount arithmetic ────────────────────────────────────────────────────


def test_promo_halves_the_first_month():
    assert promo.compute_upfront_amount(MONTHLY, percent_promo(50)) == 1000


def test_no_promo_charges_the_list_price():
    """Monthly without a promo still pays full price up front, and still gets a
    full period for it — promo and non-promo differ only in this number."""
    assert promo.compute_upfront_amount(MONTHLY, None) == MONTHLY


def test_fractional_cents_are_floored_in_the_customers_favour():
    # 50% of $29.99 is $14.995 — the customer gets the cent.
    assert promo.compute_upfront_amount(2999, percent_promo(50)) == 1499


def test_flat_discount_subtracts_cents():
    assert promo.compute_upfront_amount(MONTHLY, flat_promo(500)) == 1500


# ─── Guard rails ────────────────────────────────────────────────────────────


def rails(list_amount, upfront, interval="monthly"):
    promo.assert_within_guard_rails(
        list_amount_minor=list_amount,
        upfront_amount_minor=upfront,
        billing_interval=interval,
    )


@pytest.mark.parametrize(
    "list_amount, upfront, interval",
    [
        (MONTHLY, 1000, "monthly"),
        (MONTHLY, MONTHLY, "monthly"),
        (MONTHLY, 800, "monthly"),          # exactly the 60% ceiling
        (ANNUAL, ANNUAL, "yearly"),         # annual always at list price
    ],
)
def test_guard_rails_allow_valid_amounts(list_amount, upfront, interval):
    rails(list_amount, upfront, interval)


@pytest.mark.parametrize(
    "list_amount, upfront, interval, why",
    [
        (MONTHLY, 0, "monthly", "zero upfront"),
        (MONTHLY, -100, "monthly", "negative upfront"),
        (MONTHLY, 2500, "monthly", "upfront above list price"),
        (MONTHLY, 600, "monthly", "70% discount exceeds the 60% ceiling"),
        # The one that matters most: annual's discount is already in the list
        # price, so applying one again would double-count it.
        (ANNUAL, 10000, "yearly", "discount applied to annual"),
    ],
)
def test_guard_rails_reject_impossible_amounts(list_amount, upfront, interval, why):
    with pytest.raises(HTTPException) as exc:
        rails(list_amount, upfront, interval)
    assert exc.value.status_code == 400, why


def test_annual_is_never_promo_eligible():
    assert "yearly" not in promo.PROMO_ELIGIBLE_INTERVALS
    assert "monthly" in promo.PROMO_ELIGIBLE_INTERVALS


def test_annual_saving_is_display_copy_not_a_deduction():
    """$40 is the gap between 12 monthly payments and the annual price. It is
    already inside the $200, so nothing may subtract it again."""
    assert ANNUAL == MONTHLY * 10          # two months free, permanently
    assert MONTHLY * 12 - ANNUAL == 4000   # the advertised saving
    # The amount actually charged for a year is the list price, untouched.
    assert promo.compute_upfront_amount(ANNUAL, None) == ANNUAL


# ─── Country and currency resolution ────────────────────────────────────────


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"cf-ipcountry": "IN"}, "IN"),
        ({"cf-ipcountry": "us"}, "US"),
        # Cloudflare's own header outranks anything forwarded.
        ({"cf-ipcountry": "IN", "x-rvl-country": "US"}, "IN"),
        # The marketing site's server-side render forwards the visitor's country.
        ({"x-rvl-country": "GB"}, "GB"),
        ({}, None),
        ({"cf-ipcountry": "XX"}, None),   # Cloudflare could not geolocate
        ({"cf-ipcountry": "T1"}, None),   # Tor
    ],
)
def test_display_country_accepts_the_forwarded_header(headers, expected):
    assert resolve_display_country(FakeRequest(**headers)) == expected


def test_checkout_country_ignores_the_forwarded_header():
    """The header that decides who may be charged in USD is attacker-controllable
    if forwarded, so checkout trusts Cloudflare's alone."""
    assert resolve_checkout_country(FakeRequest(**{"x-rvl-country": "US"})) is None
    assert resolve_checkout_country(FakeRequest(**{"cf-ipcountry": "US"})) == "US"


@pytest.mark.parametrize(
    "country, currency",
    [("IN", "INR"), ("US", "USD"), ("GB", "USD"), (None, "USD")],
)
def test_currency_for_country(country, currency):
    assert currency_for_country(country) == currency


def test_only_india_is_india():
    assert is_india("IN")
    assert not is_india("US")
    assert not is_india(None)


# ─── The pricing contract must not leak gateway plan IDs ────────────────────


def test_pricing_response_exposes_no_plan_ids():
    """Checkout resolves the plan server-side from the tier and interval. If a
    plan ID could reach a client, a client could name the plan it is charged for."""

    def field_names(model, seen=None):
        seen = seen or set()
        if model in seen:
            return set()
        seen.add(model)
        names = set()
        for name, field in model.model_fields.items():
            names.add(name)
            annotation = field.annotation
            for candidate in (annotation, *(getattr(annotation, "__args__", ()) or ())):
                if hasattr(candidate, "model_fields"):
                    names |= field_names(candidate, seen)
        return names

    leaked = [
        name
        for name in field_names(PricingResponse)
        if "razorpay" in name or "plan_id" in name
    ]
    assert leaked == []
