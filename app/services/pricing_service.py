"""Pricing display — the single source of truth for what a visitor is quoted.

Both Next.js apps read prices from here. That is the whole point: three
deployables rendering prices from three copies of the same numbers drift apart
the first time a price moves, and the customer-visible symptom is seeing one
price on the marketing site and a different one after signing in.

This module is display-only. It never returns a Razorpay plan ID — those stay
server-side so a client can never name the plan it wants to be charged for.
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.geo import INR, USD, currency_for_country, resolve_display_country
from app.database.subscription_repo import SubscriptionRepository
from app.models.models import User
from app.models.plan import Plan, PlanPrice
from app.models.promo import PromoCode
from app.schemas.pricing import (
    AnnualSaving,
    PricingFeature,
    PricingPeriod,
    PricingPromo,
    PricingResponse,
    PricingTier,
)
from app.services import usd_promo_service
from app.services.billing_currency import get_locked_currency
from app.utils.billing_dates import next_period_start
from app.utils.money import CURRENCY_SYMBOLS, format_money, to_minor_units

_logger = logging.getLogger("rivollo.pricing_service")

# Exports of services are zero-rated; local sales tax is the customer's own.
USD_TAX_NOTE = "Prices exclusive of applicable local taxes."

# INR keeps whatever tax line the existing pages already render. Neither app
# shows one today, so returning an empty string here leaves that behaviour
# exactly as it is.
INR_TAX_NOTE = ""

_INTERVALS = ("monthly", "yearly")

# ── Tier cache ───────────────────────────────────────────────────────────────
#
# The tier list is a pure function of the currency and whether the intro promo
# is shown. Plans, prices and promos are configuration — they change when
# someone edits the database, not per request — so rebuilding them on every page
# view spent a database round trip per plan to produce an identical answer.
#
# Not cached, and resolved on every request: the visitor's country, whether
# their currency is locked, and whether they are still eligible for the intro
# promo. Those are per-visitor. They only decide *which* cached list is
# returned, never what is in it.
#
# There is no explicit invalidation because nothing in this application writes a
# price — they are edited directly against the database. A change therefore
# appears within PRICING_CACHE_TTL_SECONDS rather than immediately. Set the
# variable to 0 to disable caching entirely.
PRICING_CACHE_TTL_SECONDS = int(os.getenv("PRICING_CACHE_TTL_SECONDS", "60"))

# (currency, show_promo) -> (expires_at_monotonic, tiers). At most three live
# keys: INR never shows the promo, USD appears with and without it.
_tier_cache: dict[tuple[str, bool], tuple[float, list[PricingTier]]] = {}

# Serialises misses so a cold cache under concurrent load rebuilds once rather
# than once per in-flight request. Cache *hits* never touch it.
_tier_cache_lock = asyncio.Lock()


def clear_pricing_cache() -> None:
    """Drop every cached tier list.

    Nothing in the request path calls this — it exists so tests do not leak
    state between cases, and so a future admin write path has an obvious hook.
    """
    _tier_cache.clear()


async def _locked_currency(db: AsyncSession, user: Optional[User]) -> Optional[str]:
    """The currency this user is already committed to, if any."""
    if user is None:
        return None
    return await get_locked_currency(db, user.id)


def _feature_labels(plan: Plan) -> list[PricingFeature]:
    """Render a plan's features the same way /subscriptions/plans does."""
    features: list[PricingFeature] = []
    for pf in getattr(plan, "plan_features", []):
        if not pf.feature:
            continue
        label = pf.feature.name
        if pf.limit_value is not None:
            label = f"{pf.limit_value:,} {label.lower()}"
        features.append(PricingFeature(label=label, available=pf.is_available))
    return features


def _annual_saving(periods: list[PricingPeriod], currency: str) -> Optional[AnnualSaving]:
    """What paying annually saves, derived from the two amounts we just built.

    Never a stored figure — computing it from the prices themselves is what
    stops a "save 17%" badge surviving a price change that made it false.
    """
    by_interval = {p.interval: p for p in periods}
    monthly, yearly = by_interval.get("monthly"), by_interval.get("yearly")
    if not monthly or not yearly or monthly.amount_minor <= 0 or yearly.amount_minor <= 0:
        return None

    twelve_months = monthly.amount_minor * 12
    saving = twelve_months - yearly.amount_minor
    if saving <= 0:
        return None

    return AnnualSaving(
        amount_minor=saving,
        formatted=format_money(saving, currency),
        percent=round(saving * 100 / twelve_months),
    )


def _promo_display(
    promo: Optional[PromoCode],
    periods: list[PricingPeriod],
    currency: str,
) -> Optional[PricingPromo]:
    """Turn an eligible promo into copy that states the full price and its date.

    The disclosure is not decoration. Card networks treat "charged more than
    displayed" as a strong cardholder case, so the promotional price is always
    quoted alongside what happens next and when.
    """
    if promo is None:
        return None

    monthly = next((p for p in periods if p.interval == "monthly"), None)
    if monthly is None or not monthly.available:
        return None

    try:
        first_amount = usd_promo_service.compute_upfront_amount(monthly.amount_minor, promo)
    except usd_promo_service.PromoRejected as exc:
        _logger.warning("Public promo %s is not displayable: %s", promo.code, exc.reason)
        return None

    if first_amount <= 0 or first_amount >= monthly.amount_minor:
        return None

    first_formatted = format_money(first_amount, currency)
    charge_date = next_period_start(datetime.now(timezone.utc), "monthly")
    discount_pct = round((monthly.amount_minor - first_amount) * 100 / monthly.amount_minor)

    return PricingPromo(
        interval="monthly",
        code=promo.code,
        first_amount_minor=first_amount,
        first_formatted=first_formatted,
        headline=f"{discount_pct}% off your first month",
        detail=(
            f"Pay {first_formatted} today, then {monthly.formatted} per month "
            f"from {charge_date.strftime('%-d %B %Y')}."
        ),
    )


async def _usd_prices_by_plan(
    db: AsyncSession,
) -> dict[uuid.UUID, dict[str, PlanPrice]]:
    """Every active USD price row, grouped by plan and then by interval.

    One query for the whole page. The per-plan form this replaces issued a
    round trip per tier, which is what made the USD path measurably slower than
    the INR one — INR reads its rows off the already-loaded relationship.

    The currency filter is not optional: tbl_plan_prices holds INR and USD rows
    for the same (plan, interval), so dropping it would collapse two rows into
    one dictionary slot and quote rupee amounts as dollars.
    """
    result = await db.execute(
        select(PlanPrice).where(
            PlanPrice.currency == USD,
            PlanPrice.isactive.is_(True),
        )
    )

    by_plan: dict[uuid.UUID, dict[str, PlanPrice]] = {}
    for row in result.scalars():
        by_plan.setdefault(row.plan_id, {})[row.billing_interval] = row
    return by_plan


def _usd_periods(rows: dict[str, PlanPrice]) -> list[PricingPeriod]:
    periods: list[PricingPeriod] = []
    for interval in _INTERVALS:
        row = rows.get(interval)
        if row is None:
            continue
        periods.append(
            PricingPeriod(
                interval=interval,
                # price_inr holds whole units of the row's currency — whole
                # dollars here, not cents.
                amount_minor=to_minor_units(row.price_inr, USD),
                formatted=format_money(to_minor_units(row.price_inr, USD), USD),
                ai_credits=row.ai_credit_limit,
                available=bool(row.razorpay_plan_id),
            )
        )
    return periods


def _inr_periods(plan: Plan) -> list[PricingPeriod]:
    # The relationship loads every currency's row. Keyed by interval alone, a
    # USD row would overwrite the INR one and this would render $20 as Rs 20.
    rows = {
        pp.billing_interval: pp
        for pp in getattr(plan, "plan_prices", [])
        if pp.isactive and pp.currency == INR
    }

    periods: list[PricingPeriod] = []
    for interval in _INTERVALS:
        row: Optional[PlanPrice] = rows.get(interval)
        if row is None:
            continue
        # tbl_plan_prices stores whole rupees; the API speaks minor units only.
        amount_minor = to_minor_units(row.price_inr, INR)
        periods.append(
            PricingPeriod(
                interval=interval,
                amount_minor=amount_minor,
                formatted=format_money(amount_minor, INR),
                ai_credits=row.ai_credit_limit,
                available=bool(row.razorpay_plan_id),
            )
        )
    return periods


async def _build_tiers(
    db: AsyncSession, *, currency: str, show_promo: bool
) -> list[PricingTier]:
    """Assemble the tier list for one currency.

    Two queries for the whole page rather than two per plan: the prices and the
    advertised promos are each fetched once and matched up in memory.
    """
    plans = await SubscriptionRepository.get_all_plans(db)

    prices_by_plan: dict[uuid.UUID, dict[str, PlanPrice]] = {}
    promos: dict[Optional[str], PromoCode] = {}
    if currency == USD:
        prices_by_plan = await _usd_prices_by_plan(db)
        # Returning customers are not eligible for the intro promo, so it must
        # not be built into the list they are served. Quoting "$10.00 today" to
        # someone checkout will charge $20.00 is exactly the displayed-price /
        # charged-price gap this module exists to prevent.
        if show_promo:
            promos = await usd_promo_service.get_public_promos(
                db, billing_interval="monthly"
            )

    tiers: list[PricingTier] = []
    for plan in plans:
        if currency == USD:
            periods = _usd_periods(prices_by_plan.get(plan.id, {}))
            # A promo naming this plan beats one that applies to every plan.
            promo = promos.get(plan.code) or promos.get(None)
            promo_display = _promo_display(promo, periods, currency)
        else:
            periods = _inr_periods(plan)
            # INR promos run through Razorpay Offers and are rendered by the
            # existing pages; nothing to surface here.
            promo_display = None

        if not periods:
            # A tier with no price list in this currency is not purchasable
            # here. Omit it rather than showing a tier with no price.
            continue

        tiers.append(
            PricingTier(
                code=plan.code,
                name=plan.name,
                description=getattr(plan, "description", None) or f"{plan.name} Plan",
                featured=getattr(plan, "is_featured", False),
                features=_feature_labels(plan),
                periods=periods,
                annual_saving=_annual_saving(periods, currency),
                promo=promo_display,
            )
        )
    return tiers


async def _cached_tiers(
    db: AsyncSession, *, currency: str, show_promo: bool
) -> list[PricingTier]:
    """_build_tiers with a short TTL in front of it.

    One caveat worth knowing: the promo copy contains the date of the first
    full-price charge, formatted when the list is built. Within one TTL of a
    month boundary a visitor can therefore be shown the previous month's date.
    At a 60-second TTL that is a 60-second window, which is why the date is
    allowed to be cached at all — lengthen the TTL substantially and it stops
    being acceptable.
    """
    if PRICING_CACHE_TTL_SECONDS <= 0:
        return await _build_tiers(db, currency=currency, show_promo=show_promo)

    key = (currency, show_promo)

    cached = _tier_cache.get(key)
    if cached and time.monotonic() < cached[0]:
        return cached[1]

    async with _tier_cache_lock:
        # Re-check under the lock. A request that queued here while another
        # rebuilt the same key should use that result, not immediately rebuild.
        cached = _tier_cache.get(key)
        if cached and time.monotonic() < cached[0]:
            return cached[1]

        tiers = await _build_tiers(db, currency=currency, show_promo=show_promo)

        # An empty list means the query matched nothing — an unmigrated or
        # unreachable database, not a real answer. Caching it would keep the
        # pricing page blank for a full TTL after the cause was fixed.
        if tiers:
            _tier_cache[key] = (time.monotonic() + PRICING_CACHE_TTL_SECONDS, tiers)
        return tiers


async def get_pricing(
    db: AsyncSession,
    *,
    request: Request,
    current_user: Optional[User] = None,
) -> PricingResponse:
    """Resolve the visitor's currency and return everything needed to render it."""
    country = resolve_display_country(request)

    locked = await _locked_currency(db, current_user)
    currency = locked if locked in (INR, USD) else currency_for_country(country)
    currency_locked = currency == locked

    # A customer who has already paid for a subscription cannot redeem the
    # new-customer intro promo, so it must not be advertised to them.
    promo_ineligible = currency != USD or (
        current_user is not None
        and await usd_promo_service.has_prior_paid_subscription(db, current_user.id)
    )

    tiers = await _cached_tiers(
        db, currency=currency, show_promo=not promo_ineligible
    )

    return PricingResponse(
        currency=currency,
        currency_symbol=CURRENCY_SYMBOLS.get(currency, ""),
        country=country,
        currency_locked=currency_locked,
        tax_note=USD_TAX_NOTE if currency == USD else INR_TAX_NOTE,
        tiers=tiers,
    )
