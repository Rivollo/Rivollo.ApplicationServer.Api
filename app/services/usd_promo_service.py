"""USD promo validation and discount arithmetic.

Razorpay Offers are INR-locked on this account and fail silently in USD — with
"On Offer Failure = Allow Payment" the customer is simply charged full price
after being shown a discount. So USD discounts never go through the Offer
entity: the amount is computed here and applied as the subscription's upfront
amount, which inherits the plan's currency automatically.

The same functions serve the pricing page and checkout. That is the point: the
promo advertised on the page is the promo applied at checkout, computed by the
same code, so the price shown and the price charged cannot drift apart.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.geo import USD
from app.models.promo import PromoCode
from app.models.subscription import Subscription
from app.services.billing_currency import PAID_STATUSES

_logger = logging.getLogger("rivollo.usd_promo_service")

# Same vocabulary as the existing INR promos, so the two currencies cannot
# disagree about what a discount type is called.
DISCOUNT_PERCENTAGE = "percentage"
DISCOUNT_FIXED = "fixed"

# No promo may take more than this off the list price. A guard rail, not a
# business rule: it is what stops a mistyped flat value (or an INR-denominated
# one pasted into a USD row) from taking $4,000 off a $29 plan.
MAX_DISCOUNT_PCT = 60

# Annual is never eligible. Its "two months off" is permanent and already inside
# the list price (annual = 10x monthly), so a promo on top would double-count.
PROMO_ELIGIBLE_INTERVALS = frozenset({"monthly"})

# A subscription that never got past PENDING was never paid for. Someone who
# opened checkout and abandoned it is still a new customer.
_PAID_STATUSES = PAID_STATUSES


class PromoRejected(Exception):
    """A promo code was supplied but cannot be applied.

    Carries a customer-facing reason. Callers decide whether to surface it as an
    error (the customer typed a code) or fall back to full price (we tried to
    auto-apply a public promo) — but never silently, either way.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


async def has_prior_paid_subscription(db: AsyncSession, user_id: uuid.UUID) -> bool:
    """True when the user has ever held a subscription that was actually paid."""
    result = await db.execute(
        select(func.count())
        .select_from(Subscription)
        .where(
            Subscription.user_id == user_id,
            Subscription.status.in_(_PAID_STATUSES),
        )
    )
    return (result.scalar() or 0) > 0


def _is_live(promo: PromoCode, now: datetime) -> Optional[str]:
    """Return a rejection reason if the promo is not currently redeemable."""
    if not promo.is_active:
        return "This promo code is no longer active."
    if promo.valid_from and now < promo.valid_from:
        return "This promo code is not valid yet."
    if promo.valid_to and now > promo.valid_to:
        return "This promo code has expired."
    if promo.max_usage is not None and promo.used_count >= promo.max_usage:
        return "This promo code has reached its redemption limit."
    return None


def compute_upfront_amount(list_amount_minor: int, promo: Optional[PromoCode]) -> int:
    """The amount to charge for the first period, in minor units.

    With no promo this is the full list price. Every monthly subscription — promo
    or not — is created the same way (upfront amount plus a start date one period
    out), so the two paths differ only in this number.

    Fractional cents are floored, which resolves them in the customer's favour.
    """
    if promo is None:
        return list_amount_minor

    if promo.discount_type == DISCOUNT_PERCENTAGE:
        return (list_amount_minor * (100 - promo.discount_value)) // 100
    if promo.discount_type == DISCOUNT_FIXED:
        return list_amount_minor - promo.discount_value

    raise PromoRejected("This promo code is misconfigured and cannot be applied.")


def assert_within_guard_rails(
    *, list_amount_minor: int, upfront_amount_minor: int, billing_interval: str
) -> None:
    """Reject an upfront amount that cannot be right.

    Raises HTTPException — these conditions mean a bug or bad data reached
    checkout, not that the customer did anything wrong.
    """
    problem: Optional[str] = None

    if billing_interval not in PROMO_ELIGIBLE_INTERVALS and upfront_amount_minor != list_amount_minor:
        problem = (
            f"discount applied to ineligible interval '{billing_interval}' "
            f"(list={list_amount_minor}, upfront={upfront_amount_minor})"
        )
    elif upfront_amount_minor <= 0:
        problem = f"upfront amount {upfront_amount_minor} is not positive"
    elif upfront_amount_minor > list_amount_minor:
        problem = (
            f"upfront amount {upfront_amount_minor} exceeds list price {list_amount_minor}"
        )
    else:
        discount_pct = (list_amount_minor - upfront_amount_minor) * 100 / list_amount_minor
        if discount_pct > MAX_DISCOUNT_PCT:
            problem = (
                f"discount {discount_pct:.1f}% exceeds the {MAX_DISCOUNT_PCT}% ceiling "
                f"(list={list_amount_minor}, upfront={upfront_amount_minor})"
            )

    if problem is not None:
        _logger.error("USD checkout guard rail tripped: %s", problem)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This price could not be confirmed. Please contact support.",
        )


async def get_public_promo(
    db: AsyncSession, *, plan_code: str, billing_interval: str
) -> Optional[PromoCode]:
    """The promo advertised on the pricing page for this tier and interval.

    Returns None for ineligible intervals, so annual never advertises one.
    """
    if billing_interval not in PROMO_ELIGIBLE_INTERVALS:
        return None

    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(PromoCode).where(
            # USD promos only. The table holds both currencies, and an INR
            # promo auto-applied to a dollar checkout would discount the wrong
            # amount in the wrong currency.
            PromoCode.currency == USD,
            PromoCode.is_public.is_(True),
            PromoCode.is_active.is_(True),
            PromoCode.billing_interval == billing_interval,
            PromoCode.valid_from <= now,
            PromoCode.valid_to >= now,
            # NOT `plan_code.in_([plan_code, None])`. In SQL's three-valued
            # logic `x IN (a, NULL)` is NULL rather than TRUE when x is NULL, so
            # that form silently never matches an all-plans promo — which would
            # leave it advertised nowhere yet still redeemable when typed.
            or_(
                PromoCode.plan_code == plan_code,
                PromoCode.plan_code.is_(None),
            ),
        )
    )

    for promo in result.scalars():
        if _is_live(promo, now) is None:
            return promo
    return None


async def get_public_promos(
    db: AsyncSession, *, billing_interval: str
) -> dict[Optional[str], PromoCode]:
    """Every advertised promo for this interval, keyed by the plan it targets.

    The same answer as calling get_public_promo() once per plan, in one query
    instead of one per plan. The pricing page renders every tier at once, so the
    per-plan form was a round trip per tier for a table that holds a handful of
    rows.

    The key is the promo's own plan_code, so a promo that applies to every plan
    is stored under None. Callers resolve precedence explicitly:

        promos.get(plan.code) or promos.get(None)

    which prefers a promo naming this plan over a catch-all. The per-plan
    version left that choice to whatever order the database happened to return.
    """
    if billing_interval not in PROMO_ELIGIBLE_INTERVALS:
        return {}

    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(PromoCode).where(
            PromoCode.currency == USD,
            PromoCode.is_public.is_(True),
            PromoCode.is_active.is_(True),
            PromoCode.billing_interval == billing_interval,
            PromoCode.valid_from <= now,
            PromoCode.valid_to >= now,
        )
    )

    promos: dict[Optional[str], PromoCode] = {}
    for promo in result.scalars():
        if _is_live(promo, now) is not None:
            continue
        # First writer wins, matching the single-plan lookup's "return the first
        # live row" behaviour for the case where two promos target one plan.
        promos.setdefault(promo.plan_code, promo)
    return promos


async def resolve_promo_for_checkout(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    plan_code: str,
    billing_interval: str,
    submitted_code: Optional[str],
) -> Optional[PromoCode]:
    """Decide which promo, if any, applies to this checkout.

    A code the customer typed is validated strictly and raises PromoRejected on
    failure — never silently ignored, because a customer who sees a code accepted
    and is then charged full price has a strong chargeback case.

    With no code submitted, the publicly advertised promo is applied
    automatically, so the customer is charged the price the pricing page quoted.
    """
    if submitted_code:
        code = submitted_code.strip().upper()

        if billing_interval not in PROMO_ELIGIBLE_INTERVALS:
            raise PromoRejected(
                "Promo codes do not apply to annual plans — the annual price "
                "already includes two months free."
            )

        # first() rather than scalar_one_or_none(): the unique constraint on
        # `code` is case-sensitive, so 'intro50' and 'INTRO50' can both exist
        # while this lookup is case-insensitive. That would raise
        # MultipleResultsFound and 500 instead of validating a promo code.
        result = await db.execute(
            select(PromoCode)
            .where(func.upper(PromoCode.code) == code, PromoCode.currency == USD)
            .order_by(PromoCode.created_date.asc())
            .limit(1)
        )
        promo = result.scalars().first()
        if promo is None:
            raise PromoRejected("This promo code was not recognised.")

        reason = _is_live(promo, datetime.now(timezone.utc))
        if reason is not None:
            raise PromoRejected(reason)

        if promo.billing_interval != billing_interval:
            raise PromoRejected(
                f"This promo code applies to {promo.billing_interval} billing only."
            )
        if promo.plan_code is not None and promo.plan_code != plan_code:
            raise PromoRejected("This promo code does not apply to the selected plan.")
        if await has_prior_paid_subscription(db, user_id):
            raise PromoRejected("This promo code is for new customers only.")

        return promo

    if await has_prior_paid_subscription(db, user_id):
        return None

    return await get_public_promo(db, plan_code=plan_code, billing_interval=billing_interval)


async def record_redemption_by_code(db: AsyncSession, code: str) -> None:
    """Count one redemption of ``code``. Caller commits.

    Called when the payment is actually captured, not when checkout opens, so
    that abandoned checkouts do not consume a promo's max_usage. The
    webhook that calls this is idempotent on the Razorpay event ID, so a
    replayed event cannot double-count.

    Incremented in the database rather than on a loaded object so concurrent
    captures cannot lose an increment to a read-modify-write race.
    """
    await db.execute(
        update(PromoCode)
        .where(
            func.upper(PromoCode.code) == code.strip().upper(),
            PromoCode.currency == USD,
        )
        .values(used_count=PromoCode.used_count + 1)
    )
