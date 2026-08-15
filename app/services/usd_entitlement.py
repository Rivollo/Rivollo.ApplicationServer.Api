"""Entitlement for USD subscriptions — loading the subscription and granting the
licence that actually unlocks the product.

Shared by the USD webhook and the USD verify endpoint so the two can never
disagree about what a customer is entitled to. Deliberately separate from the
INR equivalent in razorpay_subscription_service, which resolves AI credits from
tbl_plan_prices (the INR price list) and would therefore both apply INR limits to
a USD customer and raise a 400 whenever a plan has no active INR row — after the
card has already been charged.
"""

import logging
import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.geo import USD
from app.models.license_assignment import LicenseAssignment
from app.models.plan import Plan, PlanFeature
from app.models.plan import PlanPrice
from app.models.subscription import Subscription
from app.models.subscription_enums import LicenseStatus

_logger = logging.getLogger("rivollo.usd_entitlement")


async def is_usd_subscription(db: AsyncSession, rz_subscription_id: str) -> bool:
    """Whether this Razorpay subscription bills in USD.

    Read from the currency recorded when it was created, not inferred from a
    webhook payload: payload fields are not present on every event, so they
    would give a different answer depending on which event arrived.

    first() rather than scalar_one_or_none(): razorpay_subscription_id carries a
    non-unique index, so duplicate rows are structurally possible, and this runs
    on the INR webhook path too. scalar_one_or_none() would raise
    MultipleResultsFound and take INR webhooks down with it.
    """
    result = await db.execute(
        select(Subscription.currency)
        .where(Subscription.razorpay_subscription_id == rz_subscription_id)
        .order_by(Subscription.created_date.desc())
        .limit(1)
    )
    return result.scalars().first() == USD


async def get_subscription(
    db: AsyncSession, rz_subscription_id: str, *, user_id: Optional[uuid.UUID] = None
) -> Optional[Subscription]:
    """Load a subscription with everything the callers touch eagerly loaded.

    plan and plan_features are selectinload-ed because callers read
    ``subscription.plan.code`` and iterate plan_features; a lazy load on an async
    session raises MissingGreenlet rather than quietly issuing a query.
    """
    stmt = (
        select(Subscription)
        .where(Subscription.razorpay_subscription_id == rz_subscription_id)
        .options(
            selectinload(Subscription.plan)
            .selectinload(Plan.plan_features)
            .selectinload(PlanFeature.feature)
        )
        .order_by(Subscription.created_date.desc())
        .limit(1)
    )
    if user_id is not None:
        stmt = stmt.where(Subscription.user_id == user_id)

    result = await db.execute(stmt)
    return result.scalars().first()


def feature_limits(subscription: Subscription) -> dict:
    """Per-feature limits for the subscription's plan. Currency-independent."""
    limits: dict = {}
    if not subscription.plan:
        return limits
    for pf in getattr(subscription.plan, "plan_features", []):
        if pf.feature and pf.limit_value is not None:
            limits[pf.feature.code] = pf.limit_value
    return limits


async def usd_ai_credit_limit(
    db: AsyncSession, plan_id: uuid.UUID, billing_interval: Optional[str], limits: dict
) -> int:
    """AI credits for a USD subscription, read from its own price row.

    Reads the USD row rather than the plan's INR row so the two cannot silently
    diverge for a USD customer, even though they are seeded identical today.
    """
    result = await db.execute(
        select(PlanPrice.ai_credit_limit).where(
            PlanPrice.plan_id == plan_id,
            PlanPrice.billing_interval == (billing_interval or "monthly"),
            PlanPrice.currency == USD,
            PlanPrice.isactive.is_(True),
        )
    )
    credits = result.scalars().first()
    if credits is not None:
        return credits
    return limits.get("max_ai_credits_month", 0)


async def upsert_usd_license(
    db: AsyncSession,
    *,
    subscription: Subscription,
    limits: dict,
    reset_usage: bool,
) -> None:
    """Create or refresh the licence that grants the customer access.

    ``reset_usage`` must only be true when a new billing period genuinely began.
    Resetting on any other path would let a caller that can be replayed — such as
    a checkout-callback verify — hand the customer a fresh quota on demand.
    """
    ai_credit_limit = await usd_ai_credit_limit(
        db, subscription.plan_id, subscription.billing_interval, limits
    )

    other_active = await db.execute(
        select(LicenseAssignment).where(
            LicenseAssignment.user_id == subscription.user_id,
            LicenseAssignment.status == LicenseStatus.ACTIVE,
            LicenseAssignment.subscription_id != subscription.id,
        )
    )
    for license_obj in other_active.scalars():
        license_obj.status = LicenseStatus.REVOKED

    result = await db.execute(
        select(LicenseAssignment).where(
            LicenseAssignment.subscription_id == subscription.id,
            LicenseAssignment.user_id == subscription.user_id,
        )
    )
    existing = result.scalars().first()

    if existing is not None:
        existing.status = LicenseStatus.ACTIVE
        existing.limit_max_products = limits.get("max_products", 0)
        existing.limit_max_ai_credits = ai_credit_limit
        existing.limit_max_public_views = limits.get("max_public_views", 0)
        existing.limit_max_galleries = limits.get("max_galleries", 0)
        if reset_usage:
            existing.usage_ai_credits = 0
            existing.usage_public_views = 0
    else:
        db.add(
            LicenseAssignment(
                subscription_id=subscription.id,
                user_id=subscription.user_id,
                status=LicenseStatus.ACTIVE,
                limit_max_products=limits.get("max_products", 0),
                limit_max_ai_credits=ai_credit_limit,
                limit_max_public_views=limits.get("max_public_views", 0),
                limit_max_galleries=limits.get("max_galleries", 0),
                usage_products=0,
                usage_ai_credits=0,
                usage_public_views=0,
                usage_galleries=0,
            )
        )

    await db.flush()
