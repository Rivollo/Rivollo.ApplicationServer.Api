"""Subscription Activation Service — activates subscription + license after payment.

This is the heart of the payment flow. It runs AFTER payment is verified.

What it does (3 things every time):
    1. Look up the plan by code (pro / enterprise)
    2. Create or extend the subscription in tbl_subscriptions
    3. Create or update the license assignment in tbl_license_assignments

Testing note:
    SUBSCRIPTION_PERIOD is set to 10 minutes for easy testing.
    Change it to timedelta(days=30) before going to production.

Design decisions:
    - If user already has an active subscription → EXTEND period_end by SUBSCRIPTION_PERIOD
    - If user has no subscription → CREATE a new subscription row
    - License row uses (subscription_id, user_id) unique key:
        * If same subscription is extended → UPDATE the existing license row
        * If new subscription row is created → INSERT a new license row
    - usage_counters are reset to {} only on new subscription, NOT on extension
      (so the user doesn't lose their usage history mid-period)
    - limits are always written fresh from PLAN_LIMITS (in case plan changes)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.license_assignment import LicenseAssignment
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.subscription_enums import LicenseStatus, SubscriptionStatus

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# TESTING PERIOD — change to timedelta(days=30) for production
# ─────────────────────────────────────────────────────────────────
SUBSCRIPTION_PERIOD = timedelta(days=10)


# ─────────────────────────────────────────────────────────────────
# Plan limits — server-side source of truth for quota values
# These are the limits that get written to tbl_license_assignments.limits
# ─────────────────────────────────────────────────────────────────
PLAN_LIMITS: dict[str, dict] = {
    "pro": {
        "max_products": 50,
        "max_ai_credits_month": 50,
        "max_public_views": 25000,
        "max_galleries": 10,
    },
    "enterprise": {
        # None = unlimited (frontend should show "Unlimited")
        "max_products": None,
        "max_ai_credits_month": None,
        "max_public_views": None,
        "max_galleries": None,
    },
}

# Plan price in paise — server-side source of truth (never trust frontend amount).
# Only "pro" is listed here because "enterprise" is a contact-sales plan
# and does NOT go through the self-serve Razorpay payment flow.
PLAN_PRICES_PAISE: dict[str, int] = {
    "pro": 199900,  # ₹1,999/month
}


class SubscriptionActivationService:
    """Service to activate a subscription after successful Razorpay payment."""

    @staticmethod
    async def activate(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        plan_code: str,
        razorpay_order_id: str,
        razorpay_payment_id: str,
    ) -> Subscription:
        """Activate or extend a user's subscription after payment.

        Steps:
            1. Fetch plan from DB (validates the plan_code exists)
            2. Look up user's existing subscription (any status)
            3. Create new OR extend existing subscription
            4. Upsert license assignment with fresh limits

        Args:
            db: Async database session (transaction managed by caller)
            user_id: UUID of the user who paid
            plan_code: "pro" or "enterprise"
            razorpay_order_id: Razorpay order ID (stored in billing JSON)
            razorpay_payment_id: Razorpay payment ID (stored in billing JSON)

        Returns:
            The activated / extended Subscription row

        Raises:
            HTTPException 400: if plan_code is not recognised
        """
        # ── 1. Validate plan_code and load plan from DB ──────────────────────
        plan = await SubscriptionActivationService._get_plan(db, plan_code)

        # ── 2. Get limits for this plan ──────────────────────────────────────
        limits = PLAN_LIMITS.get(plan_code)
        if limits is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No limits configured for plan '{plan_code}'.",
            )

        now = datetime.now(timezone.utc)
        period_start = now
        period_end = now + SUBSCRIPTION_PERIOD

        # ── 3. Billing JSON to store payment reference in subscription ───────
        billing_data = {
            "razorpay_order_id": razorpay_order_id,
            "razorpay_payment_id": razorpay_payment_id,
            "plan_code": plan_code,
            "paid_at": now.isoformat(),
        }
        billing_json = json.dumps(billing_data)

        # ── 4. Find existing subscription for this user ──────────────────────
        existing_sub = await SubscriptionActivationService._get_user_subscription(
            db, user_id
        )

        if existing_sub is not None:
            # ── EXTEND existing subscription ─────────────────────────────────
            logger.info(
                "Extending subscription %s for user %s (plan=%s)",
                existing_sub.id, user_id, plan_code,
            )
            existing_sub.plan_id = plan.id
            existing_sub.status = SubscriptionStatus.ACTIVE
            existing_sub.current_period_start = period_start
            # If subscription is being renewed (not expired yet), extend from now
            # If it was already expired, start fresh from now
            if (
                existing_sub.current_period_end
                and existing_sub.current_period_end > now
            ):
                # Still active — extend from the current end
                period_end = existing_sub.current_period_end + SUBSCRIPTION_PERIOD
            existing_sub.current_period_end = period_end
            existing_sub.billing = billing_json
            subscription = existing_sub
        else:
            # ── CREATE new subscription ──────────────────────────────────────
            logger.info(
                "Creating new subscription for user %s (plan=%s)", user_id, plan_code
            )
            subscription = Subscription(
                user_id=user_id,
                plan_id=plan.id,
                status=SubscriptionStatus.ACTIVE,
                seats_purchased=1,
                current_period_start=period_start,
                current_period_end=period_end,
                billing=billing_json,
            )
            db.add(subscription)
            await db.flush()  # get subscription.id before creating license
            await db.refresh(subscription)

        # ── 5. Upsert license assignment ─────────────────────────────────────
        await SubscriptionActivationService._upsert_license(
            db=db,
            subscription=subscription,
            user_id=user_id,
            limits=limits,
            is_new_subscription=(existing_sub is None),
        )

        logger.info(
            "Subscription %s activated for user %s until %s",
            subscription.id, user_id, period_end.isoformat(),
        )
        return subscription

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    async def _get_plan(db: AsyncSession, plan_code: str) -> Plan:
        """Fetch a plan by code from the database.

        Validation strategy:
            1. Validate plan_code against PLAN_LIMITS (server-side source of truth).
               This prevents any unknown plan from ever being activated.
            2. Query tbl_mstr_plans for the matching row.
            3. If no row exists yet (unseeded DB), create a transient Plan object
               so activation can still proceed without crashing.
        """
        # ── Step 1: validate against our authoritative plan list ─────────────
        valid_plans = list(PLAN_LIMITS.keys())
        if plan_code not in valid_plans:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Plan '{plan_code}' not found. Valid values: {', '.join(valid_plans)}.",
            )

        # ── Step 2: try to fetch from DB ─────────────────────────────────────
        result = await db.execute(select(Plan).where(Plan.code == plan_code))
        plan = result.scalar_one_or_none()

        if plan is not None:
            return plan

        # ── Step 3: DB row missing (unseeded) — create + persist it ──────────
        logger.warning(
            "Plan '%s' not found in tbl_mstr_plans — creating it now.", plan_code
        )
        plan = Plan(
            code=plan_code,
            name=plan_code.capitalize(),
        )
        db.add(plan)
        await db.flush()   # assigns plan.id without committing the outer txn
        await db.refresh(plan)
        return plan

    @staticmethod
    async def _get_user_subscription(
        db: AsyncSession, user_id: uuid.UUID
    ) -> Optional[Subscription]:
        """Fetch the most recent subscription for a user (any status)."""
        result = await db.execute(
            select(Subscription)
            .where(Subscription.user_id == user_id)
            .order_by(Subscription.current_period_end.desc().nullslast())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def _upsert_license(
        db: AsyncSession,
        *,
        subscription: Subscription,
        user_id: uuid.UUID,
        limits: dict,
        is_new_subscription: bool,
    ) -> LicenseAssignment:
        """Create or update the license assignment for this subscription.

        For a new subscription → create new license row, reset usage_counters to {}
        For an extended subscription → update limits, keep existing usage_counters
        """
        limits_json = json.dumps(limits)

        # Check if a license already exists for this (subscription_id, user_id)
        existing_license_result = await db.execute(
            select(LicenseAssignment).where(
                LicenseAssignment.subscription_id == subscription.id,
                LicenseAssignment.user_id == user_id,
            )
        )
        existing_license = existing_license_result.scalar_one_or_none()

        if existing_license is not None:
            # ── UPDATE: refresh limits but keep existing usage ───────────────
            existing_license.status = LicenseStatus.ACTIVE
            existing_license.limits = limits_json
            # Don't reset usage_counters — user keeps their usage history
            license_assignment = existing_license
        else:
            # ── CREATE: fresh license with zeroed usage ──────────────────────
            empty_usage = json.dumps({
                "ai_credits": 0,
                "public_views": 0,
                "galleries": 0,
            })
            license_assignment = LicenseAssignment(
                subscription_id=subscription.id,
                user_id=user_id,
                status=LicenseStatus.ACTIVE,
                limits=limits_json,
                usage_counters=empty_usage,
            )
            db.add(license_assignment)

        await db.flush()
        return license_assignment

    @staticmethod
    def get_plan_price_paise(plan_code: str) -> int:
        """Return the server-side price for a plan in paise.

        This is the authoritative source of truth — never trust frontend amounts.

        Args:
            plan_code: "pro" or "enterprise"

        Returns:
            Amount in paise

        Raises:
            HTTPException 400: if plan_code is not recognised
        """
        price = PLAN_PRICES_PAISE.get(plan_code)
        if price is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown plan '{plan_code}'. Valid values: {list(PLAN_PRICES_PAISE.keys())}.",
            )
        return price
