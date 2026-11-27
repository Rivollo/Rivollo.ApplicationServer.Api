"""Licensing and subscription management service."""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.agent_billing_repo import agent_billing_repository
from app.models.models import LicenseAssignment, Notification, Plan, Subscription, User
from app.models.subscription_enums import SubscriptionStatus


logger = logging.getLogger(__name__)


def carry_over_ai_credit_overdraft(usage: Optional[int], limit: Optional[int]) -> int:
    """AI-credit usage a licence starts its new period with.

    Normally 0. The last AI Assistant message of a period is allowed to finish
    past the limit (app/services/agent_billing/hold_service.py, "Overdraft"),
    so usage can end above it; that excess is carried into the new period
    instead of being forgiven by the reset. ``limit`` is the limit of the period
    that just ended; None (unlimited) can never be overdrawn.
    """
    if limit is None:
        return 0
    return max(int(usage or 0) - int(limit), 0)


class LicensingService:
    """Service for managing subscriptions and license enforcement."""

    @staticmethod
    def _quota_columns(quota_key: str) -> tuple[str, str] | None:
        if quota_key in ("max_products", "products"):
            return "limit_max_products", "usage_products"
        if quota_key in ("ai_credits", "max_ai_credits_month"):
            return "limit_max_ai_credits", "usage_ai_credits"
        if quota_key in ("public_views", "max_public_views"):
            return "limit_max_public_views", "usage_public_views"
        if quota_key in ("galleries", "max_galleries"):
            return "limit_max_galleries", "usage_galleries"
        return None

    @staticmethod
    def _quota_label(quota_key: str) -> str:
        labels = {
            "max_products": "product",
            "products": "product",
            "ai_credits": "AI credit",
            "max_ai_credits_month": "AI credit",
            "public_views": "public view",
            "max_public_views": "public view",
            "galleries": "gallery",
            "max_galleries": "gallery",
        }
        return labels.get(quota_key, quota_key.replace("_", " "))

    @staticmethod
    def _crossed_quota_thresholds(previous_usage: int, current_usage: int, limit: int) -> list[int]:
        if limit <= 0:
            return []

        previous_pct = (previous_usage / limit) * 100
        current_pct = (current_usage / limit) * 100

        for threshold in sorted(settings.quota_notification_thresholds(), reverse=True):
            if previous_pct < threshold <= current_pct:
                return [threshold]
        return []

    @staticmethod
    async def _quota_notification_exists(
        db: AsyncSession,
        user_id: uuid.UUID,
        notification_type: str,
    ) -> bool:
        result = await db.execute(
            select(Notification.id)
            .where(
                Notification.user_id == user_id,
                Notification.type == notification_type,
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    @staticmethod
    async def _notify_quota_threshold(
        db: AsyncSession,
        user_id: uuid.UUID,
        quota_key: str,
        usage: int,
        limit: int,
        threshold: int,
    ) -> None:
        notification_type = f"quota.{quota_key}.{threshold}"
        if await LicensingService._quota_notification_exists(db, user_id, notification_type):
            return

        quota_label = LicensingService._quota_label(quota_key)
        if threshold >= 100:
            title = "Quota Exhausted"
            body = f"You have used all of your {quota_label} quota."
        else:
            title = "Quota Warning"
            body = f"You have used {threshold}% of your {quota_label} quota."

        try:
            from app.services.notification_service import NotificationService

            await NotificationService.create_and_push_notification(
                db=db,
                user_id=user_id,
                notification_type=notification_type,
                title=title,
                body=body,
                data={
                    "quota": quota_key,
                    "usage": usage,
                    "limit": limit,
                    "threshold": threshold,
                    "percentage": min(100, round((usage / limit) * 100)),
                },
            )
        except Exception:
            logger.warning(
                "Failed to send quota threshold notification for user %s quota %s threshold %s",
                user_id,
                quota_key,
                threshold,
                exc_info=True,
            )
            try:
                await db.rollback()
            except Exception:
                pass

    @staticmethod
    async def get_active_license(db: AsyncSession, user_id: uuid.UUID) -> Optional[LicenseAssignment]:
        """Get active license for a user."""
        result = await db.execute(
            select(LicenseAssignment)
            .where(
                LicenseAssignment.user_id == user_id,
                LicenseAssignment.status == "active",
            )
            .order_by(LicenseAssignment.created_date.desc())
            .limit(1)
        )
        license_obj = result.scalar_one_or_none()
        return license_obj

    @staticmethod
    async def check_quota(
        db: AsyncSession,
        user_id: uuid.UUID,
        quota_key: str,
        increment: int = 1,
    ) -> tuple[bool, Optional[dict]]:
        """
        Check if user has quota available.

        Returns (allowed, limits_info)
        """
        license = await LicensingService.get_active_license(db, user_id)

        if not license:
            # No active license - deny
            return False, None

        columns = LicensingService._quota_columns(quota_key)
        if columns is None:
            # Unknown quota - fail safe by allowing or we could block
            return True, None
        limit_col, usage_col = columns

        # Dynamically fetch limit and usage from native integer columns
        limit_val = getattr(license, limit_col, 0)
        current_usage = getattr(license, usage_col, 0)
        
        # Build limits dict for legacy response struct, if API still expects it
        limits_info = {
            "max_products": license.limit_max_products,
            "max_ai_credits_month": license.limit_max_ai_credits,
            "max_public_views": license.limit_max_public_views,
            "max_galleries": license.limit_max_galleries,
        }

        # 0 or None means unlimited in this context if that's the business rule?
        # Actually in our activation service, 'enterprise' plan has hardcoded None for unlimited.
        # But wait, integer columns with default=0 means 0 is the limit.
        if limit_val is None:
            return True, limits_info

        # Credits held by running AI Assistant messages are spoken for: product
        # creation and generations must not spend them (see
        # app/services/agent_billing/hold_service.py).
        held = 0
        if usage_col == "usage_ai_credits":
            held = await LicensingService.held_ai_credits(db, user_id)

        if current_usage + held + increment > limit_val:
            # Quota exceeded
            info = {"limit": limit_val, "current": current_usage, "quota": quota_key}
            if held:
                info["held"] = held
            return False, info

        return True, limits_info

    @staticmethod
    async def held_ai_credits(db: AsyncSession, user_id: uuid.UUID) -> int:
        """AI credits held by the user's running AI Assistant messages.

        Always 0 while ENABLE_AGENT_BILLING is off, so environments without
        tbl_agent_usage (sql/create_agent_billing.sql) never query it.
        """
        if not settings.ENABLE_AGENT_BILLING:
            return 0
        return await agent_billing_repository.sum_open_holds(
            db, user_id, datetime.now(timezone.utc)
        )

    @staticmethod
    async def increment_usage(
        db: AsyncSession,
        user_id: uuid.UUID,
        quota_key: str,
        increment: int = 1,
    ) -> bool:
        """Increment usage counter for a user's license."""
        license = await LicensingService.get_active_license(db, user_id)

        if not license:
            return False

        columns = LicensingService._quota_columns(quota_key)
        if columns is None:
            return False

        limit_col, usage_col = columns
        limit_val = getattr(license, limit_col, 0)
        previous_usage = getattr(license, usage_col, 0) or 0
        current_usage = previous_usage + increment
        setattr(license, usage_col, current_usage)

        await db.commit()

        thresholds = LicensingService._crossed_quota_thresholds(
            previous_usage=previous_usage,
            current_usage=current_usage,
            limit=limit_val or 0,
        )
        for threshold in thresholds:
            await LicensingService._notify_quota_threshold(
                db=db,
                user_id=user_id,
                quota_key=quota_key,
                usage=current_usage,
                limit=limit_val,
                threshold=threshold,
            )

        return True

    @staticmethod
    async def notify_usage_thresholds(
        db: AsyncSession,
        user_id: uuid.UUID,
        quota_key: str,
        previous_usage: int,
        current_usage: int,
        limit: Optional[int],
    ) -> None:
        """Send the quota-threshold notifications a usage change crossed — the
        same ones increment_usage sends — for callers that update usage under
        their own lock and transaction (AI Assistant billing). Call after
        committing: a failed notification rolls the session back."""
        thresholds = LicensingService._crossed_quota_thresholds(
            previous_usage=previous_usage,
            current_usage=current_usage,
            limit=limit or 0,
        )
        for threshold in thresholds:
            await LicensingService._notify_quota_threshold(
                db=db,
                user_id=user_id,
                quota_key=quota_key,
                usage=current_usage,
                limit=limit,
                threshold=threshold,
            )

    @staticmethod
    async def get_user_plan_code(db: AsyncSession, user_id: uuid.UUID) -> str:
        """Get the plan code for a user (free, pro, enterprise)."""
        license = await LicensingService.get_active_license(db, user_id)

        if not license:
            return "free"

        # The plan counts only while its subscription is ACTIVE and inside its
        # paid period — the same rule GET /subscriptions/me applies. Starting a
        # checkout puts the new plan on the subscription row while it is still
        # PENDING; reading the plan without this check treated an unpaid (or
        # abandoned) checkout as Pro.
        result = await db.execute(
            select(Plan.code, Subscription.status, Subscription.current_period_end)
            .join(Subscription, Subscription.plan_id == Plan.id)
            .where(Subscription.id == license.subscription_id)
        )
        row = result.first()
        if row is None:
            return "free"
        plan_code, sub_status, period_end = row
        if sub_status != SubscriptionStatus.ACTIVE:
            return "free"
        if period_end is not None:
            if period_end.tzinfo is None:
                period_end = period_end.replace(tzinfo=timezone.utc)
            if period_end <= datetime.now(timezone.utc):
                return "free"
        return plan_code or "free"

    # Plans whose users may generate USDZ. GLB creation stays available on every
    # plan — only the GLB -> USDZ conversion is paid-tier.
    USDZ_PLAN_CODES = ("pro", "enterprise")

    # Shown to the user when the gate below denies the conversion. It must read as
    # "your GLB is here, the USDZ is not" — the upload/job paths deliberately keep
    # succeeding, unlike the /createProductFal 403 which blocks outright.
    USDZ_UPGRADE_MESSAGE = (
        "USDZ conversion requires a Pro or Enterprise plan. Your GLB was saved; "
        "please subscribe to generate a USDZ file."
    )

    @staticmethod
    async def can_create_usdz(db: AsyncSession, user_id: uuid.UUID | str) -> bool:
        """True when the user's plan includes USDZ conversion (pro/enterprise).

        Callers treat a False result as "skip the USDZ, keep the GLB" — never as
        an error. Any unusable user_id denies the conversion rather than raising,
        so a bad id can't hand out a paid feature.
        """
        try:
            uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
        except (AttributeError, TypeError, ValueError):
            logger.warning("can_create_usdz: unusable user_id %r — denying USDZ", user_id)
            return False

        plan_code = await LicensingService.get_user_plan_code(db, uid)
        allowed = plan_code in LicensingService.USDZ_PLAN_CODES
        if not allowed:
            logger.info(
                "USDZ conversion not included in plan '%s' (user %s) — GLB only",
                plan_code,
                uid,
            )
        return allowed

    @staticmethod
    async def create_free_plan_license(
        db: AsyncSession,
        user: User,
    ) -> LicenseAssignment:
        """Create a free plan license for a new user."""
        from app.database.subscription_repo import SubscriptionRepository
        
        # Get free plan with features
        free_plan = await SubscriptionRepository.get_plan_with_features(db, "free")

        if not free_plan:
            # Fallback if DB not seeded
            free_plan = Plan(
                code="free",
                name="Free",
            )
            db.add(free_plan)
            await db.flush()
            # If we created it now, it won't have features. 
            # In a real environment, it should exist with features.

        # Default limits for safety if features missing. AI credits default to 0:
        # product creation is paid-only (every AI creation path is gated on
        # credits; only /createProductFromGlb is free), so a missing or deleted
        # free-plan feature row must never hand a new signup free credits.
        limits = {
            "max_products": 2,
            "max_ai_credits_month": 0,
            "max_public_views": 1000,
            "max_galleries": 0,
        }

        # Override with database values if available
        if free_plan.plan_features:
            for pf in free_plan.plan_features:
                if pf.feature.code == "max_products":
                    limits["max_products"] = pf.limit_value or 0
                elif pf.feature.code in ["ai_credits", "max_ai_credits_month"]:
                    limits["max_ai_credits_month"] = pf.limit_value or 0
                elif pf.feature.code in ["public_views", "max_public_views"]:
                    limits["max_public_views"] = pf.limit_value or 0
                elif pf.feature.code in ["galleries", "max_galleries"]:
                    limits["max_galleries"] = pf.limit_value or 0

        # Create subscription
        subscription = Subscription(
            user_id=user.id,
            plan_id=free_plan.id,
            status="active",
            seats_purchased=1,
        )
        db.add(subscription)
        await db.flush()

        # Create license assignment
        license = LicenseAssignment(
            subscription_id=subscription.id,
            user_id=user.id,
            status="active",
            limit_max_products=limits["max_products"],
            limit_max_ai_credits=limits["max_ai_credits_month"],
            limit_max_public_views=limits["max_public_views"],
            limit_max_galleries=limits["max_galleries"],
            usage_products=0,
            usage_ai_credits=0,
            usage_public_views=0,
            usage_galleries=0,
        )
        db.add(license)
        await db.commit()

        return license
