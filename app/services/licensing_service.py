"""Licensing and subscription management service."""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import LicenseAssignment, Notification, Plan, Subscription, User


logger = logging.getLogger(__name__)


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

        if current_usage + increment > limit_val:
            # Quota exceeded
            return False, {"limit": limit_val, "current": current_usage, "quota": quota_key}

        return True, limits_info

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
    async def get_user_plan_code(db: AsyncSession, user_id: uuid.UUID) -> str:
        """Get the plan code for a user (free, pro, enterprise)."""
        license = await LicensingService.get_active_license(db, user_id)

        if not license:
            return "free"

        # Get subscription and plan
        result = await db.execute(
            select(Plan)
            .join(Subscription)
            .where(Subscription.id == license.subscription_id)
        )
        plan = result.scalar_one_or_none()

        return plan.code if plan else "free"

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

        # Default limits for safety if features missing
        limits = {
            "max_products": 2,
            "max_ai_credits_month": 50,
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

        # Free users now always start with 50 AI credits.
        limits["max_ai_credits_month"] = 50

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
