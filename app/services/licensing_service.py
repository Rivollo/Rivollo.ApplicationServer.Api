"""Licensing and subscription management service."""

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import LicenseAssignment, Plan, Subscription, User


class LicensingService:
    """Service for managing subscriptions and license enforcement."""

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

        # Map quota_key to internal column names
        # Default mapping logic based on legacy keys:
        if quota_key == "max_products":
            limit_col = "limit_max_products"
            usage_col = "usage_products"
        elif quota_key == "ai_credits" or quota_key == "max_ai_credits_month":
            limit_col = "limit_max_ai_credits"
            usage_col = "usage_ai_credits"
        elif quota_key == "public_views" or quota_key == "max_public_views":
            limit_col = "limit_max_public_views"
            usage_col = "usage_public_views"
        elif quota_key == "galleries" or quota_key == "max_galleries":
            limit_col = "limit_max_galleries"
            usage_col = "usage_galleries"
        else:
            # Unknown quota - fail safe by allowing or we could block
            return True, None

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

        if quota_key == "max_products" or quota_key == "products":
            license.usage_products += increment
        elif quota_key == "ai_credits" or quota_key == "max_ai_credits_month":
            license.usage_ai_credits += increment
        elif quota_key == "public_views" or quota_key == "max_public_views":
            license.usage_public_views += increment
        elif quota_key == "galleries" or quota_key == "max_galleries":
            license.usage_galleries += increment

        await db.commit()
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
        # Get or create free plan
        result = await db.execute(select(Plan).where(Plan.code == "free"))
        free_plan = result.scalar_one_or_none()

        if not free_plan:
            quotas_dict = {
                "max_products": 2,
                "max_ai_credits_month": 5,
                "max_public_views": 1000,
            }
            free_plan = Plan(
                code="free",
                name="Free",
                quotas=json.dumps(quotas_dict),  # Serialize to JSON string
            )
            db.add(free_plan)
            await db.flush()

        # Create subscription
        subscription = Subscription(
            user_id=user.id,
            plan_id=free_plan.id,
            status="active",
            seats_purchased=1,
        )
        db.add(subscription)
        await db.flush()

        # Parse quotas from TEXT field
        quotas_dict = json.loads(free_plan.quotas) if free_plan.quotas else {}

        # Create license assignment
        license = LicenseAssignment(
            subscription_id=subscription.id,
            user_id=user.id,
            status="active",
            limit_max_products=int(quotas_dict.get("max_products", 2) or 0),
            limit_max_ai_credits=int(quotas_dict.get("max_ai_credits_month", 5) or 0),
            limit_max_public_views=int(quotas_dict.get("max_public_views", 1000) or 0),
            limit_max_galleries=int(quotas_dict.get("max_galleries", 0) or 0),
            usage_products=0,
            usage_ai_credits=0,
            usage_public_views=0,
            usage_galleries=0,
        )
        db.add(license)
        await db.commit()

        return license
