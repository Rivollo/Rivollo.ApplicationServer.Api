"""Repository layer for subscription-related database operations.

This module contains ONLY database access logic - no business rules.
Repository functions fetch data from the database and return raw models or primitive values.

Key Concepts:
- Subscription: A user's subscription to a plan (links user to a plan)
- Plan: A subscription tier (Free, Pro, Enterprise) with defined features
- LicenseAssignment: Tracks active license with usage limits and counters
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import select, update, text
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import LicenseAssignment, Product, Subscription
from app.models.plan import Plan, PlanFeature , PlanPrice
from app.queries.subscription_queries import GET_USER_PLAN_CODE


class SubscriptionRepository:
    """Repository for subscription database operations."""

    @staticmethod
    async def get_subscription_and_plan(
        db: AsyncSession, subscription_id: uuid.UUID
    ) -> Optional[Tuple[Subscription, Plan]]:
        """
        Fetch a subscription and its associated plan from the database.

        Args:
            db: Database session
            subscription_id: ID of the subscription to fetch

        Returns:
            Tuple of (Subscription, Plan) if found, None otherwise
        """
        result = await db.execute(
            select(Subscription, Plan)
            .join(Plan, Subscription.plan_id == Plan.id)
            .where(Subscription.id == subscription_id)
        )
        row = result.first()
        return row if row else None

    @staticmethod
    async def get_user_product_count(db: AsyncSession, user_id: uuid.UUID) -> int:
        """
        Count the number of non-deleted products created by a user.

        This is used for quota calculation - products are counted separately
        from other usage counters stored in LicenseAssignment.

        Args:
            db: Database session
            user_id: ID of the user whose products to count

        Returns:
            Number of products (integer)
        """
        result = await db.execute(
            select(Product).where(
                Product.created_by == user_id,
                Product.deleted_at.is_(None),
            )
        )
        products = result.scalars().all()
        return len(products)

    @staticmethod
    async def get_plan_by_code(db: AsyncSession, plan_code: str) -> Optional[Plan]:
        """
        Fetch a plan by its code (e.g., "free", "pro", "enterprise").

        Args:
            db: Database session
            plan_code: Code of the plan to fetch

        Returns:
            Plan model if found, None otherwise
        """
        result = await db.execute(select(Plan).where(Plan.code == plan_code))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_plan_with_features(db: AsyncSession, plan_code: str) -> Optional[Plan]:
        """
        Fetch a plan by its code with all its features and limits loaded.
        """
        result = await db.execute(
            select(Plan)
            .where(Plan.code == plan_code)
            .options(
                selectinload(Plan.plan_features).selectinload(PlanFeature.feature),
                selectinload(Plan.plan_prices)
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_plans(db: AsyncSession) -> list[Plan]:
        """
        Fetch all subscription plans from the database, including features.

        Args:
            db: Database session

        Returns:
            List of Plan models
        """
        result = await db.execute(
            select(Plan)
            .where(Plan.isactive == True)
            .options(
                selectinload(Plan.plan_features).selectinload(PlanFeature.feature),
                selectinload(Plan.plan_prices)
            )
            .order_by(Plan.created_date.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_user_plan_code(
        db: AsyncSession, user_id: uuid.UUID
    ) -> Optional[str]:
        """Fetch the user's active plan code (e.g. 'pro', 'free').

        Returns 'free' if the user has no active subscription.
        """
        result = await db.execute(
            text(GET_USER_PLAN_CODE),
            {"user_id": user_id},
        )
        row = result.fetchone()
        return row[0] if row else "free"
