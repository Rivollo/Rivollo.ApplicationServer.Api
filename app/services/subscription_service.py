"""Service layer for subscription business logic.

This module contains ALL business rules and logic for subscriptions.
It orchestrates repository calls and transforms data into API-ready formats.

Key Concepts:
- Subscription: A user's active subscription to a plan (Free, Pro, Enterprise)
- License: An active LicenseAssignment that tracks usage limits and counters
- Quotas: Resource limits (products, AI credits, views, galleries) and their usage
- Free Plan Fallback: If user has no active license, we return free plan defaults
  This ensures all users always have subscription information, even new users.

Architecture:
- Repository: Fetches raw data from database
- Service: Applies business rules, calculates quotas, formats responses
- Route: Orchestrates service calls and returns HTTP responses
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import LicenseAssignment, Plan, Subscription
from app.schemas.subscriptions import QuotaInfo, QuotaUsage, SubscriptionMe, TrialInfo
from app.services.licensing_service import LicensingService
from app.database.subscription_repo import SubscriptionRepository


class SubscriptionService:
    """Service for subscription business logic."""

    @staticmethod
    def _parse_license_json_fields(license_assignment: LicenseAssignment) -> tuple[dict, dict]:
        """
        Parse JSON fields from LicenseAssignment into dictionaries.

        LicenseAssignment stores limits and usage_counters as JSON strings.
        We parse them to dictionaries and cache the result to avoid re-parsing.

        Args:
            license_assignment: The LicenseAssignment model

        Returns:
            Tuple of (limits_dict, usage_dict)
        """
        # Check if already parsed and cached
        limits = getattr(license_assignment, "_limits_dict", None)
        if limits is None:
            limits = json.loads(license_assignment.limits) if license_assignment.limits else {}
            license_assignment._limits_dict = limits  # Cache for performance

        usage = getattr(license_assignment, "_usage_dict", None)
        if usage is None:
            usage = (
                json.loads(license_assignment.usage_counters)
                if license_assignment.usage_counters
                else {}
            )
            license_assignment._usage_dict = usage  # Cache for performance

        return limits, usage

    @staticmethod
    def _is_expired(period_end: Optional[datetime], now: datetime) -> bool:
        """Check whether a subscription billing period has ended.

        Handles both timezone-aware and timezone-naive datetimes returned by
        the database driver (asyncpg returns tz-aware, but older rows may not).

        Args:
            period_end: The subscription's current_period_end timestamp, or None.
            now:        The current UTC time (must be tz-aware).

        Returns:
            True if the period has ended, False if still active or no end date.
        """
        if period_end is None:
            return False  # No end date → treat as non-expiring (free plan forever)

        # Make period_end tz-aware if the driver returned a naive datetime.
        # asyncpg always returns tz-aware, but we guard defensively.
        if period_end.tzinfo is None:
            period_end = period_end.replace(tzinfo=timezone.utc)

        return period_end < now

    @staticmethod
    def _calculate_trial_info(subscription: Subscription) -> TrialInfo:
        """
        Calculate trial period information from subscription.

        Currently, trial_end_at is always NULL in the database (virtual column),
        so trials are always inactive. This function is prepared for future trial support.

        Args:
            subscription: The Subscription model

        Returns:
            TrialInfo with trial status and remaining days
        """
        trial_active = False
        days_remaining = 0
        trial_started = None

        if subscription.trial_end_at:
            now = datetime.utcnow()
            if subscription.trial_end_at > now:
                trial_active = True
                days_remaining = max(0, (subscription.trial_end_at - now).days)
                # Calculate trial start (assumes 7-day trial period)
                trial_started = subscription.trial_end_at - timedelta(days=7)

        return TrialInfo(
            active=trial_active,
            daysRemaining=days_remaining,
            startedAt=trial_started,
        )

    @staticmethod
    def _build_quotas(
        limits: dict,
        usage: dict,
        product_count: int,
    ) -> dict:
        """
        Build quota information dictionary from limits, usage, and product count.

        Quotas track resource usage:
        - aiCredits: AI processing credits (included in plan, can be purchased)
        - publicViews: Public product view count
        - products: Number of products created (counted separately)
        - galleries: Number of galleries created

        Args:
            limits: Dictionary of quota limits from LicenseAssignment
            usage: Dictionary of current usage from LicenseAssignment
            product_count: Number of products (counted from database)

        Returns:
            Dictionary of quota information ready for API response
        """
        quotas = {
            "aiCredits": QuotaUsage(
                included=limits.get("max_ai_credits_month", 5),  # Default: 5 for free plan
                purchased=0,  # Not implemented yet
                used=usage.get("ai_credits", 0),
            ).model_dump(),
            "publicViews": QuotaUsage(
                included=limits.get("max_public_views", 1000),  # Default: 1000 for free plan
                purchased=0,  # Not implemented yet
                used=usage.get("public_views", 0),
            ).model_dump(),
            "products": QuotaInfo(
                used=product_count,
                limit=limits.get("max_products"),  # Can be None (unlimited)
            ).model_dump(),
            "galleries": QuotaInfo(
                used=usage.get("galleries", 0),
                limit=limits.get("max_galleries"),  # Can be None (unlimited)
            ).model_dump(),
        }

        return quotas

    @staticmethod
    def _get_free_plan_defaults() -> SubscriptionMe:
        """
        Get default free plan subscription information.

        This is returned when a user has no active license.
        All new users start with free plan until they subscribe to a paid plan.

        Returns:
            SubscriptionMe with free plan defaults
        """
        return SubscriptionMe(
            plan="free",
            trial=TrialInfo(active=False, daysRemaining=0, startedAt=None),
            quotas={
                "aiCredits": QuotaUsage(included=5, purchased=0, used=0).model_dump(),
                "publicViews": QuotaUsage(included=1000, purchased=0, used=0).model_dump(),
                "products": QuotaInfo(used=0, limit=2).model_dump(),
                "galleries": QuotaInfo(used=0, limit=0).model_dump(),
            },
        )

    @staticmethod
    async def get_user_subscription(
        db: AsyncSession, user_id: uuid.UUID
    ) -> SubscriptionMe:
        """
        Get complete subscription information for a user.

        This is the main service method that orchestrates:
        1. Getting active license (via LicensingService - DO NOT MODIFY)
        2. Fetching subscription and plan data
        3. Calculating quotas and trial info
        4. Returning formatted response

        Business Rules:
        - If no active license exists → return free plan defaults
        - If license exists but subscription not found → return free plan defaults
        - Otherwise → calculate quotas from license and subscription data

        Args:
            db: Database session
            user_id: ID of the user

        Returns:
            SubscriptionMe with complete subscription information
        """
        # Step 1: Get active license (read-only access via LicensingService)
        # NOTE: LicensingService MUST NOT be modified per requirements
        license_assignment = await LicensingService.get_active_license(db, user_id)

        # Step 2: If no license, return free plan defaults
        # This happens for new users who haven't subscribed yet
        if not license_assignment:
            return SubscriptionService._get_free_plan_defaults()

        # Step 3: Fetch subscription and plan data from database
        subscription_plan = await SubscriptionRepository.get_subscription_and_plan(
            db, license_assignment.subscription_id
        )

        # Step 4: If subscription not found, return free plan defaults
        # This is a safety fallback
        if not subscription_plan:
            return SubscriptionMe(
                plan="free",
                trial=TrialInfo(active=False, daysRemaining=0, startedAt=None),
                quotas={},
            )

        subscription, plan = subscription_plan

        # ── Realtime expiry check ────────────────────────────────────────────
        # The background job deactivates expired subscriptions every 5 minutes,
        # but we MUST NOT rely on it for live API correctness.
        # If period_end has passed right now, return free plan immediately —
        # the background job will clean up the DB row on its next run.
        now = datetime.now(timezone.utc)
        if SubscriptionService._is_expired(subscription.current_period_end, now):
            return SubscriptionService._get_free_plan_defaults()

        # Step 5: Calculate trial information
        trial_info = SubscriptionService._calculate_trial_info(subscription)

        # Step 6: Parse license JSON fields (limits and usage counters)
        limits, usage = SubscriptionService._parse_license_json_fields(license_assignment)

        # Step 7: Count user's products (products are counted separately)
        product_count = await SubscriptionRepository.get_user_product_count(db, user_id)

        # Step 8: Build quota information
        quotas = SubscriptionService._build_quotas(limits, usage, product_count)

        # Step 9: Build and return complete subscription response
        return SubscriptionMe(
            plan=plan.code,  # "free", "pro", or "enterprise"
            trial=trial_info,
            quotas=quotas,
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
        )

