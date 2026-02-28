"""Subscription and plan management routes.

This module contains ONLY route handlers - thin orchestration layer.
All business logic is in SubscriptionService.
All database access is in SubscriptionRepository.

Architecture:
- Route: Handles HTTP requests/responses, calls service
- Service: Contains business logic, orchestrates repository calls
- Repository: Contains database queries only
"""

from fastapi import APIRouter, Depends

from app.api.deps import CurrentUser, DB, OptionalUser
from app.schemas.subscriptions import Plan as PlanSchema, PlanFeature
from app.services.subscription_service import SubscriptionService
from app.utils.envelopes import api_success

router = APIRouter(tags=["subscriptions"])


@router.get("/subscriptions/me", response_model=dict)
async def get_my_subscription(
    current_user: CurrentUser,
    db: DB,
):
    """
    Get current user's subscription and quota information.

    This endpoint returns:
    - Current plan (free, pro, enterprise)
    - Trial status (if applicable)
    - Quota usage for all resources (AI credits, views, products, galleries)

    Returns free plan defaults if user has no active subscription.
    """
    # Delegate all logic to service layer
    subscription_data = await SubscriptionService.get_user_subscription(db, current_user.id)

    # Return formatted response
    # by_alias=True → uses camelCase aliases (periodStart, periodEnd, daysRemaining, etc.)
    # exclude_none=True → omits null fields for free-plan users (no period dates)
    return api_success(subscription_data.model_dump(by_alias=True, exclude_none=True))


@router.get("/subscriptions/plans", response_model=dict)
async def list_plans(
    current_user: OptionalUser = None,
):
    """List all available subscription plans (public endpoint)."""
    plans_data = [
        PlanSchema(
            name="Free",
            priceINR=0,
            description="Perfect for trying out Rivollo",
            features=[
                PlanFeature(label="2 product listings", available=True),
                PlanFeature(label="5 AI credits per month", available=True),
                PlanFeature(label="1,000 public views", available=True),
                PlanFeature(label="Basic analytics", available=True),
                PlanFeature(label="Galleries", available=False),
                PlanFeature(label="Advanced analytics", available=False),
                PlanFeature(label="Custom branding", available=False),
            ],
            featured=False,
        ),
        PlanSchema(
            name="Pro",
            priceINR=1999,
            description="Scale with galleries, credits, views, and advanced analytics",
            features=[
                PlanFeature(label="50 product listings", available=True),
                PlanFeature(label="50 AI credits per month", available=True),
                PlanFeature(label="25,000 public views", available=True),
                PlanFeature(label="10 galleries", available=True),
                PlanFeature(label="Advanced analytics", available=True),
                PlanFeature(label="Priority support", available=True),
                PlanFeature(label="Custom branding", available=True),
            ],
            featured=True,
        ),
        PlanSchema(
            name="Enterprise",
            priceINR=0,
            description="Unlimited everything with dedicated support. Contact sales for pricing.",
            features=[
                PlanFeature(label="Unlimited products", available=True),
                PlanFeature(label="Unlimited AI credits", available=True),
                PlanFeature(label="Unlimited public views", available=True),
                PlanFeature(label="Unlimited galleries", available=True),
                PlanFeature(label="Advanced analytics", available=True),
                PlanFeature(label="Custom branding", available=True),
                PlanFeature(label="Dedicated account manager", available=True),
                PlanFeature(label="SLA guarantee", available=True),
            ],
            featured=False,
        ),
    ]

    return api_success([p.model_dump() for p in plans_data])
