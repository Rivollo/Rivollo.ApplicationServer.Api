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
from app.database.subscription_repo import SubscriptionRepository
from app.schemas.subscriptions import Plan as PlanSchema, PlanFeature, PlanPricing
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
    db: DB,
    current_user: OptionalUser = None,
):
    """List all available subscription plans."""
    
    # ── Fetch plans and features from DB ──────────────────────────────────────
    db_plans = await SubscriptionRepository.get_all_plans(db)
    
    plans_data = []

    # ── Map Normalized DB plans to output schemas ─────────────────────────────
    for p in db_plans:
        # Build the features list based on the junction table (tbl_plan_features)
        features_list = []
        for pf in getattr(p, "plan_features", []):
            if pf.feature:
                # Format label dynamically based on explicit limits or boolean availability
                label = pf.feature.name
                if pf.limit_value is not None:
                    # Example format: "50 product listings" instead of just "Product listings"
                    label = f"{pf.limit_value:,} {label.lower()}"
                    
                features_list.append(
                    PlanFeature(
                        label=label,
                        available=pf.is_available
                    )
                )

        # Build pricing options based on configured Razorpay plan IDs
        monthly_price = getattr(p, "price_inr", None) or 0
        yearly_price = getattr(p, "price_inr_yearly", None) or 0
        has_monthly = bool(getattr(p, "razorpay_plan_id", None))
        has_yearly = bool(getattr(p, "razorpay_plan_id_yearly", None))

        pricing = []
        if has_monthly or monthly_price > 0:
            pricing.append(
                PlanPricing(interval="monthly", priceINR=monthly_price, available=has_monthly)
            )
        if has_yearly or yearly_price > 0:
            pricing.append(
                PlanPricing(interval="yearly", priceINR=yearly_price, available=has_yearly)
            )

        # Append to response list using pure database values
        plans_data.append(
            PlanSchema(
                code=p.code,
                name=p.name,
                priceINR=monthly_price,
                priceINRYearly=yearly_price,
                pricing=pricing,
                description=getattr(p, "description", None) or f"{p.name} Plan",
                features=features_list,
                featured=getattr(p, "is_featured", False),
            )
        )

    return api_success([p.model_dump(by_alias=True) for p in plans_data])
