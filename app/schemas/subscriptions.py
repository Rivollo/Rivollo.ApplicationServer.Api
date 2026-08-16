"""Subscription and plan schemas matching OpenAPI spec."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class QuotaUsage(BaseModel):
    """Quota usage information."""

    included: int = Field(..., ge=0)
    purchased: int = Field(default=0, ge=0)
    used: int = Field(..., ge=0)


class QuotaInfo(BaseModel):
    """Quota information for a resource."""

    used: int = Field(..., ge=0)
    limit: Optional[int] = Field(None, description="null means unlimited")


class TrialInfo(BaseModel):
    """Trial period information."""

    active: bool
    days_remaining: int = Field(..., ge=0, le=7, alias="daysRemaining")
    started_at: Optional[datetime] = Field(None, alias="startedAt")

    class Config:
        populate_by_name = True


class SubscriptionMe(BaseModel):
    """Current user's subscription information.

    Fields:
        plan:        Plan code — "free", "pro", or "enterprise".
        trial:       Trial period status (currently always inactive).
        quotas:      Resource usage and limits.
        period_start: UTC ISO timestamp of when the current billing period began.
                     Null for free-plan users (no billing period).
        period_end:  UTC ISO timestamp of when the current billing period ends.
                     Null for free-plan users. Frontend uses this to show
                     time remaining / countdown.
        cancel_at_period_end:
                     True when the customer has cancelled but paid access runs
                     to period_end. The plan and quotas above are unaffected --
                     they still describe full paid access, because that is what
                     the customer has until the period ends. Only the renewal
                     is gone, so the frontend must say "Cancels on <period_end>"
                     rather than "Renews on <period_end>".
    """

    plan: str = Field(..., description="Plan code: free, pro, enterprise")
    trial: TrialInfo
    quotas: dict[str, Any]
    period_start: Optional[datetime] = Field(
        None,
        alias="periodStart",
        description="Billing period start (UTC ISO). Null for free-plan users.",
    )
    period_end: Optional[datetime] = Field(
        None,
        alias="periodEnd",
        description="Billing period end (UTC ISO). Frontend uses this for countdown.",
    )
    cancel_at_period_end: bool = Field(
        False,
        alias="cancelAtPeriodEnd",
        description=(
            "True when the subscription is cancelled but paid access continues "
            "to periodEnd. Show 'Cancels on <periodEnd>', not 'Renews on'."
        ),
    )

    class Config:
        populate_by_name = True


class PlanFeature(BaseModel):
    """Plan feature description."""

    label: str
    available: bool


class PlanPricing(BaseModel):
    """Pricing details for a specific billing interval."""

    interval: str = Field(..., description="Billing interval: 'monthly' or 'yearly'.")
    price_inr: int = Field(..., ge=0, alias="priceINR")
    ai_credits: int = Field(..., ge=0, alias="aiCredits")
    available: bool = Field(..., description="True if this interval is configured for purchase.")

    class Config:
        populate_by_name = True


class Plan(BaseModel):
    """Subscription plan details."""

    code: str = Field(..., description="Plan code: free, pro, enterprise.")
    name: str
    price_inr: int = Field(..., ge=0, alias="priceINR", description="Monthly price (kept for backward compatibility).")
    price_inr_yearly: int = Field(0, ge=0, alias="priceINRYearly", description="Yearly price.")
    pricing: list[PlanPricing] = Field(default_factory=list, description="Available billing intervals with pricing.")
    description: str = Field(..., max_length=500)
    features: list[PlanFeature]
    featured: bool = False

    class Config:
        populate_by_name = True


class PlanList(BaseModel):
    """List of available plans."""

    plans: list[Plan]
