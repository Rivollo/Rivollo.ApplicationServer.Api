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

    class Config:
        populate_by_name = True


class PlanFeature(BaseModel):
    """Plan feature description."""

    label: str
    available: bool


class Plan(BaseModel):
    """Subscription plan details."""

    name: str
    price_inr: int = Field(..., ge=0, alias="priceINR")
    description: str = Field(..., max_length=500)
    features: list[PlanFeature]
    featured: bool = False

    class Config:
        populate_by_name = True


class PlanList(BaseModel):
    """List of available plans."""

    plans: list[Plan]
