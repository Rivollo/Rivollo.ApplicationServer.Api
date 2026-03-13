"""Schemas for Razorpay Subscription APIs (create, verify, cancel).

Pydantic v2 models with camelCase <-> snake_case alias mapping.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    """Base model that accepts camelCase input and snake_case attribute access."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Create Subscription
# ---------------------------------------------------------------------------


class CreateSubscriptionRequest(_CamelModel):
    """Request body for creating a Razorpay subscription.

    The amount is derived server-side from plan_code — never trust frontend.
    """

    plan_code: str = Field(
        ...,
        description="Plan to subscribe: 'pro'.",
        examples=["pro"],
    )
    billing_interval: Literal["monthly", "yearly"] = Field(
        default="monthly",
        description="Billing interval: 'monthly' or 'yearly'.",
        examples=["monthly", "yearly"],
    )
    offer_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional Razorpay offer ID for discounted billing cycles.",
        examples=["offer_JHD834hdjsl"],
    )


class CreateSubscriptionResponse(_CamelModel):
    """Response after creating a Razorpay subscription."""

    subscription_id: str = Field(
        ..., description="Razorpay subscription ID (use in checkout)."
    )
    plan_code: str = Field(..., description="Plan code subscribed to.")
    key_id: str = Field(
        ..., description="Razorpay Key ID — pass to frontend checkout widget."
    )
    status: str = Field(..., description="Subscription status from Razorpay.")
    short_url: Optional[str] = Field(
        None, description="Razorpay hosted checkout URL (optional fallback)."
    )


# ---------------------------------------------------------------------------
# Verify Subscription
# ---------------------------------------------------------------------------


class VerifySubscriptionRequest(_CamelModel):
    """Request body for verifying a Razorpay subscription payment.

    After user completes Razorpay checkout, the handler callback returns
    razorpay_payment_id, razorpay_subscription_id, razorpay_signature.
    """

    razorpay_payment_id: str = Field(
        ...,
        description="Payment ID returned by Razorpay after checkout.",
        examples=["pay_PwFakePaymentId"],
    )
    razorpay_subscription_id: str = Field(
        ...,
        description="Subscription ID returned by Razorpay.",
        examples=["sub_PwFakeSubId"],
    )
    razorpay_signature: str = Field(
        ...,
        description="HMAC-SHA256 signature from Razorpay checkout handler.",
        examples=["abc123hexsignature"],
    )


class VerifySubscriptionResponse(_CamelModel):
    """Result of subscription verification + activation."""

    verified: bool = Field(..., description="True if the signature is valid.")
    message: str = Field(..., description="Human-readable result.")
    plan: Optional[str] = Field(None, description="Activated plan code.")
    subscription_id: Optional[str] = Field(
        None, description="Internal subscription UUID."
    )
    period_end: Optional[datetime] = Field(
        None, description="When the current billing period ends."
    )


# ---------------------------------------------------------------------------
# Cancel Subscription
# ---------------------------------------------------------------------------


class CancelSubscriptionRequest(_CamelModel):
    """Request body for cancelling a Razorpay subscription."""

    cancel_at_cycle_end: bool = Field(
        default=True,
        description="If true, user keeps access until current period ends. "
        "If false, cancel immediately.",
    )


class CancelSubscriptionResponse(_CamelModel):
    """Result of subscription cancellation."""

    cancelled: bool = Field(..., description="True if cancellation succeeded.")
    message: str = Field(..., description="Human-readable result.")
    access_until: Optional[datetime] = Field(
        None, description="User retains access until this date."
    )
