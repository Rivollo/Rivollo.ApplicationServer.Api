"""Schemas for USD subscription creation and promo validation."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _StrictCamelModel(BaseModel):
    """camelCase in, snake_case out — and nothing the server did not ask for.

    ``extra="forbid"`` is load-bearing rather than tidiness: it is what rejects a
    request carrying an amount, a currency or a plan ID. Price is resolved
    server-side from the tier and interval, so a client that tries to name its
    own price gets a validation error instead of being quietly ignored.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class CreateUsdSubscriptionRequest(_StrictCamelModel):
    """Everything the client is allowed to choose."""

    plan_code: str = Field(..., description="Tier to subscribe to, e.g. 'pro'.", examples=["pro"])
    billing_interval: Literal["monthly", "yearly"] = Field(
        default="monthly",
        description="'monthly' or 'yearly'. Promo codes apply to monthly only.",
    )
    promo_code: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional promo code. Monthly only; rejected on annual.",
    )


class CreateUsdSubscriptionResponse(_CamelModel):
    """Result of creating a USD subscription."""

    subscription_id: str = Field(..., description="Razorpay subscription ID (use in checkout).")
    plan_code: str
    key_id: str = Field(..., description="Razorpay Key ID for the checkout widget.")
    status: str = Field(..., description="Subscription status from Razorpay.")
    short_url: Optional[str] = Field(None, description="Razorpay hosted checkout URL.")
    currency: Literal["USD"]
    billing_interval: Literal["monthly", "yearly"]
    full_amount: int = Field(..., description="List price in cents.")
    upfront_amount: Optional[int] = Field(
        None,
        description="Charged at authentication, in cents. Null for annual, which "
        "is billed at its list price immediately.",
    )
    promo_code: Optional[str] = None
    first_charge_at: Optional[datetime] = Field(
        None,
        description="When the first full-price charge falls. Null for annual.",
    )


class VerifyUsdSubscriptionRequest(_StrictCamelModel):
    """The three values Razorpay's checkout handler returns."""

    razorpay_payment_id: str = Field(..., examples=["pay_PwFakePaymentId"])
    razorpay_subscription_id: str = Field(..., examples=["sub_PwFakeSubId"])
    razorpay_signature: str = Field(..., examples=["abc123hexsignature"])


class VerifyUsdSubscriptionResponse(_CamelModel):
    """Result of verifying and activating a USD subscription."""

    verified: bool
    message: str
    plan: Optional[str] = None
    subscription_id: Optional[str] = None
    period_end: Optional[datetime] = None


class ValidateUsdPromoResponse(_CamelModel):
    """Result of checking a promo code before checkout."""

    valid: bool
    code: str
    currency: Literal["USD"]
    full_amount: int = Field(..., description="List price in cents.")
    upfront_amount: int = Field(..., description="First-period price in cents.")
    description: Optional[str] = None
