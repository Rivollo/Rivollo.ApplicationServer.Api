"""Payment schemas for Razorpay integration.

Pydantic v2 models for create order and verify payment flows.

Aliases (camelCase ↔ snake_case) are applied globally via model_config +
alias_generator so every field is automatically available under both names.
This avoids the Pydantic v2 UnsupportedFieldAttributeWarning that occurs
when `alias=` is passed inside Field().

Input:  camelCase JSON from the frontend  (e.g. planCode, razorpayOrderId)
Output: snake_case Python attrs            (e.g. plan_code, razorpay_order_id)
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


# ---------------------------------------------------------------------------
# Shared base — all payment schemas accept camelCase input
# ---------------------------------------------------------------------------


class _CamelModel(BaseModel):
    """Base model that accepts camelCase input and snake_case attribute access."""

    model_config = ConfigDict(
        alias_generator=to_camel,   # planCode ↔ plan_code, razorpayOrderId ↔ razorpay_order_id
        populate_by_name=True,      # also accept snake_case field names
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Create Order
# ---------------------------------------------------------------------------


class CreateOrderRequest(_CamelModel):
    """Request body for creating a Razorpay order.

    The `plan_code` field is required. The server derives the correct amount
    from PLAN_PRICES_PAISE — never trust a frontend-supplied amount.
    """

    plan_code: str = Field(
        ...,
        description="Plan to purchase: 'pro' or 'enterprise'.",
        examples=["pro"],
    )
    currency: str = Field(
        default="INR",
        max_length=3,
        description="ISO 4217 currency code.",
        examples=["INR"],
    )
    receipt: Optional[str] = Field(
        default=None,
        max_length=40,
        description="Your internal receipt / order reference (max 40 chars).",
        examples=["receipt_pro_001"],
    )
    notes: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arbitrary key-value notes attached to the order.",
        examples=[{"user": "pratik"}],
    )

    promo_code: str | None = Field(
        default=None,
        max_length=50,
        description="Optional promo or discount code applied to the order.",
        examples=["NEWUSER20"],
    )

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "planCode": "pro",
                "currency": "INR",
                "receipt": "receipt_pro_001",
            }
        },
    )


class CreateOrderResponse(_CamelModel):
    """Razorpay order details returned to the client."""

    order_id: str = Field(..., description="Razorpay order ID (use in checkout).")
    amount: int = Field(..., description="Order amount in paise.")
    currency: str = Field(..., description="Currency code.")
    receipt: Optional[str] = Field(None, description="Receipt / reference ID.")
    status: str = Field(..., description="Order status from Razorpay (e.g. 'created').")
    key_id: str = Field(
        ...,
        description="Razorpay Key ID — pass this to the frontend Razorpay checkout widget.",
    )
    plan_code: str = Field(
        ...,
        description="Plan code for which this order was created.",
    )


# ---------------------------------------------------------------------------
# Verify Payment
# ---------------------------------------------------------------------------


class VerifyPaymentRequest(_CamelModel):
    """Request body for verifying a completed Razorpay payment.

    Send the 3 values from the Razorpay checkout handler callback + the
    plan_code so the backend knows which subscription to activate.

    Frontend sends camelCase (razorpayOrderId, razorpayPaymentId,
    razorpaySignature, planCode) — all mapped automatically via alias_generator.
    """

    razorpay_order_id: str = Field(
        ...,
        description="Razorpay order ID returned when the order was created.",
        examples=["order_PwFakeExampleId"],
    )
    razorpay_payment_id: str = Field(
        ...,
        description="Payment ID returned by Razorpay after checkout.",
        examples=["pay_PwFakePaymentId"],
    )
    razorpay_signature: str = Field(
        ...,
        description="HMAC-SHA256 signature from Razorpay checkout handler.",
        examples=["abc123hexsignature"],
    )
    plan_code: str = Field(
        ...,
        description="Plan to activate: 'pro' or 'enterprise'.",
        examples=["pro"],
    )

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "razorpayOrderId": "order_PwFakeExampleId",
                "razorpayPaymentId": "pay_PwFakePaymentId",
                "razorpaySignature": "abc123hexsignature",
                "planCode": "pro",
            }
        },
    )


# ---------------------------------------------------------------------------
# Promo Code Validation
# ---------------------------------------------------------------------------


class ValidatePromoResponse(_CamelModel):
    """Response for promo code validation."""

    valid: bool = Field(..., description="True if the promo code is valid and applicable.")
    code: str = Field(..., description="The promo code that was validated.")
    discount_type: Optional[str] = Field(None, description="'percentage' or 'fixed'.")
    discount_value: Optional[int] = Field(None, description="Discount amount or percentage.")
    description: Optional[str] = Field(None, description="Human-readable description of the promo code.")
    razorpay_offer_id: Optional[str] = Field(
        None,
        description="Razorpay offer ID to pass in the checkout payload (if applicable).",
    )
    message: str = Field(..., description="Human-readable validation result.")


class VerifyPaymentResponse(_CamelModel):
    """Result of signature verification + subscription activation."""

    verified: bool = Field(..., description="True if the payment signature is valid.")
    message: str = Field(..., description="Human-readable verification result.")

    # Razorpay echo
    razorpay_payment_id: Optional[str] = Field(
        None,
        description="Echo of the verified payment ID (only on success).",
    )
    razorpay_order_id: Optional[str] = Field(
        None,
        description="Echo of the verified order ID (only on success).",
    )

    # Subscription activation result
    plan: Optional[str] = Field(
        None,
        description="Activated plan code (e.g. 'pro'). Present on success.",
    )
    subscription_id: Optional[str] = Field(
        None,
        description="UUID of the activated subscription. Present on success.",
    )
    period_end: Optional[datetime] = Field(
        None,
        description="When the subscription period expires. Present on success.",
    )
