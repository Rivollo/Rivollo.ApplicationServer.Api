"""Payment schemas — promo code validation.

All subscription schemas are in razorpay_subscriptions.py.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _CamelModel(BaseModel):
    """Base model that accepts camelCase input and snake_case attribute access."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
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


