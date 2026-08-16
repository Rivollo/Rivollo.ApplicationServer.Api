"""Schemas for the public pricing endpoint.

This is display data only. It deliberately carries no Razorpay plan IDs — those
stay server-side, so a client can never name the plan it wants to be charged
for. Checkout takes a tier and a period and resolves the plan ID itself.
"""

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


class PricingFeature(_CamelModel):
    label: str
    available: bool


class PricingPeriod(_CamelModel):
    """One purchasable billing interval for a tier."""

    interval: Literal["monthly", "yearly"]
    amount_minor: int = Field(
        ..., ge=0, description="List price in minor units (paise for INR, cents for USD)."
    )
    formatted: str = Field(..., description="Display-ready list price, e.g. '$20.00'.")
    ai_credits: int = Field(0, ge=0)
    available: bool = Field(
        ..., description="False when no gateway plan is configured for this interval."
    )


class AnnualSaving(_CamelModel):
    """What the customer saves by paying annually instead of monthly.

    Derived from the two amounts in `periods` — never a stored or hardcoded
    figure, so it cannot drift when a price moves. For USD this saving is
    already inside the annual list price (annual = 10x monthly, i.e. two months
    free, permanently). It is copy, not a discount to apply at checkout.
    """

    amount_minor: int
    formatted: str
    percent: int


class PricingPromo(_CamelModel):
    """A promotional first period, advertised and auto-applied together.

    Present only for intervals that are actually eligible. Annual never carries
    one: its discount is permanent and already in the list price.
    """

    interval: Literal["monthly"]
    code: str
    first_amount_minor: int
    first_formatted: str
    headline: str
    detail: str = Field(
        ...,
        description="Full-price disclosure: what is charged now, and the exact "
        "amount and date of the first full-price charge.",
    )


class PricingTier(_CamelModel):
    code: str
    name: str
    description: str
    featured: bool = False
    features: list[PricingFeature] = Field(default_factory=list)
    periods: list[PricingPeriod] = Field(default_factory=list)
    annual_saving: Optional[AnnualSaving] = None
    promo: Optional[PricingPromo] = None


class PricingResponse(_CamelModel):
    """Everything a pricing page needs to render, for one visitor."""

    currency: Literal["INR", "USD"]
    currency_symbol: str
    country: Optional[str] = Field(
        None, description="Resolved ISO country code, or null when undetermined."
    )
    currency_locked: bool = Field(
        False,
        description="True when the currency came from the user's existing "
        "subscription rather than their location. Currency is locked at first "
        "subscription, so a USD subscriber browsing from India still sees USD.",
    )
    tax_note: str
    tiers: list[PricingTier] = Field(default_factory=list)
