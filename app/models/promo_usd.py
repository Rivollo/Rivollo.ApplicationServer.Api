"""USD promo codes.

Separate from ``tbl_promo_codes`` because the two work by different mechanisms.
INR promos are Razorpay Offers — the server validates eligibility and hands the
frontend a ``razorpay_offer_id``, and Razorpay does the discount arithmetic.
Razorpay Offers are INR-locked on this account and fail silently in USD, so USD
discounts are computed here and applied as a subscription upfront amount
instead. There is deliberately no ``razorpay_offer_id`` column on this table.

``is_public`` marks the promo that is advertised on the pricing page. The public
promo is auto-applied at checkout when the customer submits no code, so the
price shown is always the price charged.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base

# Values match tbl_promo_codes' existing discount_type vocabulary rather than
# inventing a second one for the same concept in a sibling table.
DISCOUNT_PERCENTAGE = "percentage"
DISCOUNT_FIXED = "fixed"


class PromoCodeUsd(Base):
    """A USD-only promotional discount on the first billing period."""

    __tablename__ = "tbl_promo_codes_usd"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # "percentage" (0-100) or "fixed" (cents off). Prefer percentage — a fixed
    # value carries a currency, and that is the failure mode that takes $4,000
    # off a $29 plan when someone reuses an INR-denominated number.
    discount_type: Mapped[str] = mapped_column(String(20), nullable=False)
    discount_value: Mapped[int] = mapped_column(Integer, nullable=False)

    # Annual is never eligible: its two-months-free discount is permanent and
    # already inside the list price, so a promo on top would double-count.
    billing_interval: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="monthly"
    )
    plan_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    max_redemptions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    used_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")

    # Advertised on the pricing page and auto-applied when no code is submitted.
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
