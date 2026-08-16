"""
PromoCode model — represents discount promo codes.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.models.base import Base


class PromoCode(Base):
    """SQLAlchemy model for tbl_promo_codes."""

    __tablename__ = "tbl_promo_codes"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    code = Column(
        String(50),
        unique=True,
        nullable=False,
    )

    discount_type = Column(
        String(20),
        nullable=False,
    )

    discount_value = Column(
        Integer,
        nullable=False,
    )

    max_usage = Column(
        Integer,
        nullable=True,
    )

    used_count = Column(
        Integer,
        default=0,
        nullable=False,
    )

    plan_code = Column(
        String(50),
        nullable=True,
    )

    # NULL = applies to every billing interval of the plan.
    billing_interval = Column(
        String(20),
        nullable=True,
    )

    valid_from = Column(
        DateTime(timezone=True),
        nullable=False,
    )

    valid_to = Column(
        DateTime(timezone=True),
        nullable=False,
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
    )

    description = Column(
        String(255),
        nullable=True,
    )

    razorpay_offer_id = Column(
        String(255),
        nullable=True,
    )

    # INR promos are applied through a Razorpay Offer; USD promos are computed
    # server-side and charged as a subscription addon, because Offers are
    # INR-locked on this account and fail silently against a USD plan. Every
    # lookup filters on this so a code for one currency cannot be redeemed
    # against the other.
    currency = Column(
        String(3),
        nullable=False,
        server_default="INR",
    )

    # Marks the promo advertised on the pricing page. It is auto-applied at
    # checkout when no code is submitted, so the displayed price and the charged
    # price cannot drift apart.
    is_public = Column(
        Boolean,
        nullable=False,
        server_default="false",
    )

    created_date = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )