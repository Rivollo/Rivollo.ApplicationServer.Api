"""Subscription model - User's subscription to a plan.

This module contains the Subscription model which links a user to a plan
and tracks subscription status, billing periods, and related data.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Enum, ForeignKey, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, column_property, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.types import TIMESTAMP

from typing import TYPE_CHECKING

from app.models.base import Base
from app.models.models import UUIDMixin
from app.models.subscription_enums import SubscriptionStatus

if TYPE_CHECKING:
    from app.models.models import User
    from app.models.plan import Plan
    from app.models.license_assignment import LicenseAssignment


class Subscription(UUIDMixin, Base):
    """User subscription model.

    Represents a user's subscription to a plan. Links a user to a plan and tracks
    subscription status, billing period, and other subscription-related data.
    """

    __tablename__ = "tbl_subscriptions"
    __table_args__ = (Index("ix_subscriptions_user", "user_id"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    plan_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_mstr_plans.id"), nullable=False
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus, name="subscription_status", native_enum=False), nullable=False
    )
    seats_purchased: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    billing_interval: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    offer_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # These columns exist in the database as nullable timestamps
    current_period_start: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    current_period_end: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Razorpay subscription tracking
    razorpay_subscription_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    razorpay_customer_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    # Audit fields that exist in the database
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(PGUUID(as_uuid=True))
    created_date: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(PGUUID(as_uuid=True))
    updated_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Virtual columns - these don't exist in the actual database table
    # Keeping as properties for backward compatibility with code that references them
    trial_end_at = column_property(literal_column("NULL::timestamptz"))
    renews_at = column_property(literal_column("NULL::timestamptz"))

    user: Mapped["User"] = relationship("User", back_populates="subscriptions")
    plan: Mapped["Plan"] = relationship("Plan", back_populates="subscriptions")
    licenses: Mapped[list["LicenseAssignment"]] = relationship("LicenseAssignment", back_populates="subscription")

