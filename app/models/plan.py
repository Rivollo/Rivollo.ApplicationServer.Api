"""Plan model - Subscription plan definitions.

This module contains the Plan model which defines subscription tiers
(Free, Pro, Enterprise) with their features and quotas.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text , UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.models import AuditMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.subscription import Subscription


class Feature(UUIDMixin, Base):
    """Master list of available subscription features."""

    __tablename__ = "tbl_mstr_features"

    code: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_date: Mapped[datetime] = mapped_column(nullable=False)

    plan_features: Mapped[list["PlanFeature"]] = relationship("PlanFeature", back_populates="feature")


class PlanFeature(Base):
    """Junction table mapping features to plans with specific limits."""

    __tablename__ = "tbl_plan_features"

    plan_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tbl_mstr_plans.id", ondelete="CASCADE"), primary_key=True)
    feature_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tbl_mstr_features.id", ondelete="CASCADE"), primary_key=True)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    limit_value: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    plan: Mapped["Plan"] = relationship("Plan", back_populates="plan_features")
    feature: Mapped["Feature"] = relationship("Feature", back_populates="plan_features")

class PlanPrice(Base):
    """Pricing for a plan at a specific billing interval."""

    __tablename__ = "tbl_plan_prices"
    __table_args__ = (
        UniqueConstraint("plan_id", "billing_interval", name="tbl_plan_prices_plan_interval_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tbl_mstr_plans.id", ondelete="CASCADE"), nullable=False)
    billing_interval: Mapped[str] = mapped_column(String(20), nullable=False)
    price_inr: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="INR")
    razorpay_plan_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    trial_period_days: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)

    plan: Mapped["Plan"] = relationship("Plan", back_populates="plan_prices")
    total_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1200")



class Plan(UUIDMixin, AuditMixin, Base):
    """Subscription plan model (Free, Pro, Enterprise).

    Plans define the features and quotas available to users.
    Each plan has a code (e.g., "free", "pro", "enterprise") and quotas stored as JSON.
    """

    __tablename__ = "tbl_mstr_plans"

    code: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    # The legacy 'quotas' JSON column has been removed in favor of normalized plan_features.
    # New normalized columns
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_featured: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")


    # Property for backward compatibility
    @property
    def created_at(self) -> datetime:
        return self.created_date

    subscriptions: Mapped[list["Subscription"]] = relationship("Subscription", back_populates="plan")
    plan_features: Mapped[list["PlanFeature"]] = relationship("PlanFeature", back_populates="plan", cascade="all, delete-orphan")
    plan_prices: Mapped[list["PlanPrice"]] = relationship("PlanPrice", back_populates="plan", cascade="all, delete-orphan")


