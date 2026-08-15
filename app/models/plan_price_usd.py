"""USD pricing for a plan at a specific billing interval.

Deliberately a separate table from ``tbl_plan_prices`` rather than extra rows or
columns on it. ``tbl_plan_prices`` carries a unique constraint on
(plan_id, billing_interval) and the INR lookup in razorpay_subscription_service
queries it by that pair with no currency filter — a USD row at the same interval
would either violate the constraint or make that existing query ambiguous.

No ORM relationship back to Plan is declared: adding a ``back_populates`` would
mean editing plan.py, which is on the frozen INR path. Queries join on plan_id.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PlanPriceUsd(Base):
    """USD price for a plan at one billing interval. Amounts are in cents."""

    __tablename__ = "tbl_plan_prices_usd"
    __table_args__ = (
        UniqueConstraint(
            "plan_id", "billing_interval", name="tbl_plan_prices_usd_plan_interval_key"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_mstr_plans.id", ondelete="CASCADE"), nullable=False
    )
    billing_interval: Mapped[str] = mapped_column(String(20), nullable=False)

    # Cents, not dollars. $20.00 is stored as 2000.
    price_usd: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    ai_credit_limit: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # Razorpay plan ID for the USD plan. NULL until the plans are created in the
    # Razorpay dashboard — the USD checkout route rejects the interval while it
    # is NULL rather than falling back to anything.
    razorpay_plan_id_usd: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    total_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1200")
    description: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_date: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)
