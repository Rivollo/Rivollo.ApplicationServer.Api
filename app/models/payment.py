"""Payment model - Tracks every Razorpay payment attempt.

This is the permanent receipt book for all payments.
Every payment — success or failure — is recorded here.

Key fields:
    razorpay_order_id: Unique key from Razorpay when order is created (used for idempotency)
    razorpay_payment_id: Filled after user completes payment
    razorpay_signature: HMAC proof that the payment is genuine
    status: created -> captured (success) OR created -> failed
    subscription_id: Linked after subscription is activated (nullable initially)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from typing import TYPE_CHECKING

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.models import User
    from app.models.subscription import Subscription


class PaymentStatus:
    """Payment status constants.

    Using plain string constants instead of Enum so we are compatible
    with the TEXT column in the database (no native enum type needed).
    """

    CREATED = "created"     # Order created, user hasn't paid yet
    CAPTURED = "captured"   # ✅ Payment success — money received
    FAILED = "failed"       # ❌ Payment failed (card declined, timeout, etc.)
    REFUNDED = "refunded"   # Money returned to user


class Payment(Base):
    """Razorpay payment record.

    One row is created when an order is created (status=created).
    The same row is updated when the payment is verified (status=captured or failed).
    The subscription_id is linked after subscription activation.

    Unique constraint on razorpay_order_id ensures idempotency — you cannot
    accidentally create two payment records for the same Razorpay order.
    """

    __tablename__ = "tbl_payments"
    __table_args__ = (
        Index("ix_payments_user", "user_id"),
        Index("ix_payments_order", "razorpay_order_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Who made this payment
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tbl_users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Linked after subscription is successfully activated (nullable initially)
    subscription_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tbl_subscriptions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Razorpay identifiers
    razorpay_order_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    razorpay_payment_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    razorpay_signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Payment details
    amount: Mapped[int] = mapped_column(Integer, nullable=False)  # in paise
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="INR"
    )
    plan_code: Mapped[str] = mapped_column(
        String(50), nullable=False  # "pro" | "enterprise"
    )

    # Status — TEXT column (see PaymentStatus constants above)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=PaymentStatus.CREATED
    )

    # Failure details (only set when status = "failed")
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_date: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_date: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User")
    subscription: Mapped[Optional["Subscription"]] = relationship("Subscription")
