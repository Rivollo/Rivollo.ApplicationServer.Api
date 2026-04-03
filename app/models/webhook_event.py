"""WebhookEvent model - Permanent log of every Razorpay webhook received."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from app.models.base import Base


class WebhookEvent(Base):
    """One row per Razorpay webhook event received.

    event_id is Razorpay's unique ID for the event — used as idempotency key.
    Inserting a duplicate event_id will raise a unique constraint violation,
    which we catch to skip already-processed events.
    """

    __tablename__ = "tbl_webhook_events"
    __table_args__ = (
        Index("ix_webhook_events_rz_sub_id", "rz_sub_id"),
        Index("ix_webhook_events_event", "event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    event: Mapped[str] = mapped_column(String(100), nullable=False)
    rz_sub_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
