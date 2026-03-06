"""
PromoUsage model — tracks which user used which promo code.
"""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.models.base import Base


class PromoUsage(Base):
    """SQLAlchemy model for tbl_promo_usages."""

    __tablename__ = "tbl_promo_usages"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    promo_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tbl_promo_codes.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tbl_users.id", ondelete="CASCADE"),
        nullable=False,
    )

    payment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tbl_payments.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_date = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )