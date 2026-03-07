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

    created_date = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )