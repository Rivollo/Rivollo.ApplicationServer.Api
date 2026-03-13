
"""
Promo repository — database operations for promo codes.

Responsibilities:
    - Fetch a valid promo code
    - Increment promo usage counter
    - Record promo usage per payment
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.promo import PromoCode
from app.models.promo_usage import PromoUsage


class PromoRepository:
    """Repository for promo code database operations."""

    # ─────────────────────────────────────────────────────────────
    # Fetch promo by code
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    async def get_by_code(
        db: AsyncSession,
        code: str,
    ) -> Optional[PromoCode]:

        now = datetime.now(timezone.utc)

        result = await db.execute(
            select(PromoCode).where(
                PromoCode.code == code,
                PromoCode.is_active.is_(True),
                PromoCode.valid_from <= now,
                PromoCode.valid_to >= now,
            )
        )

        return result.scalar_one_or_none()

    # ─────────────────────────────────────────────────────────────
    # Increment promo usage counter
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    async def increment_usage(
        db: AsyncSession,
        promo: PromoCode,
    ) -> None:

        promo.used_count += 1
        await db.flush()

    # ─────────────────────────────────────────────────────────────
    # Record promo usage
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    async def create_usage(
        db: AsyncSession,
        *,
        promo_id: uuid.UUID,
        user_id: uuid.UUID,
        payment_id: uuid.UUID | None = None,
        created_by: uuid.UUID | None = None,
    ) -> PromoUsage:

        usage = PromoUsage(
            promo_id=promo_id,
            user_id=user_id,
            payment_id=payment_id,
            created_by=created_by or user_id,
            is_active=True,
        )

        db.add(usage)

        await db.flush()
        await db.refresh(usage)

        return usage