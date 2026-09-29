"""Data access for tbl_api_keys.

Statements only. None of these functions commit; the service owns the
transaction, matching the convention in app/database/*_repo.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey
from app.models.models import User


class ApiKeyRepository:
    """Repository for tbl_api_keys."""

    @staticmethod
    def add(db: AsyncSession, api_key: ApiKey) -> None:
        db.add(api_key)

    @staticmethod
    async def count_usable_for_user(db: AsyncSession, user_id: uuid.UUID, now: datetime) -> int:
        """Keys that still authenticate: active and not expired."""
        result = await db.execute(
            select(func.count())
            .select_from(ApiKey)
            .where(
                ApiKey.user_id == user_id,
                ApiKey.isactive.is_(True),
                or_(ApiKey.expires_at.is_(None), ApiKey.expires_at > now),
            )
        )
        return int(result.scalar_one() or 0)

    @staticmethod
    async def list_for_user(db: AsyncSession, user_id: uuid.UUID) -> list[ApiKey]:
        """Every key the user has created, revoked ones included, newest first."""
        result = await db.execute(
            select(ApiKey)
            .where(ApiKey.user_id == user_id)
            .order_by(ApiKey.created_date.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_for_user(
        db: AsyncSession, key_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[ApiKey]:
        """One key, only if it belongs to ``user_id``. Ownership is in the WHERE clause."""
        result = await db.execute(
            select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_hash(db: AsyncSession, key_hash: str) -> Optional[ApiKey]:
        """The key with this hash, in any state. The service decides what the state means."""
        result = await db.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
        """The key's owner, soft-deleted or not, so the service can tell the cases apart."""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def touch_last_used(
        db: AsyncSession, key_id: uuid.UUID, now: datetime, not_since: datetime
    ) -> None:
        """Record use, unless another request already did within the resolution window.

        The guard is in the WHERE clause so concurrent requests on one key write
        at most once per window without reading first.
        """
        await db.execute(
            update(ApiKey)
            .where(
                ApiKey.id == key_id,
                or_(ApiKey.last_used_at.is_(None), ApiKey.last_used_at < not_since),
            )
            .values(last_used_at=now)
        )


api_key_repository = ApiKeyRepository()
