"""Repository layer for user database operations."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import User


class UserRepository:
    """Repository for user database operations."""

    @staticmethod
    async def get_for_restore(db: AsyncSession, email: str) -> Optional[User]:
        """Load a user by email for restore, INCLUDING soft-deleted rows.

        Every other user lookup in this codebase filters ``deleted_at IS NULL``
        (AuthService.get_user_by_email, authenticate_email, both get_current_user
        variants), which is correct for them and useless here: the only accounts
        restore ever operates on are exactly the ones those filters hide. Restore
        therefore needs its own lookup rather than a flag on an existing one, so
        the soft-delete filter can never be bypassed by accident elsewhere.

        Takes ``FOR UPDATE`` on the row. Two things race for it — a second restore
        request, and a delete request arriving in the same instant — and the lock
        makes them queue rather than interleave, so the loser observes the
        winner's committed state instead of a half-applied one.

        Must be called inside a transaction, which the caller's session provides.
        """
        result = await db.execute(
            select(User).where(User.email == email.lower()).with_for_update()
        )
        return result.scalar_one_or_none()
