"""Account management service — handles account deletion and related operations."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_password
from app.models.models import AuthIdentity, Product, User

logger = logging.getLogger(__name__)


class AccountService:

    @staticmethod
    async def delete_account(
        db: AsyncSession,
        user: User,
        password: Optional[str],
        confirmation: Optional[str],
    ) -> datetime:
        """Soft-delete a user account and all their products in one transaction.

        Identity verification rules:
        - Email/password users must supply their current password.
        - Google OAuth users (password_hash is None) must supply confirmation = "DELETE MY ACCOUNT".

        What this does:
        1. Verifies identity.
        2. Soft-deletes all products owned by the user (created_by = user.id).
        3. Hard-deletes AuthIdentity rows — frees up the OAuth provider slot so the
           provider_user_id is not orphaned. The user row itself stays for audit/analytics.
        4. Soft-deletes the user row (sets deleted_at).
        5. Commits — single atomic transaction; any failure rolls everything back.

        After this call:
        - All JWTs for this user return 401 (deps.py filters deleted_at IS NULL).
        - Login with the same email returns 401.
        - All published product links go dark (products are soft-deleted).
        """
        _verify_identity(user, password, confirmation)

        now = datetime.now(timezone.utc)

        # 1. Soft-delete all products owned by this user
        await db.execute(
            update(Product)
            .where(
                Product.created_by == user.id,
                Product.deleted_at.is_(None),
            )
            .values(deleted_at=now, updated_date=now, updated_by=user.id)
        )

        # 2. Hard-delete OAuth identities to free up provider slots
        await db.execute(
            delete(AuthIdentity).where(AuthIdentity.user_id == user.id)
        )

        # 3. Soft-delete the user
        await db.execute(
            update(User)
            .where(User.id == user.id)
            .values(deleted_at=now, updated_date=now)
        )

        await db.commit()

        logger.info(
            "Account deleted | user_id=%s | email=%s",
            user.id,
            user.email,
        )
        return now


def _verify_identity(
    user: User,
    password: Optional[str],
    confirmation: Optional[str],
) -> None:
    """Raise HTTP 400 if the caller cannot prove they own this account."""
    if user.password_hash:
        # Email / password account
        if not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Your current password is required to delete your account.",
            )
        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect password.",
            )
    else:
        # Google OAuth account — no password, require explicit typed confirmation
        if not confirmation or confirmation.strip().upper() != "DELETE MY ACCOUNT":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Please type "DELETE MY ACCOUNT" to confirm.',
            )
