"""FastAPI dependencies for authentication and database sessions."""

import uuid
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.security import decode_access_token
from app.models.models import User
from app.services.activity_service import ActivityService
from app.services.auth_service import AuthService

bearer_scheme = HTTPBearer(auto_error=False)
security = HTTPBearer()

# Shown when an account was deactivated by us (is_active = false) and the owner
# cannot do anything about it themselves.
ACCOUNT_DEACTIVATED_DETAIL = "Your account has been deactivated. Please contact support."

# Shown when the owner deleted their OWN account and is still inside the 30-day
# recovery window. Deliberately distinct from ACCOUNT_DEACTIVATED_DETAIL: telling
# someone to contact support about a state they entered on purpose, and can undo
# on purpose, sends a self-service action into a support queue.
ACCOUNT_PENDING_DELETION_DETAIL = (
    "Your account is scheduled for deletion. You can restore it until the recovery "
    "period ends."
)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise credentials_exception

    user_id_str: Optional[str] = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise credentials_exception

    # Loaded WITHOUT the usual `deleted_at IS NULL` filter, purely so the two
    # rejection cases below can be told apart. Filtering here instead would fold
    # a soft-deleted account into `user is None` and answer 401 "Could not
    # validate credentials" — indistinguishable from an expired token, which
    # sends someone who deleted their own account looking for a login bug
    # instead of the restore flow.
    #
    # Every path below either raises or returns a user whose deleted_at is NULL.
    # Do not add a `return user` above those guards.
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    # Pending deletion, still recoverable. 403 rather than 401: the token is
    # perfectly valid and the caller is who they say they are — they just have no
    # access while the account is scheduled for erasure. Answering 401 would
    # invite clients to treat it as "refresh your token and retry", which no
    # amount of retrying fixes.
    if user.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ACCOUNT_PENDING_DELETION_DETAIL,
        )

    # Deactivated by us: reject tokens issued before deactivation.
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ACCOUNT_DEACTIVATED_DETAIL,
        )

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None."""
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        payload = decode_access_token(token)
        if payload is None:
            return None

        user_id_str: Optional[str] = payload.get("sub")
        if user_id_str is None:
            return None

        user_id = uuid.UUID(user_id_str)
        result = await db.execute(
            select(User).where(
                User.id == user_id,
                User.deleted_at.is_(None),
                User.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()
    except Exception:
        return None


async def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> str:
    """Legacy compatibility: Get current user ID as string (for old routes)."""
    if credentials is None or not credentials.scheme.lower() == "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user = await get_current_user(credentials, db)
    return str(user.id)


async def verify_app_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Verify that the request carries a valid app token stored in the database."""
    invalid_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing app token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None or payload.get("type") != "app_token":
        await ActivityService.log_activity(
            db=db,
            action="apptoken.validation_failed",
            target_type="app_token",
            metadata={"reason": "invalid_jwt"},
            request=request,
        )
        raise invalid_exc

    if not await AuthService.validate_app_token(db, token):
        await ActivityService.log_activity(
            db=db,
            action="apptoken.validation_failed",
            target_type="app_token",
            metadata={"reason": "token_not_found_or_inactive", "client_key": payload.get("sub")},
            request=request,
        )
        raise invalid_exc


# Convenience type aliases
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[Optional[User], Depends(get_current_user_optional)]
DB = Annotated[AsyncSession, Depends(get_db)]
AppTokenVerified = Annotated[None, Depends(verify_app_token)]
