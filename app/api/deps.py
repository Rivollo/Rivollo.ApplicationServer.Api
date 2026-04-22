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

    # Fetch user from database
    result = await db.execute(select(User).where(User.id == user_id, User.deleted_at.is_(None)))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

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
        result = await db.execute(select(User).where(User.id == user_id, User.deleted_at.is_(None)))
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
