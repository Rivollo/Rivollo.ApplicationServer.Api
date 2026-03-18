"""Authentication routes for signup, login, and OAuth."""

import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError

from app.api.deps import DB
from app.schemas.auth import (
    AuthResponse,
    ForgotPasswordRequest,
    GoogleAuthRequest,
    LoginRequest,
    ResetPasswordRequest,
    SignupRequest,
    UserResponse,
    VerifyOTPRequest,
)
from app.services.activity_service import ActivityService
from app.services.auth_service import AuthService
from app.services.email_service import EmailService
from app.utils.envelopes import api_success
from app.core.config import settings

router = APIRouter(tags=["auth"])


@router.post("/auth/signup", response_model=dict)
async def signup(
    payload: SignupRequest,
    request: Request,
    db: DB,
):
    """Create new user account."""
    # Check if user already exists
    existing_user = await AuthService.get_user_by_email(db, payload.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    try:
        # Create user
        user = await AuthService.create_user(
            db=db,
            email=payload.email,
            password=payload.password,
            name=payload.name,
        )

        # Generate token
        token = AuthService.generate_token(user.id, payload.remember_me)

        # Log activity
        await ActivityService.log_auth_action(
            db=db,
            action="user.signup",
            user_id=user.id,
            request=request,
        )

        # Prepare response
        user_data = UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            avatar_url=user.avatar_url,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

        # Send welcome email (non-blocking — failure must not affect signup response)
        try:
            await EmailService.send_welcome_email(
                to_email=user.email,
                name=user.name or user.email,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to send welcome email to %s", user.email, exc_info=True
            )

        return api_success(
            AuthResponse(
                user=user_data,
                token=token,
            ).model_dump()
        )

    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )


@router.post("/auth/login", response_model=dict)
async def login(
    payload: LoginRequest,
    request: Request,
    db: DB,
):
    """Login with email and password."""
    # Authenticate user
    user = await AuthService.authenticate_email(
        db=db,
        email=payload.email,
        password=payload.password,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Generate token
    token = AuthService.generate_token(user.id, payload.remember_me)

    # Log activity
    await ActivityService.log_auth_action(
        db=db,
        action="user.login",
        user_id=user.id,
        request=request,
    )

    # Prepare response
    user_data = UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )

    return api_success(
        AuthResponse(
            user=user_data,
            token=token,
        ).model_dump()
    )


@router.post("/auth/forgot-password", response_model=dict)
async def forgot_password(
    payload: ForgotPasswordRequest,
    db: DB,
):
    """Initiate a password reset by sending a 6-digit OTP to the user's email."""
    user = await AuthService.get_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    otp = await AuthService.create_password_reset_otp(db, payload.email)
    await EmailService.send_otp_email(
        to_email=user.email,
        name=user.name or user.email,
        otp=otp,
        expires_minutes=settings.PASSWORD_RESET_OTP_EXPIRES_MINUTES,
    )
    return api_success({"message": "OTP sent to your email", "expires_in_minutes": settings.PASSWORD_RESET_OTP_EXPIRES_MINUTES})


@router.post("/auth/verify-otp", response_model=dict)
async def verify_otp(
    payload: VerifyOTPRequest,
    db: DB,
):
    """Verify the OTP and return a secure reset token to use in reset-password."""
    reset_token = await AuthService.verify_otp(db, payload.email, payload.otp)
    if reset_token is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP",
        )
    return api_success({"reset_token": reset_token, "expires_in_minutes": 15})


@router.post("/auth/reset-password", response_model=dict)
async def reset_password(
    payload: ResetPasswordRequest,
    db: DB,
):
    """Reset password using the verified reset token."""
    user = await AuthService.reset_password(db, payload.token, payload.new_password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    await EmailService.send_password_reset_success_email(
        to_email=user.email,
        name=user.name or user.email,
    )
    return api_success({"message": "Password reset successfully"})


@router.post("/auth/google", response_model=dict)
async def google_auth(
    payload: GoogleAuthRequest,
    request: Request,
    db: DB,
):
    """Login or signup with Google OAuth."""
    logger = logging.getLogger(__name__)
    token_info: Dict[str, Any]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": payload.credential},
            )
    except httpx.RequestError as exc:
        logger.exception("Failed to verify Google credential: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify Google credential",
        ) from exc

    if resp.status_code != 200:
        logger.warning("Google tokeninfo rejected credential with status %s", resp.status_code)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credential",
        )

    token_info = resp.json()
    aud = token_info.get("aud")
    if aud != settings.GOOGLE_CLIENT_ID:
      raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Google token was not issued for this application",
    )

    google_user_id = token_info.get("sub")
    email = token_info.get("email")

    if not google_user_id or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Google credential payload",
        )

    email_verified = str(token_info.get("email_verified", "")).lower()
    if email_verified not in {"true", "1", "yes"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google email is not verified",
        )

    display_name = token_info.get("name")
    avatar_url = token_info.get("picture")

    user, is_new_user = await AuthService.get_or_create_google_user(
        db=db,
        google_id=str(google_user_id),
        email=email,
        name=display_name,
    )

    updated = False
    if display_name and user.name != display_name:
        user.name = display_name
        updated = True
    if avatar_url and user.avatar_url != avatar_url:
        user.avatar_url = avatar_url
        updated = True

    if updated:
        await db.commit()
        await db.refresh(user)

    token = AuthService.generate_token(user.id, payload.remember_me)

    await ActivityService.log_auth_action(
        db=db,
        action="user.login.google",
        user_id=user.id,
        request=request,
        metadata={"provider": "google"},
    )

    user_data = UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )

    # Send welcome email for newly created Google accounts (non-blocking)
    if is_new_user:
        try:
            await EmailService.send_welcome_email(
                to_email=user.email,
                name=user.name or user.email,
            )
        except Exception:
            logger.warning(
                "Failed to send welcome email to %s", user.email, exc_info=True
            )

    return api_success(
        AuthResponse(
            user=user_data,
            token=token,
        ).model_dump()
    )
