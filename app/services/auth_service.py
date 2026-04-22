"""Authentication service for user signup, login, and OAuth."""

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import create_access_token, decode_access_token, generate_token, hash_password, hash_token, verify_password
from app.models.models import AppToken, AuthIdentity, AuthProvider, PasswordReset, SignupOtp, User
from app.services.licensing_service import LicensingService


class AuthService:
    """Service for authentication operations."""

    @staticmethod
    async def create_user(
        db: AsyncSession,
        email: str,
        password: Optional[str] = None,
        name: Optional[str] = None,
        provider: AuthProvider = AuthProvider.EMAIL,
        provider_user_id: Optional[str] = None,
    ) -> User:
        """Create a new user with email/password or OAuth."""
        # Create user
        user = User(
            email=email.lower(),
            password_hash=hash_password(password) if password else None,
            name=name or email.split("@")[0],
        )
        db.add(user)
        await db.flush()

        # Create auth identity
        identity = AuthIdentity(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id or str(user.id),
            email=email.lower(),
        )
        db.add(identity)

        # Create free plan license for user
        await LicensingService.create_free_plan_license(db, user)

        await db.commit()
        await db.refresh(user)

        return user

    @staticmethod
    async def authenticate_email(
        db: AsyncSession,
        email: str,
        password: str,
    ) -> Optional[User]:
        """Authenticate user with email and password."""
        result = await db.execute(
            select(User).where(
                User.email == email.lower(),
                User.deleted_at.is_(None),
            )
        )
        user = result.scalar_one_or_none()

        if not user or not user.password_hash:
            return None

        if not verify_password(password, user.password_hash):
            return None

        return user

    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email."""
        result = await db.execute(
            select(User).where(
                User.email == email.lower(),
                User.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_or_create_google_user(
        db: AsyncSession,
        google_id: str,
        email: str,
        name: Optional[str] = None,
    ) -> Tuple[User, bool]:
        """Get or create user from Google OAuth. Returns (user, is_new)."""
        # Check if identity exists
        result = await db.execute(
            select(AuthIdentity).where(
                AuthIdentity.provider == AuthProvider.GOOGLE,
                AuthIdentity.provider_user_id == google_id,
            )
        )
        identity = result.scalar_one_or_none()

        if identity:
            # Get existing user
            result = await db.execute(
                select(User).where(User.id == identity.user_id, User.deleted_at.is_(None))
            )
            user = result.scalar_one_or_none()
            if user:
                return user, False

        # Check if user with email exists
        existing_user = await AuthService.get_user_by_email(db, email)
        if existing_user:
            # Link Google identity to existing user
            identity = AuthIdentity(
                user_id=existing_user.id,
                provider=AuthProvider.GOOGLE,
                provider_user_id=google_id,
                email=email.lower(),
            )
            db.add(identity)
            await db.commit()
            return existing_user, False

        # Create new user
        user = await AuthService.create_user(
            db=db,
            email=email,
            name=name,
            provider=AuthProvider.GOOGLE,
            provider_user_id=google_id,
        )
        return user, True

    @staticmethod
    def generate_token(user_id: uuid.UUID, remember_me: bool = False) -> str:
        """Generate JWT token for user."""
        expires_delta = timedelta(days=30) if remember_me else None
        return create_access_token(
            data={"sub": str(user_id)},
            expires_delta=expires_delta,
        )

    @staticmethod
    def is_valid_client_key(client_key: str) -> bool:
        """Check if the client_key is in the configured allowed list."""
        return client_key.lower() in settings.get_allowed_client_keys()

    @staticmethod
    async def generate_app_token(db: AsyncSession, client_key: str) -> str:
        """Upsert a JWT for the given client_key — one row per client, updated in place.
        Only the SHA-256 hash of the token is persisted; the raw JWT is returned to the caller.
        """
        expires_delta = timedelta(minutes=settings.APP_TOKEN_EXPIRES_MINUTES)
        token = create_access_token(
            data={
                "sub": client_key.lower(),
                "type": "app_token",
            },
            expires_delta=expires_delta,
        )
        expires_at = datetime.now(timezone.utc) + expires_delta
        token_hash = hash_token(token)

        stmt = pg_insert(AppToken).values(
            client_key=client_key.lower(),
            token=token_hash,
            expires_at=expires_at,
            isactive=True,
        ).on_conflict_do_update(
            constraint="uq_app_tokens_client_key",
            set_={
                "token": token_hash,
                "expires_at": expires_at,
                "isactive": True,
                "updated_date": datetime.now(timezone.utc),
            },
        )
        await db.execute(stmt)
        await db.commit()
        return token

    @staticmethod
    async def validate_app_token(db: AsyncSession, token: str) -> bool:
        """Return True if the token hash exists in DB, is active, and is not expired."""
        now = datetime.now(timezone.utc)
        token_hash = hash_token(token)
        result = await db.execute(
            select(AppToken).where(
                AppToken.token == token_hash,
                AppToken.isactive.is_(True),
                AppToken.expires_at > now,
            )
        )
        return result.scalar_one_or_none() is not None

    @staticmethod
    async def create_signup_otp(db: AsyncSession, email: str) -> str:
        """Generate a 6-digit OTP for signup email verification and store it.

        Returns the OTP string. Expires in SIGNUP_OTP_EXPIRES_MINUTES.
        """
        otp = str(random.randint(100000, 999999))
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.SIGNUP_OTP_EXPIRES_MINUTES)

        record = SignupOtp(
            email=email.lower(),
            otp=otp,
            expires_at=expires_at,
        )
        db.add(record)
        await db.commit()

        return otp

    @staticmethod
    async def verify_signup_otp(db: AsyncSession, email: str, otp: str) -> Optional[str]:
        """Verify the signup OTP for the given email.

        Returns a short-lived JWT signup_token on success, or None if invalid/expired.
        """
        now = datetime.now(timezone.utc)

        result = await db.execute(
            select(SignupOtp).where(
                SignupOtp.email == email.lower(),
                SignupOtp.otp == otp,
                SignupOtp.isactive.is_(True),
                SignupOtp.expires_at > now,
            )
        )
        record = result.scalar_one_or_none()
        if not record:
            return None

        record.used_at = now
        record.isactive = False
        await db.commit()

        signup_token = create_access_token(
            data={"sub": email.lower(), "type": "email_verified"},
            expires_delta=timedelta(minutes=settings.SIGNUP_TOKEN_EXPIRES_MINUTES),
        )
        return signup_token

    @staticmethod
    def validate_signup_token(signup_token: str, email: str) -> bool:
        """Return True if the signup_token is valid for the given email."""
        payload = decode_access_token(signup_token)
        if not payload:
            return False
        return (
            payload.get("type") == "email_verified"
            and payload.get("sub", "").lower() == email.lower()
        )

    @staticmethod
    async def create_password_reset_otp(db: AsyncSession, email: str) -> Optional[str]:
        """Generate a 6-digit OTP for password reset and store it.

        Returns the OTP string, or None if no user with that email exists.
        OTP expires in 10 minutes.
        """
        user = await AuthService.get_user_by_email(db, email)
        if not user:
            return None

        otp = str(random.randint(100000, 999999))
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_OTP_EXPIRES_MINUTES)

        reset = PasswordReset(
            user_id=user.id,
            token=otp,
            expires_at=expires_at,
        )
        db.add(reset)
        await db.commit()

        return otp

    @staticmethod
    async def verify_otp(db: AsyncSession, email: str, otp: str) -> Optional[str]:
        """Verify the OTP for the given email.

        Returns a secure reset token on success, or None if OTP is invalid/expired.
        The OTP record is replaced with the secure token for use in reset-password.
        """
        now = datetime.now(timezone.utc)

        user = await AuthService.get_user_by_email(db, email)
        if not user:
            return None

        result = await db.execute(
            select(PasswordReset).where(
                PasswordReset.user_id == user.id,
                PasswordReset.token == otp,
                PasswordReset.used_at.is_(None),
                PasswordReset.expires_at > now,
            )
        )
        reset = result.scalar_one_or_none()
        if not reset:
            return None

        # Replace OTP with a secure reset token valid for 15 minutes
        secure_token = generate_token(32)
        reset.token = secure_token
        reset.expires_at = now + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRES_MINUTES)
        await db.commit()

        return secure_token

    @staticmethod
    async def reset_password(db: AsyncSession, token: str, new_password: str) -> Optional[User]:
        """Consume a verified reset token and update the user's password.

        Returns the User on success, or None if the token is invalid, expired, or already used.
        """
        now = datetime.now(timezone.utc)

        result = await db.execute(
            select(PasswordReset).where(
                PasswordReset.token == token,
                PasswordReset.used_at.is_(None),
                PasswordReset.expires_at > now,
            )
        )
        reset = result.scalar_one_or_none()
        if not reset:
            return None

        result = await db.execute(
            select(User).where(User.id == reset.user_id, User.deleted_at.is_(None))
        )
        user = result.scalar_one_or_none()
        if not user:
            return None

        user.password_hash = hash_password(new_password)
        reset.used_at = now
        await db.commit()

        return user
