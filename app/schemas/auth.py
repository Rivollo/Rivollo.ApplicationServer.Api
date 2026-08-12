"""Authentication schemas matching OpenAPI spec."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, model_validator, field_validator

from app.utils.email_domain_check import MESSAGES, Verdict, is_disposable


def is_valid_email_domain(email: str) -> bool:
    """Returns True if the email domain is not a known disposable provider.

    Set-based lookup against the maintained `disposable-email-domains` list,
    walking up subdomains so wildcards like xyz.mailinator.com are caught.
    Replaces the old substring keyword blacklist, which both missed unlisted
    temp-mail services and falsely rejected legitimate domains that merely
    contained a keyword.

    Disposable-only, deliberately: this runs inside a sync pydantic validator,
    and the MX/DNS half of the check is async and lives in the signup route.
    """
    try:
        domain = email.strip().lower().split("@")[1]
    except IndexError:
        return False
    return not is_disposable(domain)


def _reject_disposable(v: str) -> str:
    """Shared `email` field validator for the SIGNUP path only.

    Not applied to login / password-reset: those addresses are already in the
    database, and gating them would lock an existing user out of their own
    account if their domain later landed on the blocklist.
    """
    if not is_valid_email_domain(v):
        raise ValueError(MESSAGES[Verdict.DISPOSABLE])
    return v


class SendSignupOtpRequest(BaseModel):
    """Request to send an OTP to verify an email address before signup."""

    email: EmailStr

    _check_email = field_validator("email", mode="after")(_reject_disposable)


class VerifySignupOtpRequest(BaseModel):
    """Request to verify the signup OTP and obtain a signup token."""

    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)

    _check_email = field_validator("email", mode="after")(_reject_disposable)


class LoginRequest(BaseModel):
    """Login request with email and password."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    remember_me: bool = False
    # No disposable-domain gate here: the address is already registered, so
    # rejecting it would lock an existing user out of their own account.


class SignupRequest(BaseModel):
    """Signup request — requires a verified signup_token obtained via /auth/verify-signup-otp."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    remember_me: bool = False
    signup_token: str = Field(..., min_length=1, description="Token returned by /auth/verify-signup-otp")

    _check_email = field_validator("email", mode="after")(_reject_disposable)


class GoogleAuthRequest(BaseModel):
    """Google OAuth authentication request."""

    credential: str = Field(..., min_length=1, description="Google OAuth credential token")
    remember_me: bool = False


class UserResponse(BaseModel):
    """User response model."""

    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    # "email" for email/password accounts, "google" for Google OAuth accounts.
    # Frontend uses this to decide which delete-account confirmation to show.
    auth_provider: str = "email"

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    """Authentication response with user and token."""

    user: UserResponse
    token: str
    expires_at: Optional[datetime] = None


class UserUpdateRequest(BaseModel):
    """Request to update user profile."""

    name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    """Request to initiate a password reset."""

    email: EmailStr
    # No disposable-domain gate — existing-user path, see LoginRequest.


class VerifyOTPRequest(BaseModel):
    """Request to verify the OTP sent to the user's email."""

    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)
    # No disposable-domain gate — existing-user path, see LoginRequest.


class ResetPasswordRequest(BaseModel):
    """Request to complete a password reset using a verified token."""

    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)

    @model_validator(mode="after")
    def passwords_match(self) -> "ResetPasswordRequest":
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class AppTokenRequest(BaseModel):
    clientKey: str = Field(..., min_length=1, max_length=100)


class AppTokenResponse(BaseModel):
    token: str
    client_key: str
    expires_in_minutes: int

