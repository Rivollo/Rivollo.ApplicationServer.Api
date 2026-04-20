"""Authentication schemas matching OpenAPI spec."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, model_validator, field_validator

# Blocked disposable / temporary email keywords (blacklist approach).
# If the domain contains any of these keywords it is rejected —
# regardless of TLD (e.g. mailinator.com, mailinator.in, mailinator.xyz all blocked).
BLOCKED_DOMAIN_KEYWORDS: frozenset[str] = frozenset({
    "mailinator",
    "guerrillamail",
    "guerrillamailblock",
    "trashmail",
    "trash-mail",
    "tempmail",
    "temp-mail",
    "throwam",
    "throwaway",
    "yopmail",
    "maildrop",
    "mailnull",
    "mailnesia",
    "mailscrap",
    "fakeinbox",
    "dispostable",
    "sharklasers",
    "spamgourmet",
    "spamhereplease",
    "spambox",
    "spamcannon",
    "spamfree",
    "spamgob",
    "spamhole",
    "spamify",
    "spaminator",
    "spamkill",
    "spammotel",
    "spamslicer",
    "spamspot",
    "spamtroll",
    "tempalias",
    "tempemail",
    "tempinbox",
    "tempmailer",
    "tmailinator",
    "trashdevil",
    "trashemail",
    "trashmailer",
    "trashspam",
    "discard",
    "disposable",
    "filzmail",
    "shiftmail",
    "sneakemail",
    "willselfdestruct",
    "whyspam",
})


def is_valid_email_domain(email: str) -> bool:
    """
    Returns True if the email domain does NOT contain any blocked keyword.
    Blocks known disposable / temporary email providers by keyword —
    so mailinator.com, mailinator.in, mailinator.xyz are all blocked.
    All other domains (including business / company emails) are allowed.
    """
    try:
        domain = email.strip().lower().split("@")[1]
        return not any(keyword in domain for keyword in BLOCKED_DOMAIN_KEYWORDS)
    except IndexError:
        return False


class SendSignupOtpRequest(BaseModel):
    """Request to send an OTP to verify an email address before signup."""

    email: EmailStr

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address. Disposable or temporary email addresses are not allowed."
            )
        return v


class VerifySignupOtpRequest(BaseModel):
    """Request to verify the signup OTP and obtain a signup token."""

    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address. Disposable or temporary email addresses are not allowed."
            )
        return v


class LoginRequest(BaseModel):
    """Login request with email and password."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    remember_me: bool = False

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address. Disposable or temporary email addresses are not allowed."
            )
        return v


class SignupRequest(BaseModel):
    """Signup request — requires a verified signup_token obtained via /auth/verify-signup-otp."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    remember_me: bool = False
    signup_token: str = Field(..., min_length=1, description="Token returned by /auth/verify-signup-otp")

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address. Disposable or temporary email addresses are not allowed."
            )
        return v


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

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address. Disposable or temporary email addresses are not allowed."
            )
        return v


class VerifyOTPRequest(BaseModel):
    """Request to verify the OTP sent to the user's email."""

    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address. Disposable or temporary email addresses are not allowed."
            )
        return v


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

