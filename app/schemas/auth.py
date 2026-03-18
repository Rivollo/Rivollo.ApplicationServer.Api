"""Authentication schemas matching OpenAPI spec."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, model_validator, field_validator

# Allowed personal / business email domains for sign-up (whitelist approach).
ALLOWED_EMAIL_DOMAINS: frozenset[str] = frozenset({
    # Google
    "gmail.com",
    "googlemail.com",

    # Microsoft
    "outlook.com",
    "hotmail.com",
    "hotmail.in",
    "hotmail.co.uk",
    "hotmail.fr",
    "hotmail.de",
    "live.com",
    "live.in",
    "live.co.uk",
    "msn.com",

    # Yahoo
    "yahoo.com",
    "yahoo.in",
    "yahoo.co.in",
    "yahoo.co.uk",
    "yahoo.fr",
    "yahoo.de",
    "yahoo.com.au",
    "yahoo.ca",
    "ymail.com",
    "rocketmail.com",

    # Apple
    "icloud.com",
    "me.com",
    "mac.com",

    # Indian Providers
    "rediffmail.com",
    "sify.com",
    "indiatimes.com",
    "in.com",

    # ProtonMail (privacy-focused but legitimate)
    "protonmail.com",
    "protonmail.ch",
    "proton.me",
    "pm.me",

    # Zoho
    "zoho.com",
    "zohomail.com",
    "zohomail.in",

    # Other Legitimate Providers
    "aol.com",
    "aim.com",
    "mail.com",
    "email.com",
    "fastmail.com",
    "fastmail.fm",
    "hushmail.com",
    "tutanota.com",
    "tutamail.com",
    "tuta.io",
    "gmx.com",
    "gmx.net",
    "gmx.de",
    "gmx.us",
    "iinet.net.au",
    "bigpond.com",
    "bigpond.net.au",
    "optusnet.com.au",
    "virginmedia.com",
    "btinternet.com",
    "sky.com",
    "talktalk.net",
    "ntlworld.com",
    "o2.co.uk",
    "orange.fr",
    "sfr.fr",
    "free.fr",
    "laposte.net",
    "web.de",
    "t-online.de",
    "freenet.de",
    "arcor.de",
    "bluewin.ch",
    "hispeed.ch",
    "sunrise.ch",
    "tiscali.it",
    "libero.it",
    "virgilio.it",
    "tin.it",
    "alice.it",
    "telenet.be",
    "skynet.be",
    "shaw.ca",
    "rogers.com",
    "bell.net",
    "sympatico.ca",
    "videotron.ca",
    "terra.com.br",
    "uol.com.br",
    "bol.com.br",
    "ig.com.br",
    "globo.com",
    "naver.com",
    "hanmail.net",
    "daum.net",
    "nate.com",
    "qq.com",
    "163.com",
    "126.com",
    "sina.com",
    "sohu.com",
    "aliyun.com",
})


def is_valid_email_domain(email: str) -> bool:
    """
    Returns True if the email belongs to an allowed (real) domain.
    Returns False if it's not in the whitelist.
    """
    try:
        domain = email.strip().lower().split("@")[1]
        return domain in ALLOWED_EMAIL_DOMAINS
    except IndexError:
        return False


class LoginRequest(BaseModel):
    """Login request with email and password."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    remember_me: bool = False


class SignupRequest(BaseModel):
    """Signup request with email, password, and optional name."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    remember_me: bool = False

    @field_validator("email", mode="after")
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        if not is_valid_email_domain(v):
            raise ValueError(
                "Please use a valid email address from a recognized provider."
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


class VerifyOTPRequest(BaseModel):
    """Request to verify the OTP sent to the user's email."""

    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


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

