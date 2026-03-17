"""Authentication schemas matching OpenAPI spec."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


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

    @field_validator("avatar_url")
    @classmethod
    def validate_avatar_size(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Allow regular URLs (Google OAuth avatar)
        if v.startswith("http://") or v.startswith("https://"):
            return v
        # For base64 data URIs, enforce 2MB limit
        max_bytes = 2 * 1024 * 1024  # 2MB
        if len(v.encode("utf-8")) > max_bytes:
            raise ValueError("Avatar image must be smaller than 2MB")
        return v
