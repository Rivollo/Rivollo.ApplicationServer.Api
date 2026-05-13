from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class TokenRegistrationRequest(BaseModel):
    user_id: Optional[UUID] = Field(
        default=None,
        description="Accepted for client compatibility; authenticated user is used by the API.",
    )
    fcm_token: str = Field(..., min_length=1)
    device_type: str = Field(..., min_length=1)

    @field_validator("fcm_token", "device_type")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value


class TokenUnregisterRequest(BaseModel):
    fcm_token: str = Field(..., min_length=1)

    @field_validator("fcm_token")
    @classmethod
    def strip_fcm_token(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("fcm_token cannot be empty")
        return value


class DirectNotificationRequest(BaseModel):
    user_id: UUID
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=1000)
    data: Optional[dict[str, Any]] = None

    @field_validator("title", "body")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value
