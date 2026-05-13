from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TokenRegistrationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: Optional[UUID] = Field(
        default=None,
        description="Accepted for client compatibility; authenticated user is used by the API.",
    )
    fcm_token: str = Field(..., min_length=1, alias="fcmToken")
    device_type: str = Field(..., min_length=1, alias="deviceInfo")

    @field_validator("fcm_token", "device_type")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value


class TokenUnregisterRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fcm_token: str = Field(..., min_length=1, alias="fcmToken")

    @field_validator("fcm_token")
    @classmethod
    def strip_fcm_token(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("fcm_token cannot be empty")
        return value


class DirectNotificationRequest(BaseModel):
    user_id: UUID
    notification_type: str = Field(default="push.direct", alias="type")
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=1000)
    device_type: Optional[str] = Field(default=None, alias="deviceType")
    data: Optional[dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("notification_type", "title", "body", "device_type")
    @classmethod
    def strip_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value
