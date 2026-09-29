"""Request and response contracts for API key management (docs/api_keys.md).

snake_case with no aliases, as in every newer schema module.

The raw key appears in exactly one schema, ``ApiKeyCreatedResponse``, which is
returned once by ``POST /api-keys``. Every other response carries only the
non-secret ``key_prefix``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.api_key import ALL_SCOPES

ApiKeyScope = Literal["read", "write", "convert"]
ApiKeyStatus = Literal["active", "revoked", "expired"]

NAME_MAX_LENGTH = 100
MAX_EXPIRY_DAYS = 365


class ApiKeyCreateRequest(BaseModel):
    """``POST /api-keys``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=NAME_MAX_LENGTH, description="A label, e.g. 'Shopify - my-store'.")
    scopes: list[ApiKeyScope] = Field(
        default_factory=lambda: list(ALL_SCOPES),
        min_length=1,
        description="What the key may do. Defaults to every scope.",
    )
    expires_in_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_EXPIRY_DAYS,
        description="Days until the key stops working. Omit for a key that never expires.",
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must not be blank")
        return cleaned

    @field_validator("scopes")
    @classmethod
    def _dedupe_scopes(cls, value: list[str]) -> list[str]:
        # Stable order (the canonical one), duplicates dropped.
        return [scope for scope in ALL_SCOPES if scope in set(value)]


class ApiKeyResponse(BaseModel):
    """One key, as the owner sees it. Never carries the secret."""

    id: uuid.UUID
    name: str
    key_prefix: str
    scopes: list[str]
    status: ApiKeyStatus
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


class ApiKeyCreatedResponse(ApiKeyResponse):
    """The create response: the only place the raw key is ever returned."""

    key: str = Field(..., description="The full API key. Shown once; it cannot be retrieved again.")


class ApiKeyOwner(BaseModel):
    id: uuid.UUID
    name: Optional[str] = None
    email: str
    avatar_url: Optional[str] = None


class ApiKeyCredits(BaseModel):
    """AI credits on the owner's active licence. ``limit``/``remaining`` null = unlimited."""

    limit: Optional[int] = None
    used: int = 0
    remaining: Optional[int] = None


class CurrentApiKeyResponse(BaseModel):
    """``GET /api-keys/current``: who a key belongs to, for an integration's
    "Connected as ..." screen. Authenticated BY the key, so it only ever
    describes its caller's own account."""

    api_key: ApiKeyResponse
    user: ApiKeyOwner
    credits: ApiKeyCredits
