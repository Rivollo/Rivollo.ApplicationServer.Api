"""User account schemas (non-auth)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator


class DeleteAccountRequest(BaseModel):
    # Email/password users: supply current password
    password: Optional[str] = None
    # Google OAuth users (no password): type exactly "DELETE MY ACCOUNT"
    confirmation: Optional[str] = None

    @field_validator("confirmation")
    @classmethod
    def _validate_confirmation(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip().upper() != "DELETE MY ACCOUNT":
            raise ValueError('confirmation must be exactly "DELETE MY ACCOUNT"')
        return v


class DeleteAccountResponse(BaseModel):
    message: str
    deleted_at: datetime
