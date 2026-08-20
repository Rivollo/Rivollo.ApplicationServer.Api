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
    # When the account stops being restorable. Returned rather than left for the
    # client to compute as deleted_at + 30 days, because that would copy the
    # retention window into every client and they would all have to be redeployed
    # to change it — and any that lagged would show a date the server disagrees
    # with. ACCOUNT_RETENTION_DAYS stays the single source of truth.
    #
    # The purge job runs daily at 00:00 UTC, so erasure happens at the first run
    # AFTER this instant, not exactly on it. Do not present it to the second.
    purge_after: datetime
