"""User account schemas (non-auth)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DeleteAccountRequest(BaseModel):
    # Every account types the same phrase; there is no password branch any more.
    #
    # Left optional and unvalidated here on purpose. AccountService is the single
    # gate, so a missing phrase and a mistyped one both come back as one 400
    # carrying the same actionable message. Validating the text here as well
    # would answer a mistyped phrase with a 422 and a missing one with a 400,
    # for what is the same user mistake.
    confirmation: Optional[str] = None


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
