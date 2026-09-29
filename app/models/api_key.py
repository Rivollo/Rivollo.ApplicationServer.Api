"""ORM model for long-lived API keys used by third-party integrations.

A seller creates a key in the portal and pastes it into an integration (the
Shopify app is the first). The integration then sends it as
``Authorization: Bearer riv_live_...`` on every call, in place of a 60-minute
JWT it has no way to refresh. See docs/api_keys.md.

Lives in its own module, not ``models/models.py``, as every newer domain does.

ONLY A HASH IS STORED
---------------------
``key_hash`` is the SHA-256 of the raw key (``app.core.security.hash_token``,
the same helper tbl_app_tokens uses). The raw key is returned exactly once, by
the create call, and exists nowhere after that response. A slow password hash
(bcrypt, argon2) buys nothing here: the key carries 192 random bits, so there is
no dictionary to slow down, and a fast hash lets authentication be a single
indexed lookup on the unique ``key_hash`` rather than a scan-and-compare.

``key_prefix`` is the first characters of the raw key (``riv_live_`` plus eight
hex digits). It is not secret; it exists so a seller can tell their keys apart
in a list.

OWNERSHIP: user_id -> tbl_users ON DELETE CASCADE
-------------------------------------------------
A key belongs to exactly one user and dies with them. The account purge deletes
the user row last, so the cascade removes every key; Rivollo.AccountPurge.Job
must allow-list this FK in USER_CASCADE_FKS (docs/account-purge-job-changes.md)
before the table reaches production, or its contract check (A14) aborts.

Authentication still re-reads the owner on every request and refuses one that
is deactivated or pending deletion, which the FK alone does not cover.

created_by / updated_by stay plain UUIDs, as AuditMixin declares everywhere.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TIMESTAMP

from app.models.base import Base
from app.models.models import AuditMixin, UUIDMixin

# What a key may do. A route that accepts API keys names the scope it needs
# (app.api.deps.require_api_key_scope); a key without it gets 403.
SCOPE_READ = "read"
SCOPE_WRITE = "write"
SCOPE_CONVERT = "convert"
ALL_SCOPES: tuple[str, ...] = (SCOPE_READ, SCOPE_WRITE, SCOPE_CONVERT)


class ApiKey(UUIDMixin, AuditMixin, Base):
    """One API key belonging to one user."""

    __tablename__ = "tbl_api_keys"
    __table_args__ = (
        # Authentication is one lookup on this index.
        Index("ux_api_keys_key_hash", "key_hash", unique=True),
        # The portal's list, newest first.
        Index("ix_api_keys_user_created", "user_id", "created_date"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False)
    key_prefix: Mapped[str] = mapped_column(Text, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("ARRAY['read','write','convert']::text[]"),
    )

    # NULL = never expires.
    expires_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    # Written at most once a minute per key (API_KEY_LAST_USED_RESOLUTION_SECONDS),
    # so a busy integration does not turn every request into a write.
    last_used_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Revocation is a soft switch: the row is kept for the audit trail and so
    # the portal can still show that the key existed.
    isactive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true"), default=True
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        # Never the hash: this object reaches log lines and tracebacks.
        return f"<ApiKey id={self.id} prefix={self.key_prefix} isactive={self.isactive}>"
