"""3D Agent (AI Assistant) billing — usage ledger and LLM price list.

Sellers pay for AI Assistant chat messages in AI credits, metered by the LLM
tokens each message actually uses (``sql/create_agent_billing.sql`` holds the
DDL, the formula and the seed prices):

    cost_usd = sum over the message's LLM calls of tokens x model price
    credits  = CEIL(cost_usd x multiplier / credit_usd_value), at least 1

A message runs on a credit hold (``status = 'holding'``) that grows chunk by
chunk while it runs. Settling charges the actual credits to
``tbl_license_assignments.usage_ai_credits`` and releases the rest, so the
licence table itself is never altered: open holds are summed from here.

No foreign keys on ``user_id`` / ``product_id`` / audit columns, on purpose —
a billing record outlives the product and user it describes, and the account
purge job needs no change (ADR-010).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func, text
from sqlalchemy.types import TIMESTAMP

from app.models.base import Base
from app.models.models import AuditMixin, UUIDMixin

AGENT_USAGE_STATUSES = ("holding", "settled", "released", "expired")


class AgentUsage(UUIDMixin, AuditMixin, Base):
    """One AI Assistant chat message: its credit hold, then its final bill."""

    __tablename__ = "tbl_agent_usage"
    __table_args__ = (
        CheckConstraint(
            "status IN ('holding', 'settled', 'released', 'expired')",
            name="ck_agent_usage_status",
        ),
        CheckConstraint("credits_held >= 0", name="ck_agent_usage_credits_held"),
        CheckConstraint("credits_charged >= 0", name="ck_agent_usage_credits_charged"),
        CheckConstraint("max_credits > 0", name="ck_agent_usage_max_credits"),
        CheckConstraint("cost_usd >= 0", name="ck_agent_usage_cost"),
        CheckConstraint(
            "input_tokens >= 0 AND cached_tokens >= 0 AND output_tokens >= 0 "
            "AND llm_calls >= 0",
            name="ck_agent_usage_tokens",
        ),
        Index("ix_agent_usage_user_created", "user_id", text("created_date DESC")),
        # Open holds per seller: summed into every available-credits check.
        Index(
            "ix_agent_usage_user_holding",
            "user_id",
            postgresql_where=text("status = 'holding'"),
        ),
        # The expiry sweep's query.
        Index(
            "ix_agent_usage_holding_expires",
            "expires_at",
            postgresql_where=text("status = 'holding'"),
        ),
    )

    # Issued with the hold, echoed back on settle: a retried settle is a no-op.
    hold_key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    product_id: Mapped[Optional[uuid.UUID]] = mapped_column(PGUUID(as_uuid=True))

    session_id: Mapped[Optional[str]] = mapped_column(Text)
    agent_model_id: Mapped[Optional[str]] = mapped_column(Text)
    mode: Mapped[Optional[str]] = mapped_column(Text)

    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'holding'")
    )

    credits_held: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    credits_charged: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    max_credits: Mapped[int] = mapped_column(Integer, nullable=False)

    cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, server_default=text("0")
    )
    input_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    cached_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    output_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    llm_calls: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    breakdown: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONB)

    multiplier: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 3))
    credit_usd_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))

    error: Mapped[Optional[str]] = mapped_column(Text)

    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    settled_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))


class LlmModelPrice(UUIDMixin, AuditMixin, Base):
    """What one LLM deployment costs Rivollo, in USD per 1M tokens.

    Repricing is a new row with a later ``effective_from``, never an update,
    so past messages keep the price they were billed at.
    """

    __tablename__ = "tbl_llm_model_prices"
    __table_args__ = (
        UniqueConstraint(
            "model_key", "effective_from", name="uq_llm_model_prices_model_effective"
        ),
        CheckConstraint(
            "input_per_m >= 0 AND cached_input_per_m >= 0 AND output_per_m >= 0",
            name="ck_llm_model_prices_non_negative",
        ),
    )

    # The Azure AI Foundry deployment name, exactly as on its Deployments page.
    model_key: Mapped[str] = mapped_column(Text, nullable=False)

    input_per_m: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    cached_input_per_m: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    output_per_m: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    currency: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'USD'")
    )

    effective_from: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    isactive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    notes: Mapped[Optional[str]] = mapped_column(Text)
