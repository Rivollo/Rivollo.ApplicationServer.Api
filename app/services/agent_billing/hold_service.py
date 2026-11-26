"""Credit holds for 3D Agent chat messages ("taxi meter").

A message does not reserve a fixed cap up front. It opens a hold of one chunk
(AGENT_HOLD_CHUNK_CREDITS) and, while it runs, the agent asks to extend the
hold one chunk at a time before an LLM call it could not otherwise cover. An
extension is refused when the seller's available credits cannot cover the next
chunk, or the message would pass AGENT_MAX_CREDITS_PER_MESSAGE — the agent then
stops cleanly between steps. Settling (charging the actual credits and
releasing the rest) is a later step.

    available = limit_max_ai_credits - usage_ai_credits - credits held by the
                seller's running messages            (limit None = unlimited)

Concurrency: every open / extend takes ``SELECT ... FOR UPDATE`` on the
seller's licence row FIRST, then reads the open holds, then writes. Two
messages from one seller therefore serialise and can never both spend the same
credits. Settling takes the same lock in the same order.

Refusing to OPEN is an HTTP error (the portal shows it before anything runs).
Refusing to EXTEND is a normal result (``granted=False`` with a reason): the
agent expects it and stops gracefully.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.agent_billing_repo import agent_billing_repository as repo
from app.models.agent_billing import AgentUsage

logger = logging.getLogger(__name__)

STATUS_HOLDING = "holding"

# Extension refusals the agent turns into a message to the seller.
REASON_INSUFFICIENT_CREDITS = "insufficient_credits"
REASON_MESSAGE_LIMIT = "message_limit"

BILLING_DISABLED_DETAIL = "The AI Assistant is not available right now."
NO_PLAN_DETAIL = "You need an active plan to use the AI Assistant."
HOLD_NOT_FOUND_DETAIL = "Credit hold not found."
HOLD_CLOSED_DETAIL = "This credit hold is no longer open."


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class HoldState:
    """A hold as the caller needs it."""

    hold_key: str
    credits_held: int
    max_credits: int
    # Credits the seller can still use after this hold; None = unlimited plan.
    available_credits: Optional[int]
    expires_at: datetime


@dataclass(frozen=True)
class ExtendResult:
    granted: bool
    hold: HoldState
    # Why an extension was refused (REASON_*); None when granted.
    reason: Optional[str] = None


class AgentHoldService:
    """Opens and extends AI Assistant credit holds."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def require_enabled() -> None:
        """404 while ENABLE_AGENT_BILLING is off (the tables may not exist)."""
        if not settings.ENABLE_AGENT_BILLING:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=BILLING_DISABLED_DETAIL)

    @staticmethod
    async def _available(db: AsyncSession, license_obj, user_id: uuid.UUID, now: datetime) -> Optional[int]:
        """Credits free to hold right now; None = unlimited plan. Call under the
        licence lock."""
        limit = license_obj.limit_max_ai_credits
        if limit is None:
            return None
        used = int(license_obj.usage_ai_credits or 0)
        held = await repo.sum_open_holds(db, user_id, now)
        return max(int(limit) - used - held, 0)

    @staticmethod
    def _state(usage: AgentUsage, available: Optional[int]) -> HoldState:
        return HoldState(
            hold_key=usage.hold_key,
            credits_held=int(usage.credits_held),
            max_credits=int(usage.max_credits),
            available_credits=available,
            expires_at=usage.expires_at,
        )

    # ------------------------------------------------------------------ #
    # Open
    # ------------------------------------------------------------------ #
    @staticmethod
    async def open_hold(
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        product_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        agent_model_id: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> HoldState:
        """Start a message: hold its first chunk. Raises 400 when the seller
        cannot cover it, 404 when agent billing is disabled. Commits."""
        AgentHoldService.require_enabled()
        now = _utcnow()
        max_credits = int(settings.AGENT_MAX_CREDITS_PER_MESSAGE)
        chunk = min(int(settings.AGENT_HOLD_CHUNK_CREDITS), max_credits)

        license_obj = await repo.lock_active_license(db, user_id)
        if license_obj is None:
            await db.rollback()
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=NO_PLAN_DETAIL)

        available = await AgentHoldService._available(db, license_obj, user_id, now)
        if available is not None and available < chunk:
            await db.rollback()
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Not enough AI credits. At least {chunk} credits are required "
                    "to send a message to the AI Assistant."
                ),
            )

        usage = AgentUsage(
            hold_key=secrets.token_urlsafe(24),
            user_id=user_id,
            product_id=product_id,
            session_id=session_id,
            agent_model_id=agent_model_id,
            mode=mode,
            status=STATUS_HOLDING,
            credits_held=chunk,
            credits_charged=0,
            max_credits=max_credits,
            multiplier=Decimal(str(settings.AGENT_BILLING_MULTIPLIER)),
            credit_usd_value=Decimal(str(settings.AGENT_CREDIT_USD_VALUE)),
            expires_at=now + timedelta(minutes=int(settings.AGENT_HOLD_TTL_MINUTES)),
            created_by=user_id,
        )
        repo.add_usage(db, usage)
        await db.commit()

        remaining = None if available is None else available - chunk
        return AgentHoldService._state(usage, remaining)

    # ------------------------------------------------------------------ #
    # Extend
    # ------------------------------------------------------------------ #
    @staticmethod
    async def extend_hold(db: AsyncSession, hold_key: str) -> ExtendResult:
        """Add one chunk to a running message's hold, or refuse with a reason.

        Raises 404 for an unknown hold and 409 for one that is no longer open
        (settled, released, expired, or past its expiry time). Commits when it
        grants; a refusal changes nothing.
        """
        AgentHoldService.require_enabled()
        now = _utcnow()

        # Find the owner without a lock, then lock in the fixed order:
        # licence first, usage row second (the same order open and settle use).
        found = await repo.get_usage_by_hold_key(db, hold_key)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=HOLD_NOT_FOUND_DETAIL)
        user_id = found.user_id

        license_obj = await repo.lock_active_license(db, user_id)
        usage = await repo.get_usage_by_hold_key(db, hold_key, for_update=True)
        if usage is None:
            await db.rollback()
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=HOLD_NOT_FOUND_DETAIL)
        if usage.status != STATUS_HOLDING or _as_aware(usage.expires_at) <= now:
            await db.rollback()
            raise HTTPException(status.HTTP_409_CONFLICT, detail=HOLD_CLOSED_DETAIL)

        # Refusals build their result BEFORE rolling back: a rollback expires
        # the loaded rows, and reading them afterwards would hit the database.
        if license_obj is None:
            # Plan ended mid-message: nothing more can be held.
            refused = ExtendResult(
                granted=False,
                hold=AgentHoldService._state(usage, 0),
                reason=REASON_INSUFFICIENT_CREDITS,
            )
            await db.rollback()
            return refused

        available = await AgentHoldService._available(db, license_obj, user_id, now)
        chunk = min(
            int(settings.AGENT_HOLD_CHUNK_CREDITS),
            int(usage.max_credits) - int(usage.credits_held),
        )
        reason = None
        if chunk <= 0:
            reason = REASON_MESSAGE_LIMIT
        elif available is not None and available < chunk:
            reason = REASON_INSUFFICIENT_CREDITS
        if reason is not None:
            refused = ExtendResult(
                granted=False, hold=AgentHoldService._state(usage, available), reason=reason
            )
            await db.rollback()
            return refused

        usage.credits_held = int(usage.credits_held) + chunk
        usage.updated_date = now
        await db.commit()

        remaining = None if available is None else available - chunk
        return ExtendResult(granted=True, hold=AgentHoldService._state(usage, remaining))


agent_hold_service = AgentHoldService()
