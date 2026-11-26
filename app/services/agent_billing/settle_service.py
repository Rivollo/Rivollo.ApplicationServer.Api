"""Settle a 3D Agent chat message: charge its actual credits, release the rest.

The agent reports, once the message ends, how it ended and the token counts of
every LLM call it made. The cost is recomputed here from tbl_llm_model_prices —
the agent's own estimate is never trusted for billing:

    credits = CEIL(cost x multiplier / credit_usd_value), at least min_credits,
              never above what the message held

Outcomes:
    completed  the message finished          -> charged
    stopped    it hit the hold / message limit -> charged (the work ran)
    failed     agent or LLM provider error   -> released, nothing charged
A message that made no LLM call is released whatever the outcome.

Idempotent: settling a hold that is already settled or released returns the
saved result and changes nothing, so the agent can safely retry.

A hold the sweep already expired (the agent finished after AGENT_HOLD_TTL) is
still settled: the work was done, so it is charged — the seller may end
slightly over their limit, exactly as an in-flight message is allowed to.

Locking: licence row first, usage row second — the order hold_service uses —
and usage_ai_credits is incremented on the locked licence row, so concurrent
settles and holds of one seller serialise instead of losing an update.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.agent_billing_repo import agent_billing_repository as repo
from app.services.agent_billing.hold_service import (
    HOLD_NOT_FOUND_DETAIL,
    STATUS_HOLDING,
    AgentHoldService,
)
from app.services.agent_billing.pricing_service import (
    AgentPricingService,
    BillingRates,
    LlmCallUsage,
    credits_for_cost,
    quantize_cost,
)
from app.services.licensing_service import LicensingService

logger = logging.getLogger(__name__)

OUTCOME_COMPLETED = "completed"
OUTCOME_STOPPED = "stopped"
OUTCOME_FAILED = "failed"
OUTCOMES = (OUTCOME_COMPLETED, OUTCOME_STOPPED, OUTCOME_FAILED)

STATUS_SETTLED = "settled"
STATUS_RELEASED = "released"
STATUS_EXPIRED = "expired"

_MAX_ERROR_CHARS = 500
_QUOTA_KEY = "ai_credits"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SettleResult:
    hold_key: str
    status: str  # settled | released
    credits_charged: int
    credits_released: int
    cost_usd: Decimal
    # True when this hold had already been settled: nothing changed now.
    already_settled: bool = False


class AgentSettleService:
    """Turns a finished message's LLM usage into a charge."""

    @staticmethod
    def _result(usage, already_settled: bool) -> SettleResult:
        charged = int(usage.credits_charged)
        return SettleResult(
            hold_key=usage.hold_key,
            status=usage.status,
            credits_charged=charged,
            credits_released=max(int(usage.credits_held) - charged, 0),
            cost_usd=Decimal(usage.cost_usd or 0),
            already_settled=already_settled,
        )

    @staticmethod
    async def settle(
        db: AsyncSession,
        hold_key: str,
        outcome: str,
        calls: list[LlmCallUsage],
        *,
        error: Optional[str] = None,
    ) -> SettleResult:
        """Charge (or release) a message. Raises 404 for an unknown hold, 422
        for an unknown outcome or impossible token counts. Commits."""
        AgentHoldService.require_enabled()
        if outcome not in OUTCOMES:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"outcome must be one of: {', '.join(OUTCOMES)}",
            )
        now = _utcnow()

        found = await repo.get_usage_by_hold_key(db, hold_key)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=HOLD_NOT_FOUND_DETAIL)
        if found.status in (STATUS_SETTLED, STATUS_RELEASED):
            return AgentSettleService._result(found, already_settled=True)
        user_id = found.user_id

        # Fixed lock order: licence, then the usage row.
        license_obj = await repo.lock_active_license(db, user_id)
        usage = await repo.get_usage_by_hold_key(db, hold_key, for_update=True)
        if usage is None:
            await db.rollback()
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=HOLD_NOT_FOUND_DETAIL)
        if usage.status in (STATUS_SETTLED, STATUS_RELEASED):
            # A concurrent retry won the lock first.
            result = AgentSettleService._result(usage, already_settled=True)
            await db.rollback()
            return result

        # Price every call, even for a failed message: the cost is recorded
        # for reconciliation with Azure whether or not it is charged.
        try:
            cost = await AgentPricingService.price_message(db, calls, at=usage.created_date)
        except ValueError as exc:
            await db.rollback()
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

        charge = outcome in (OUTCOME_COMPLETED, OUTCOME_STOPPED) and cost.llm_calls > 0
        credits = 0
        if charge:
            # The rates recorded when the hold was opened, not today's settings.
            current = BillingRates.from_settings()
            rates = BillingRates(
                multiplier=Decimal(usage.multiplier) if usage.multiplier is not None else current.multiplier,
                credit_usd_value=(
                    Decimal(usage.credit_usd_value)
                    if usage.credit_usd_value is not None
                    else current.credit_usd_value
                ),
                min_credits=current.min_credits,
            )
            # Never above what was held. An expired hold no longer counts as
            # held, but the work it covered still ran: cap at what it held then.
            credits = credits_for_cost(cost.cost_usd, rates, cap=max(int(usage.credits_held), 1))

        previous_usage = None
        limit = None
        if credits and license_obj is not None:
            previous_usage = int(license_obj.usage_ai_credits or 0)
            limit = license_obj.limit_max_ai_credits
            license_obj.usage_ai_credits = previous_usage + credits
        elif credits:
            # The plan ended while the message ran: there is no balance to
            # charge. Record what it cost; charge nothing.
            logger.warning(
                "agent billing: hold %s for user %s settled with no active licence; "
                "%s credits not charged",
                hold_key,
                user_id,
                credits,
            )
            credits = 0

        usage.status = STATUS_SETTLED if credits else STATUS_RELEASED
        usage.credits_charged = credits
        usage.cost_usd = quantize_cost(cost.cost_usd)
        usage.input_tokens = cost.input_tokens
        usage.cached_tokens = cost.cached_tokens
        usage.output_tokens = cost.output_tokens
        usage.llm_calls = cost.llm_calls
        usage.breakdown = cost.breakdown
        if error:
            usage.error = error.strip()[:_MAX_ERROR_CHARS]
        usage.settled_at = now
        usage.updated_date = now

        result = AgentSettleService._result(usage, already_settled=False)
        await db.commit()

        if previous_usage is not None:
            await LicensingService.notify_usage_thresholds(
                db,
                user_id,
                _QUOTA_KEY,
                previous_usage=previous_usage,
                current_usage=previous_usage + credits,
                limit=limit,
            )
        return result


agent_settle_service = AgentSettleService()
