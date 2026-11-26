"""LLM token usage -> USD cost -> AI credits, for 3D Agent chat messages.

The agent reports, per LLM call, the deployment it called and the token counts
Azure returned. Azure's responses carry no price, so the cost is computed here
from tbl_llm_model_prices — the same tokens x meter price Azure bills:

    uncached = input_tokens - cached_tokens
    cost     = (uncached x input_per_m + cached x cached_input_per_m
                + output_tokens x output_per_m) / 1,000,000

``input_tokens`` INCLUDES the cached ones (that is how the OpenAI-compatible
``usage`` block reports them); ``output_tokens`` includes reasoning tokens.

Credits for a whole message are rounded up once, never per call:

    credits = CEIL(cost x multiplier / credit_usd_value), at least min_credits,
              and never above the message's cap

Money is Decimal throughout; floats only at the edges (config values).

The pure functions take no database and raise ValueError on bad input; the
service class loads prices through the repository and decides the policy for a
model with no price (bill it at the most expensive active price, so a missing
row can never make a message free).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
from typing import Iterable, Mapping, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.agent_billing_repo import agent_billing_repository as repo
from app.models.agent_billing import LlmModelPrice

logger = logging.getLogger(__name__)

_PER_MILLION = Decimal(1_000_000)
# Stored precision of tbl_agent_usage.cost_usd (NUMERIC(12, 6)).
_COST_QUANTUM = Decimal("0.000001")


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens for one deployment."""

    model_key: str
    input_per_m: Decimal
    cached_input_per_m: Decimal
    output_per_m: Decimal

    @classmethod
    def from_row(cls, row: LlmModelPrice) -> "ModelPrice":
        return cls(
            model_key=row.model_key,
            input_per_m=Decimal(row.input_per_m),
            cached_input_per_m=Decimal(row.cached_input_per_m),
            output_per_m=Decimal(row.output_per_m),
        )

    @property
    def weight(self) -> Decimal:
        """Orders prices by expensiveness, for the missing-price fallback."""
        return self.input_per_m + self.cached_input_per_m + self.output_per_m


@dataclass(frozen=True)
class LlmCallUsage:
    """Token counts Azure returned for one LLM call."""

    model: str
    input_tokens: int
    cached_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelCostLine:
    """One model's share of a message: summed tokens and cost."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    # Billed at the fallback price because the model had no price row.
    price_fallback: bool = False

    def as_breakdown(self) -> dict:
        """The JSON stored in tbl_agent_usage.breakdown."""
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": str(self.cost_usd.quantize(_COST_QUANTUM)),
            "price_fallback": self.price_fallback,
        }


@dataclass
class MessageCost:
    """The whole message: totals plus a line per model."""

    cost_usd: Decimal = Decimal(0)
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    lines: dict[str, ModelCostLine] = field(default_factory=dict)

    @property
    def breakdown(self) -> list[dict]:
        return [line.as_breakdown() for line in self.lines.values()]


@dataclass(frozen=True)
class BillingRates:
    """The rates a message is billed at (recorded on its usage row)."""

    multiplier: Decimal
    credit_usd_value: Decimal
    min_credits: int

    @classmethod
    def from_settings(cls) -> "BillingRates":
        # str() first: Decimal(0.01) is 0.01000000000000000020816681711721685...
        return cls(
            multiplier=Decimal(str(settings.AGENT_BILLING_MULTIPLIER)),
            credit_usd_value=Decimal(str(settings.AGENT_CREDIT_USD_VALUE)),
            min_credits=int(settings.AGENT_MIN_CREDITS),
        )


# --------------------------------------------------------------------------- #
# Pure calculation
# --------------------------------------------------------------------------- #
def _validate(usage: LlmCallUsage) -> None:
    if usage.input_tokens < 0 or usage.cached_tokens < 0 or usage.output_tokens < 0:
        raise ValueError(f"negative token count for model '{usage.model}'")
    if usage.cached_tokens > usage.input_tokens:
        raise ValueError(
            f"cached tokens ({usage.cached_tokens}) exceed input tokens "
            f"({usage.input_tokens}) for model '{usage.model}'"
        )


def call_cost_usd(usage: LlmCallUsage, price: ModelPrice) -> Decimal:
    """Exact USD cost of one LLM call (not rounded)."""
    _validate(usage)
    uncached = usage.input_tokens - usage.cached_tokens
    return (
        Decimal(uncached) * price.input_per_m
        + Decimal(usage.cached_tokens) * price.cached_input_per_m
        + Decimal(usage.output_tokens) * price.output_per_m
    ) / _PER_MILLION


def message_cost(
    calls: Iterable[LlmCallUsage],
    prices: Mapping[str, ModelPrice],
    fallback: Optional[ModelPrice] = None,
) -> MessageCost:
    """Sum every call of a message, per model and in total.

    A call whose model is not in ``prices`` is billed at ``fallback``; with no
    fallback it raises ValueError, so an unpriced call is never silently free.
    """
    total = MessageCost()
    for usage in calls:
        price = prices.get(usage.model)
        used_fallback = False
        if price is None:
            if fallback is None:
                raise ValueError(f"no price for model '{usage.model}'")
            price, used_fallback = fallback, True

        cost = call_cost_usd(usage, price)
        line = total.lines.setdefault(usage.model, ModelCostLine(model=usage.model))
        line.calls += 1
        line.input_tokens += usage.input_tokens
        line.cached_tokens += usage.cached_tokens
        line.output_tokens += usage.output_tokens
        line.cost_usd += cost
        line.price_fallback = line.price_fallback or used_fallback

        total.llm_calls += 1
        total.input_tokens += usage.input_tokens
        total.cached_tokens += usage.cached_tokens
        total.output_tokens += usage.output_tokens
        total.cost_usd += cost
    return total


def credits_for_cost(
    cost_usd: Decimal, rates: BillingRates, cap: Optional[int] = None
) -> int:
    """AI credits for a message's total cost: rounded up once, at least
    ``rates.min_credits``, and never above ``cap`` when one is given."""
    if cost_usd < 0:
        raise ValueError("cost cannot be negative")
    if rates.credit_usd_value <= 0:
        raise ValueError("credit_usd_value must be positive")
    raw = cost_usd * rates.multiplier / rates.credit_usd_value
    credits = int(raw.to_integral_value(rounding=ROUND_CEILING))
    credits = max(credits, rates.min_credits)
    if cap is not None:
        credits = min(credits, cap)
    return credits


def quantize_cost(cost_usd: Decimal) -> Decimal:
    """Cost at the precision tbl_agent_usage.cost_usd stores."""
    return cost_usd.quantize(_COST_QUANTUM, rounding=ROUND_CEILING)


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class AgentPricingService:
    """Prices a message's LLM calls against tbl_llm_model_prices."""

    @staticmethod
    async def load_prices(
        db: AsyncSession, model_keys: Iterable[str], at: Optional[datetime] = None
    ) -> tuple[dict[str, ModelPrice], Optional[ModelPrice]]:
        """Prices in force at ``at`` for ``model_keys``, plus the fallback for a
        model with none: the most expensive active price, or None if the price
        list is empty."""
        at = at or datetime.now(timezone.utc)
        keys = set(model_keys)
        rows = await repo.list_prices_in_force(db, keys, at)
        prices = {row.model_key: ModelPrice.from_row(row) for row in rows}

        fallback: Optional[ModelPrice] = None
        if keys - prices.keys():
            every = [ModelPrice.from_row(r) for r in await repo.list_all_prices_in_force(db, at)]
            fallback = max(every, key=lambda p: p.weight, default=None)
        return prices, fallback

    @staticmethod
    async def price_message(
        db: AsyncSession,
        calls: list[LlmCallUsage],
        at: Optional[datetime] = None,
    ) -> MessageCost:
        """Cost of a message's calls. A model with no price row is billed at the
        most expensive active price (and logged); with no prices at all, raises
        ValueError — billing must not go ahead on an empty price list."""
        prices, fallback = await AgentPricingService.load_prices(
            db, (c.model for c in calls), at
        )
        missing = sorted({c.model for c in calls} - prices.keys())
        if missing:
            if fallback is None:
                raise ValueError("tbl_llm_model_prices has no active price to bill with")
            logger.warning(
                "agent billing: no price for %s; billed at the %s price",
                ", ".join(missing),
                fallback.model_key,
            )
        return message_cost(calls, prices, fallback)

    @staticmethod
    def credits(
        cost_usd: Decimal, cap: Optional[int] = None, rates: Optional[BillingRates] = None
    ) -> int:
        """Credits for a message's cost at the configured (or given) rates."""
        return credits_for_cost(cost_usd, rates or BillingRates.from_settings(), cap)


agent_pricing_service = AgentPricingService()
