"""Request and response contracts for AI Assistant (3D Agent) billing.

snake_case with no aliases, as in every newer schema module.

Two audiences:
* the portal (seller JWT, through its /api/agent3d proxy): opens a hold before
  a message and reads the seller's usage history — credits only;
* the 3D Agent (service key): extends a hold while a message runs and settles
  it with the token counts of every LLM call — never shown to sellers.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AgentMode = Literal["animate", "split", "repair", "rig", "deliver", "refine", "auto"]
SettleOutcome = Literal["completed", "stopped", "failed"]
ExtendRefusal = Literal["insufficient_credits", "message_limit"]

# Generous bounds that still stop a malformed or hostile body.
MAX_CALLS_PER_MESSAGE = 500
MAX_TOKENS_PER_CALL = 10_000_000


# --------------------------------------------------------------------------- #
# Portal: open a hold
# --------------------------------------------------------------------------- #
class AgentHoldCreateRequest(BaseModel):
    """``POST /agent/holds`` — sent by the portal before forwarding a message."""

    model_config = ConfigDict(extra="forbid")

    product_id: Optional[uuid.UUID] = Field(default=None, description="The product being edited.")
    session_id: Optional[str] = Field(default=None, max_length=128, description="The agent chat session.")
    agent_model_id: Optional[str] = Field(default=None, max_length=128, description="The agent's model id (u-..., cut-...).")
    mode: Optional[AgentMode] = Field(default=None, description="The work mode the seller picked.")


class AgentHoldResponse(BaseModel):
    hold_key: str = Field(..., description="Pass to the agent; it extends and settles the hold with it.")
    credits_held: int
    max_credits: int = Field(..., description="The most one message can cost.")
    available_credits: Optional[int] = Field(
        ..., description="Credits left after this hold; null on an unlimited plan."
    )
    expires_at: datetime


# --------------------------------------------------------------------------- #
# Agent: extend and settle
# --------------------------------------------------------------------------- #
class AgentHoldExtendResponse(AgentHoldResponse):
    """``POST /internal/agent/holds/{hold_key}/extend``. A refusal is a normal
    200 response with ``granted: false`` — the agent stops gracefully."""

    granted: bool
    reason: Optional[ExtendRefusal] = None


class AgentLlmCall(BaseModel):
    """Token counts Azure returned for one LLM call."""

    # Extra fields (e.g. the agent's own cost estimate) are ignored: the cost is
    # always recomputed server-side from the price list.
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., min_length=1, max_length=128, description="Azure deployment name.")
    input_tokens: int = Field(..., ge=0, le=MAX_TOKENS_PER_CALL, description="Includes cached tokens.")
    cached_tokens: int = Field(default=0, ge=0, le=MAX_TOKENS_PER_CALL)
    output_tokens: int = Field(default=0, ge=0, le=MAX_TOKENS_PER_CALL, description="Includes reasoning tokens.")


class AgentSettleRequest(BaseModel):
    """``POST /internal/agent/usage/settle`` — sent once, when a message ends."""

    model_config = ConfigDict(extra="forbid")

    hold_key: str = Field(..., min_length=1, max_length=128)
    outcome: SettleOutcome
    calls: list[AgentLlmCall] = Field(default_factory=list, max_length=MAX_CALLS_PER_MESSAGE)
    error: Optional[str] = Field(default=None, max_length=2000, description="Seller-safe text only.")


class AgentSettleResponse(BaseModel):
    hold_key: str
    status: Literal["settled", "released"]
    credits_charged: int
    credits_released: int
    cost_usd: str = Field(..., description="Internal. Decimal as a string, e.g. '0.034485'.")
    already_settled: bool


# --------------------------------------------------------------------------- #
# Portal: usage history (credits only)
# --------------------------------------------------------------------------- #
class AgentUsageItem(BaseModel):
    id: uuid.UUID
    product_id: Optional[uuid.UUID]
    mode: Optional[str]
    status: Literal["holding", "settled", "released", "expired"]
    credits_charged: int
    created_at: datetime
    settled_at: Optional[datetime]


class AgentUsageListResponse(BaseModel):
    items: list[AgentUsageItem]
    total: int
    limit: int
    offset: int
