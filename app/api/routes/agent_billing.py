"""AI Assistant (3D Agent) billing routes — sql/create_agent_billing.sql.

Portal (seller JWT, through the portal's /api/agent3d proxy):
    POST /agent/holds                              open a message's credit hold
    GET  /agent/usage                              the seller's usage (credits only)

3D Agent (X-Agent-Usage-Key service key):
    POST /internal/agent/holds/{hold_key}/extend   one more chunk, or a refusal
    POST /internal/agent/usage/settle              charge actual credits, release the rest

Every route answers 404 while ENABLE_AGENT_BILLING is off.

A router of its own so the feature can be removed with one include_router line.
"""

from fastapi import APIRouter, Query, status

from app.api.deps import DB, AgentServiceAuth, CurrentUser
from app.schemas.agent_billing import (
    AgentHoldCreateRequest,
    AgentHoldExtendResponse,
    AgentHoldResponse,
    AgentSettleRequest,
    AgentSettleResponse,
    AgentUsageItem,
    AgentUsageListResponse,
)
from app.services.agent_billing.hold_service import AgentHoldService, HoldState
from app.services.agent_billing.pricing_service import LlmCallUsage
from app.services.agent_billing.settle_service import AgentSettleService
from app.services.agent_billing.usage_service import MAX_PAGE_SIZE, AgentUsageService
from app.utils.envelopes import api_success

router = APIRouter(tags=["agent-billing"])


def _hold_fields(hold: HoldState) -> dict:
    return {
        "hold_key": hold.hold_key,
        "credits_held": hold.credits_held,
        "max_credits": hold.max_credits,
        "available_credits": hold.available_credits,
        "expires_at": hold.expires_at,
    }


@router.post("/agent/holds", response_model=dict, status_code=status.HTTP_201_CREATED)
async def open_agent_hold(body: AgentHoldCreateRequest, current_user: CurrentUser, db: DB):
    """Hold the first credits for an AI Assistant message (400 when the seller
    cannot cover them)."""
    hold = await AgentHoldService.open_hold(
        db,
        current_user.id,
        product_id=body.product_id,
        session_id=body.session_id,
        agent_model_id=body.agent_model_id,
        mode=body.mode,
    )
    return api_success(AgentHoldResponse(**_hold_fields(hold)).model_dump(mode="json"))


@router.get("/agent/usage", response_model=dict)
async def list_agent_usage(
    current_user: CurrentUser,
    db: DB,
    limit: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
):
    """The seller's AI Assistant messages, newest first — credits only."""
    page = await AgentUsageService.list_for_user(db, current_user.id, limit=limit, offset=offset)
    items = [
        AgentUsageItem(
            id=row.id,
            product_id=row.product_id,
            mode=row.mode,
            status=row.status,
            credits_charged=row.credits_charged,
            created_at=row.created_date,
            settled_at=row.settled_at,
        )
        for row in page.items
    ]
    response = AgentUsageListResponse(
        items=items, total=page.total, limit=page.limit, offset=page.offset
    )
    return api_success(response.model_dump(mode="json"))


@router.post("/internal/agent/holds/{hold_key}/extend", response_model=dict)
async def extend_agent_hold(hold_key: str, _: AgentServiceAuth, db: DB):
    """Add one chunk to a running message's hold. A refusal is a 200 with
    ``granted: false`` and a reason; an unknown hold is 404, a closed one 409."""
    result = await AgentHoldService.extend_hold(db, hold_key)
    response = AgentHoldExtendResponse(
        **_hold_fields(result.hold), granted=result.granted, reason=result.reason
    )
    return api_success(response.model_dump(mode="json"))


@router.post("/internal/agent/usage/settle", response_model=dict)
async def settle_agent_usage(body: AgentSettleRequest, _: AgentServiceAuth, db: DB):
    """Charge a finished message's actual credits and release the rest.
    Idempotent: settling again returns the first result."""
    calls = [
        LlmCallUsage(
            model=c.model,
            input_tokens=c.input_tokens,
            cached_tokens=c.cached_tokens,
            output_tokens=c.output_tokens,
        )
        for c in body.calls
    ]
    result = await AgentSettleService.settle(
        db, body.hold_key, body.outcome, calls, error=body.error
    )
    response = AgentSettleResponse(
        hold_key=result.hold_key,
        status=result.status,
        credits_charged=result.credits_charged,
        credits_released=result.credits_released,
        cost_usd=str(result.cost_usd),
        already_settled=result.already_settled,
    )
    return api_success(response.model_dump(mode="json"))
