"""A seller's AI Assistant usage history (credits only).

Sellers see credits, never raw tokens or USD costs — those stay internal for
reconciliation with Azure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.agent_billing_repo import agent_billing_repository as repo
from app.models.agent_billing import AgentUsage
from app.services.agent_billing.hold_service import AgentHoldService

MAX_PAGE_SIZE = 100


@dataclass(frozen=True)
class UsagePage:
    items: list[AgentUsage]
    total: int
    limit: int
    offset: int


class AgentUsageService:
    @staticmethod
    async def list_for_user(
        db: AsyncSession, user_id: uuid.UUID, *, limit: int = 20, offset: int = 0
    ) -> UsagePage:
        """The seller's messages, newest first. 404 while billing is disabled."""
        AgentHoldService.require_enabled()
        limit = max(1, min(int(limit), MAX_PAGE_SIZE))
        offset = max(0, int(offset))
        items = await repo.list_usage_for_user(db, user_id, limit=limit, offset=offset)
        total = await repo.count_usage_for_user(db, user_id)
        return UsagePage(items=items, total=total, limit=limit, offset=offset)


agent_usage_service = AgentUsageService()
