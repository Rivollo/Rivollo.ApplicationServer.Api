"""Data access for 3D Agent billing: tbl_llm_model_prices and tbl_agent_usage.

Statements only. None of these functions commit; the service owns the
transaction, matching the convention in app/database/*_repo.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_billing import AgentUsage, LlmModelPrice
from app.models.license_assignment import LicenseAssignment
from app.models.subscription_enums import LicenseStatus


class AgentBillingRepository:
    """Repository for tbl_llm_model_prices and tbl_agent_usage."""

    # ------------------------------------------------------------------ #
    # Prices
    # ------------------------------------------------------------------ #
    @staticmethod
    async def list_prices_in_force(
        db: AsyncSession, model_keys: Iterable[str], at: datetime
    ) -> list[LlmModelPrice]:
        """The price row in force at ``at`` for each of ``model_keys``.

        In force = active, ``effective_from <= at``, and the latest such row for
        its model. Models with no such row are simply absent from the result.
        """
        keys = sorted(set(model_keys))
        if not keys:
            return []
        result = await db.execute(
            select(LlmModelPrice)
            .where(
                LlmModelPrice.model_key.in_(keys),
                LlmModelPrice.isactive.is_(True),
                LlmModelPrice.effective_from <= at,
            )
            .order_by(LlmModelPrice.model_key, LlmModelPrice.effective_from.desc())
            .distinct(LlmModelPrice.model_key)
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_all_prices_in_force(db: AsyncSession, at: datetime) -> list[LlmModelPrice]:
        """The price row in force at ``at`` for every model that has one."""
        result = await db.execute(
            select(LlmModelPrice)
            .where(
                LlmModelPrice.isactive.is_(True),
                LlmModelPrice.effective_from <= at,
            )
            .order_by(LlmModelPrice.model_key, LlmModelPrice.effective_from.desc())
            .distinct(LlmModelPrice.model_key)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------ #
    # Licence (the credit balance) — locked, so holds and charges serialise
    # ------------------------------------------------------------------ #
    @staticmethod
    async def lock_active_license(
        db: AsyncSession, user_id: uuid.UUID
    ) -> Optional[LicenseAssignment]:
        """The user's active licence, row-locked (``SELECT ... FOR UPDATE``) until
        the transaction ends. Same selection as LicensingService.get_active_license.

        Every hold open / extend / settle takes this lock FIRST, so two messages
        from one seller cannot both spend the same credits.
        """
        result = await db.execute(
            select(LicenseAssignment)
            .where(
                LicenseAssignment.user_id == user_id,
                LicenseAssignment.status == LicenseStatus.ACTIVE,
            )
            .order_by(LicenseAssignment.created_date.desc())
            .limit(1)
            .with_for_update()
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------ #
    # Usage rows (holds and bills)
    # ------------------------------------------------------------------ #
    @staticmethod
    async def sum_open_holds(db: AsyncSession, user_id: uuid.UUID, now: datetime) -> int:
        """Credits held by the user's running messages.

        A hold past ``expires_at`` no longer counts even before the sweep marks
        it expired, so a crashed agent never keeps a seller's credits blocked.
        """
        result = await db.execute(
            select(func.coalesce(func.sum(AgentUsage.credits_held), 0)).where(
                AgentUsage.user_id == user_id,
                AgentUsage.status == "holding",
                AgentUsage.expires_at > now,
            )
        )
        return int(result.scalar_one() or 0)

    @staticmethod
    def add_usage(db: AsyncSession, usage: AgentUsage) -> None:
        db.add(usage)

    @staticmethod
    async def get_usage_by_hold_key(
        db: AsyncSession, hold_key: str, *, for_update: bool = False
    ) -> Optional[AgentUsage]:
        """The usage row for a hold. ``for_update`` row-locks it — take the
        licence lock first (see lock_active_license)."""
        statement = select(AgentUsage).where(AgentUsage.hold_key == hold_key)
        if for_update:
            # populate_existing: if the row is already in the session (read
            # before the lock), overwrite it with the locked, current values.
            statement = statement.with_for_update().execution_options(populate_existing=True)
        result = await db.execute(statement)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_usage_for_user(
        db: AsyncSession, user_id: uuid.UUID, *, limit: int, offset: int
    ) -> list[AgentUsage]:
        """The user's messages, newest first. Ownership is in the WHERE clause."""
        result = await db.execute(
            select(AgentUsage)
            .where(AgentUsage.user_id == user_id)
            .order_by(AgentUsage.created_date.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    @staticmethod
    async def count_usage_for_user(db: AsyncSession, user_id: uuid.UUID) -> int:
        result = await db.execute(
            select(func.count()).select_from(AgentUsage).where(AgentUsage.user_id == user_id)
        )
        return int(result.scalar_one() or 0)


agent_billing_repository = AgentBillingRepository()
