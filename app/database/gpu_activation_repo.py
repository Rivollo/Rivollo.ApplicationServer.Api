"""Repository for GPU activation tracking."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.gpu_activation import GpuActivation

_SERVICE = "3d_model"


async def get_activation(db: AsyncSession) -> GpuActivation | None:
    result = await db.execute(
        select(GpuActivation).where(GpuActivation.service == _SERVICE)
    )
    return result.scalar_one_or_none()


async def upsert_activation(db: AsyncSession, started_at: datetime) -> None:
    stmt = (
        insert(GpuActivation)
        .values(service=_SERVICE, activation_started_at=started_at)
        .on_conflict_do_update(
            index_elements=["service"],
            set_={"activation_started_at": started_at},
        )
    )
    await db.execute(stmt)
    await db.commit()
