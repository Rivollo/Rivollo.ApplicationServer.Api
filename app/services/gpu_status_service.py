"""Shared GPU status and cold-start estimate service."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.gpu_activation_repo import get_activation, upsert_activation
from app.integrations.threed_model_client import threed_model_client


class GpuStatusService:
    """Central place for GPU warm/cold detection and estimate calculation."""

    async def get_status(
        self,
        db: AsyncSession,
        *,
        touch_activation: bool = True,
    ) -> dict[str, Any]:
        """
        Return current GPU status and estimate.

        When ``touch_activation`` is true, a cold result starts or refreshes the
        activation timer only if no fresh activation is already in progress.
        """
        is_warm = await threed_model_client.fast_health_check()
        if is_warm:
            return {
                "gpu_status": "warm",
                "estimated_time": 20,
                "message": "GPU is ready. Your 3D model is being generated now.",
            }

        cold_seconds = settings.GPU_COLD_START_SECONDS
        stale_threshold = cold_seconds + 120
        now = datetime.now(timezone.utc)
        remaining = float(cold_seconds)

        activation = await get_activation(db)
        if activation is not None:
            elapsed = (now - activation.activation_started_at).total_seconds()
            if elapsed < stale_threshold:
                remaining = max(30.0, cold_seconds - elapsed)
            elif touch_activation:
                await upsert_activation(db, now)
        elif touch_activation:
            await upsert_activation(db, now)

        minutes = math.ceil(remaining / 60)
        plural = "s" if minutes != 1 else ""

        return {
            "gpu_status": "cold",
            "estimated_time": int(remaining),
            "message": f"GPU is loading up. Approximately {minutes} more minute{plural} remaining.",
        }


gpu_status_service = GpuStatusService()
