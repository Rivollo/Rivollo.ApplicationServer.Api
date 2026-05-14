import math
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.gpu_activation_repo import get_activation, upsert_activation
from app.integrations.threed_model_client import threed_model_client


class GPUStatusService:
    async def get_status(
            self,
            db: AsyncSession,
            *,
            touch_activation: bool = True,
    ) -> dict:
        is_warm = await threed_model_client.fast_health_check()

        if is_warm:
            return {
                "gpu_status": "warm",
                "estimated_seconds": 20,
                "estimated_time": "20 seconds",
                "message": "GPU is ready. Your 3D model is being generated now.",
            }
        
       
        cold_seconds = settings.GPU_COLD_START_SECONDS
        stale_threshold = cold_seconds + 120
        now = datetime.now(timezone.utc)

        activation = await get_activation(db)

        if activation is not None:
            elapsed = (now - activation.activation_started_at).total_seconds()
            if elapsed < stale_threshold:
                remaining = max(30, cold_seconds - elapsed)
            else:
                remaining = cold_seconds
                if touch_activation:
                    await upsert_activation(db, now)
        else:
            remaining = cold_seconds
            if touch_activation:
                await upsert_activation(db, now)

        minutes = math.ceil(remaining / 60)

        return {
            "gpu_status": "cold",
            "estimated_seconds": int(remaining),
            "estimated_time": f"~{minutes} minute{'s' if minutes != 1 else ''}",
            "message": f"GPU is loading up. Approximately {minutes} more minute{'s' if minutes != 1 else ''} remaining.",
        }


gpu_status_service = GPUStatusService()
