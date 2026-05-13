from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user_device import UserDevice


class UserDeviceService:
    """Persistence operations for user FCM device tokens."""

    @staticmethod
    async def register_device(
        db: AsyncSession,
        user_id: uuid.UUID,
        fcm_token: str,
        device_type: str,
    ) -> UserDevice:
        result = await db.execute(
            select(UserDevice).where(UserDevice.fcm_token == fcm_token)
        )
        device = result.scalar_one_or_none()

        if device:
            device.user_id = user_id
            device.device_type = device_type
            device.updated_by = user_id
            device.updated_date = datetime.now(timezone.utc)
        else:
            device = UserDevice(
                user_id=user_id,
                fcm_token=fcm_token,
                device_type=device_type,
                created_by=user_id,
            )
            db.add(device)

        await db.commit()
        await db.refresh(device)
        return device

    @staticmethod
    async def unregister_device(
        db: AsyncSession,
        user_id: uuid.UUID,
        fcm_token: str,
    ) -> int:
        result = await db.execute(
            delete(UserDevice).where(
                UserDevice.user_id == user_id,
                UserDevice.fcm_token == fcm_token,
            )
        )
        await db.commit()
        return result.rowcount or 0
