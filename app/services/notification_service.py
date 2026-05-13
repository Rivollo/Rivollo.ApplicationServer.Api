"""Notification service for user notifications."""

import uuid
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import json

from app.models.models import Notification, NotificationChannel, UserNotificationPreference
from app.services.push_notification_service import PushNotificationService


class NotificationService:
    """Service for managing user notifications."""

    @staticmethod
    async def create_notification(
        db: AsyncSession,
        user_id: uuid.UUID,
        notification_type: str,
        title: str,
        body: str,
        data: Optional[dict[str, Any]] = None,
        channel: NotificationChannel = NotificationChannel.IN_APP,
    ) -> Notification:
        """Create a notification for a user."""
        # Check if user has muted this notification type
        # Respect user mutes (best-effort; prefs schema stores muted_types as TEXT)
        if await NotificationService.is_muted(db, user_id, notification_type):
            # User has muted this notification type - skip
            return None

        # Serialize data to TEXT for DB
        serialized = json.dumps(data) if data is not None else None

        notification = Notification(
            user_id=user_id,
            type=notification_type,
            title=title,
            body=body,
            data=serialized,
            channel=channel,
        )

        db.add(notification)
        await db.commit()
        await db.refresh(notification)

        return notification

    @staticmethod
    async def create_and_push_notification(
        db: AsyncSession,
        user_id: uuid.UUID,
        notification_type: str,
        title: str,
        body: str,
        data: Optional[dict[str, Any]] = None,
        device_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Store the app notification, then send push best-effort."""
        notification = await NotificationService.create_notification(
            db=db,
            user_id=user_id,
            notification_type=notification_type,
            title=title,
            body=body,
            data=data,
            device_type=device_type,
        )

        if notification is None:
            return {
                "notification_id": None,
                "stored": False,
                "push": {
                    "tokens_found": 0,
                    "messages_sent": 0,
                    "messages_failed": 0,
                    "stale_tokens_removed": 0,
                    "skipped": True,
                },
            }

        push_result = await PushNotificationService.send_to_user(
            db=db,
            user_id=user_id,
            title=title,
            body=body,
            data=data,
        )

        return {
            "notification_id": str(notification.id) if notification else None,
            "stored": notification is not None,
            "push": push_result,
        }

    @staticmethod
    async def get_user_preferences(
        db: AsyncSession,
        user_id: uuid.UUID,
    ) -> Optional[UserNotificationPreference]:
        """Get user notification preferences for a specific type."""
        result = await db.execute(
            select(UserNotificationPreference).where(
                UserNotificationPreference.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def is_muted(
        db: AsyncSession,
        user_id: uuid.UUID,
        notification_type: str,
    ) -> bool:
        prefs = await NotificationService.get_user_preferences(db, user_id)
        if not prefs or not prefs.muted_types:
            return False
        # Try JSON array first; fallback to comma-separated list
        try:
            muted = json.loads(prefs.muted_types)
            if isinstance(muted, list):
                return notification_type in muted
        except Exception:
            pass
        # Fallback CSV parsing
        return any(nt.strip() == notification_type for nt in prefs.muted_types.split(","))

    @staticmethod
    async def notify_job_completed(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_name: str,
        job_id: uuid.UUID,
    ) -> Optional[Notification]:
        """Notify user that a 3D job has completed."""
        result = await NotificationService.create_and_push_notification(
            db=db,
            user_id=user_id,
            notification_type="job.completed",
            title="3D Model Ready",
            body=f"Your 3D model for '{product_name}' is ready to view and configure.",
            data={"job_id": str(job_id), "product_name": product_name},
        )
        notification_id = result.get("notification_id")
        if not notification_id:
            return None
        return await db.get(Notification, uuid.UUID(notification_id))

    @staticmethod
    async def notify_quota_warning(
        db: AsyncSession,
        user_id: uuid.UUID,
        quota_type: str,
        percentage: int,
    ) -> Optional[Notification]:
        """Notify user about quota usage warning."""
        result = await NotificationService.create_and_push_notification(
            db=db,
            user_id=user_id,
            notification_type="quota.warning",
            title="Quota Warning",
            body=f"You've used {percentage}% of your {quota_type} quota.",
            data={"quota_type": quota_type, "percentage": percentage},
        )
        notification_id = result.get("notification_id")
        if not notification_id:
            return None
        return await db.get(Notification, uuid.UUID(notification_id))
