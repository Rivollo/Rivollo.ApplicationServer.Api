from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user_device import UserDevice

logger = logging.getLogger(__name__)

_MAX_FCM_TOKENS_PER_BATCH = 500


def _build_firebase_credential(credentials_module):
    if settings.FIREBASE_SERVICE_ACCOUNT_JSON_B64:
        try:
            decoded = base64.b64decode(settings.FIREBASE_SERVICE_ACCOUNT_JSON_B64)
            service_account_info = json.loads(decoded.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON_B64 is not valid base64 JSON.") from exc
        return credentials_module.Certificate(service_account_info)

    if settings.FIREBASE_JSON_PATH:
        return credentials_module.Certificate(settings.FIREBASE_JSON_PATH)

    raise RuntimeError(
        "Firebase credentials are not configured. Set FIREBASE_SERVICE_ACCOUNT_JSON_B64 "
        "or FIREBASE_JSON_PATH."
    )


def _get_firebase_messaging():
    """Initialize Firebase lazily and return the messaging module."""
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging
    except ImportError as exc:
        raise RuntimeError("firebase-admin is not installed.") from exc

    if not firebase_admin._apps:
        cred = _build_firebase_credential(credentials)
        firebase_admin.initialize_app(cred)

    return messaging


def _stringify_data(data: dict[str, Any] | None) -> dict[str, str] | None:
    if not data:
        return None
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _is_stale_token_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    code = str(getattr(exc, "code", "") or "").upper()
    message = str(exc).upper()
    return any(
        marker in code or marker in message
        for marker in ("UNREGISTERED", "INVALID_ARGUMENT", "REGISTRATION_TOKEN_NOT_REGISTERED")
    )


class PushNotificationService:
    """FCM push notification dispatch and stale-token cleanup."""

    @staticmethod
    async def send_to_user(
        db: AsyncSession,
        user_id: uuid.UUID,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
        device_type: str | None = None,
    ) -> dict[str, int | bool]:
        query = select(UserDevice.fcm_token).where(UserDevice.user_id == user_id)
        if device_type:
            query = query.where(UserDevice.device_type == device_type)

        result = await db.execute(query)
        tokens = [token for token in result.scalars().all() if token]

        if not tokens:
            return {
                "tokens_found": 0,
                "messages_sent": 0,
                "messages_failed": 0,
                "stale_tokens_removed": 0,
                "dry_run": settings.FCM_DRY_RUN,
            }

        try:
            messaging = _get_firebase_messaging()
        except RuntimeError as exc:
            logger.warning("FCM dispatch skipped for user %s: %s", user_id, exc)
            return {
                "tokens_found": len(tokens),
                "messages_sent": 0,
                "messages_failed": len(tokens),
                "stale_tokens_removed": 0,
                "dry_run": settings.FCM_DRY_RUN,
            }

        success_count = 0
        failure_count = 0
        stale_tokens: list[str] = []
        message_data = _stringify_data(data)

        for start in range(0, len(tokens), _MAX_FCM_TOKENS_PER_BATCH):
            batch_tokens = tokens[start:start + _MAX_FCM_TOKENS_PER_BATCH]
            messages = [
                messaging.Message(
                    notification=messaging.Notification(title=title, body=body),
                    data=message_data,
                    token=token,
                )
                for token in batch_tokens
            ]

            try:
                response = await asyncio.to_thread(
                    messaging.send_each,
                    messages,
                    dry_run=settings.FCM_DRY_RUN,
                )
            except Exception:
                logger.exception("FCM batch dispatch failed for user %s", user_id)
                failure_count += len(batch_tokens)
                continue

            success_count += response.success_count
            failure_count += response.failure_count

            for token, send_response in zip(batch_tokens, response.responses):
                if not send_response.success and _is_stale_token_error(send_response.exception):
                    stale_tokens.append(token)

        if stale_tokens:
            await db.execute(
                delete(UserDevice).where(UserDevice.fcm_token.in_(stale_tokens))
            )
            await db.commit()

        return {
            "tokens_found": len(tokens),
            "messages_sent": success_count,
            "messages_failed": failure_count,
            "stale_tokens_removed": len(stale_tokens),
            "dry_run": settings.FCM_DRY_RUN,
        }
