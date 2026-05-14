from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import CurrentUser, DB
from app.schemas.notifications import (
    DirectNotificationRequest,
    TokenRegistrationRequest,
    TokenUnregisterRequest,
)
from app.services.notification_service import NotificationService
from app.services.user_device_service import UserDeviceService
from app.utils.envelopes import api_success

router = APIRouter(tags=["notifications"])


@router.get("/notifications", response_model=dict, status_code=status.HTTP_200_OK)
async def list_notifications(
    current_user: CurrentUser,
    db: DB,
    unread_only: bool = Query(False, alias="unreadOnly"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    result = await NotificationService.list_notifications(
        db=db,
        user_id=current_user.id,
        unread_only=unread_only,
        limit=limit,
        offset=offset,
    )
    return api_success(result)


@router.get("/notifications/unread", response_model=dict, status_code=status.HTTP_200_OK)
async def list_unread_notifications(
    current_user: CurrentUser,
    db: DB,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    result = await NotificationService.list_notifications(
        db=db,
        user_id=current_user.id,
        unread_only=True,
        limit=limit,
        offset=offset,
    )
    return api_success(result)


@router.patch("/notifications/{notification_id}/read", response_model=dict, status_code=status.HTTP_200_OK)
async def mark_notification_read(
    notification_id: UUID,
    current_user: CurrentUser,
    db: DB,
):
    notification = await NotificationService.mark_notification_read(
        db=db,
        user_id=current_user.id,
        notification_id=notification_id,
    )
    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found",
        )
    return api_success(NotificationService.serialize_notification(notification))


@router.patch("/notifications/read-all", response_model=dict, status_code=status.HTTP_200_OK)
async def mark_all_notifications_read(
    current_user: CurrentUser,
    db: DB,
):
    updated_count = await NotificationService.mark_all_notifications_read(
        db=db,
        user_id=current_user.id,
    )
    return api_success({"updated_count": updated_count})


@router.post("/devices/register", response_model=dict, status_code=status.HTTP_200_OK)
# @router.post("/fcmtoken/register", response_model=dict, status_code=status.HTTP_200_OK)
async def register_device(
    payload: TokenRegistrationRequest,
    current_user: CurrentUser,
    db: DB,
):
    device = await UserDeviceService.register_device(
        db=db,
        user_id=current_user.id,
        fcm_token=payload.fcm_token,
        device_type=payload.device_type,
    )
    return api_success(
        {
            "message": "Device synced successfully.",
            "device_id": device.id,
        }
    )


@router.post("/devices/unregister", response_model=dict, status_code=status.HTTP_200_OK)
async def unregister_device(
    payload: TokenUnregisterRequest,
    current_user: CurrentUser,
    db: DB,
):
    deleted_count = await UserDeviceService.unregister_device(
        db=db,
        user_id=current_user.id,
        fcm_token=payload.fcm_token,
    )
    return api_success(
        {
            "message": "Device removed successfully.",
            "deleted_count": deleted_count,
        }
    )


@router.post("/notifications/send", response_model=dict, status_code=status.HTTP_200_OK)
async def dispatch_notification(
    payload: DirectNotificationRequest,
    current_user: CurrentUser,
    db: DB,
):
    if payload.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only send test notifications to your own user account.",
        )

    result = await NotificationService.create_and_push_notification(
        db=db,
        user_id=payload.user_id,
        notification_type=payload.notification_type,
        title=payload.title,
        body=payload.body,
        data=payload.data,
        device_type=payload.device_type,
    )
    return api_success(result)
