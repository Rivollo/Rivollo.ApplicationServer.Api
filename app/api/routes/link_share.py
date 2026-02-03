from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_current_user
import logging

from app.schemas.whatsapp import (
    SendWhatsAppLinkRequest,
    SendWhatsAppLinkResponse,
)
from app.services.link_share_service import LinkShareService
from app.core.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/link-share",
    tags=["Link Share"],
)


@router.post(
    "/whatsapp",
    response_model=SendWhatsAppLinkResponse,
)
async def share_link_via_whatsapp(
    payload: SendWhatsAppLinkRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Share a product link via WhatsApp and log the event.
    """

    try:
        result = await LinkShareService.share_link_via_whatsapp(
            db=db,
            payload=payload,
            shared_by_user_id=current_user.id,
        )

        # business failure → client error
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error_message"],
            )

        return SendWhatsAppLinkResponse(
            success=True,
            message_id=result["message_id"],
            error_message=None,
        )

    except HTTPException:
        # already correct → pass through
        raise

    except Exception as e:
        logger.exception("Unexpected error in WhatsApp share route")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error",
        )
