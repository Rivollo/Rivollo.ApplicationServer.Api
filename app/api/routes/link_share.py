"""Link sharing via WhatsApp routes."""

import logging
from fastapi import APIRouter, Depends

from app.api.deps import CurrentUser, DB, get_current_user
from app.schemas.whatsapp import (
    SendWhatsAppLinkRequest,
    SendWhatsAppLinkResponse,
)
from app.services.link_share_service import LinkShareService
from app.utils.envelopes import api_success

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/link-share",
    tags=["Link Share"],
    dependencies=[Depends(get_current_user)],
)


@router.post(
    "/whatsapp",
    response_model=dict,
)
async def share_link_via_whatsapp(
    payload: SendWhatsAppLinkRequest,
    current_user: CurrentUser, 
    db: DB,
):
    """
    Share a product link via WhatsApp and log the event.
    """
    try:
        logger.info(
            f"Processing WhatsApp share request for user {current_user.id}"
        )
        
        result = await LinkShareService.share_link_via_whatsapp(
            db=db,
            payload=payload,
            shared_by_user_id=current_user.id,
        )

        # Business failure → client error
        if not result["success"]:
            logger.warning(
                f"WhatsApp share failed: {result['error_message']}"
            )
            return {"error": result["error_message"]}

        logger.info("WhatsApp share request processed successfully")
        return api_success(result)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(
            "A system failure occurred in WhatsApp share",
            exc_info=True
        )
        return {"error": error_message}