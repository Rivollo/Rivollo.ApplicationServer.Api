import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.whatsapp import SendWhatsAppLinkRequest
from app.database.link_share_repository import LinkShareRepository
from app.integrations.whatsapp_client import WhatsAppClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class LinkShareService:
    """Service for handling link sharing via WhatsApp."""

    @staticmethod
    async def share_link_via_whatsapp(
        db: AsyncSession,
        payload: SendWhatsAppLinkRequest,
        *,
        product_id: uuid.UUID | None = None,
        shared_by_user_id: uuid.UUID | None = None,
    ) -> dict:

      
        # Build WhatsApp payload
      
        whatsapp_payload = {
            "messaging_product": "whatsapp",
            "to": payload.phone_number,
            "type": "template",
            "template": {
                "name": settings.WHATSAPP_TEMPLATE_NAME,
                "language": {
                    "code": settings.WHATSAPP_TEMPLATE_LANGUAGE
                },
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": payload.product_name},
                            {"type": "text", "text": payload.company_name},
                            {"type": "text", "text": payload.product_link},
                        ],
                    }
                ],
            },
        }

 
        # TRY — External API call 

        try:
            response = await WhatsAppClient.send_template_message(
                whatsapp_payload
            )

        except Exception as api_error:
            logger.exception("WhatsApp API call crashed")

            await LinkShareService._safe_log(
                db,
                payload,
                product_id,
                shared_by_user_id,
                status="failed",
                message_id=None,
                error=str(api_error),
            )

            return {
                "success": False,
                "error_message": "WhatsApp send failed",
            }

      
        # Handle API failure response 
      
        if response["status_code"] >= 400:

            await LinkShareService._safe_log(
                db,
                payload,
                product_id,
                shared_by_user_id,
                status="failed",
                message_id=None,
                error=response.get("text"),
            )

            return {
                "success": False,
                "error_message": response.get("text"),
            }

    
        # Safe message id extraction
      
        message_id = None
        try:
            if response["json"] and "messages" in response["json"]:
                message_id = response["json"]["messages"][0].get("id")
        except Exception:
            logger.warning("Message ID missing in WhatsApp response")

      
        # Log success safely
       
        await LinkShareService._safe_log(
            db,
            payload,
            product_id,
            shared_by_user_id,
            status="sent",
            message_id=message_id,
            error=None,
        )

        return {
            "success": True,
            "message_id": message_id,
        }

    # SAFE LOG HELPER
   
    @staticmethod
    async def _safe_log(
        db,
        payload,
        product_id,
        shared_by_user_id,
        status,
        message_id,
        error,
    ):
        try:
            await LinkShareRepository.create_link_share_log(
                db,
                product_id=product_id,
                shared_by_user_id=shared_by_user_id,
                link=payload.product_link,
                channel="whatsapp",
                destination=payload.phone_number,
                template_variant=payload.template_varient.value,
                external_message_id=message_id,
                status=status,
                error_message=error,
            )
        except Exception:
            logger.exception("DB log write failed")
