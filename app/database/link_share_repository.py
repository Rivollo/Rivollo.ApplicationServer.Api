import uuid
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.link_share_log import LinkShareLog

logger = logging.getLogger(__name__)


class LinkShareRepository:
    """Repository for managing link share logs in the database."""

    @staticmethod
    async def create_link_share_log(
        db: AsyncSession,
        *,
        product_id: Optional[uuid.UUID],
        shared_by_user_id: Optional[uuid.UUID],
        link: str,
        channel: str,
        destination: str,
        template_variant: str,
        external_message_id: Optional[str],
        status: str,
        error_message: Optional[str],
    ) -> LinkShareLog:

        log_entry = LinkShareLog(
            product_id=product_id,
            shared_by_user_id=shared_by_user_id,
            link=link,
            channel=channel,
            destination=destination,
            template_variant=template_variant,
            external_message_id=external_message_id,
            status=status,
            error_message=error_message,
        )

        db.add(log_entry)

        try:
            await db.commit()
        except Exception:
            await db.rollback()   
            logger.exception("DB commit failed for link_share_log")
            raise                 

        await db.refresh(log_entry)

        return log_entry
