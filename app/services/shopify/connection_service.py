"""Binding an API key to a Shopify shop, and resolving the shop for every call.

The shop for any Shopify request comes from the key's active connection, never
from the request body, so a key bound to one shop cannot act on another.

Rules:
  * one active connection per shop, one active shop per key
  * a shop connected to ANOTHER Rivollo account is refused (409); the merchant
    must disconnect it there first
  * re-connecting the same shop with a new key of the same account moves the
    binding to the new key; connecting a key to a different shop moves the key
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.shopify_repo import shopify_repository as repo
from app.models.shopify import ShopifyConnection
from app.services.api_key_service import ApiKeyPrincipal

logger = logging.getLogger(__name__)

NOT_CONNECTED = (
    "This API key is not connected to a Shopify store. "
    "Call POST /integrations/shopify/connect first."
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ShopifyContext:
    """Who is calling and for which shop."""

    principal: ApiKeyPrincipal
    connection: ShopifyConnection

    @property
    def user_id(self) -> uuid.UUID:
        return self.principal.user.id

    @property
    def shop_domain(self) -> str:
        return self.connection.shop_domain


class ShopifyConnectionService:
    @staticmethod
    def require_enabled() -> None:
        """404 while the integration is switched off, so its routes look absent."""
        if not settings.ENABLE_SHOPIFY_INTEGRATION:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    @staticmethod
    async def connect(
        db: AsyncSession, principal: ApiKeyPrincipal, shop_domain: str
    ) -> ShopifyConnection:
        user_id = principal.user.id
        key_id = principal.api_key.id
        now = _utcnow()

        for_shop = await repo.get_active_connection_by_shop(db, shop_domain, for_update=True)
        if for_shop is not None and for_shop.user_id != user_id:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This store is already connected to another Rivollo account. "
                    "Disconnect it there first."
                ),
            )
        if for_shop is not None and for_shop.api_key_id == key_id:
            await db.rollback()
            return for_shop  # already connected: idempotent

        for_key = await repo.get_active_connection_by_key(db, key_id, for_update=True)
        for stale in (for_shop, for_key):
            if stale is not None and stale.isactive:
                stale.isactive = False
                stale.disconnected_at = now
                stale.updated_by = user_id
                stale.updated_date = now
        if for_shop is not None or for_key is not None:
            # Release the partial unique indexes before inserting the new row.
            await repo.flush(db)

        connection = ShopifyConnection(
            id=uuid.uuid4(),
            user_id=user_id,
            api_key_id=key_id,
            shop_domain=shop_domain,
            isactive=True,
            connected_at=now,
            created_by=user_id,
            created_date=now,
        )
        repo.add(db, connection)
        await db.commit()
        logger.info("API key %s connected to shop %s", principal.api_key.key_prefix, shop_domain)
        return connection

    @staticmethod
    async def get_context(db: AsyncSession, principal: ApiKeyPrincipal) -> ShopifyContext:
        connection = await repo.get_active_connection_by_key(db, principal.api_key.id)
        if connection is None or connection.user_id != principal.user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NOT_CONNECTED)
        return ShopifyContext(principal=principal, connection=connection)

    @staticmethod
    async def disconnect(db: AsyncSession, principal: ApiKeyPrincipal) -> None:
        """Shopify app/uninstalled. Idempotent. Data is kept until shop/redact."""
        connection = await repo.get_active_connection_by_key(db, principal.api_key.id, for_update=True)
        if connection is None:
            await db.rollback()
            return
        now = _utcnow()
        connection.isactive = False
        connection.disconnected_at = now
        connection.updated_by = principal.user.id
        connection.updated_date = now
        await db.commit()
        logger.info("Shop %s disconnected", connection.shop_domain)

    @staticmethod
    async def redact_shop(db: AsyncSession, context: ShopifyContext) -> int:
        """Shopify shop/redact: delete every Shopify row for the shop, then disconnect.

        Rivollo products created for the shop are NOT deleted: they belong to the
        Rivollo account, not to Shopify (spec D3).
        """
        now = _utcnow()
        deleted = await repo.delete_products_for_shop(db, context.shop_domain, context.user_id)
        for connection in await repo.get_active_connections_for_shop(
            db, context.shop_domain, context.user_id
        ):
            connection.isactive = False
            connection.disconnected_at = now
            connection.updated_by = context.user_id
            connection.updated_date = now
        await db.commit()
        logger.info("Shop %s redacted: %d products removed", context.shop_domain, deleted)
        return deleted


shopify_connection_service = ShopifyConnectionService()
