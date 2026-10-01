"""Data access for the Shopify integration. Statements only; never commits.

Every seller-side lookup is scoped by BOTH the shop and the Rivollo user in the
WHERE clause, so a key bound to one shop can never read another shop's rows,
and a shop row can never be reached by a user who does not own it.
"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.configurator import ModelVariantGeneration
from app.models.models import (
    Product,
    ProductAsset,
    ProductAssetMapping,
    PublishLink,
)
from app.models.shopify import ShopifyConnection, ShopifyLayout, ShopifyProduct

THUMBNAIL_ASSET_ID = 1
MESH_ASSET_ID = 9
USDZ_ASSET_ID = 11


class ShopifyRepository:
    # ------------------------------------------------------------------ #
    # Connections
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_active_connection_by_key(
        db: AsyncSession, api_key_id: uuid.UUID, *, for_update: bool = False
    ) -> Optional[ShopifyConnection]:
        stmt = select(ShopifyConnection).where(
            ShopifyConnection.api_key_id == api_key_id, ShopifyConnection.isactive.is_(True)
        )
        if for_update:
            stmt = stmt.with_for_update()
        return (await db.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def get_active_connection_by_shop(
        db: AsyncSession, shop_domain: str, *, for_update: bool = False
    ) -> Optional[ShopifyConnection]:
        stmt = select(ShopifyConnection).where(
            ShopifyConnection.shop_domain == shop_domain, ShopifyConnection.isactive.is_(True)
        )
        if for_update:
            stmt = stmt.with_for_update()
        return (await db.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def get_active_connections_for_shop(
        db: AsyncSession, shop_domain: str, user_id: uuid.UUID
    ) -> list[ShopifyConnection]:
        result = await db.execute(
            select(ShopifyConnection).where(
                ShopifyConnection.shop_domain == shop_domain,
                ShopifyConnection.user_id == user_id,
                ShopifyConnection.isactive.is_(True),
            )
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------ #
    # Shopify products
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_product(
        db: AsyncSession,
        shop_domain: str,
        user_id: uuid.UUID,
        shopify_product_id: int,
        *,
        for_update: bool = False,
    ) -> Optional[ShopifyProduct]:
        stmt = select(ShopifyProduct).where(
            ShopifyProduct.shop_domain == shop_domain,
            ShopifyProduct.user_id == user_id,
            ShopifyProduct.shopify_product_id == shopify_product_id,
        )
        if for_update:
            stmt = stmt.with_for_update(of=ShopifyProduct)
        return (await db.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def get_product_any_owner(
        db: AsyncSession, shop_domain: str, shopify_product_id: int
    ) -> Optional[ShopifyProduct]:
        """For the sync collision check only (the unique key has no user in it)."""
        result = await db.execute(
            select(ShopifyProduct).where(
                ShopifyProduct.shop_domain == shop_domain,
                ShopifyProduct.shopify_product_id == shopify_product_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_products(
        db: AsyncSession, shop_domain: str, user_id: uuid.UUID
    ) -> list[ShopifyProduct]:
        result = await db.execute(
            select(ShopifyProduct)
            .where(ShopifyProduct.shop_domain == shop_domain, ShopifyProduct.user_id == user_id)
            .order_by(ShopifyProduct.synced_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete_products_for_shop(
        db: AsyncSession, shop_domain: str, user_id: uuid.UUID
    ) -> int:
        """Delete every Shopify row for a shop (variants and layouts cascade)."""
        result = await db.execute(
            delete(ShopifyProduct).where(
                ShopifyProduct.shop_domain == shop_domain, ShopifyProduct.user_id == user_id
            )
        )
        return int(result.rowcount or 0)

    @staticmethod
    async def get_by_rivollo_product(
        db: AsyncSession, product_id: uuid.UUID
    ) -> Optional[ShopifyProduct]:
        """Public payload lookup. Newest link wins if a product was ever re-linked."""
        result = await db.execute(
            select(ShopifyProduct)
            .where(ShopifyProduct.rivollo_product_id == product_id)
            .order_by(ShopifyProduct.synced_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_linked_product(
        db: AsyncSession, product_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[ShopifyProduct]:
        """Portal lookup by Rivollo product, scoped to the user. Newest link wins."""
        result = await db.execute(
            select(ShopifyProduct)
            .where(
                ShopifyProduct.rivollo_product_id == product_id,
                ShopifyProduct.user_id == user_id,
            )
            .order_by(ShopifyProduct.synced_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_linked_product_ids(
        db: AsyncSession, product_ids: Sequence[uuid.UUID], user_id: uuid.UUID
    ) -> set[uuid.UUID]:
        """Which of these products are linked to a shop the user still has connected."""
        if not product_ids:
            return set()
        result = await db.execute(
            select(ShopifyProduct.rivollo_product_id)
            .join(
                ShopifyConnection,
                and_(
                    ShopifyConnection.shop_domain == ShopifyProduct.shop_domain,
                    ShopifyConnection.user_id == ShopifyProduct.user_id,
                    ShopifyConnection.isactive.is_(True),
                ),
            )
            .where(
                ShopifyProduct.rivollo_product_id.in_(list(product_ids)),
                ShopifyProduct.user_id == user_id,
            )
            .distinct()
        )
        return set(result.scalars().all())

    # ------------------------------------------------------------------ #
    # Rivollo product (read; ownership in the WHERE clause)
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_live_product(
        db: AsyncSession, product_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[Product]:
        result = await db.execute(
            select(Product).where(
                Product.id == product_id,
                Product.created_by == user_id,
                Product.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_published_product(db: AsyncSession, product_id: uuid.UUID) -> Optional[Product]:
        result = await db.execute(
            select(Product).where(Product.id == product_id, Product.deleted_at.is_(None))
        )
        product = result.scalar_one_or_none()
        if product is None or getattr(product.status, "value", product.status) != "published":
            return None
        return product

    @staticmethod
    async def get_mapped_asset(
        db: AsyncSession, product_id: uuid.UUID, asset_id: int
    ) -> Optional[ProductAsset]:
        """The newest active mapped asset of one format, as every reader resolves it."""
        result = await db.execute(
            select(ProductAsset)
            .join(ProductAssetMapping, ProductAsset.id == ProductAssetMapping.product_asset_id)
            .where(
                ProductAssetMapping.productid == product_id,
                ProductAsset.asset_id == asset_id,
                ProductAssetMapping.isactive.is_(True),
            )
            .order_by(ProductAssetMapping.created_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_enabled_public_id(db: AsyncSession, product_id: uuid.UUID) -> Optional[str]:
        result = await db.execute(
            select(PublishLink.public_id).where(
                PublishLink.product_id == product_id, PublishLink.is_enabled.is_(True)
            )
        )
        return result.scalars().first()

    @staticmethod
    async def get_layout_generations(
        db: AsyncSession, product_id: uuid.UUID, client_refs: Sequence[str]
    ) -> list[ModelVariantGeneration]:
        if not client_refs:
            return []
        result = await db.execute(
            select(ModelVariantGeneration)
            .where(
                ModelVariantGeneration.product_id == product_id,
                ModelVariantGeneration.client_ref.in_(list(client_refs)),
            )
            .order_by(ModelVariantGeneration.created_date.desc())
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------ #
    # Write — no commits
    # ------------------------------------------------------------------ #
    @staticmethod
    def add(db: AsyncSession, instance: object) -> None:
        db.add(instance)

    @staticmethod
    async def flush(db: AsyncSession) -> None:
        await db.flush()

    @staticmethod
    async def delete(db: AsyncSession, instance: object) -> None:
        await db.delete(instance)


shopify_repository = ShopifyRepository()


def layouts_by_id(product: ShopifyProduct) -> dict[uuid.UUID, ShopifyLayout]:
    return {layout.id: layout for layout in product.layouts}
