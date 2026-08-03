"""Color variant repository — database access for colourways and baked assets."""

import uuid
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    AssetStatic,
    Product,
    ProductAsset,
    ProductAssetMapping,
    ProductColorVariant,
    VariantAsset,
)

# A product's uploaded GLB is stored in tbl_product_assets with asset_id 9.
# This mirrors `ProductCreate.mesh_asset_id` (default 9) on the write side and
# `products_repo.get_primary_asset_for_product` (asset_id 1 = image) on the read
# side — it is the id the pipeline actually writes meshes under.
#
# Note: the legacy `tbl_products.model_asset_id -> tbl_assets` relation still
# exists in the SQLAlchemy models but NOT in the live database, so it must not
# be used for lookups.
MESH_ASSET_ID = 9


class ColorVariantRepository:
    """Database access layer for the color configurator."""

    # ---------- Read ----------

    @staticmethod
    async def get_product_by_id(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> Optional[Product]:
        return await db.get(Product, product_id)

    @staticmethod
    async def get_product_model_url(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> Optional[str]:
        """CDN URL of the product's source GLB — what every colourway is baked from.

        Resolved through tbl_product_assets / tbl_product_asset_mapping exactly
        as the rest of the app does. Returns the most recently mapped mesh, so a
        re-uploaded model wins over the one it replaced.
        """
        result = await db.execute(
            select(ProductAsset.image)
            .join(
                ProductAssetMapping,
                ProductAsset.id == ProductAssetMapping.product_asset_id,
            )
            .join(AssetStatic, ProductAsset.asset_id == AssetStatic.id)
            .where(
                ProductAssetMapping.productid == str(product_id),
                ProductAsset.asset_id == MESH_ASSET_ID,
                ProductAssetMapping.isactive.is_(True),
            )
            .order_by(ProductAssetMapping.created_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_variant_by_id(
        db: AsyncSession,
        variant_id: uuid.UUID,
    ) -> Optional[ProductColorVariant]:
        result = await db.execute(
            select(ProductColorVariant)
            .options(selectinload(ProductColorVariant.assets))
            .where(ProductColorVariant.id == variant_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_variants_for_product(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> list[ProductColorVariant]:
        result = await db.execute(
            select(ProductColorVariant)
            .options(selectinload(ProductColorVariant.assets))
            .where(ProductColorVariant.product_id == product_id)
            .order_by(ProductColorVariant.order_index.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_original_variant(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> Optional[ProductColorVariant]:
        result = await db.execute(
            select(ProductColorVariant).where(
                ProductColorVariant.product_id == product_id,
                ProductColorVariant.is_original.is_(True),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def slug_exists(
        db: AsyncSession,
        product_id: uuid.UUID,
        slug: str,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> bool:
        stmt = select(ProductColorVariant.id).where(
            ProductColorVariant.product_id == product_id,
            ProductColorVariant.slug == slug,
        )
        if exclude_id is not None:
            stmt = stmt.where(ProductColorVariant.id != exclude_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none() is not None

    @staticmethod
    async def get_next_order_index(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> int:
        result = await db.execute(
            select(func.max(ProductColorVariant.order_index))
            .where(ProductColorVariant.product_id == product_id)
        )
        max_order = result.scalar()
        return 0 if max_order is None else max_order + 1

    @staticmethod
    async def get_asset_for_format(
        db: AsyncSession,
        variant_id: uuid.UUID,
        fmt: str,
    ) -> Optional[VariantAsset]:
        result = await db.execute(
            select(VariantAsset).where(
                VariantAsset.variant_id == variant_id,
                VariantAsset.format == fmt,
            )
        )
        return result.scalar_one_or_none()

    # ---------- Write ----------

    @staticmethod
    def add(db: AsyncSession, entity: ProductColorVariant | VariantAsset) -> None:
        db.add(entity)

    @staticmethod
    async def delete_variant(
        db: AsyncSession,
        variant: ProductColorVariant,
    ) -> None:
        await db.delete(variant)

    @staticmethod
    async def clear_default(
        db: AsyncSession,
        product_id: uuid.UUID,
        keep_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Unset is_default on every other variant of the product.

        A partial unique index enforces one default per product, so this must
        run *before* the new default is flushed.
        """
        stmt = (
            update(ProductColorVariant)
            .where(
                ProductColorVariant.product_id == product_id,
                ProductColorVariant.is_default.is_(True),
            )
            .values(is_default=False)
        )
        if keep_id is not None:
            stmt = stmt.where(ProductColorVariant.id != keep_id)
        await db.execute(stmt)

    @staticmethod
    async def set_order(
        db: AsyncSession,
        variant_id: uuid.UUID,
        order_index: int,
    ) -> None:
        await db.execute(
            update(ProductColorVariant)
            .where(ProductColorVariant.id == variant_id)
            .values(order_index=order_index)
        )


color_variant_repository = ColorVariantRepository()
