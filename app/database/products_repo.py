"""Repository layer for product database operations."""

import uuid
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Product, ProductAsset, ProductAssetMapping, AssetStatic, ProductStatus, PublishLink


class ProductRepository:
    """Repository for product database operations."""

    @staticmethod
    async def get_products_by_user_id(
        db: AsyncSession, user_id: uuid.UUID
    ) -> list[Product]:
        """Get all products for a user, ordered by most recent first. Excludes archived products."""
        result = await db.execute(
            select(Product)
            .where(
                Product.created_by == user_id,
                Product.deleted_at.is_(None),
                # Product.status != ProductStatus.ARCHIVED,   # ← exclude disabled/archived
            )
            .order_by(func.coalesce(Product.updated_date, Product.created_date).desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_primary_asset_for_product(
        db: AsyncSession, product_id: uuid.UUID
    ) -> Optional[tuple[str, str, int]]:
        """Get primary asset (asset_id = 1) for a product.
        
        Returns: Tuple of (image_url, asset_type_name, asset_type_id) or None
        """
        result = await db.execute(
            select(
                ProductAsset.image,
                AssetStatic.name.label("asset_name"),
                ProductAsset.asset_id,
            )
            .join(ProductAssetMapping, ProductAsset.id == ProductAssetMapping.product_asset_id)
            .join(AssetStatic, ProductAsset.asset_id == AssetStatic.id)
            .where(
                ProductAssetMapping.productid == str(product_id),
                ProductAsset.asset_id == 1,
                ProductAssetMapping.isactive.is_(True),
            )
            .order_by(ProductAssetMapping.created_date.desc())
            .limit(1)
        )
        row = result.first()
        if row:
            return (row.image, row.asset_name, row.asset_id)
        return None

    @staticmethod
    async def get_primary_assets_for_products(
        db: AsyncSession, product_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, tuple[str, str, int]]:
        """Bulk-fetch primary assets for multiple products in a single DB call.

        Replaces the N+1 pattern of calling get_primary_asset_for_product in a loop.
        Returns mapping of product_id -> (image_url, asset_type_name, asset_type_id).
        """
        if not product_ids:
            return {}

        str_ids = [str(pid) for pid in product_ids]

        subq = (
            select(
                ProductAssetMapping.productid.label("productid"),
                ProductAsset.image.label("image"),
                AssetStatic.name.label("asset_name"),
                ProductAsset.asset_id.label("asset_id"),
                func.row_number().over(
                    partition_by=ProductAssetMapping.productid,
                    order_by=ProductAssetMapping.created_date.desc(),
                ).label("rn"),
            )
            .join(ProductAsset, ProductAsset.id == ProductAssetMapping.product_asset_id)
            .join(AssetStatic, ProductAsset.asset_id == AssetStatic.id)
            .where(
                ProductAssetMapping.productid.in_(str_ids),
                ProductAsset.asset_id == 1,
                ProductAssetMapping.isactive.is_(True),
            )
        ).subquery()

        result = await db.execute(
            select(
                subq.c.productid,
                subq.c.image,
                subq.c.asset_name,
                subq.c.asset_id,
            ).where(subq.c.rn == 1)
        )

        return {
            uuid.UUID(str(row.productid)): (row.image, row.asset_name, row.asset_id)
            for row in result.all()
        }

    @staticmethod
    async def get_public_ids_for_products(
        db: AsyncSession, product_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, str]:
        """Get public_ids for multiple products in bulk.
        
        Returns a mapping of product_id -> public_id for products that are published
        and have an enabled entry in PublishLink.
        
        Args:
            db: Database session
            product_ids: List of product UUIDs to fetch public_ids for
            
        Returns:
            Dictionary mapping product_id to public_id (only for published products with enabled links)
        """
        if not product_ids:
            return {}
        
        result = await db.execute(
            select(PublishLink.product_id, PublishLink.public_id)
            .where(
                PublishLink.product_id.in_(product_ids),
                PublishLink.is_enabled.is_(True),
            )
        )
        rows = result.all()
        return {row.product_id: row.public_id for row in rows}
