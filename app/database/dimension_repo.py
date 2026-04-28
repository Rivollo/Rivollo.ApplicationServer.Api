"""Repository layer for dimension-related database operations.

"""

import uuid
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.models import (
    Hotspot,
    ProductDimensionGroup,
    ProductDimensions,
)


class DimensionRepository:
    """Repository for dimension database operations."""

    @staticmethod
    async def delete_existing_dimensions(
        db: AsyncSession, product_id: uuid.UUID
    ) -> None:
        """
        Delete all existing dimension data for a product.

        """
        await db.execute(
            delete(ProductDimensionGroup).where(
                ProductDimensionGroup.product_id == product_id
            )
        )
        await db.execute(
            delete(ProductDimensions).where(
                ProductDimensions.product_id == product_id
            )
        )
        await db.execute(
            delete(Hotspot).where(
                Hotspot.product_id == product_id,
                Hotspot.description.like("Dimension marker:%"),
            )
        )
        await db.flush()

    @staticmethod
    async def get_max_hotspot_order(
        db: AsyncSession, product_id: uuid.UUID
    ) -> int:
        """
        Get the current maximum order_index for hotspots in a product.


        Returns:
            Next available order_index (max + 1)
        """
        result = await db.execute(
            select(func.max(Hotspot.order_index)).where(
                Hotspot.product_id == product_id
            )
        )
        return (result.scalar() or -1) + 1

    @staticmethod
    async def create_dimension_group(
        db: AsyncSession,
        product_id: uuid.UUID,
        name: str,
        order_index: int,
        created_by: uuid.UUID,
    ) -> ProductDimensionGroup:
        """
        Create a new dimension group.

        """
        group = ProductDimensionGroup(
            product_id=product_id,
            name=name,
            order_index=order_index,
            created_by=created_by,
        )
        db.add(group)
        await db.flush()
        return group

    @staticmethod
    async def create_hotspot(
        db: AsyncSession,
        product_id: uuid.UUID,
        title: str,
        description: str,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        order_index: int,
        created_by: uuid.UUID,
    ) -> uuid.UUID:
        """
        Create a hotspot for a dimension marker.

        """
        hotspot = Hotspot(
            product_id=product_id,
            label=title,
            description=description,
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            order_index=order_index,
            created_by=created_by,
        )
        hotspot.set_position_to_geometry(pos_x, pos_y, pos_z)
        db.add(hotspot)
        await db.flush()
        return hotspot.id

    @staticmethod
    async def create_dimension(
        db: AsyncSession,
        product_id: uuid.UUID,
        dimension_group_id: uuid.UUID,
        dimension_name: str,
        value: float,
        unit: str,
        start_hotspot_id: uuid.UUID,
        end_hotspot_id: uuid.UUID,
        order_index: int,
        created_by: uuid.UUID,
    ) -> ProductDimensions:
        """
        Create a dimension record.

        """
        dimension = ProductDimensions(
            product_id=product_id,
            dimension_group_id=dimension_group_id,
            dimension_name=dimension_name,
            dimension_type=None,
            value=value,
            unit=unit,
            start_hotspot_id=start_hotspot_id,
            end_hotspot_id=end_hotspot_id,
            order_index=order_index,
            created_by=created_by,
        )
        db.add(dimension)
        return dimension

    @staticmethod
    async def get_dimension_groups(
        db: AsyncSession, product_id: uuid.UUID
    ) -> list[ProductDimensionGroup]:
        """
        Get all dimension groups for a product.

        """
        result = await db.execute(
            select(ProductDimensionGroup)
            .where(ProductDimensionGroup.product_id == product_id)
            .order_by(ProductDimensionGroup.order_index)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_dimensions_by_group(
        db: AsyncSession, group_id: uuid.UUID
    ) -> list[ProductDimensions]:
        """
        Get all dimensions for a dimension group.

        Returns:
            List of ProductDimensions instances
        """
        result = await db.execute(
            select(ProductDimensions)
            .where(ProductDimensions.dimension_group_id == group_id)
            .order_by(ProductDimensions.dimension_type, ProductDimensions.order_index)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_hotspot(
        db: AsyncSession, hotspot_id: uuid.UUID
    ) -> Optional[Hotspot]:
        """
        Get a hotspot by ID.
        """
        return await db.get(Hotspot, hotspot_id)

    @staticmethod
    async def get_dimensions_with_hotspots(
        db: AsyncSession, product_id: uuid.UUID
    ) -> list:
        """Fetch all dimension groups, dimensions, and both hotspots in one JOIN query.

        Returns rows of (ProductDimensionGroup, ProductDimensions, start_hotspot, end_hotspot).
        start_hotspot and end_hotspot are None when the foreign key is NULL.
        """
        start_hs = aliased(Hotspot, name="start_hotspot")
        end_hs = aliased(Hotspot, name="end_hotspot")

        result = await db.execute(
            select(ProductDimensionGroup, ProductDimensions, start_hs, end_hs)
            .join(ProductDimensions, ProductDimensions.dimension_group_id == ProductDimensionGroup.id)
            .outerjoin(start_hs, start_hs.id == ProductDimensions.start_hotspot_id)
            .outerjoin(end_hs, end_hs.id == ProductDimensions.end_hotspot_id)
            .where(ProductDimensionGroup.product_id == product_id)
            .order_by(ProductDimensionGroup.order_index, ProductDimensions.order_index)
        )
        return result.all()
