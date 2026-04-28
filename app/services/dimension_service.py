"""Service layer for dimension business logic.

This module contains ALL business rules and logic for dimensions.
It orchestrates repository calls and transforms data into API-ready formats.

Architecture:
- Route: Handles HTTP requests/responses, calls service
- Service: Contains business logic, orchestrates repository calls
- Repository: Contains database queries only
"""

import uuid
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.dimension_repo import DimensionRepository
from app.models.models import Product
from app.schemas.dimensions import DimensionItem


class DimensionService:
    """Service for dimension business logic."""

    @staticmethod
    async def save_product_dimensions(
        db: AsyncSession,
        product_id: uuid.UUID,
        dimensions: list[DimensionItem],
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Save product dimensions using a list-based input.

        This method:
        1. Deletes existing dimension data (replace mode)
        2. Creates a dimension group
        3. Creates hotspots for each dimension's start/end points
        4. Creates dimension records linking to the hotspots

        Args:
            db: Database session
            product_id: ID of the product
            dimensions: List of dimension items with hotspots
            user_id: ID of the user performing the action

        Returns:
            Dictionary with product_id and success message

        Raises:
            ValueError: If product not found or dimension has wrong number of hotspots
        """
        # Validate product exists
        product = await db.get(Product, product_id)
        if not product:
            raise ValueError("Product not found")

        # Delete existing dimension data (replace mode)
        await DimensionRepository.delete_existing_dimensions(db, product_id)

        # Get current max order index for hotspots
        current_order = await DimensionRepository.get_max_hotspot_order(db, product_id)

        # Create dimension group (single group for all dimensions)
        group = await DimensionRepository.create_dimension_group(
            db=db,
            product_id=product_id,
            name="Product Measurements",
            order_index=0,
            created_by=user_id,
        )

        # Process each dimension
        for dim in dimensions:
            # Validate hotspots count
            if len(dim.hotspots) != 2:
                raise ValueError(
                    f"Dimension '{dim.name}' must have exactly 2 hotspots"
                )

            # Find start and end hotspots
            start = next((h for h in dim.hotspots if h.type == "start"), None)
            end = next((h for h in dim.hotspots if h.type == "end"), None)

            if not start or not end:
                raise ValueError(
                    f"Dimension '{dim.name}' must have both 'start' and 'end' hotspots"
                )

            # Create start hotspot
            start_id = await DimensionRepository.create_hotspot(
                db=db,
                product_id=product_id,
                title=start.title,
                description=f"Dimension marker: {start.title}",
                pos_x=start.position.x,
                pos_y=start.position.y,
                pos_z=start.position.z,
                order_index=current_order,
                created_by=user_id,
            )

            # Create end hotspot
            end_id = await DimensionRepository.create_hotspot(
                db=db,
                product_id=product_id,
                title=end.title,
                description=f"Dimension marker: {end.title}",
                pos_x=end.position.x,
                pos_y=end.position.y,
                pos_z=end.position.z,
                order_index=current_order + 1,
                created_by=user_id,
            )

            # Create dimension record
            await DimensionRepository.create_dimension(
                db=db,
                product_id=product_id,
                dimension_group_id=group.id,
                dimension_name=dim.name,
                value=dim.value,
                unit=dim.unit or "cm",
                start_hotspot_id=start_id,
                end_hotspot_id=end_id,
                order_index=current_order,
                created_by=user_id,
            )

            current_order += 2

        return {
            "product_id": str(product_id),
            "message": "Dimensions saved successfully",
        }

    @staticmethod
    async def get_product_dimensions(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> Optional[dict[str, Any]]:
        """
        Get dimension data for a product in the API response format.

        This method fetches dimension groups and dimensions, then transforms
        them into the format expected by the product assets response.

        Args:
            db: Database session
            product_id: ID of the product

        Returns:
            Dictionary with dimensions data if found, None otherwise.
            Format: {"dimensions": {dimension_type: {value, unit, hotspots}, ...}}
        """
        # Single JOIN query: groups + dimensions + both hotspots
        rows = await DimensionRepository.get_dimensions_with_hotspots(db, product_id)

        if not rows:
            return None

        # Take only the first group (rows are ordered by group.order_index)
        first_group = rows[0][0]
        group_rows = [r for r in rows if r[0].id == first_group.id]

        dimensions_dict: dict[str, Any] = {}
        seen_types: set[str] = set()

        for group, dim, start_hotspot, end_hotspot in group_rows:
            dim_type = (
                dim.dimension_type.lower()
                if dim.dimension_type
                else dim.dimension_name.lower()
                if dim.dimension_name
                else "unknown"
            )

            if dim_type in seen_types:
                continue
            seen_types.add(dim_type)

            dim_hotspots = []
            if start_hotspot:
                dim_hotspots.append({
                    "id": str(start_hotspot.id),
                    "title": start_hotspot.label,
                    "position": {
                        "x": start_hotspot.pos_x,
                        "y": start_hotspot.pos_y,
                        "z": start_hotspot.pos_z,
                    },
                })
            if end_hotspot:
                dim_hotspots.append({
                    "id": str(end_hotspot.id),
                    "title": end_hotspot.label,
                    "position": {
                        "x": end_hotspot.pos_x,
                        "y": end_hotspot.pos_y,
                        "z": end_hotspot.pos_z,
                    },
                })

            dimensions_dict[dim_type] = {
                "value": float(dim.value),
                "unit": dim.unit or "cm",
                "hotspots": dim_hotspots,
            }

        if dimensions_dict:
            dimensions_dict["dimension_name"] = first_group.name
            return {"dimensions": dimensions_dict}

        return None

    @staticmethod
    async def get_dimensions_list(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """
        Get dimensions for a product in list-based format.

        This method returns dimensions in the same format as the POST payload,
        with hotspots including start/end type indicators.

        Args:
            db: Database session
            product_id: ID of the product

        Returns:
            List of dimensions with hotspots including type field
        """
        # Validate product exists
        product = await db.get(Product, product_id)
        if not product:
            raise ValueError("Product not found")

        # Fetch dimension groups
        groups = await DimensionRepository.get_dimension_groups(db, product_id)

        if not groups:
            return []

        result: list[dict[str, Any]] = []

        for group in groups:
            dimensions = await DimensionRepository.get_dimensions_by_group(db, group.id)

            for dim in dimensions:
                dim_hotspots: list[dict[str, Any]] = []

                # Add start hotspot
                if dim.start_hotspot_id:
                    start_hotspot = await DimensionRepository.get_hotspot(
                        db, dim.start_hotspot_id
                    )
                    if start_hotspot:
                        dim_hotspots.append({
                            "id": str(start_hotspot.id),
                            "type": "start",
                            "title": start_hotspot.label,
                            "position": {
                                "x": start_hotspot.pos_x,
                                "y": start_hotspot.pos_y,
                                "z": start_hotspot.pos_z,
                            },
                        })

                # Add end hotspot
                if dim.end_hotspot_id:
                    end_hotspot = await DimensionRepository.get_hotspot(
                        db, dim.end_hotspot_id
                    )
                    if end_hotspot:
                        dim_hotspots.append({
                            "id": str(end_hotspot.id),
                            "type": "end",
                            "title": end_hotspot.label,
                            "position": {
                                "x": end_hotspot.pos_x,
                                "y": end_hotspot.pos_y,
                                "z": end_hotspot.pos_z,
                            },
                        })

                result.append({
                    "name": dim.dimension_name or dim.dimension_type or "unknown",
                    "value": float(dim.value),
                    "unit": dim.unit or "cm",
                    "hotspots": dim_hotspots,
                })

        return result

    @staticmethod
    async def delete_dimensions(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Delete all dimensions for a product.

        This deletes:
        - Product dimensions
        - Dimension groups
        - Dimension-created hotspots only (identified by 'Dimension marker:' description)

        Normal hotspots are preserved.

        Args:
            db: Database session
            product_id: ID of the product

        Returns:
            Dictionary with product_id and success message

        Raises:
            ValueError: If product not found
        """
        # Validate product exists
        product = await db.get(Product, product_id)
        if not product:
            raise ValueError("Product not found")

        # Delete dimension data (reuse existing repo method)
        await DimensionRepository.delete_existing_dimensions(db, product_id)

        return {
            "product_id": str(product_id),
            "message": "Dimensions deleted successfully",
        }
