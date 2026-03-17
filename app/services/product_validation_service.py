# 



# app/services/product_validation_service.py

import uuid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Product


class ProductValidationService:

    @staticmethod
    async def is_product_name_duplicate(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_name: str,
        exclude_product_id: uuid.UUID | None = None,
    ) -> bool:
        """
        Returns True if the given user already has a product with the same name
        (case-insensitive). Pass exclude_product_id when validating on UPDATE
        so the product being edited is not matched against itself.
        """
        query = select(func.count()).select_from(Product).where(
            Product.created_by == user_id,
            func.lower(Product.name) == func.lower(product_name.strip()),
        )

        if exclude_product_id is not None:
            query = query.where(Product.id != exclude_product_id)

        result = await db.execute(query)
        return (result.scalar() or 0) > 0

    @staticmethod
    async def validate_product_name(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_name: str,
        exclude_product_id: uuid.UUID | None = None,
    ) -> None:
        """
        Raises ValueError if the product name is already taken by this user.
        Call this before INSERT or UPDATE. Catches both CREATE and UPDATE flows
        via the optional exclude_product_id.
        """
        if not product_name or not product_name.strip():
            raise ValueError("Product name must not be empty.")

        is_duplicate = await ProductValidationService.is_product_name_duplicate(
            db=db,
            user_id=user_id,
            product_name=product_name,
            exclude_product_id=exclude_product_id,
        )

        if is_duplicate:
            raise ValueError(
                f"A product named '{product_name.strip()}' already exists for this user. "
                "Please choose a different name."
            )