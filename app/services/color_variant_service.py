"""Color variant service — business logic for product colourways.

Owns the rules that keep a product's colour scheme coherent:

  * every product with a model has exactly one **Original** variant, created on
    demand and never deletable — this is the "always keep the real one" guarantee
  * exactly one variant is the default (what the viewer loads first)
  * slugs are unique per product
  * changing a variant's colours changes its config hash, which marks it stale
    and queues a re-bake

Baking itself lives in ``variant_bake_service``; this module only decides *when*
a bake is needed.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.color_variant_repo import color_variant_repository as repo
from app.models.models import ProductColorVariant
from app.schemas.color_variants import (
    ColorVariantCreate,
    ColorVariantResponse,
    ColorVariantUpdate,
    MaterialOverride,
    MaterialPart,
    VariantAssetResponse,
)
from app.services.variant_bake_service import variant_bake_service

ORIGINAL_NAME = "Original"
ORIGINAL_SLUG = "original"
# Sentinel hash for the original variant — it is never baked, it *is* the source.
ORIGINAL_HASH = "original"
MAX_VARIANTS_PER_PRODUCT = 24


class ColorVariantService:
    """Business logic for colourway operations."""

    # ---------- Read ----------

    @staticmethod
    async def get_materials(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> list[MaterialPart]:
        """The colourable parts of a product's model, with a suggested method each."""
        model_url = await ColorVariantService._get_model_url_or_400(db, product_id)
        parts = await variant_bake_service.inspect_model(model_url)
        return [MaterialPart(**p) for p in parts]

    @staticmethod
    async def list_variants(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[ColorVariantResponse]:
        await ColorVariantService._ensure_product_exists(db, product_id)
        await ColorVariantService._ensure_original_exists(db, product_id, user_id)
        variants = await repo.get_variants_for_product(db, product_id)
        original_url = await ColorVariantService._original_model_url(db, product_id)
        return [ColorVariantService._to_response(v, original_url) for v in variants]

    @staticmethod
    async def get_variant(
        db: AsyncSession,
        variant_id: uuid.UUID,
    ) -> ColorVariantResponse:
        variant = await ColorVariantService._get_variant_or_404(db, variant_id)
        original_url = await ColorVariantService._original_model_url(db, variant.product_id)
        return ColorVariantService._to_response(variant, original_url)

    # ---------- Create ----------

    @staticmethod
    async def create_variant(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        payload: ColorVariantCreate,
    ) -> ColorVariantResponse:
        await ColorVariantService._ensure_product_exists(db, product_id)
        model_url = await ColorVariantService._get_model_url_or_400(db, product_id)
        await ColorVariantService._ensure_original_exists(db, product_id, user_id)

        existing = await repo.get_variants_for_product(db, product_id)
        if len(existing) >= MAX_VARIANTS_PER_PRODUCT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A product can have at most {MAX_VARIANTS_PER_PRODUCT} colourways",
            )

        if not payload.overrides:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A colourway needs at least one coloured part",
            )

        overrides = [o.model_dump() for o in payload.overrides]
        # Freeze "auto" into a concrete method now, so the preview the seller
        # approved and the file we bake are produced by the same rule. The portal
        # normally sends concrete methods, in which case this is a no-op.
        overrides = await variant_bake_service.resolve_methods(model_url, overrides)

        slug = await ColorVariantService._unique_slug(db, product_id, payload.name)
        order_index = await repo.get_next_order_index(db, product_id)

        variant = ProductColorVariant(
            product_id=product_id,
            name=payload.name.strip(),
            slug=slug,
            swatch_hex=payload.swatch_hex or overrides[0]["color"],
            is_default=False,
            is_original=False,
            isactive=payload.isactive,
            order_index=order_index,
            overrides=overrides,
            config_hash=variant_bake_service.compute_config_hash(model_url, overrides),
            bake_status="pending",
            created_by=user_id,
        )

        if payload.is_default:
            await repo.clear_default(db, product_id)
            variant.is_default = True

        repo.add(db, variant)
        await db.commit()
        await db.refresh(variant)

        original_url = await ColorVariantService._original_model_url(db, product_id)
        return ColorVariantService._to_response(variant, original_url)

    # ---------- Update ----------

    @staticmethod
    async def update_variant(
        db: AsyncSession,
        variant_id: uuid.UUID,
        user_id: uuid.UUID,
        payload: ColorVariantUpdate,
    ) -> tuple[ColorVariantResponse, bool]:
        """Update a colourway. Returns (variant, needs_rebake)."""
        variant = await ColorVariantService._get_variant_or_404(db, variant_id)
        needs_rebake = False

        if payload.name is not None:
            if variant.is_original:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The original colourway cannot be renamed",
                )
            variant.name = payload.name.strip()
            variant.slug = await ColorVariantService._unique_slug(
                db, variant.product_id, payload.name, exclude_id=variant.id
            )

        if payload.overrides is not None:
            if variant.is_original:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The original colourway cannot be recoloured",
                )
            model_url = await ColorVariantService._get_model_url_or_400(
                db, variant.product_id
            )
            overrides = [o.model_dump() for o in payload.overrides]
            overrides = await variant_bake_service.resolve_methods(model_url, overrides)
            new_hash = variant_bake_service.compute_config_hash(model_url, overrides)

            # Only a real change to the output invalidates the baked file.
            if new_hash != variant.config_hash:
                variant.overrides = overrides
                variant.config_hash = new_hash
                variant.bake_status = "pending"
                variant.bake_error = None
                needs_rebake = True

        if payload.swatch_hex is not None:
            variant.swatch_hex = payload.swatch_hex

        if payload.isactive is not None:
            if variant.is_default and not payload.isactive:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The default colourway cannot be hidden. Set another as default first.",
                )
            variant.isactive = payload.isactive

        if payload.is_default is True:
            if not variant.isactive:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A hidden colourway cannot be the default",
                )
            await repo.clear_default(db, variant.product_id, keep_id=variant.id)
            variant.is_default = True

        variant.updated_by = user_id
        await db.commit()
        await db.refresh(variant)

        original_url = await ColorVariantService._original_model_url(db, variant.product_id)
        return ColorVariantService._to_response(variant, original_url), needs_rebake

    @staticmethod
    async def reorder_variants(
        db: AsyncSession,
        product_id: uuid.UUID,
        variant_ids: list[str],
    ) -> list[ColorVariantResponse]:
        await ColorVariantService._ensure_product_exists(db, product_id)
        known = {str(v.id) for v in await repo.get_variants_for_product(db, product_id)}

        for index, raw_id in enumerate(variant_ids):
            if raw_id not in known:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Variant {raw_id} does not belong to this product",
                )
            await repo.set_order(db, uuid.UUID(raw_id), index)

        await db.commit()
        variants = await repo.get_variants_for_product(db, product_id)
        original_url = await ColorVariantService._original_model_url(db, product_id)
        return [ColorVariantService._to_response(v, original_url) for v in variants]

    # ---------- Delete ----------

    @staticmethod
    async def delete_variant(
        db: AsyncSession,
        variant_id: uuid.UUID,
    ) -> None:
        variant = await ColorVariantService._get_variant_or_404(db, variant_id)

        if variant.is_original:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The original colourway cannot be deleted",
            )

        was_default = variant.is_default
        product_id = variant.product_id

        # Purge blobs before the rows go, so a failure here can't orphan files.
        await variant_bake_service.purge_variant_assets(variant)
        await repo.delete_variant(db, variant)
        await db.flush()

        # A product must always have a default; fall back to the original.
        if was_default:
            original = await repo.get_original_variant(db, product_id)
            if original is not None:
                original.is_default = True

        await db.commit()

    # ---------- Helpers ----------

    @staticmethod
    async def _ensure_product_exists(db: AsyncSession, product_id: uuid.UUID) -> None:
        if not await repo.get_product_by_id(db, product_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found",
            )

    @staticmethod
    async def _get_model_url_or_400(db: AsyncSession, product_id: uuid.UUID) -> str:
        model_url = await repo.get_product_model_url(db, product_id)
        if not model_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This product has no 3D model yet. Upload a model before adding colours.",
            )
        return model_url

    @staticmethod
    async def _get_variant_or_404(
        db: AsyncSession,
        variant_id: uuid.UUID,
    ) -> ProductColorVariant:
        variant = await repo.get_variant_by_id(db, variant_id)
        if variant is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Colourway not found",
            )
        return variant

    @staticmethod
    async def _ensure_original_exists(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ProductColorVariant:
        """Create the Original colourway on first use.

        Products created before this feature existed have no variants, so the
        first read or write lazily backfills one. It carries no overrides and is
        never baked — it points straight at the untouched product model.
        """
        original = await repo.get_original_variant(db, product_id)
        if original is not None:
            return original

        has_default = any(
            v.is_default for v in await repo.get_variants_for_product(db, product_id)
        )
        original = ProductColorVariant(
            product_id=product_id,
            name=ORIGINAL_NAME,
            slug=ORIGINAL_SLUG,
            swatch_hex="#FFFFFF",
            is_default=not has_default,
            is_original=True,
            isactive=True,
            order_index=0,
            overrides=[],
            config_hash=ORIGINAL_HASH,
            bake_status="ready",
            created_by=user_id,
        )
        repo.add(db, original)
        await db.commit()
        await db.refresh(original)
        return original

    @staticmethod
    async def _original_model_url(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> Optional[str]:
        return await repo.get_product_model_url(db, product_id)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
        return slug or "colour"

    @staticmethod
    async def _unique_slug(
        db: AsyncSession,
        product_id: uuid.UUID,
        name: str,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> str:
        base = ColorVariantService._slugify(name)
        # "original" is reserved for the untouched model.
        if base == ORIGINAL_SLUG and exclude_id is None:
            base = "original-colour"

        slug = base
        suffix = 2
        while await repo.slug_exists(db, product_id, slug, exclude_id=exclude_id):
            slug = f"{base}-{suffix}"
            suffix += 1
        return slug

    @staticmethod
    def _to_response(
        variant: ProductColorVariant,
        original_model_url: Optional[str],
    ) -> ColorVariantResponse:
        assets = [
            VariantAssetResponse(
                format=a.format, url=a.url, size_bytes=a.size_bytes
            )
            for a in (variant.assets or [])
            # A file whose hash no longer matches is stale — hide it rather than
            # serve the wrong colour while the re-bake is in flight.
            if a.config_hash == variant.config_hash
        ]

        if variant.is_original:
            model_url = original_model_url
        else:
            model_url = next((a.url for a in assets if a.format == "glb"), None)

        return ColorVariantResponse(
            id=str(variant.id),
            product_id=str(variant.product_id),
            name=variant.name,
            slug=variant.slug,
            swatch_hex=variant.swatch_hex,
            is_default=variant.is_default,
            is_original=variant.is_original,
            isactive=variant.isactive,
            order_index=variant.order_index,
            overrides=[MaterialOverride(**o) for o in (variant.overrides or [])],
            bake_status=variant.bake_status,
            bake_error=variant.bake_error,
            baked_at=variant.baked_at,
            model_url=model_url,
            assets=assets,
            created_at=variant.created_date,
        )


color_variant_service = ColorVariantService()
