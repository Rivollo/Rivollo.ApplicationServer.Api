"""Create the main GLB and the layout GLBs for a synced Shopify product.

MAIN GLB — the Rivollo product's own model (the "Original" layout). Runs the
existing fal product pipeline, unchanged, on the draft product sync created:
ProductService._run_fal_3d_generation_background moves it
draft -> queue -> processing -> ready (WebSocket, Draco, notification, USDZ),
exactly as /createProductFal does, and back to draft on failure.

LAYOUT GLB — every other layout value. A photo generation (ADR-015) tagged
``client_ref = "shopify-layout:<layout id>"``: candidate -> preview -> accept ->
model variant. Needs only the product row, so it can run in parallel with the
main GLB.

Both take the image from this product's synced Shopify images only, and copy it
into Rivollo storage first (image_importer): fal is never given a Shopify URL,
and a merchant editing their gallery cannot break a Rivollo product.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.shopify_repo import (
    MESH_ASSET_ID,
    THUMBNAIL_ASSET_ID,
    shopify_repository as repo,
)
from app.models.configurator import ModelVariantGeneration
from app.models.models import Product, ProductStatus
from app.models.shopify import ShopifyLayout, ShopifyProduct, layout_client_ref
from app.services.configurator.model_variant_generation_service import (
    AcceptedGeneration,
    ModelVariantGenerationService,
    RequestedGeneration,
)
from app.services.generation_estimate_service import generation_estimate_service
from app.services.generation_gate import authorize_generation, charge_generation
from app.services.product_service import ProductService
from app.services.shopify.connection_service import ShopifyContext
from app.services.shopify.image_importer import image_importer
from app.services.shopify.sync_service import ShopifySyncService, require_allowed_image

logger = logging.getLogger(__name__)

PRODUCT_DELETED = "The linked Rivollo product was deleted. Sync the product again to recreate it."
LAYOUT_NOT_FOUND = "Layout not found."
GENERATION_NOT_FOUND = "Generation not found."

GENERATING_STATUSES = (ProductStatus.QUEUE.value, ProductStatus.PROCESSING.value)
DONE_STATUSES = (ProductStatus.READY.value, ProductStatus.PUBLISHED.value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def status_value(product: Product) -> str:
    return getattr(product.status, "value", product.status)


def main_glb_stalled(shopify_product: ShopifyProduct, product: Product, now: Optional[datetime] = None) -> bool:
    """Generating for longer than any real generation takes: its replica died."""
    if status_value(product) not in GENERATING_STATUSES:
        return False
    requested = shopify_product.main_glb_requested_at
    if requested is None:
        return False
    if requested.tzinfo is None:
        requested = requested.replace(tzinfo=timezone.utc)
    cutoff = (now or _utcnow()) - timedelta(seconds=settings.GENERATION_STALE_AFTER_SECONDS)
    return requested < cutoff


@dataclass(frozen=True)
class MainGlbStarted:
    product: Product
    estimate: Optional[dict[str, Any]]


class ShopifyGlbService:
    # ------------------------------------------------------------------ #
    # Main GLB
    # ------------------------------------------------------------------ #
    @staticmethod
    async def start_main_glb(
        db: AsyncSession,
        context: ShopifyContext,
        shopify_product_id: int,
        *,
        image_url: str,
        model_key: Optional[str],
        retry: bool,
        background_tasks: BackgroundTasks,
    ) -> MainGlbStarted:
        user_id = context.user_id
        shopify_product = await ShopifySyncService.get_product(db, context, shopify_product_id)
        product = await ShopifyGlbService._live_product(db, shopify_product, user_id)
        require_allowed_image(shopify_product, image_url)

        current = status_value(product)
        if current in GENERATING_STATUSES:
            if not (retry and main_glb_stalled(shopify_product, product)):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The 3D model is already being generated.",
                )
        elif current in DONE_STATUSES or await repo.get_mapped_asset(db, product.id, MESH_ASSET_ID):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This product already has a 3D model.",
            )

        rivollo_image = await image_importer.copy_to_uploads(user_id, image_url)
        spec = await authorize_generation(db, user_id, model_key)

        now = _utcnow()
        if current in GENERATING_STATUSES:
            product.status = ProductStatus.DRAFT  # the stalled attempt is abandoned
        if await repo.get_mapped_asset(db, product.id, THUMBNAIL_ASSET_ID) is None:
            await ShopifySyncService.add_thumbnail(db, product, rivollo_image, user_id)
        shopify_product.main_glb_requested_at = now
        await db.commit()

        await charge_generation(db, user_id, spec.credit_cost)
        # The product pipeline, unchanged. It opens its own sessions.
        background_tasks.add_task(
            ProductService._run_fal_3d_generation_background,
            user_id=user_id,
            product_id=product.id,
            mesh_asset_id=MESH_ASSET_ID,
            name=product.name,
            blob_url=rivollo_image,
            spec=spec,
        )
        logger.info(
            "Main GLB requested for Shopify product %s -> Rivollo product %s (model=%s)",
            shopify_product_id, product.id, spec.key,
        )

        estimate = None
        try:
            estimate = (
                await generation_estimate_service.estimate(db, spec.key, spec.baseline_estimate_seconds)
            ).to_payload()
        except Exception:  # noqa: BLE001
            logger.warning("Could not estimate main GLB generation", exc_info=True)
        return MainGlbStarted(product=product, estimate=estimate)

    # ------------------------------------------------------------------ #
    # Layout GLB
    # ------------------------------------------------------------------ #
    @staticmethod
    async def start_layout_glb(
        db: AsyncSession,
        context: ShopifyContext,
        shopify_product_id: int,
        layout_id: uuid.UUID,
        *,
        image_url: str,
        model_key: Optional[str],
        auto_accept: bool,
    ) -> RequestedGeneration:
        user_id = context.user_id
        shopify_product = await ShopifySyncService.get_product(db, context, shopify_product_id)
        layout = ShopifyGlbService._layout(shopify_product, layout_id)
        if layout.is_original:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The original layout uses the product's main 3D model. Use POST .../glb instead.",
            )
        product = await ShopifyGlbService._live_product(db, shopify_product, user_id)
        require_allowed_image(shopify_product, image_url)
        if auto_accept and await ShopifyGlbService.layout_has_model(db, product.id, layout, user_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This layout already has a 3D model.",
            )

        rivollo_image = await image_importer.copy_to_uploads(user_id, image_url)
        return await ModelVariantGenerationService.request(
            db,
            product.id,
            user_id,
            name=layout.option_value,
            image_url=rivollo_image,
            model_key=model_key,
            client_ref=layout_client_ref(layout.id),
            auto_accept=auto_accept,
        )

    # ------------------------------------------------------------------ #
    # Accept / discard a layout candidate
    # ------------------------------------------------------------------ #
    @staticmethod
    async def accept(
        db: AsyncSession, context: ShopifyContext, shopify_product_id: int, generation_id: uuid.UUID
    ) -> AcceptedGeneration:
        shopify_product, layout, generation = await ShopifyGlbService._layout_generation(
            db, context, shopify_product_id, generation_id
        )
        if generation.status != "accepted" and await ShopifyGlbService.layout_has_model(
            db, shopify_product.rivollo_product_id, layout, context.user_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This layout already has a 3D model. Replacing it is not supported yet.",
            )
        return await ModelVariantGenerationService.accept(db, generation_id, context.user_id)

    @staticmethod
    async def discard(
        db: AsyncSession, context: ShopifyContext, shopify_product_id: int, generation_id: uuid.UUID
    ) -> ModelVariantGeneration:
        await ShopifyGlbService._layout_generation(db, context, shopify_product_id, generation_id)
        return await ModelVariantGenerationService.discard(db, generation_id, context.user_id)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _live_product(db: AsyncSession, shopify_product: ShopifyProduct, user_id: uuid.UUID) -> Product:
        product = await repo.get_live_product(db, shopify_product.rivollo_product_id, user_id)
        if product is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=PRODUCT_DELETED)
        return product

    @staticmethod
    def _layout(shopify_product: ShopifyProduct, layout_id: uuid.UUID) -> ShopifyLayout:
        layout = next((l for l in shopify_product.layouts if l.id == layout_id), None)
        if layout is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=LAYOUT_NOT_FOUND)
        return layout

    @staticmethod
    async def _layout_generation(
        db: AsyncSession, context: ShopifyContext, shopify_product_id: int, generation_id: uuid.UUID
    ) -> tuple[ShopifyProduct, ShopifyLayout, ModelVariantGeneration]:
        """The generation, only if it belongs to one of THIS product's layouts."""
        shopify_product = await ShopifySyncService.get_product(db, context, shopify_product_id)
        generation = await ModelVariantGenerationService.get(db, generation_id, context.user_id)
        refs = {layout_client_ref(l.id): l for l in shopify_product.layouts}
        layout = refs.get(generation.client_ref or "")
        if layout is None or generation.product_id != shopify_product.rivollo_product_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=GENERATION_NOT_FOUND)
        return shopify_product, layout, generation

    @staticmethod
    async def layout_has_model(
        db: AsyncSession, product_id: uuid.UUID, layout: ShopifyLayout, user_id: uuid.UUID
    ) -> bool:
        from app.database.configurator_repo import configurator_repository

        for generation in await repo.get_layout_generations(db, product_id, [layout_client_ref(layout.id)]):
            if generation.status == "accepted" and generation.accepted_variant_id is not None:
                if await configurator_repository.get_owned_model_variant(
                    db, generation.accepted_variant_id, user_id
                ):
                    return True
        return False


shopify_glb_service = ShopifyGlbService()
