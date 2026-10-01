"""Read models for the Shopify integration: the merchant state and the shopper payload.

A layout's 3D model is DERIVED, never stored: the Original's is the Rivollo
product's main GLB; any other layout's is the live model variant its newest
accepted generation produced (found via client_ref). So an auto-accepted
generation, or a variant deleted in the portal, is reflected with no
bookkeeping in the Shopify tables.
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Iterable, Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.configurator_repo import configurator_repository
from app.database.shopify_repo import (
    MESH_ASSET_ID,
    THUMBNAIL_ASSET_ID,
    USDZ_ASSET_ID,
    shopify_repository as repo,
)
from app.models.configurator import ModelVariantGeneration, ProductModelVariant
from app.models.models import Product
from app.models.shopify import ROLE_LAYOUT, ShopifyLayout, ShopifyProduct, layout_client_ref
from app.schemas.configurator import ModelVariantGenerationResponse
from app.schemas.shopify import (
    PublicShopifyLayout,
    PublicShopifyOption,
    PublicShopifyProduct,
    PublicShopifyVariant,
    RivolloProductOut,
    ShopifyLayoutModelOut,
    ShopifyLayoutOut,
    ShopifyProductStateResponse,
    ShopifyVariantOut,
)
from app.services.shopify.glb_service import GENERATING_STATUSES, main_glb_stalled, status_value
from app.services.shopify.sync_service import ShopifySyncService

logger = logging.getLogger(__name__)

ORIGINAL = "original"
SOURCE_RIVOLLO = "rivollo"
SOURCE_SHOPIFY = "shopify"


def money(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else f"{Decimal(value):.2f}"


def viewer_url(public_id: Optional[str]) -> Optional[str]:
    base = (settings.VIEWER_BASE_URL or "").rstrip("/")
    return f"{base}/{public_id}" if base and public_id else None


async def product_sources(
    db: AsyncSession, product_ids: Iterable[uuid.UUID], user_id: uuid.UUID
) -> dict[uuid.UUID, str]:
    """``source`` for the core product responses: "shopify" or "rivollo".

    "shopify" exactly when GET /products/{id}/shopify answers 200 for the
    owner: linked, and the shop still connected. The core product routes call
    this, so it must never break them: with the integration off it does not
    query, and a failure (e.g. tables not created yet) is contained in a
    savepoint and reported as "rivollo".
    """
    ids = list(product_ids)
    linked: set[uuid.UUID] = set()
    if ids and settings.ENABLE_SHOPIFY_INTEGRATION:
        try:
            async with db.begin_nested():
                linked = await repo.get_linked_product_ids(db, ids, user_id)
        except SQLAlchemyError:
            logger.warning("Shopify source lookup failed; reporting products as rivollo", exc_info=True)
    return {pid: SOURCE_SHOPIFY if pid in linked else SOURCE_RIVOLLO for pid in ids}


def generation_dict(generation: ModelVariantGeneration) -> dict:
    return ModelVariantGenerationResponse(
        id=generation.id,
        product_id=generation.product_id,
        name=generation.name,
        source_image_url=generation.source_image_url,
        model=generation.model_key,
        credit_cost=generation.credit_cost,
        status=generation.status,
        error=generation.error,
        candidate_glb_url=generation.candidate_glb_url if generation.status == "ready" else None,
        accepted_variant_id=generation.accepted_variant_id,
        auto_accept=bool(generation.auto_accept),
        client_ref=generation.client_ref,
        started_at=generation.started_at,
        completed_at=generation.completed_at,
        created_at=generation.created_date,
    ).model_dump(mode="json")


class _LayoutModels:
    """Resolves each layout to its live model variant (or the Original)."""

    def __init__(self, generations: list[ModelVariantGeneration], variants: list[ProductModelVariant]):
        self.by_ref: dict[str, list[ModelVariantGeneration]] = {}
        for generation in generations:  # newest first
            self.by_ref.setdefault(generation.client_ref or "", []).append(generation)
        self.variants = {v.id: v for v in variants}

    def generations(self, layout: ShopifyLayout) -> list[ModelVariantGeneration]:
        return self.by_ref.get(layout_client_ref(layout.id), [])

    def variant(self, layout: ShopifyLayout) -> Optional[ProductModelVariant]:
        for generation in self.generations(layout):
            if generation.status == "accepted" and generation.accepted_variant_id in self.variants:
                return self.variants[generation.accepted_variant_id]
        return None


class ShopifyViewService:
    @staticmethod
    async def _layout_models(db: AsyncSession, shopify_product: ShopifyProduct) -> _LayoutModels:
        generations = await repo.get_layout_generations(
            db,
            shopify_product.rivollo_product_id,
            [layout_client_ref(l.id) for l in shopify_product.layouts],
        )
        variants = await configurator_repository.get_model_variants(db, shopify_product.rivollo_product_id)
        return _LayoutModels(generations, variants)

    @staticmethod
    async def _variant_glb_urls(db: AsyncSession, variants: list[ProductModelVariant]) -> dict[uuid.UUID, str]:
        assets = await configurator_repository.get_assets_by_ids(db, [v.glb_asset_id for v in variants])
        return {
            v.id: assets[v.glb_asset_id].image
            for v in variants
            if v.glb_asset_id is not None and v.glb_asset_id in assets
        }

    # ------------------------------------------------------------------ #
    # Merchant state
    # ------------------------------------------------------------------ #
    @staticmethod
    async def state(
        db: AsyncSession, shopify_product: ShopifyProduct, user_id: uuid.UUID
    ) -> ShopifyProductStateResponse:
        product = await repo.get_live_product(db, shopify_product.rivollo_product_id, user_id)

        rivollo: Optional[RivolloProductOut] = None
        main_state = "none"
        glb_url: Optional[str] = None
        layouts_out: list[ShopifyLayoutOut] = []

        if product is not None:
            mesh = await repo.get_mapped_asset(db, product.id, MESH_ASSET_ID)
            usdz = await repo.get_mapped_asset(db, product.id, USDZ_ASSET_ID)
            thumb = await repo.get_mapped_asset(db, product.id, THUMBNAIL_ASSET_ID)
            glb_url = mesh.image if mesh is not None else None
            if glb_url:
                main_state = "ready"
            elif status_value(product) in GENERATING_STATUSES:
                main_state = "stalled" if main_glb_stalled(shopify_product, product) else "generating"
            public_id = None
            if status_value(product) == "published":
                public_id = await repo.get_enabled_public_id(db, product.id)
            rivollo = RivolloProductOut(
                id=product.id,
                status=status_value(product),
                main_glb_state=main_state,
                glb_url=glb_url,
                usdz_url=usdz.image if usdz is not None else None,
                thumbnail_url=thumb.image if thumb is not None else None,
                public_id=public_id,
                viewer_url=viewer_url(public_id),
            )

            models = await ShopifyViewService._layout_models(db, shopify_product)
            variant_urls = await ShopifyViewService._variant_glb_urls(db, list(models.variants.values()))
            for layout in shopify_product.layouts:
                layouts_out.append(
                    ShopifyViewService._layout_out(
                        shopify_product, layout, models, variant_urls, product, main_state, glb_url
                    )
                )

        return ShopifyProductStateResponse(
            id=shopify_product.id,
            shopify_product_id=str(shopify_product.shopify_product_id),
            shop_domain=shopify_product.shop_domain,
            title=shopify_product.title,
            handle=shopify_product.handle,
            currency=shopify_product.currency,
            shopify_status=shopify_product.shopify_status,
            synced_at=shopify_product.synced_at,
            images=list(shopify_product.images or []),
            options=list(shopify_product.options or []),
            option_roles=dict(shopify_product.option_roles or {}),
            variants=[
                ShopifyVariantOut(
                    shopify_variant_id=str(v.shopify_variant_id),
                    title=v.title,
                    sku=v.sku,
                    price=money(v.price),
                    compare_at_price=money(v.compare_at_price),
                    inventory=v.inventory_quantity,
                    available=v.available,
                    image_urls=list(v.image_urls or []),
                    options={o["name"]: o["value"] for o in v.options or []},
                )
                for v in shopify_product.shopify_variants
            ],
            rivollo_product=rivollo,
            layouts=layouts_out,
        )

    @staticmethod
    def _layout_out(
        shopify_product: ShopifyProduct,
        layout: ShopifyLayout,
        models: _LayoutModels,
        variant_urls: dict[uuid.UUID, str],
        product: Product,
        main_state: str,
        main_glb_url: Optional[str],
    ) -> ShopifyLayoutOut:
        generations = models.generations(layout)
        model: Optional[ShopifyLayoutModelOut] = None
        if layout.is_original:
            state = {"ready": "ready", "generating": "generating", "stalled": "failed"}.get(main_state, "none")
            if main_glb_url:
                model = ShopifyLayoutModelOut(id=ORIGINAL, name=product.name, glb_url=main_glb_url)
        else:
            variant = models.variant(layout)
            if variant is not None:
                state = "ready"
                model = ShopifyLayoutModelOut(
                    id=str(variant.id),
                    name=variant.name,
                    glb_url=variant_urls.get(variant.id),
                    thumbnail_url=variant.thumbnail_url,
                )
            elif any(g.status == "ready" for g in generations):
                state = "ready_for_review"
            elif any(g.status in ("queued", "generating") for g in generations):
                state = "generating"
            elif generations and generations[0].status == "failed":
                state = "failed"
            else:
                state = "none"

        return ShopifyLayoutOut(
            id=layout.id,
            option_value=layout.option_value,
            is_original=layout.is_original,
            position=layout.position,
            state=state,
            model=model,
            candidate_image_urls=ShopifyViewService.candidate_images(shopify_product, layout),
            generations=[generation_dict(g) for g in generations],
        )

    @staticmethod
    def candidate_images(shopify_product: ShopifyProduct, layout: ShopifyLayout) -> list[str]:
        """This layout value's variant images first, then the product gallery."""
        name = ShopifySyncService.layout_option(shopify_product)
        urls: list[str] = []
        for variant in shopify_product.shopify_variants:
            values = {o["name"]: o["value"] for o in variant.options or []}
            if name is not None and values.get(name) == layout.option_value:
                urls.extend(u for u in variant.image_urls or [] if u not in urls)
        for image in shopify_product.images or []:
            if image.get("url") and image["url"] not in urls:
                urls.append(image["url"])
        return urls

    # ------------------------------------------------------------------ #
    # Public shopper payload (Phase 3)
    # ------------------------------------------------------------------ #
    @staticmethod
    async def public_payload(db: AsyncSession, product_id: uuid.UUID) -> PublicShopifyProduct:
        """For a PUBLISHED product linked to a Shopify product on a live connection. 404 otherwise."""
        not_found = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

        product = await repo.get_published_product(db, product_id)
        if product is None or await repo.get_enabled_public_id(db, product_id) is None:
            raise not_found
        shopify_product = await repo.get_by_rivollo_product(db, product_id)
        if shopify_product is None:
            raise not_found
        connection = await repo.get_active_connection_by_shop(db, shopify_product.shop_domain)
        if connection is None or connection.user_id != shopify_product.user_id:
            raise not_found

        shop = shopify_product.shop_domain
        layout_name = ShopifySyncService.layout_option(shopify_product)

        layouts: list[PublicShopifyLayout] = []
        if layout_name is not None:
            models = await ShopifyViewService._layout_models(db, shopify_product)
            has_main = await repo.get_mapped_asset(db, product_id, MESH_ASSET_ID) is not None
            for layout in shopify_product.layouts:
                if layout.is_original:
                    if has_main:
                        layouts.append(PublicShopifyLayout(value=layout.option_value, model=ORIGINAL))
                    continue
                variant = models.variant(layout)
                if variant is not None:  # unaccepted layouts are omitted
                    layouts.append(PublicShopifyLayout(value=layout.option_value, model=str(variant.id)))

        roles = shopify_product.option_roles or {}
        return PublicShopifyProduct(
            title=shopify_product.title,
            currency=shopify_product.currency,
            product_url=f"https://{shop}/products/{shopify_product.handle}",
            layout_option=layout_name,
            options=[
                PublicShopifyOption(
                    name=o["name"],
                    role=ROLE_LAYOUT if roles.get(o["name"]) == ROLE_LAYOUT else "info",
                    values=list(o["values"]),
                )
                for o in shopify_product.options or []
            ],
            layouts=layouts,
            variants=[
                PublicShopifyVariant(
                    id=str(v.shopify_variant_id),
                    title=v.title,
                    options={o["name"]: o["value"] for o in v.options or []},
                    price=money(v.price),
                    compare_at_price=money(v.compare_at_price),
                    available=v.available,
                    image_url=(v.image_urls or [None])[0],
                    add_to_cart_url=f"https://{shop}/cart/{v.shopify_variant_id}:1",
                )
                for v in shopify_product.shopify_variants
            ],
        )


shopify_view_service = ShopifyViewService()
