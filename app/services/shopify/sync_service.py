"""Product sync, option roles and unlink for the Shopify integration.

FIRST SYNC creates an ordinary draft Rivollo product — the same rows
/createProductFal writes before it starts generating (tbl_products DRAFT, an
asset-1 thumbnail, its mapping) — and links it by id. The portal lists it as a
normal draft. No AI credits, no plan requirement.

LATER SYNCS update only the Shopify mirror (variants, prices, images, options).
They never overwrite the Rivollo product's name, description or thumbnail,
which the seller may have edited in the portal (spec D10). If the linked
product was deleted in the portal, the next sync creates a fresh draft.

The submitted variants REPLACE the stored set: variants the merchant no longer
selects are removed. Rows are updated in place by Shopify id rather than
deleted and re-inserted, because a same-flush delete + insert of the same
unique key is not ordered safely by the unit of work.
"""

from __future__ import annotations

import html
import logging
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.shopify_repo import THUMBNAIL_ASSET_ID, shopify_repository as repo
from app.models.models import Product, ProductAsset, ProductAssetMapping, ProductStatus
from app.models.shopify import (
    ROLE_COLOUR,
    ROLE_INFO,
    ROLE_LAYOUT,
    ShopifyLayout,
    ShopifyProduct,
    ShopifyProductVariant,
)
from app.schemas.shopify import ShopifyOptionsRequest, ShopifySyncRequest
from app.services.product_service import ProductService
from app.services.shopify.connection_service import ShopifyContext
from app.services.shopify.image_importer import image_importer

logger = logging.getLogger(__name__)

PRODUCT_NOT_FOUND = "Shopify product not found. Sync it first."
DESCRIPTION_MAX = 5000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def html_to_text(value: Optional[str]) -> Optional[str]:
    """Merchant HTML -> plain text for tbl_products.description.

    The portal and the viewer must never render merchant HTML; the original is
    kept in the Shopify mirror (description_html).
    """
    if not value:
        return None
    # Script and style bodies are code, not description: drop them whole.
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:DESCRIPTION_MAX] or None


def derive_options(variants: list[ShopifyProductVariant]) -> list[dict]:
    """[{name, values[]}] in first-seen order, from the synced variants."""
    order: list[str] = []
    values: dict[str, list[str]] = {}
    for variant in variants:
        for option in variant.options or []:
            name, value = option["name"], option["value"]
            if name not in values:
                order.append(name)
                values[name] = []
            if value not in values[name]:
                values[name].append(value)
    return [{"name": name, "values": values[name]} for name in order]


def allowed_image_urls(product: ShopifyProduct) -> list[str]:
    """Every image URL the merchant synced for this product, deduplicated, in order."""
    urls: list[str] = []
    for image in product.images or []:
        if image.get("url") and image["url"] not in urls:
            urls.append(image["url"])
    for variant in product.shopify_variants:
        for url in variant.image_urls or []:
            if url not in urls:
                urls.append(url)
    return urls


def require_allowed_image(product: ShopifyProduct, url: str) -> None:
    if url not in allowed_image_urls(product):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_url must be one of this product's synced Shopify images.",
        )


class ShopifySyncService:
    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_product(
        db: AsyncSession, context: ShopifyContext, shopify_product_id: int, *, for_update: bool = False
    ) -> ShopifyProduct:
        product = await repo.get_product(
            db, context.shop_domain, context.user_id, shopify_product_id, for_update=for_update
        )
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)
        return product

    # ------------------------------------------------------------------ #
    # Sync
    # ------------------------------------------------------------------ #
    @staticmethod
    async def sync(
        db: AsyncSession, context: ShopifyContext, payload: ShopifySyncRequest
    ) -> tuple[ShopifyProduct, Product, bool]:
        """Upsert. Returns (shopify product, Rivollo product, created)."""
        user_id = context.user_id
        existing = await repo.get_product_any_owner(db, context.shop_domain, payload.shopify_product_id)
        if existing is not None and existing.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This Shopify product is linked to another Rivollo account. "
                    "Unlink it there first."
                ),
            )

        rivollo_product: Optional[Product] = None
        if existing is not None:
            rivollo_product = await repo.get_live_product(db, existing.rivollo_product_id, user_id)

        # Network work (the thumbnail copy) happens before any row is written.
        thumbnail_url: Optional[str] = None
        if rivollo_product is None:
            thumbnail_url = await ShopifySyncService._import_thumbnail(user_id, payload)

        now = _utcnow()
        if rivollo_product is None:
            rivollo_product = await ShopifySyncService._create_draft_product(
                db, user_id, payload, thumbnail_url, now
            )

        created = existing is None
        product = existing or ShopifyProduct(
            id=uuid.uuid4(),
            user_id=user_id,
            shop_domain=context.shop_domain,
            shopify_product_id=payload.shopify_product_id,
            option_roles={},
            created_by=user_id,
            created_date=now,
        )
        product.title = payload.title
        product.handle = payload.handle
        product.description_html = payload.description
        product.vendor = payload.vendor
        product.product_type = payload.product_type
        product.tags = list(payload.tags)
        product.shopify_status = payload.status
        product.currency = payload.currency
        product.images = [
            {"id": image.id, "url": image.url, "alt_text": image.alt_text} for image in payload.images
        ]
        product.rivollo_product_id = rivollo_product.id
        product.synced_at = now
        if not created:
            product.updated_by = user_id
            product.updated_date = now
            if existing.rivollo_product_id != rivollo_product.id:
                # Re-linked to a fresh draft: the old main-GLB request is moot.
                product.main_glb_requested_at = None
        if created:
            repo.add(db, product)

        ShopifySyncService._replace_variants(product, payload, user_id, now)
        product.options = derive_options(product.shopify_variants)
        ShopifySyncService._prune_roles(product)
        ShopifySyncService._reconcile_layouts(product, user_id, now)

        await db.commit()
        logger.info(
            "Shopify product %s (%s) synced: %d variants, rivollo product %s, created=%s",
            payload.shopify_product_id, context.shop_domain, len(product.shopify_variants),
            rivollo_product.id, created,
        )
        return product, rivollo_product, created

    @staticmethod
    async def _import_thumbnail(user_id: uuid.UUID, payload: ShopifySyncRequest) -> Optional[str]:
        """The main Shopify image, copied into Rivollo. None (logged) if that fails.

        A thumbnail is a nicety: failing to copy it must not fail the sync.
        """
        first = payload.images[0].url if payload.images else next(
            (url for variant in payload.variants for url in variant.image_urls), None
        )
        if first is None:
            return None
        try:
            return await image_importer.copy_to_uploads(user_id, first)
        except HTTPException as exc:
            logger.warning("Sync continues without a thumbnail: %s", exc.detail)
            return None

    @staticmethod
    async def _create_draft_product(
        db: AsyncSession,
        user_id: uuid.UUID,
        payload: ShopifySyncRequest,
        thumbnail_url: Optional[str],
        now: datetime,
    ) -> Product:
        """The rows /createProductFal writes before generating, minus the generation."""
        slug = await ProductService._generate_unique_slug(db, ProductService._slugify(payload.title))
        product = Product(
            id=uuid.uuid4(),
            name=payload.title,
            slug=slug,
            status=ProductStatus.DRAFT,
            description=html_to_text(payload.description),
            created_by=user_id,
        )
        repo.add(db, product)
        await repo.flush(db)
        if thumbnail_url:
            await ShopifySyncService.add_thumbnail(db, product, thumbnail_url, user_id)
        return product

    @staticmethod
    async def add_thumbnail(
        db: AsyncSession, product: Product, image_url: str, user_id: uuid.UUID
    ) -> None:
        """An asset-1 row + mapping, exactly as create_product_with_fal_image_urls writes it."""
        asset = ProductAsset(id=uuid.uuid4(), asset_id=THUMBNAIL_ASSET_ID, image=image_url, created_by=user_id)
        repo.add(db, asset)
        await repo.flush(db)
        repo.add(
            db,
            ProductAssetMapping(
                name=product.name,
                productid=product.id,
                product_asset_id=asset.id,
                isactive=True,
                created_by=user_id,
            ),
        )

    @staticmethod
    def _replace_variants(
        product: ShopifyProduct, payload: ShopifySyncRequest, user_id: uuid.UUID, now: datetime
    ) -> None:
        current = {v.shopify_variant_id: v for v in product.shopify_variants}
        kept: list[ShopifyProductVariant] = []
        for position, incoming in enumerate(payload.variants):
            row = current.get(incoming.shopify_variant_id)
            if row is None:
                row = ShopifyProductVariant(
                    id=uuid.uuid4(),
                    shopify_variant_id=incoming.shopify_variant_id,
                    created_by=user_id,
                    created_date=now,
                )
            else:
                row.updated_by = user_id
                row.updated_date = now
            row.title = incoming.title
            row.sku = incoming.sku
            row.price = Decimal(incoming.price)
            row.compare_at_price = Decimal(incoming.compare_at_price) if incoming.compare_at_price else None
            row.inventory_quantity = incoming.inventory
            row.available = incoming.available
            row.image_urls = list(incoming.image_urls)
            row.options = [{"name": o.name, "value": o.value} for o in incoming.options]
            row.position = position
            kept.append(row)
        # delete-orphan removes the variants no longer selected.
        product.shopify_variants = kept

    # ------------------------------------------------------------------ #
    # Option roles and layouts
    # ------------------------------------------------------------------ #
    @staticmethod
    def layout_option(product: ShopifyProduct) -> Optional[str]:
        return next(
            (name for name, role in (product.option_roles or {}).items() if role == ROLE_LAYOUT), None
        )

    @staticmethod
    def _prune_roles(product: ShopifyProduct) -> None:
        """Drop roles for options the product no longer has."""
        names = {option["name"] for option in product.options or []}
        product.option_roles = {k: v for k, v in (product.option_roles or {}).items() if k in names}

    @staticmethod
    def _reconcile_layouts(
        product: ShopifyProduct,
        user_id: uuid.UUID,
        now: datetime,
        *,
        original_value: Optional[str] = None,
    ) -> None:
        """One layout row per value of the layout option; none without one.

        Removed values lose their row, but any model variant they produced is
        left in Rivollo untouched (the seller can delete it in the portal).
        ``original_value`` re-points the Original; otherwise it is kept.
        """
        name = ShopifySyncService.layout_option(product)
        values: list[str] = []
        if name is not None:
            values = next((o["values"] for o in product.options if o["name"] == name), [])

        existing = {layout.option_value: layout for layout in product.layouts}
        if original_value is None:
            original_value = next((l.option_value for l in product.layouts if l.is_original), None)

        rows: list[ShopifyLayout] = []
        for position, value in enumerate(values):
            row = existing.get(value)
            if row is None:
                row = ShopifyLayout(
                    id=uuid.uuid4(), option_value=value, created_by=user_id, created_date=now
                )
            row.position = position
            row.is_original = value == original_value
            rows.append(row)
        product.layouts = rows

    @staticmethod
    async def set_options(
        db: AsyncSession, context: ShopifyContext, shopify_product_id: int, request: ShopifyOptionsRequest
    ) -> ShopifyProduct:
        product = await ShopifySyncService.get_product(db, context, shopify_product_id, for_update=True)
        options = {o["name"]: o["values"] for o in product.options or []}

        unknown = [name for name in request.roles if name not in options]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown option(s): {', '.join(unknown)}. Sync the product first.",
            )
        if any(role == ROLE_COLOUR for role in request.roles.values()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The colour role is not available yet. Use 'info' for colour options for now.",
            )
        layouts = [name for name, role in request.roles.items() if role == ROLE_LAYOUT]
        if len(layouts) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only one option can be the layout (3D model) option.",
            )

        original_value: Optional[str] = None
        if layouts:
            original_value = request.original_layout_value
            if original_value is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="original_layout_value is required when an option has the layout role.",
                )
            if original_value not in options[layouts[0]]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"'{original_value}' is not a value of the '{layouts[0]}' option.",
                )

        now = _utcnow()
        # Every option gets an explicit role; unlisted ones are "info".
        product.option_roles = {name: request.roles.get(name, ROLE_INFO) for name in options}
        product.updated_by = context.user_id
        product.updated_date = now
        ShopifySyncService._reconcile_layouts(
            product, context.user_id, now, original_value=original_value or ""
        )
        await db.commit()
        return product

    # ------------------------------------------------------------------ #
    # Unlink
    # ------------------------------------------------------------------ #
    @staticmethod
    async def unlink(db: AsyncSession, context: ShopifyContext, shopify_product_id: int) -> None:
        """Shopify products/delete: drop the Shopify rows; the Rivollo product stays."""
        product = await ShopifySyncService.get_product(db, context, shopify_product_id, for_update=True)
        await repo.delete(db, product)
        await db.commit()


shopify_sync_service = ShopifySyncService()
