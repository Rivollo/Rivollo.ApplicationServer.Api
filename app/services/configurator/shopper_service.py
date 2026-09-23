"""The shopper-facing view of a product's configuration.

Separate from the seller services because the filtering is itself a business
rule, and because the two audiences want opposite things: a seller needs to see
a half-finished option in order to finish it, and a shopper must never see one.

Every exclusion below is specified in api-spec.md section 9. They are applied
HERE, before serialisation — not by leaving fields out of a response model —
so that a future field added to the seller schema cannot leak by omission.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.configurator_repo import USDZ_ASSET_ID
from app.database.configurator_repo import configurator_repository as repo
from app.models.configurator import PartOption, ProductPart
from app.services.configurator.material_service import material_service
from app.services.configurator.option_service import OptionService

logger = logging.getLogger(__name__)

PRODUCT_NOT_FOUND = "Product not found"
THUMBNAIL_ASSET_ID = 1
ORIGINAL_TOKEN = "original"
ORIGINAL_NAME = "Default"

# Only a completed bake has a texture a shopper can be shown.
_VISIBLE_BAKE_STATUS = "completed"


@dataclass(frozen=True)
class PublicOptionView:
    option: PartOption
    textures: list


@dataclass(frozen=True)
class PublicPartView:
    part: ProductPart
    options: list[PublicOptionView]
    default_option_id: Optional[uuid.UUID]


@dataclass(frozen=True)
class PublicVariantView:
    """One shape: the original model (id "original") or an extra variant."""

    id: str
    name: str
    glb_url: str
    usdz_url: Optional[str]
    thumbnail_url: Optional[str]
    is_default: bool
    order_index: int
    width_m: Optional[float]
    depth_m: Optional[float]
    height_m: Optional[float]
    parts: list[PublicPartView]


@dataclass(frozen=True)
class PublicConfiguratorView:
    product_id: uuid.UUID
    product_name: str
    model_url: Optional[str]
    ar_model_url: Optional[str]
    parts: list[PublicPartView]
    # None unless the product has extra model variants (ADR-014), so a
    # single-model product's payload is exactly what it was before.
    variants: Optional[list[PublicVariantView]] = None


class ShopperService:
    """Assembles the viewer's payload. Read-only."""

    @staticmethod
    async def get_public_configurator(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> PublicConfiguratorView:
        """What the 3D viewer needs, with everything seller-only removed.

        404s for a product that is missing, soft-deleted or unpublished — the
        shopper tier must not confirm that an unpublished product exists.
        """
        product = await repo.get_published_product(db, product_id)
        if product is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND
            )

        mesh_asset = await repo.get_product_mesh_asset(db, product_id)
        usdz_asset = await repo.get_product_asset(db, product_id, USDZ_ASSET_ID)

        current_glb_version = (
            material_service.build_glb_version(mesh_asset.id)
            if mesh_asset is not None and mesh_asset.image
            else None
        )

        # The ORIGINAL model's parts (variant_id NULL) — unchanged behaviour.
        parts = await ShopperService._visible_parts(db, product_id, None, current_glb_version)
        model_url = mesh_asset.image if mesh_asset is not None else None
        ar_model_url = usdz_asset.image if usdz_asset is not None else None

        return PublicConfiguratorView(
            product_id=product.id,
            product_name=product.name,
            model_url=model_url,
            ar_model_url=ar_model_url,
            parts=parts,
            variants=await ShopperService._variants(
                db, product_id, model_url, ar_model_url, parts
            ),
        )

    @staticmethod
    async def _visible_parts(
        db: AsyncSession,
        product_id: uuid.UUID,
        variant_id: Optional[uuid.UUID],
        current_glb_version: Optional[str],
    ) -> list[PublicPartView]:
        parts: list[PublicPartView] = []
        for part in await repo.get_parts_for_product(db, product_id, variant_id=variant_id):
            view = ShopperService._filter_part(part, current_glb_version)
            if view is not None:
                parts.append(view)
        return parts

    @staticmethod
    async def _variants(
        db: AsyncSession,
        product_id: uuid.UUID,
        model_url: Optional[str],
        ar_model_url: Optional[str],
        original_parts: list[PublicPartView],
    ) -> Optional[list[PublicVariantView]]:
        """The original model first, then each extra variant with its own parts.

        None — the key is then omitted — when the feature is off, when the
        product has no extra variants, or when the original has no GLB.
        """
        if not settings.ENABLE_MODEL_VARIANTS or not model_url:
            return None
        rows = await repo.get_model_variants(db, product_id)
        if not rows:
            return None

        assets = await repo.get_assets_by_ids(
            db, [a for v in rows for a in (v.glb_asset_id, v.usdz_asset_id)]
        )
        thumbnail = await repo.get_product_asset(db, product_id, THUMBNAIL_ASSET_ID)
        views = [
            PublicVariantView(
                id=ORIGINAL_TOKEN,
                name=ORIGINAL_NAME,
                glb_url=model_url,
                usdz_url=ar_model_url,
                thumbnail_url=thumbnail.image if thumbnail is not None else None,
                is_default=True,
                order_index=0,
                width_m=None,
                depth_m=None,
                height_m=None,
                parts=original_parts,
            )
        ]
        for variant in rows:
            glb = assets.get(variant.glb_asset_id)
            if glb is None or not glb.image:
                # A variant whose GLB is gone cannot be shown.
                continue
            usdz = assets.get(variant.usdz_asset_id)
            views.append(
                PublicVariantView(
                    id=str(variant.id),
                    name=variant.name,
                    glb_url=glb.image,
                    usdz_url=usdz.image if usdz is not None else None,
                    thumbnail_url=variant.thumbnail_url,
                    is_default=False,
                    order_index=variant.order_index,
                    width_m=variant.width_m,
                    depth_m=variant.depth_m,
                    height_m=variant.height_m,
                    parts=await ShopperService._visible_parts(
                        db, product_id, variant.id,
                        material_service.build_glb_version(glb.id),
                    ),
                )
            )
        return views if len(views) > 1 else None

    @staticmethod
    def _filter_part(
        part: ProductPart,
        current_glb_version: Optional[str],
    ) -> Optional[PublicPartView]:
        """Apply api-spec section 9's exclusions. None means "drop this part"."""
        if not part.isactive or not part.shopper_selectable:
            return None

        # A part authored against a superseded model would paint the wrong mesh.
        if current_glb_version is None or part.glb_version != current_glb_version:
            return None

        options = [
            PublicOptionView(option=option, textures=OptionService.current_textures(option))
            for option in (part.options or [])
            if option.isactive and option.bake_status == _VISIBLE_BAKE_STATUS
        ]
        options.sort(key=lambda v: (v.option.order_index, v.option.name))

        if not options:
            # Nothing to choose — the part would render as an empty swatch row.
            return None

        # Computed over the SURVIVING options only, and NOT substituted when the
        # configured default is filtered out.
        #
        # `default_option_id` means "the starting option this seller chose". When
        # the seller has chosen none - or their choice is re-baking after a recipe
        # change - the honest answer is null, which the viewer renders as Original:
        # the model as uploaded. The first surviving option is merely the first;
        # returning it would tell the viewer a seller made a choice they did not make.
        default_option_id = next(
            (v.option.id for v in options if v.option.is_default), None
        )

        return PublicPartView(
            part=part, options=options, default_option_id=default_option_id
        )


shopper_service = ShopperService()
