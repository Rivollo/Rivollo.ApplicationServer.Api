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

from app.database.configurator_repo import USDZ_ASSET_ID
from app.database.configurator_repo import configurator_repository as repo
from app.models.models import PartOption, ProductPart
from app.services.configurator.material_service import material_service
from app.services.configurator.option_service import OptionService

logger = logging.getLogger(__name__)

PRODUCT_NOT_FOUND = "Product not found"

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
class PublicConfiguratorView:
    product_id: uuid.UUID
    product_name: str
    model_url: Optional[str]
    ar_model_url: Optional[str]
    parts: list[PublicPartView]


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

        parts: list[PublicPartView] = []
        for part in await repo.get_parts_for_product(db, product_id):
            view = ShopperService._filter_part(part, current_glb_version)
            if view is not None:
                parts.append(view)

        return PublicConfiguratorView(
            product_id=product.id,
            product_name=product.name,
            model_url=mesh_asset.image if mesh_asset is not None else None,
            ar_model_url=usdz_asset.image if usdz_asset is not None else None,
            parts=parts,
        )

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
