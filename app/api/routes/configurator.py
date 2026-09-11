"""Product Configurator routes.

Thin by construction: parse the path id, hand the authenticated user's id to a
service, map the returned domain objects onto response schemas, wrap in
``api_success``. No business rules, no database access, no ownership checks, no
baking. Services raise ``HTTPException`` with the documented status codes and
those propagate untouched.

Paths live under a ``/configurator`` namespace rather than at the bare root.
That is what keeps them clear of the colour-variant feature, which already owns
``/products/{id}/materials`` and ``/products/{id}/color-variants`` — see
docs/configurator/decisions.md Q6. No colour-variant route is modified.

Reference: docs/configurator/api-spec.md.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import CurrentUser, DB, get_current_user
from app.models.models import PartOption, ProductPart
from app.schemas.configurator import (
    BakeProgress,
    BakeStatusResponse,
    MaterialResponse,
    MaterialsResponse,
    PartOptionCreate,
    PartOptionResponse,
    PartOptionTextureResponse,
    PartOptionUpdate,
    ProductPartCreate,
    ProductPartResponse,
    ProductPartUpdate,
    ProductPartUpdateResponse,
    PublicConfiguratorResponse,
    PublicOptionTexture,
    PublicPartOption,
    PublicProductPart,
)
from app.services.configurator.bake_service import bake_service
from app.services.configurator.material_service import material_service
from app.services.configurator.option_service import OptionService, option_service
from app.services.configurator.part_service import PartService, part_service
from app.services.configurator.shopper_service import shopper_service
from app.utils.envelopes import api_success

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["configurator"],
    dependencies=[Depends(get_current_user)],
)

# No auth dependency, deliberately. The shopper payload is consumed by the
# separate Viewer Portal and by any browser opening a published product, so it
# must be reachable without credentials.
#
# It previously carried a Basic-auth dependency from app/api/routes/products.py,
# which depends on `HTTPBasic()`. That scheme defaults to auto_error=True, so a
# request with no Authorization header was answered 401 with
# `WWW-Authenticate: Basic` — which is exactly what makes a browser show a
# username/password popup.
#
# Safe to remove because every field this router exposes is already filtered for
# a shopper audience in ShopperService: published products only, no inactive or
# non-shopper-selectable parts, no option that is not `completed`, no texture
# whose recipe_hash is stale, and a separate response schema that cannot carry a
# seller-only field. The seller router above is untouched and still requires a
# bearer token.
public_router = APIRouter(tags=["configurator"])


def _parse_uuid(raw: str, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {label} format"
        )


# --------------------------------------------------------------------------- #
# Serialisation — presentation only; the computed bits come from services
# --------------------------------------------------------------------------- #
def _option_response(option: PartOption) -> PartOptionResponse:
    return PartOptionResponse(
        id=option.id,
        part_id=option.part_id,
        name=option.name,
        slug=option.slug,
        swatch_hex=option.swatch_hex,
        recipe=option.recipe or {},
        recipe_hash=option.recipe_hash,
        order_index=option.order_index,
        is_default=option.is_default,
        isactive=option.isactive,
        bake_status=option.bake_status,
        bake_error=option.bake_error,
        bake_started_at=option.bake_started_at,
        bake_completed_at=option.bake_completed_at,
        bake_attempts=option.bake_attempts,
        # Stale textures are hidden rather than served as the wrong colour.
        textures=[
            PartOptionTextureResponse.model_validate(t)
            for t in OptionService.current_textures(option)
        ],
        created_at=option.created_date,
    )


def _part_response(
    part: ProductPart,
    current_glb_version: str | None,
) -> ProductPartResponse:
    return ProductPartResponse(
        id=part.id,
        product_id=part.product_id,
        name=part.name,
        slug=part.slug,
        material_indices=list(part.material_indices or []),
        material_type=part.material_type,
        order_index=part.order_index,
        shopper_selectable=part.shopper_selectable,
        isactive=part.isactive,
        glb_version=part.glb_version,
        # Both computed, never stored — ADR-011 and ADR-006.
        default_option_id=PartService.default_option_id(part),
        glb_stale=PartService.is_glb_stale(part, current_glb_version),
        options=[_option_response(o) for o in (part.options or [])],
        created_at=part.created_date,
        updated_at=part.updated_date,
    )


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #
@router.get("/products/{product_id}/configurator/materials", response_model=dict)
async def list_product_materials(
    product_id: str,
    current_user: CurrentUser,
    db: DB,
):
    """Colourable materials of the product's current GLB, for the Part Editor.

    Downloads and parses the GLB, so it is markedly slower than the other
    endpoints. Intended to be called once when the editor opens.
    """
    prod_uuid = _parse_uuid(product_id, "productId")
    await PartService.require_owned_product_for_read(db, prod_uuid, current_user.id)

    glb_version, model_url, materials = await material_service.list_materials(
        db, prod_uuid
    )

    parts, _ = await part_service.list_parts(db, prod_uuid, current_user.id)
    claimed: dict[int, uuid.UUID] = {
        index: part.id
        for part in parts
        if part.isactive
        for index in (part.material_indices or [])
    }

    payload = MaterialsResponse(
        glb_version=glb_version,
        model_url=model_url,
        material_count=len(materials),
        materials=[
            MaterialResponse(
                **material,
                assigned_part_id=claimed.get(material["material_index"]),
                eligible_for_part=material["material_index"] not in claimed,
            )
            for material in materials
        ],
    )
    return api_success(payload.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #
@router.get("/products/{product_id}/configurator/parts", response_model=dict)
async def list_parts(
    product_id: str,
    current_user: CurrentUser,
    db: DB,
):
    prod_uuid = _parse_uuid(product_id, "productId")
    parts, current_glb_version = await part_service.list_parts(
        db, prod_uuid, current_user.id
    )
    return api_success(
        [_part_response(p, current_glb_version).model_dump(mode="json") for p in parts]
    )


@router.post(
    "/products/{product_id}/configurator/parts",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def create_part(
    product_id: str,
    payload: ProductPartCreate,
    current_user: CurrentUser,
    db: DB,
):
    prod_uuid = _parse_uuid(product_id, "productId")
    part = await part_service.create_part(db, prod_uuid, current_user.id, payload)
    return api_success(_part_response(part, part.glb_version).model_dump(mode="json"))


@router.get("/configurator/parts/{part_id}", response_model=dict)
async def get_part(
    part_id: str,
    current_user: CurrentUser,
    db: DB,
):
    part_uuid = _parse_uuid(part_id, "partId")
    part = await part_service.get_part(db, part_uuid, current_user.id)
    current = await part_service.current_glb_version(db, part.product_id)
    return api_success(_part_response(part, current).model_dump(mode="json"))


@router.patch("/configurator/parts/{part_id}", response_model=dict)
async def update_part(
    part_id: str,
    payload: ProductPartUpdate,
    current_user: CurrentUser,
    db: DB,
):
    part_uuid = _parse_uuid(part_id, "partId")
    part, invalidated = await part_service.update_part(
        db, part_uuid, current_user.id, payload
    )
    current = await part_service.current_glb_version(db, part.product_id)
    body = ProductPartUpdateResponse(
        **_part_response(part, current).model_dump(),
        invalidated_option_ids=invalidated,
    )
    return api_success(body.model_dump(mode="json"))


@router.delete("/configurator/parts/{part_id}", response_model=dict)
async def delete_part(
    part_id: str,
    current_user: CurrentUser,
    db: DB,
):
    part_uuid = _parse_uuid(part_id, "partId")
    await part_service.delete_part(db, part_uuid, current_user.id)
    return api_success({"message": "Part deleted successfully"})


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
@router.get("/configurator/parts/{part_id}/options", response_model=dict)
async def list_options(
    part_id: str,
    current_user: CurrentUser,
    db: DB,
):
    part_uuid = _parse_uuid(part_id, "partId")
    options = await option_service.list_options(db, part_uuid, current_user.id)
    return api_success([_option_response(o).model_dump(mode="json") for o in options])


@router.post(
    "/configurator/parts/{part_id}/options",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def create_option(
    part_id: str,
    payload: PartOptionCreate,
    current_user: CurrentUser,
    db: DB,
):
    part_uuid = _parse_uuid(part_id, "partId")
    option = await option_service.create_option(db, part_uuid, current_user.id, payload)
    return api_success(_option_response(option).model_dump(mode="json"))


@router.get("/configurator/options/{option_id}", response_model=dict)
async def get_option(
    option_id: str,
    current_user: CurrentUser,
    db: DB,
):
    opt_uuid = _parse_uuid(option_id, "optionId")
    option = await option_service.get_option(db, opt_uuid, current_user.id)
    return api_success(_option_response(option).model_dump(mode="json"))


@router.patch("/configurator/options/{option_id}", response_model=dict)
async def update_option(
    option_id: str,
    payload: PartOptionUpdate,
    current_user: CurrentUser,
    db: DB,
):
    opt_uuid = _parse_uuid(option_id, "optionId")
    option, _needs_rebake = await option_service.update_option(
        db, opt_uuid, current_user.id, payload
    )
    # `needs_rebake` is deliberately not surfaced yet: it is the signal the
    # Phase-4 runner consumes, and advertising it before anything acts on it
    # would promise a re-bake that never happens. `bake_status` already tells
    # the client the option went back to `pending`.
    return api_success(_option_response(option).model_dump(mode="json"))


@router.delete("/configurator/options/{option_id}", response_model=dict)
async def delete_option(
    option_id: str,
    current_user: CurrentUser,
    db: DB,
):
    opt_uuid = _parse_uuid(option_id, "optionId")
    await option_service.delete_option(db, opt_uuid, current_user.id)
    return api_success({"message": "Option deleted successfully"})


# --------------------------------------------------------------------------- #
# Bake
# --------------------------------------------------------------------------- #
@router.post(
    "/configurator/options/{option_id}/bake",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_bake(
    option_id: str,
    current_user: CurrentUser,
    db: DB,
    force: bool = False,
):
    """Queue a texture bake. Returns `202` immediately; poll `bake-status`.

    Nothing expensive happens in this request. `BakeService` records the intent
    and hands the option to the runner, which opens its own database session —
    this one is closed the moment the response is sent.

    Idempotent: an already-current result reports `already_current: true` without
    re-baking, and an in-flight bake returns its existing ticket rather than
    starting a second one. `force=true` re-bakes a current result (for a blob
    deleted out of band) but never duplicates an in-flight bake.
    """
    opt_uuid = _parse_uuid(option_id, "optionId")
    ticket = await bake_service.request_bake(
        db, opt_uuid, current_user.id, force=force
    )
    return api_success(
        {
            "option_id": str(ticket.option_id),
            "bake_status": ticket.bake_status,
            "recipe_hash": ticket.recipe_hash,
            "already_current": ticket.already_current,
            "poll_url": f"/configurator/options/{ticket.option_id}/bake-status",
        }
    )


@router.get("/configurator/options/{option_id}/bake-status", response_model=dict)
async def get_bake_status(
    option_id: str,
    current_user: CurrentUser,
    db: DB,
):
    """Poll target. Reads stored bake state — no baking, no business rules here."""
    opt_uuid = _parse_uuid(option_id, "optionId")
    view = await bake_service.get_bake_status(db, opt_uuid, current_user.id)

    payload = BakeStatusResponse(
        option_id=view.option_id,
        bake_status=view.bake_status,
        bake_error=view.bake_error,
        bake_started_at=view.bake_started_at,
        bake_completed_at=view.bake_completed_at,
        bake_attempts=view.bake_attempts,
        recipe_hash=view.recipe_hash,
        progress=BakeProgress(
            textures_total=view.textures_total, textures_done=view.textures_done
        ),
        textures=[PartOptionTextureResponse.model_validate(t) for t in view.textures],
    )
    return api_success(payload.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Shopper
# --------------------------------------------------------------------------- #
@public_router.get("/public/products/{product_id}/configurator", response_model=dict)
async def get_public_configurator(
    product_id: str,
    db: DB,
):
    """The viewer's payload. Filtering happens in the service, before this runs."""
    prod_uuid = _parse_uuid(product_id, "productId")
    view = await shopper_service.get_public_configurator(db, prod_uuid)

    payload = PublicConfiguratorResponse(
        product_id=view.product_id,
        product_name=view.product_name,
        model_url=view.model_url,
        ar_model_url=view.ar_model_url,
        parts=[
            PublicProductPart(
                id=p.part.id,
                name=p.part.name,
                slug=p.part.slug,
                material_indices=list(p.part.material_indices or []),
                order_index=p.part.order_index,
                default_option_id=p.default_option_id,
                options=[
                    PublicPartOption(
                        id=o.option.id,
                        name=o.option.name,
                        slug=o.option.slug,
                        swatch_hex=o.option.swatch_hex,
                        order_index=o.option.order_index,
                        textures=[
                            PublicOptionTexture.model_validate(t) for t in o.textures
                        ],
                    )
                    for o in p.options
                ],
            )
            for p in view.parts
        ],
    )
    return api_success(payload.model_dump(mode="json"))
