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

from typing import Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Response, UploadFile, status

from app.api.deps import CurrentUser, DB, get_current_user
from app.core.config import settings
from app.models.configurator import PartOption, ProductPart
from app.schemas.configurator import (
    CLIENT_REF_MAX,
    BakeProgress,
    BakeStatusResponse,
    MaterialResponse,
    MaterialsResponse,
    ModelVariantAcceptRequest,
    ModelVariantCreateResponse,
    ModelVariantGenerateRequest,
    ModelVariantGenerationResponse,
    ModelVariantReorder,
    ModelVariantResponse,
    ModelVariantUpdate,
    PartOptionCreate,
    PartOptionResponse,
    PartOptionTextureResponse,
    PartOptionUpdate,
    ProductPartCreate,
    ProductPartResponse,
    ProductPartUpdate,
    ProductPartUpdateResponse,
    PublicConfiguratorResponse,
    PublicModelVariant,
    PublicOptionTexture,
    PublicPartOption,
    PublicProductPart,
)
from app.services.configurator.bake_service import bake_service
from app.services.configurator.material_service import material_service
from app.services.configurator.model_variant_generation_service import (
    model_variant_generation_service,
)
from app.services.configurator.model_variant_service import (
    ModelEntry,
    UploadedFile,
    model_variant_service,
)
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
        variant_id=part.variant_id,
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
# Model variants (ADR-014)
# --------------------------------------------------------------------------- #
_READ_CHUNK = 1024 * 1024


async def _read_upload(upload: UploadFile, limit: int) -> UploadedFile:
    """Read at most ``limit + 1`` bytes, so an oversized upload is never held whole.

    The size RULE is the service's; reading one byte past it is what lets the
    service see that the limit was exceeded.
    """
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        chunk = await upload.read(min(_READ_CHUNK, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return UploadedFile(
        filename=upload.filename or "",
        content_type=upload.content_type,
        data=b"".join(chunks),
    )


@router.post(
    "/products/{product_id}/configurator/model-variants",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def create_model_variant(
    product_id: str,
    current_user: CurrentUser,
    db: DB,
    name: str = Form(...),
    glb: UploadFile = File(..., description="The variant's .glb model"),
    thumbnail: Optional[UploadFile] = File(
        None, description="Optional poster image (PNG, JPEG or WebP)"
    ),
):
    """Add an extra model variant (another shape) to a product.

    The product's original model is untouched and stays its default. The GLB is
    Draco-compressed server-side; see ModelVariantService for the pipeline.
    """
    model_variant_service.require_enabled()
    prod_uuid = _parse_uuid(product_id, "productId")

    glb_file = await _read_upload(glb, settings.MAX_VARIANT_GLB_BYTES)
    thumb_file = (
        await _read_upload(thumbnail, settings.MAX_VARIANT_THUMBNAIL_BYTES)
        if thumbnail is not None and thumbnail.filename
        else None
    )

    created = await model_variant_service.create_variant(
        db,
        prod_uuid,
        current_user.id,
        name=name,
        glb=glb_file,
        thumbnail=thumb_file,
    )
    variant = created.variant
    payload = ModelVariantCreateResponse(
        id=str(variant.id),
        product_id=variant.product_id,
        name=variant.name,
        glb_url=created.glb_url,
        usdz_url=None,
        thumbnail_url=variant.thumbnail_url,
        order_index=variant.order_index,
        is_original=False,
        isactive=variant.isactive,
        compression_status=variant.compression_status,
        compression_error=variant.compression_error,
        original_size_bytes=variant.original_size_bytes,
        compressed_size_bytes=variant.compressed_size_bytes,
        width_m=variant.width_m,
        depth_m=variant.depth_m,
        height_m=variant.height_m,
        created_at=variant.created_date,
        warnings=created.warnings,
    )
    return api_success(payload.model_dump(mode="json"))

def _variant_response(variant, *, glb_url, usdz_url=None) -> ModelVariantResponse:
    return ModelVariantResponse(
        id=str(variant.id),
        product_id=variant.product_id,
        name=variant.name,
        glb_url=glb_url,
        usdz_url=usdz_url,
        thumbnail_url=variant.thumbnail_url,
        order_index=variant.order_index,
        is_original=False,
        isactive=variant.isactive,
        compression_status=variant.compression_status,
        compression_error=variant.compression_error,
        original_size_bytes=variant.original_size_bytes,
        compressed_size_bytes=variant.compressed_size_bytes,
        width_m=variant.width_m,
        depth_m=variant.depth_m,
        height_m=variant.height_m,
        created_at=variant.created_date,
    )


def _entry_response(entry: ModelEntry) -> ModelVariantResponse:
    if entry.variant is not None:
        return _variant_response(entry.variant, glb_url=entry.glb_url, usdz_url=entry.usdz_url)
    return ModelVariantResponse(
        id="original",
        product_id=entry.product_id,
        name=entry.name,
        glb_url=entry.glb_url,
        usdz_url=entry.usdz_url,
        thumbnail_url=entry.thumbnail_url,
        order_index=entry.order_index,
        is_original=True,
    )


@router.get("/products/{product_id}/configurator/model-variants", response_model=dict)
async def list_model_variants(
    product_id: str,
    current_user: CurrentUser,
    db: DB,
):
    """The original model (always first, always the default) and the extra variants."""
    model_variant_service.require_enabled()
    prod_uuid = _parse_uuid(product_id, "productId")
    entries = await model_variant_service.list_models(db, prod_uuid, current_user.id)
    return api_success([_entry_response(e).model_dump(mode="json") for e in entries])


@router.patch("/configurator/model-variants/{variant_id}", response_model=dict)
async def rename_model_variant(
    variant_id: str,
    payload: ModelVariantUpdate,
    current_user: CurrentUser,
    db: DB,
):
    model_variant_service.require_enabled()
    var_uuid = _parse_uuid(variant_id, "variantId")
    variant = await model_variant_service.rename(db, var_uuid, current_user.id, payload.name)
    return api_success(
        _variant_response(variant, glb_url=None).model_dump(mode="json", exclude={"glb_url"})
    )


@router.post("/products/{product_id}/configurator/model-variants/reorder", response_model=dict)
async def reorder_model_variants(
    product_id: str,
    payload: ModelVariantReorder,
    current_user: CurrentUser,
    db: DB,
):
    model_variant_service.require_enabled()
    prod_uuid = _parse_uuid(product_id, "productId")
    await model_variant_service.reorder(db, prod_uuid, current_user.id, payload.variant_ids)
    entries = await model_variant_service.list_models(db, prod_uuid, current_user.id)
    return api_success([_entry_response(e).model_dump(mode="json") for e in entries])


@router.put("/configurator/model-variants/{variant_id}/thumbnail", response_model=dict)
async def set_model_variant_thumbnail(
    variant_id: str,
    current_user: CurrentUser,
    db: DB,
    thumbnail: UploadFile = File(..., description="PNG, JPEG or WebP, e.g. model-viewer toBlob()"),
):
    model_variant_service.require_enabled()
    var_uuid = _parse_uuid(variant_id, "variantId")
    thumb_file = await _read_upload(thumbnail, settings.MAX_VARIANT_THUMBNAIL_BYTES)
    variant = await model_variant_service.set_thumbnail(db, var_uuid, current_user.id, thumb_file)
    return api_success({"id": str(variant.id), "thumbnail_url": variant.thumbnail_url})


@router.delete("/configurator/model-variants/{variant_id}", response_model=dict)
async def delete_model_variant(
    variant_id: str,
    current_user: CurrentUser,
    db: DB,
):
    """Soft delete. The original model is not a variant and cannot be deleted."""
    model_variant_service.require_enabled()
    var_uuid = _parse_uuid(variant_id, "variantId")
    await model_variant_service.delete(db, var_uuid, current_user.id)
    return api_success({"message": "Model variant deleted successfully"})


# --------------------------------------------------------------------------- #
# Layout from photo: model-variant generations (ADR-015)
# --------------------------------------------------------------------------- #
def _generation_response(generation, *, estimate=None) -> ModelVariantGenerationResponse:
    return ModelVariantGenerationResponse(
        id=generation.id,
        product_id=generation.product_id,
        name=generation.name,
        source_image_url=generation.source_image_url,
        model=generation.model_key,
        credit_cost=generation.credit_cost,
        status=generation.status,
        error=generation.error,
        # Only a ready candidate has a file worth previewing.
        candidate_glb_url=generation.candidate_glb_url if generation.status == "ready" else None,
        accepted_variant_id=generation.accepted_variant_id,
        auto_accept=bool(generation.auto_accept),
        client_ref=generation.client_ref,
        started_at=generation.started_at,
        completed_at=generation.completed_at,
        created_at=generation.created_date,
        estimate=estimate,
    )


@router.post(
    "/products/{product_id}/configurator/model-variants/generate",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_model_variant(
    product_id: str,
    payload: ModelVariantGenerateRequest,
    current_user: CurrentUser,
    db: DB,
):
    """Generate a model variant candidate from one of the seller's uploaded photos.

    Charges the model's AI credits and returns at once with status ``queued``.
    Poll the generation until ``ready``, preview ``candidate_glb_url``, then
    accept or discard it. With ``auto_accept`` the candidate becomes a variant
    as soon as it is ready.
    """
    prod_uuid = _parse_uuid(product_id, "productId")
    requested = await model_variant_generation_service.request(
        db,
        prod_uuid,
        current_user.id,
        name=payload.name,
        image_url=payload.image_url,
        model_key=payload.model,
        client_ref=payload.client_ref,
        auto_accept=payload.auto_accept,
    )
    body = _generation_response(requested.generation, estimate=requested.estimate)
    return api_success(body.model_dump(mode="json"))


@router.get(
    "/products/{product_id}/configurator/model-variants/generations",
    response_model=dict,
)
async def list_model_variant_generations(
    product_id: str,
    current_user: CurrentUser,
    db: DB,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    client_ref: Optional[str] = Query(default=None, max_length=CLIENT_REF_MAX),
):
    """A product's generations, newest first. Filter by ``status`` or ``client_ref``."""
    prod_uuid = _parse_uuid(product_id, "productId")
    generations = await model_variant_generation_service.list_for_product(
        db, prod_uuid, current_user.id, status_filter=status_filter, client_ref=client_ref
    )
    return api_success([_generation_response(g).model_dump(mode="json") for g in generations])


@router.get("/configurator/model-variant-generations/{generation_id}", response_model=dict)
async def get_model_variant_generation(generation_id: str, current_user: CurrentUser, db: DB):
    gen_uuid = _parse_uuid(generation_id, "generationId")
    generation = await model_variant_generation_service.get(db, gen_uuid, current_user.id)
    return api_success(_generation_response(generation).model_dump(mode="json"))


@router.post(
    "/configurator/model-variant-generations/{generation_id}/accept",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def accept_model_variant_generation(
    generation_id: str,
    current_user: CurrentUser,
    db: DB,
    response: Response,
    payload: Optional[ModelVariantAcceptRequest] = Body(default=None),
):
    """Turn a ready candidate into a model variant (a Layout tile).

    201 with the new variant. Accepting an already-accepted generation is
    idempotent and answers 200 with the variant it became.
    """
    gen_uuid = _parse_uuid(generation_id, "generationId")
    accepted = await model_variant_generation_service.accept(
        db, gen_uuid, current_user.id, name=payload.name if payload else None
    )
    if not accepted.created:
        response.status_code = status.HTTP_200_OK

    variant_body = None
    if accepted.variant is not None:
        if accepted.created:
            variant_body = ModelVariantCreateResponse(
                **_variant_response(accepted.variant, glb_url=accepted.glb_url).model_dump(),
                warnings=accepted.warnings,
            )
        else:
            variant_body = _variant_response(accepted.variant, glb_url=accepted.glb_url)
    return api_success(
        {
            "generation": _generation_response(accepted.generation).model_dump(mode="json"),
            "model_variant": variant_body.model_dump(mode="json") if variant_body else None,
        }
    )


@router.delete("/configurator/model-variant-generations/{generation_id}", response_model=dict)
async def discard_model_variant_generation(generation_id: str, current_user: CurrentUser, db: DB):
    """Discard a candidate (or cancel one still generating). Credits are not refunded."""
    gen_uuid = _parse_uuid(generation_id, "generationId")
    generation = await model_variant_generation_service.discard(db, gen_uuid, current_user.id)
    return api_success(_generation_response(generation).model_dump(mode="json"))


@router.get(
    "/products/{product_id}/configurator/model-variants/{model}/materials",
    response_model=dict,
)
async def list_model_variant_materials(
    product_id: str,
    model: str,
    current_user: CurrentUser,
    db: DB,
):
    """Materials of one model's GLB. ``model`` is ``original`` or a variant id."""
    prod_uuid = _parse_uuid(product_id, "productId")
    variant_id = await model_variant_service.resolve_model(db, prod_uuid, model, current_user.id)
    return api_success(
        await _materials_payload(db, prod_uuid, current_user.id, variant_id)
    )


@router.get(
    "/products/{product_id}/configurator/model-variants/{model}/parts",
    response_model=dict,
)
async def list_model_variant_parts(
    product_id: str,
    model: str,
    current_user: CurrentUser,
    db: DB,
):
    prod_uuid = _parse_uuid(product_id, "productId")
    variant_id = await model_variant_service.resolve_model(db, prod_uuid, model, current_user.id)
    parts, current_glb_version = await part_service.list_parts(
        db, prod_uuid, current_user.id, variant_id=variant_id
    )
    return api_success(
        [_part_response(p, current_glb_version).model_dump(mode="json") for p in parts]
    )


@router.post(
    "/products/{product_id}/configurator/model-variants/{model}/parts",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def create_model_variant_part(
    product_id: str,
    model: str,
    payload: ProductPartCreate,
    current_user: CurrentUser,
    db: DB,
):
    prod_uuid = _parse_uuid(product_id, "productId")
    variant_id = await model_variant_service.resolve_model(db, prod_uuid, model, current_user.id)
    part = await part_service.create_part(
        db, prod_uuid, current_user.id, payload, variant_id=variant_id
    )
    return api_success(_part_response(part, part.glb_version).model_dump(mode="json"))


async def _materials_payload(db, prod_uuid, user_id, variant_id) -> dict:
    """The Part Editor's materials view for one model — shared by both routes."""
    glb_version, model_url, materials = await material_service.list_materials(
        db, prod_uuid, variant_id
    )
    parts, _ = await part_service.list_parts(db, prod_uuid, user_id, variant_id=variant_id)
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
    return payload.model_dump(mode="json")


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
    # The product's original model — unchanged behaviour (ADR-014).
    return api_success(await _materials_payload(db, prod_uuid, current_user.id, None))


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
    current = await part_service.current_glb_version(db, part.product_id, part.variant_id)
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
    current = await part_service.current_glb_version(db, part.product_id, part.variant_id)
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
        parts=_public_parts(view.parts),
        variants=(
            [
                PublicModelVariant(
                    id=v.id,
                    name=v.name,
                    glb_url=v.glb_url,
                    usdz_url=v.usdz_url,
                    thumbnail_url=v.thumbnail_url,
                    is_default=v.is_default,
                    order_index=v.order_index,
                    width_m=v.width_m,
                    depth_m=v.depth_m,
                    height_m=v.height_m,
                    parts=_public_parts(v.parts),
                )
                for v in view.variants
            ]
            if view.variants
            else None
        ),
    )
    body = payload.model_dump(mode="json")
    if body.get("variants") is None:
        # Single-model product: the payload stays exactly as it was before
        # model variants existed (ADR-014).
        body.pop("variants", None)
    return api_success(body)


def _public_parts(parts) -> list[PublicProductPart]:
    return [
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
                    textures=[PublicOptionTexture.model_validate(t) for t in o.textures],
                )
                for o in p.options
            ],
        )
        for p in parts
    ]

