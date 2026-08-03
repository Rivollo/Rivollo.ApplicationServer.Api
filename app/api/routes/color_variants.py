"""Color configurator routes.

A colourway is created instantly (the recipe is a few hundred bytes) and its GLB
is baked in the background. Create/update return ``bake_status: "pending"``; the
client polls the list endpoint until it flips to ``"ready"`` and ``model_url``
appears.
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.api.deps import CurrentUser, DB, get_current_user
from app.schemas.color_variants import (
    ColorVariantCreate,
    ColorVariantReorder,
    ColorVariantUpdate,
)
from app.services.color_variant_service import color_variant_service
from app.services.variant_bake_service import variant_bake_service
from app.utils.envelopes import api_success

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["color-variants"],
    dependencies=[Depends(get_current_user)],
)


def _parse_uuid(raw: str, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {label} format")


# ---------- Model inspection ----------

@router.get("/products/{product_id}/materials", response_model=dict)
async def list_product_materials(
    product_id: str,
    current_user: CurrentUser,
    db: DB,
):
    """Colourable parts of the product's 3D model.

    Downloads and parses the GLB, so it is noticeably slower than the other
    endpoints. The portal calls it once when the Colour tab opens; day-to-day
    editing reads materials from the already-loaded viewer instead.
    """
    prod_uuid = _parse_uuid(product_id, "productId")
    parts = await color_variant_service.get_materials(db=db, product_id=prod_uuid)
    return api_success([p.model_dump() for p in parts])


# ---------- List ----------

@router.get("/products/{product_id}/color-variants", response_model=dict)
async def list_color_variants(
    product_id: str,
    current_user: CurrentUser,
    db: DB,
):
    prod_uuid = _parse_uuid(product_id, "productId")
    variants = await color_variant_service.list_variants(
        db=db,
        product_id=prod_uuid,
        user_id=current_user.id,
    )
    return api_success([v.model_dump(mode="json") for v in variants])


# ---------- Create ----------

@router.post(
    "/products/{product_id}/color-variants",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def create_color_variant(
    product_id: str,
    payload: ColorVariantCreate,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DB,
):
    """Save a colourway and queue its GLB bake."""
    prod_uuid = _parse_uuid(product_id, "productId")
    variant = await color_variant_service.create_variant(
        db=db,
        product_id=prod_uuid,
        user_id=current_user.id,
        payload=payload,
    )
    background_tasks.add_task(variant_bake_service.bake_variant, uuid.UUID(variant.id))
    return api_success(variant.model_dump(mode="json"))


# ---------- Update ----------

@router.patch("/color-variants/{variant_id}", response_model=dict)
async def update_color_variant(
    variant_id: str,
    payload: ColorVariantUpdate,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DB,
):
    var_uuid = _parse_uuid(variant_id, "variantId")
    variant, needs_rebake = await color_variant_service.update_variant(
        db=db,
        variant_id=var_uuid,
        user_id=current_user.id,
        payload=payload,
    )
    # Renaming or reordering doesn't change the bytes — only a colour change does.
    if needs_rebake:
        background_tasks.add_task(variant_bake_service.bake_variant, var_uuid)
    return api_success(variant.model_dump(mode="json"))


# ---------- Reorder ----------

@router.post("/products/{product_id}/color-variants/reorder", response_model=dict)
async def reorder_color_variants(
    product_id: str,
    payload: ColorVariantReorder,
    current_user: CurrentUser,
    db: DB,
):
    prod_uuid = _parse_uuid(product_id, "productId")
    variants = await color_variant_service.reorder_variants(
        db=db,
        product_id=prod_uuid,
        variant_ids=payload.variant_ids,
    )
    return api_success([v.model_dump(mode="json") for v in variants])


# ---------- Rebake ----------

@router.post("/color-variants/{variant_id}/rebake", response_model=dict)
async def rebake_color_variant(
    variant_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DB,
):
    """Retry a failed bake, or force one after the source model changed."""
    var_uuid = _parse_uuid(variant_id, "variantId")
    variant = await color_variant_service.get_variant(db=db, variant_id=var_uuid)
    background_tasks.add_task(variant_bake_service.bake_variant, var_uuid)
    return api_success({**variant.model_dump(mode="json"), "bake_status": "pending"})


# ---------- Delete ----------

@router.delete(
    "/color-variants/{variant_id}",
    response_model=dict,
    status_code=status.HTTP_200_OK,
)
async def delete_color_variant(
    variant_id: str,
    current_user: CurrentUser,
    db: DB,
):
    var_uuid = _parse_uuid(variant_id, "variantId")
    await color_variant_service.delete_variant(db=db, variant_id=var_uuid)
    return api_success({"message": "Colourway deleted successfully"})
