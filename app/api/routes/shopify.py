"""Shopify integration routes (docs/shopify-integration/spec.md, ADR-016).

Called by the Rivollo Shopify app with ``Authorization: Bearer riv_live_...``
(an API key, docs/api_keys.md). Every route except /connect acts for the shop
the key is connected to; the shop never comes from the request body.

    POST   /integrations/shopify/connect                       bind this key to a shop      read
    GET    /integrations/shopify/connection                    the current binding          read
    POST   /integrations/shopify/uninstall                     app/uninstalled webhook      read
    POST   /integrations/shopify/shop/redact                   shop/redact webhook          write
    GET    /integrations/shopify/models                        models, credit cost, ETA     read
    POST   /integrations/shopify/products/sync                 upsert a product             write
    GET    /integrations/shopify/products                      synced products              read
    GET    /integrations/shopify/products/{id}                 full state (poll this)       read
    PUT    /integrations/shopify/products/{id}/options         option roles                 write
    POST   /integrations/shopify/products/{id}/glb             create the main GLB          convert
    POST   /integrations/shopify/products/{id}/layouts/{layout_id}/glb   layout GLB          convert
    POST   /integrations/shopify/products/{id}/generations/{gid}/accept  accept candidate    write
    DELETE /integrations/shopify/products/{id}/generations/{gid}         discard candidate   write
    DELETE /integrations/shopify/products/{id}                 unlink (products/delete)     write

    GET    /public/products/{product_id}/shopify               shopper payload (no auth)

``{id}`` is the NUMERIC Shopify product id (strip ``gid://shopify/Product/``).
A GID cannot be used in the path: the server decodes %2F back to "/" before
routing. Request BODIES accept either form.

Behind ENABLE_SHOPIFY_INTEGRATION (on by default): set it false and every route
answers 404 before authentication runs. Removing the feature is one
include_router line in app/main.py plus migration c9e5a3b1d8f6's downgrade.
Publishing is NOT here: it stays in Rivollo.Viewer.Api.
"""


import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status

from app.api.deps import DB, get_api_key_principal, require_api_key_scope
from app.models.api_key import SCOPE_CONVERT, SCOPE_READ, SCOPE_WRITE
from app.schemas.configurator import ModelVariantCreateResponse, ModelVariantResponse
from app.schemas.shopify import (
    ShopifyConnectionResponse,
    ShopifyConnectRequest,
    ShopifyLayoutGlbRequest,
    ShopifyMainGlbRequest,
    ShopifyOptionsRequest,
    ShopifyProductSummary,
    ShopifySyncRequest,
    ShopifySyncResponse,
    parse_shopify_id,
)
from app.services.api_key_service import ApiKeyPrincipal
from app.services.shopify.connection_service import ShopifyConnectionService, ShopifyContext
from app.services.shopify.glb_service import ShopifyGlbService, status_value
from app.services.shopify.sync_service import ShopifySyncService
from app.services.shopify.view_service import ShopifyViewService, generation_dict
from app.utils.envelopes import api_success


def _require_enabled() -> None:
    ShopifyConnectionService.require_enabled()


router = APIRouter(tags=["shopify"], dependencies=[Depends(_require_enabled)])
public_router = APIRouter(tags=["shopify"], dependencies=[Depends(_require_enabled)])


def _context(scope: str):
    """The caller's key (with ``scope``) and the shop it is connected to."""

    async def _dependency(
        principal: Annotated[ApiKeyPrincipal, Depends(require_api_key_scope(scope))],
        db: DB,
    ) -> ShopifyContext:
        return await ShopifyConnectionService.get_context(db, principal)

    return _dependency


ReadContext = Annotated[ShopifyContext, Depends(_context(SCOPE_READ))]
WriteContext = Annotated[ShopifyContext, Depends(_context(SCOPE_WRITE))]
ConvertContext = Annotated[ShopifyContext, Depends(_context(SCOPE_CONVERT))]


def _product_id(raw: str) -> int:
    try:
        return parse_shopify_id(raw, "Product")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Shopify product id.")


def _uuid(raw: str, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {label}.")


def _connection(connection) -> dict:
    return ShopifyConnectionResponse(
        id=connection.id,
        shop_domain=connection.shop_domain,
        api_key_id=connection.api_key_id,
        isactive=connection.isactive,
        connected_at=connection.connected_at,
        disconnected_at=connection.disconnected_at,
    ).model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Connection lifecycle
# --------------------------------------------------------------------------- #
@router.post("/integrations/shopify/connect", response_model=dict)
async def connect_shop(
    payload: ShopifyConnectRequest,
    principal: Annotated[ApiKeyPrincipal, Depends(require_api_key_scope(SCOPE_READ))],
    db: DB,
):
    """Bind this API key to a shop. Idempotent; moves the binding if needed."""
    connection = await ShopifyConnectionService.connect(db, principal, payload.shop_domain)
    return api_success(_connection(connection))


@router.get("/integrations/shopify/connection", response_model=dict)
async def get_connection(context: ReadContext):
    return api_success(_connection(context.connection))


@router.post("/integrations/shopify/uninstall", response_model=dict)
async def uninstall(
    principal: Annotated[ApiKeyPrincipal, Depends(get_api_key_principal)],
    db: DB,
):
    """Shopify app/uninstalled: unbind the shop. The key itself stays valid."""
    await ShopifyConnectionService.disconnect(db, principal)
    return api_success({"disconnected": True})


@router.post("/integrations/shopify/shop/redact", response_model=dict)
async def redact_shop(context: WriteContext, db: DB):
    """Shopify shop/redact: delete this shop's integration data. Rivollo products are kept."""
    deleted = await ShopifyConnectionService.redact_shop(db, context)
    return api_success({"shop_domain": context.shop_domain, "products_deleted": deleted})


@router.get("/integrations/shopify/models", response_model=dict)
async def list_models(context: ReadContext, db: DB):
    """The 3D models the app can offer, with credit cost and ETA.

    The same data as GET /ai/3d-models (which needs a portal login), for the
    app's model picker and the cost shown on its "Create 3D" buttons.
    """
    from app.integrations.fal import list_model_specs
    from app.services.generation_estimate_service import generation_estimate_service

    models = []
    for spec in await list_model_specs(db):
        estimate = await generation_estimate_service.estimate(db, spec.key, spec.baseline_estimate_seconds)
        models.append(
            {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "credit_cost": spec.credit_cost,
                "is_default": spec.is_default,
                "estimated_seconds": estimate.seconds,
                "estimated_time": estimate.display,
                "estimate_is_measured": estimate.is_measured,
                "free_plan_eligible": spec.free_plan_eligible,
            }
        )
    return api_success(models)


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #
@router.post("/integrations/shopify/products/sync", response_model=dict)
async def sync_product(payload: ShopifySyncRequest, context: WriteContext, db: DB, response: Response):
    """Upsert a Shopify product. The first sync creates a draft Rivollo product (201)."""
    shopify_product, product, created = await ShopifySyncService.sync(db, context, payload)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    body = ShopifySyncResponse(
        id=shopify_product.id,
        shopify_product_id=str(shopify_product.shopify_product_id),
        rivollo_product_id=product.id,
        rivollo_status=status_value(product),
        synced_at=shopify_product.synced_at,
        variants_synced=len(shopify_product.shopify_variants),
        created=created,
    )
    return api_success(body.model_dump(mode="json"))


@router.get("/integrations/shopify/products", response_model=dict)
async def list_products(context: ReadContext, db: DB):
    from app.database.shopify_repo import shopify_repository

    products = await shopify_repository.list_products(db, context.shop_domain, context.user_id)
    return api_success(
        [
            ShopifyProductSummary(
                id=p.id,
                shopify_product_id=str(p.shopify_product_id),
                title=p.title,
                rivollo_product_id=p.rivollo_product_id,
                synced_at=p.synced_at,
            ).model_dump(mode="json")
            for p in products
        ]
    )


@router.get("/integrations/shopify/products/{shopify_product_id}", response_model=dict)
async def get_product_state(shopify_product_id: str, context: ReadContext, db: DB):
    """Everything the app shows: Rivollo status, GLB URLs, viewer link, layouts, candidates."""
    shopify_product = await ShopifySyncService.get_product(db, context, _product_id(shopify_product_id))
    state = await ShopifyViewService.state(db, shopify_product, context.user_id)
    return api_success(state.model_dump(mode="json"))


@router.put("/integrations/shopify/products/{shopify_product_id}/options", response_model=dict)
async def set_options(
    shopify_product_id: str, payload: ShopifyOptionsRequest, context: WriteContext, db: DB
):
    shopify_product = await ShopifySyncService.set_options(
        db, context, _product_id(shopify_product_id), payload
    )
    state = await ShopifyViewService.state(db, shopify_product, context.user_id)
    return api_success(state.model_dump(mode="json"))


@router.delete("/integrations/shopify/products/{shopify_product_id}", response_model=dict)
async def unlink_product(shopify_product_id: str, context: WriteContext, db: DB):
    """Shopify products/delete: remove the link. The Rivollo product is kept."""
    await ShopifySyncService.unlink(db, context, _product_id(shopify_product_id))
    return api_success({"unlinked": True})


# --------------------------------------------------------------------------- #
# GLBs
# --------------------------------------------------------------------------- #
@router.post(
    "/integrations/shopify/products/{shopify_product_id}/glb",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_main_glb(
    shopify_product_id: str,
    payload: ShopifyMainGlbRequest,
    context: ConvertContext,
    db: DB,
    background_tasks: BackgroundTasks,
):
    """Generate the product's main 3D model from one of its Shopify images. Charges AI credits."""
    started = await ShopifyGlbService.start_main_glb(
        db,
        context,
        _product_id(shopify_product_id),
        image_url=payload.image_url,
        model_key=payload.model,
        retry=payload.retry,
        background_tasks=background_tasks,
    )
    return api_success(
        {
            "rivollo_product_id": str(started.product.id),
            "status": "queue",
            "estimate": started.estimate,
        }
    )


@router.post(
    "/integrations/shopify/products/{shopify_product_id}/layouts/{layout_id}/glb",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_layout_glb(
    shopify_product_id: str,
    layout_id: str,
    payload: ShopifyLayoutGlbRequest,
    context: ConvertContext,
    db: DB,
):
    """Generate a candidate 3D model for one layout value. Charges AI credits."""
    requested = await ShopifyGlbService.start_layout_glb(
        db,
        context,
        _product_id(shopify_product_id),
        _uuid(layout_id, "layout id"),
        image_url=payload.image_url,
        model_key=payload.model,
        auto_accept=payload.auto_accept,
    )
    body = generation_dict(requested.generation)
    body["estimate"] = requested.estimate
    return api_success(body)


@router.post(
    "/integrations/shopify/products/{shopify_product_id}/generations/{generation_id}/accept",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
)
async def accept_layout_candidate(
    shopify_product_id: str, generation_id: str, context: WriteContext, db: DB, response: Response
):
    accepted = await ShopifyGlbService.accept(
        db, context, _product_id(shopify_product_id), _uuid(generation_id, "generation id")
    )
    if not accepted.created:
        response.status_code = status.HTTP_200_OK
    variant = None
    if accepted.variant is not None:
        v = accepted.variant
        base = ModelVariantResponse(
            id=str(v.id),
            product_id=v.product_id,
            name=v.name,
            glb_url=accepted.glb_url,
            thumbnail_url=v.thumbnail_url,
            order_index=v.order_index,
            is_original=False,
            isactive=v.isactive,
            compression_status=v.compression_status,
            compression_error=v.compression_error,
            original_size_bytes=v.original_size_bytes,
            compressed_size_bytes=v.compressed_size_bytes,
            width_m=v.width_m,
            depth_m=v.depth_m,
            height_m=v.height_m,
            created_at=v.created_date,
        )
        variant = (
            ModelVariantCreateResponse(**base.model_dump(), warnings=accepted.warnings)
            if accepted.created
            else base
        ).model_dump(mode="json")
    return api_success({"generation": generation_dict(accepted.generation), "model_variant": variant})


@router.delete(
    "/integrations/shopify/products/{shopify_product_id}/generations/{generation_id}",
    response_model=dict,
)
async def discard_layout_candidate(shopify_product_id: str, generation_id: str, context: WriteContext, db: DB):
    generation = await ShopifyGlbService.discard(
        db, context, _product_id(shopify_product_id), _uuid(generation_id, "generation id")
    )
    return api_success(generation_dict(generation))


# --------------------------------------------------------------------------- #
# Public shopper payload (Phase 3)
# --------------------------------------------------------------------------- #
@public_router.get("/public/products/{product_id}/shopify", response_model=dict)
async def get_public_shopify_payload(product_id: str, db: DB):
    """Price, availability and add-to-cart links per layout + option, for the viewer.

    No auth, like /public/products/{id}/configurator. 404 unless the product is
    published and linked to a Shopify product on a live connection. Never
    carries inventory counts, SKUs, Shopify GIDs, connection or user ids.
    """
    payload = await ShopifyViewService.public_payload(db, _uuid(product_id, "product id"))
    return api_success(payload.model_dump(mode="json"))
