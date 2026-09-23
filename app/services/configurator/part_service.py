"""Product Part business rules.

Owns the invariants that keep a product's configuration coherent:

  * a part belongs to a product the CALLER owns, resolved from the part's own
    foreign key — never from a client-supplied product_id
  * every material index exists on its model's current GLB
  * no material index is claimed by two active parts of the same model
  * a part is pinned to the GLB it was authored against

A "model" is the product's original GLB (``variant_id`` NULL) or one of its
live extra model variants (ADR-014). The product-level routes pass no
``variant_id`` and behave exactly as before model variants existed.

Ownership failures raise 404, never 403: a 403 confirms the resource exists and
belongs to someone else, which is an enumeration oracle over every seller's
catalogue (ADR-008).

Transactions: this service owns them. Repositories never commit.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.configurator_repo import configurator_repository as repo
from app.models.configurator import ProductPart
from app.models.models import Product
from app.schemas.configurator import ProductPartCreate, ProductPartUpdate
from app.services.configurator.material_service import MeshContext, material_service

logger = logging.getLogger(__name__)

PRODUCT_NOT_FOUND = "Product not found"
PART_NOT_FOUND = "Part not found"
MODEL_VARIANT_NOT_FOUND = "Model variant not found"



def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "part"


class PartService:
    """Business rules for Product Parts."""

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    @staticmethod
    async def list_parts(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        variant_id: Optional[uuid.UUID] = None,
    ) -> tuple[list[ProductPart], Optional[str]]:
        """Every part of one model, plus that model's current glb_version.

        The version is returned so the caller can compute ``glb_stale`` per part
        without re-resolving the mesh. It is None when the model has no GLB
        yet — a product can legitimately have no GLB and therefore no parts.
        """
        await PartService._require_owned_product(db, product_id, user_id)
        await PartService.require_model(db, product_id, variant_id, user_id)
        parts = await repo.get_parts_for_product(db, product_id, variant_id=variant_id)

        asset = await repo.get_model_mesh_asset(db, product_id, variant_id)
        current_version = (
            material_service.build_glb_version(asset.id)
            if asset is not None and asset.image
            else None
        )
        return parts, current_version

    @staticmethod
    async def get_part(
        db: AsyncSession,
        part_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ProductPart:
        return await PartService.require_owned_part(db, part_id, user_id)

    @staticmethod
    async def require_owned_product_for_read(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Product:
        """Public ownership gate for read-only endpoints that are not part CRUD.

        The materials endpoint needs it: inspection downloads the product's GLB,
        so ownership has to be settled before that work starts, or an
        unauthorised caller can make the server fetch another seller's model.
        """
        return await PartService._require_owned_product(db, product_id, user_id)

    @staticmethod
    async def current_glb_version(
        db: AsyncSession,
        product_id: uuid.UUID,
        variant_id: Optional[uuid.UUID] = None,
    ) -> Optional[str]:
        """One model's current glb_version, or None when it has no GLB.

        Cheap — one indexed lookup, no GLB download. Used to compute `glb_stale`
        on a single part without paying for inspection.
        """
        asset = await repo.get_model_mesh_asset(db, product_id, variant_id)
        if asset is None or not asset.image:
            return None
        return material_service.build_glb_version(asset.id)

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #
    @staticmethod
    async def create_part(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        payload: ProductPartCreate,
        *,
        variant_id: Optional[uuid.UUID] = None,
    ) -> ProductPart:
        # 1. Ownership WITHOUT a lock, so an unauthorised caller never reaches
        #    the expensive step below. A variant must be a live variant of THIS
        #    product — never trusted from the path alone.
        await PartService._require_owned_product(db, product_id, user_id)
        await PartService.require_model(db, product_id, variant_id, user_id)

        # 2. Inspect the GLB before taking the lock. Inspection downloads tens
        #    of megabytes; holding a row lock across that would serialise every
        #    write to the product behind a network transfer.
        mesh = await material_service.get_mesh_context(db, product_id, variant_id)
        PartService._validate_material_indices(payload.material_indices, mesh)

        # 3. Now the short critical section: re-read the product FOR UPDATE, so
        #    the sibling overlap check below cannot race a concurrent write
        #    claiming the same material index (ADR-012 — there is no database
        #    constraint backing this rule).
        await PartService._require_owned_product(db, product_id, user_id, lock=True)

        siblings = await repo.get_sibling_parts(db, product_id, variant_id=variant_id)
        PartService._reject_material_overlap(payload.material_indices, siblings)

        slug = await PartService._unique_slug(db, product_id, payload.name)
        order_index = (
            payload.order_index
            if payload.order_index is not None
            else await repo.get_next_part_order_index(db, product_id)
        )

        part = ProductPart(
            product_id=product_id,
            variant_id=variant_id,
            name=payload.name,
            slug=slug,
            material_indices=list(payload.material_indices),
            material_type=payload.material_type,
            order_index=order_index,
            shopper_selectable=payload.shopper_selectable,
            glb_version=mesh.glb_version,
            isactive=True,
            created_by=user_id,
        )
        repo.add(db, part)
        await db.commit()
        await db.refresh(part)
        return part

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    @staticmethod
    async def update_part(
        db: AsyncSession,
        part_id: uuid.UUID,
        user_id: uuid.UUID,
        payload: ProductPartUpdate,
    ) -> tuple[ProductPart, list[uuid.UUID]]:
        """Update a part. Returns (part, invalidated_option_ids).

        Changing ``material_indices`` invalidates every option's bake: the set
        of textures an option must produce is derived from the part's materials,
        so an option baked against the old set is incomplete against the new
        one. Those options are reset to ``pending`` and their ids returned.
        """
        part = await PartService.require_owned_part(db, part_id, user_id)
        invalidated: list[uuid.UUID] = []

        materials_changed = (
            payload.material_indices is not None
            and sorted(payload.material_indices) != sorted(part.material_indices or [])
        )

        if materials_changed:
            assert payload.material_indices is not None
            mesh = await material_service.get_mesh_context(
                db, part.product_id, part.variant_id
            )
            PartService.require_current_glb(part, mesh)
            PartService._validate_material_indices(payload.material_indices, mesh)

            await PartService._require_owned_product(
                db, part.product_id, user_id, lock=True
            )
            siblings = await repo.get_sibling_parts(
                db, part.product_id, variant_id=part.variant_id, exclude_id=part.id
            )
            PartService._reject_material_overlap(payload.material_indices, siblings)

            part.material_indices = list(payload.material_indices)
            invalidated = await PartService._invalidate_option_bakes(db, part)

        if payload.name is not None and payload.name != part.name:
            part.name = payload.name
            part.slug = await PartService._unique_slug(
                db, part.product_id, payload.name, exclude_id=part.id
            )

        if payload.material_type is not None:
            part.material_type = payload.material_type
        if payload.order_index is not None:
            part.order_index = payload.order_index
        if payload.shopper_selectable is not None:
            part.shopper_selectable = payload.shopper_selectable
        if payload.isactive is not None:
            part.isactive = payload.isactive

        part.updated_by = user_id
        await db.commit()
        await db.refresh(part)
        return part, invalidated

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #
    @staticmethod
    async def delete_part(
        db: AsyncSession,
        part_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Delete a part and everything under it.

        Texture blobs are purged BEFORE the rows, because the rows carry the
        only URLs that can find them — the database has no reach into Azure
        Blob Storage, and `ON DELETE CASCADE` does not delete files.
        """
        part = await PartService.require_owned_part(db, part_id, user_id)

        # Imported here rather than at module scope: option_service imports this
        # module for its own ownership helpers, and a top-level import would be
        # circular.
        from app.services.configurator.option_service import OptionService

        for option in part.options or []:
            await OptionService.purge_option_blobs(db, option)

        await repo.delete(db, part)
        await db.commit()

    # ------------------------------------------------------------------ #
    # Ownership helpers — used by OptionService too
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _require_owned_product(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        lock: bool = False,
    ) -> Product:
        product = await repo.get_owned_product(db, product_id, user_id, for_update=lock)
        if product is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND
            )
        return product

    @staticmethod
    async def require_model(
        db: AsyncSession,
        product_id: uuid.UUID,
        variant_id: Optional[uuid.UUID],
        user_id: uuid.UUID,
    ) -> None:
        """``variant_id`` is None (the original model) or a live variant of
        ``product_id`` owned by ``user_id``. Anything else is 404 (ADR-008)."""
        if variant_id is None:
            return
        variant = await repo.get_owned_model_variant(db, variant_id, user_id)
        if variant is None or variant.product_id != product_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=MODEL_VARIANT_NOT_FOUND
            )

    @staticmethod
    async def require_owned_part(
        db: AsyncSession,
        part_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ProductPart:
        part = await repo.get_owned_part(db, part_id, user_id)
        if part is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=PART_NOT_FOUND
            )
        return part

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_material_indices(indices: list[int], mesh: MeshContext) -> None:
        """Every index must exist on the product's CURRENT model."""
        for index in indices:
            if not mesh.is_valid_index(index):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Material index {index} does not exist on this model "
                        f"({mesh.material_count} materials)."
                    ),
                )

    @staticmethod
    def _reject_material_overlap(
        indices: list[int],
        siblings: list[ProductPart],
    ) -> None:
        """No material index may belong to two active parts of one model.

        Two parts claiming material 3 would let two options paint the same mesh
        with conflicting textures, and the viewer would show whichever loaded
        last. Enforced here rather than by a constraint (ADR-012); this is also
        the only place that can name the conflicting part in the error.
        """
        wanted = set(indices)
        for sibling in siblings:
            clash = wanted.intersection(sibling.material_indices or [])
            if clash:
                index = sorted(clash)[0]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Material index {index} already belongs to part "
                        f"'{sibling.name}'."
                    ),
                )

    @staticmethod
    def require_current_glb(part: ProductPart, mesh: MeshContext) -> None:
        """A part authored against a superseded model must be re-mapped first.

        Its material indices may no longer mean what the seller chose — the
        "Seat" could silently become the legs.
        """
        if part.glb_version != mesh.glb_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This part was created against an earlier version of the "
                    "product's 3D model. Re-map its materials before editing it."
                ),
            )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _invalidate_option_bakes(
        db: AsyncSession,
        part: ProductPart,
    ) -> list[uuid.UUID]:
        invalidated: list[uuid.UUID] = []
        for option in part.options or []:
            if option.bake_status != "pending":
                option.bake_status = "pending"
                option.bake_error = None
                option.bake_started_at = None
                option.bake_completed_at = None
            invalidated.append(option.id)
        return invalidated

    @staticmethod
    async def _unique_slug(
        db: AsyncSession,
        product_id: uuid.UUID,
        name: str,
        *,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> str:
        base = _slugify(name)
        slug = base
        suffix = 2
        while await repo.part_slug_exists(db, product_id, slug, exclude_id=exclude_id):
            slug = f"{base}-{suffix}"
            suffix += 1
        return slug

    @staticmethod
    def default_option_id(part: ProductPart) -> Optional[uuid.UUID]:
        """Computed, never stored — ADR-011. The API exposes this field."""
        return next(
            (o.id for o in (part.options or []) if o.is_default),
            None,
        )

    @staticmethod
    def is_glb_stale(part: ProductPart, current_glb_version: Optional[str]) -> bool:
        """Computed, never stored."""
        if current_glb_version is None:
            return True
        return part.glb_version != current_glb_version


part_service = PartService()
