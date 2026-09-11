"""Part Option business rules.

One option model carries both kinds of appearance, discriminated by
``recipe.method`` (ADR-013). Everything below that reads "recipe" applies to a
generated recolour and an uploaded texture alike; the only branches are the
three the specification actually calls for — image_url validation, the swatch
default, and which fields feed the hash.

Owns:
  * ownership resolved option -> part -> product, never a client product_id
  * recipe semantics, including the server-side image_url namespace check
  * ``auto`` resolved to a concrete method BEFORE the row is written
  * recipe_hash, and the re-bake decision that hangs off it
  * the default-option rules, backed by ux_part_options_one_default

Transactions: this service owns them. Repositories never commit.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.configurator_repo import configurator_repository as repo
from app.models.models import PartOption, PartOptionTexture, ProductPart
from app.schemas.configurator import (
    MAX_OPTIONS_PER_PART,
    PartOptionCreate,
    PartOptionUpdate,
    Recipe,
    StoredRecipe,
)
from app.services.configurator.material_service import MeshContext, material_service
from app.services.configurator.part_service import PartService
from app.services.configurator.recipe import (
    compute_recipe_hash,
    validate_image_url_ownership,
)
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

OPTION_NOT_FOUND = "Option not found"

# An option must be baked before it can be the look a shopper loads first.
_DEFAULTABLE_BAKE_STATUS = "completed"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "option"


class OptionService:
    """Business rules for Part Options."""

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    @staticmethod
    async def list_options(
        db: AsyncSession,
        part_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[PartOption]:
        await PartService.require_owned_part(db, part_id, user_id)
        return await repo.get_options_for_part(db, part_id)

    @staticmethod
    async def get_option(
        db: AsyncSession,
        option_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> PartOption:
        return await OptionService._require_owned_option(db, option_id, user_id)

    @staticmethod
    def current_textures(option: PartOption) -> list[PartOptionTexture]:
        """Only textures baked from the option's CURRENT recipe.

        A texture whose hash no longer matches was baked from a superseded
        recipe. Hiding it is deliberate: serving it would show the wrong colour
        while the re-bake is in flight, which is worse than showing nothing.
        """
        return [
            texture
            for texture in (option.textures or [])
            if texture.recipe_hash == option.recipe_hash
        ]

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #
    @staticmethod
    async def create_option(
        db: AsyncSession,
        part_id: uuid.UUID,
        user_id: uuid.UUID,
        payload: PartOptionCreate,
    ) -> PartOption:
        part = await PartService.require_owned_part(db, part_id, user_id)

        existing = await repo.count_options_for_part(db, part_id)
        if existing >= MAX_OPTIONS_PER_PART:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A part can have at most {MAX_OPTIONS_PER_PART} options.",
            )

        # No set_as_default here by design: a new option is always `pending`, and
        # is_default is only valid on a completed one. The part gets its default
        # from the first bake that completes — see promote_default_if_absent and
        # api-spec.md 7.4.
        mesh = await material_service.get_mesh_context(db, part.product_id)
        PartService.require_current_glb(part, mesh)

        stored_recipe = await OptionService._prepare_recipe(
            payload.recipe, part=part, mesh=mesh, user_id=user_id
        )

        slug = await OptionService._unique_slug(db, part_id, payload.name)
        order_index = (
            payload.order_index
            if payload.order_index is not None
            else await repo.get_next_option_order_index(db, part_id)
        )

        option = PartOption(
            part_id=part_id,
            name=payload.name,
            slug=slug,
            swatch_hex=payload.resolved_swatch_hex(),
            recipe=stored_recipe.model_dump(mode="json"),
            recipe_hash=compute_recipe_hash(stored_recipe, part.glb_version),
            order_index=order_index,
            is_default=False,
            isactive=True,
            bake_status="pending",
            bake_attempts=0,
            created_by=user_id,
        )
        repo.add(db, option)
        await db.commit()
        await db.refresh(option)

        # Committed first: the row is durable intent even if scheduling fails.
        await OptionService.schedule_bake(option.id)
        return option

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    @staticmethod
    async def update_option(
        db: AsyncSession,
        option_id: uuid.UUID,
        user_id: uuid.UUID,
        payload: PartOptionUpdate,
    ) -> tuple[PartOption, bool]:
        """Update an option. Returns (option, needs_rebake).

        Only a change that alters the output bytes triggers a re-bake. A rename
        or a reorder must not: re-baking on every edit is how a seller nudging a
        name ends up waiting a minute for nothing.
        """
        option = await OptionService._require_owned_option(db, option_id, user_id)
        part = option.part
        needs_rebake = False

        if payload.recipe is not None:
            mesh = await material_service.get_mesh_context(db, part.product_id)
            PartService.require_current_glb(part, mesh)

            # Stored-state rule: switching an EXISTING option to an image recipe
            # needs a swatch, and only here can we see whether the stored row
            # already has a usable one. The schema catches the case where the
            # request itself supplies neither.
            if payload.recipe.is_image and payload.swatch_hex is None:
                if not option.swatch_hex:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            "swatch_hex must be supplied when changing "
                            "recipe.method to 'image'."
                        ),
                    )

            stored_recipe = await OptionService._prepare_recipe(
                payload.recipe, part=part, mesh=mesh, user_id=user_id
            )
            new_hash = compute_recipe_hash(stored_recipe, part.glb_version)

            if new_hash != option.recipe_hash:
                option.recipe = stored_recipe.model_dump(mode="json")
                option.recipe_hash = new_hash
                option.bake_status = "pending"
                option.bake_error = None
                option.bake_started_at = None
                option.bake_completed_at = None
                needs_rebake = True

        if payload.name is not None and payload.name != option.name:
            option.name = payload.name
            option.slug = await OptionService._unique_slug(
                db, option.part_id, payload.name, exclude_id=option.id
            )

        if payload.swatch_hex is not None:
            option.swatch_hex = payload.swatch_hex

        if payload.order_index is not None:
            option.order_index = payload.order_index

        if payload.isactive is not None:
            if option.is_default and not payload.isactive:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "The default option cannot be hidden. Set another "
                        "option as default first."
                    ),
                )
            option.isactive = payload.isactive

        # Only `true` is in the contract (api-spec.md section 7). `false` is
        # ignored rather than errored: there is no documented operation that
        # removes a default without naming its replacement, and inventing one
        # here would be a rule nobody specified.
        if payload.set_as_default is True:
            await OptionService._promote_to_default(db, option)

        option.updated_by = user_id
        await db.commit()
        await db.refresh(option)

        # Only a real output change re-bakes; a rename or reorder must not.
        if needs_rebake:
            await OptionService.schedule_bake(option.id)
        return option, needs_rebake

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #
    @staticmethod
    async def delete_option(
        db: AsyncSession,
        option_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Delete an option, purging its blobs first and promoting a new default."""
        option = await OptionService._require_owned_option(db, option_id, user_id)
        was_default = option.is_default
        part_id = option.part_id

        await OptionService.purge_option_blobs(db, option)
        await repo.delete(db, option)
        await db.flush()

        if was_default:
            replacement = await repo.get_default_promotion_candidate(
                db, part_id, exclude_id=option_id
            )
            if replacement is not None:
                replacement.is_default = True

        await db.commit()

    @staticmethod
    async def purge_option_blobs(db: AsyncSession, option: PartOption) -> None:
        """Delete an option's texture blobs. Call BEFORE deleting the rows.

        The rows carry the only URLs that can find these files; `ON DELETE
        CASCADE` removes the rows and leaves the blobs paying rent forever.
        Failures are logged, not raised — a blob we cannot delete must not block
        the seller from deleting their option.
        """
        for texture in option.textures or []:
            if not texture.url:
                continue
            try:
                await asyncio.to_thread(
                    storage_service.delete_blob_by_cdn_url, texture.url
                )
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                logger.warning(
                    "Could not delete texture blob for option %s material %s",
                    option.id,
                    texture.material_index,
                    exc_info=True,
                )

    # ------------------------------------------------------------------ #
    # Recipe preparation
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _prepare_recipe(
        recipe: Recipe,
        *,
        part: ProductPart,
        mesh: MeshContext,
        user_id: uuid.UUID,
    ) -> StoredRecipe:
        """Validate and resolve an incoming recipe into one safe to persist."""
        part_materials = set(part.material_indices or [])

        for override in recipe.overrides:
            if override.material_index not in part_materials:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Material index {override.material_index} is not part "
                        f"of '{part.name}'."
                    ),
                )

        if recipe.is_image:
            # 🔴 Security-critical. The baker dereferences this URL server-side.
            validate_image_url_ownership(recipe.image_url, user_id)
            return StoredRecipe(
                version=recipe.version,
                method="image",
                image_url=recipe.image_url,
            )

        resolved = recipe.model_copy(deep=True)
        resolved.method = OptionService._resolve_method(
            recipe.method, part.material_indices or [], mesh
        )
        for override in resolved.overrides:
            override.method = OptionService._resolve_method(
                override.method, [override.material_index], mesh
            )

        # StoredRecipe's own validator is the backstop: if anything above left
        # an "auto" behind, this raises rather than persisting it — the failure
        # mode the colour-variant implementation has today.
        return StoredRecipe(**resolved.model_dump())

    @staticmethod
    def _resolve_method(
        method: str,
        material_indices: list[int],
        mesh: MeshContext,
    ) -> str:
        """Turn "auto" into a concrete method, so preview and bake cannot diverge.

        Resolved from the model's own albedo at save time.

        **Ambiguity is an error, not a guess.** When a part's materials suggest
        different methods there is no specified rule for choosing between them,
        and inventing one — first wins, majority wins — would silently give the
        seller a treatment they did not ask for on some of the part's materials.
        The write is rejected and the seller re-sends a concrete method.
        """
        if method != "auto":
            return method

        missing = [i for i in material_indices if i not in mesh.suggested_methods]
        if missing:
            # Indices are validated against material_count before this runs, so
            # a gap means inspection disagrees with itself rather than that the
            # client sent something wrong.
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "The product's 3D model could not be read well enough to "
                    "resolve 'auto'. Specify an explicit method."
                ),
            )

        suggestions = {mesh.suggested_methods[i] for i in material_indices}
        if len(suggestions) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "'auto' is ambiguous for this part: its materials suggest "
                    f"{', '.join(sorted(suggestions))}. Specify an explicit "
                    "method instead."
                ),
            )
        return suggestions.pop()

    @staticmethod
    async def schedule_bake(option_id: uuid.UUID) -> bool:
        """Hand a newly pending option to the bake runner. False if none is installed.

        api-spec.md §7 says creating an option "sets bake_status: pending and
        enqueues a bake" — and it has to happen here, because
        ``BakeService.request_bake`` correctly refuses to enqueue anything already
        `pending`. Without this, a new option would sit pending forever and an
        explicit bake request would be turned away as "already in flight".

        The seam is imported lazily and through the module, not the name:
        ``bake_service`` imports this class, so a module-level import here would
        be circular, and binding the attribute at call time is what lets tests and
        DI replace the runner.
        """
        from app.services.configurator import bake_service as _bake_service

        try:
            await _bake_service.enqueue_bake(option_id)
            return True
        except _bake_service.BakeNotAvailable:
            logger.warning(
                "Option %s is pending but no bake runner is installed; it will "
                "not be processed until one is.",
                option_id,
            )
            return False

    # ------------------------------------------------------------------ #
    # Defaults
    # ------------------------------------------------------------------ #
    @staticmethod
    async def promote_default_if_absent(
        db: AsyncSession,
        option: PartOption,
    ) -> bool:
        """First-baked-wins: give the part a default if it has none.

        Called by the bake pipeline when an option reaches ``completed``. This is
        what lets a part acquire a working default without the seller making a
        second API call, and it is why ``set_as_default`` is not accepted on
        create (api-spec.md 7.4).

        **Never replaces an existing default.** Promotion fills a vacancy; it
        does not compete for an occupied slot. A seller who wants a different
        default asks for one explicitly via PATCH.

        Returns True if this call made the option the default. Does not commit —
        the caller owns the transaction.
        """
        if option.is_default:
            return False
        if not option.isactive or option.bake_status != _DEFAULTABLE_BAKE_STATUS:
            return False

        existing = await repo.get_default_option(db, option.part_id)
        if existing is not None:
            return False

        option.is_default = True
        return True

    @staticmethod
    async def _promote_to_default(db: AsyncSession, option: PartOption) -> None:
        if not option.isactive:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A hidden option cannot be the default.",
            )
        if option.bake_status != _DEFAULTABLE_BAKE_STATUS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "An option can become the default only after its bake "
                    f"completes (currently '{option.bake_status}')."
                ),
            )
        # Cleared as a bulk UPDATE before setting the new one, so the partial
        # unique index never sees two defaults mid-flush.
        await repo.clear_default(db, option.part_id, keep_id=option.id)
        option.is_default = True

    # ------------------------------------------------------------------ #
    # Ownership
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _require_owned_option(
        db: AsyncSession,
        option_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> PartOption:
        option = await repo.get_owned_option(db, option_id, user_id)
        if option is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=OPTION_NOT_FOUND
            )
        return option

    @staticmethod
    async def _unique_slug(
        db: AsyncSession,
        part_id: uuid.UUID,
        name: str,
        *,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> str:
        base = _slugify(name)
        slug = base
        suffix = 2
        while await repo.option_slug_exists(db, part_id, slug, exclude_id=exclude_id):
            slug = f"{base}-{suffix}"
            suffix += 1
        return slug


option_service = OptionService()
