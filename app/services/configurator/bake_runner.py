"""Phase-1 in-process bake execution (ADR-007).

The runner owns the *workflow*; ``bake_service`` owns every database state
transition. Nothing here decides whether a bake should happen, what counts as
stale, or what the row looks like afterwards — it fetches, bakes, uploads, and
hands the results back.

    enqueue(option_id)          schedule, return immediately
      └─ run_bake(option_id)    own session, own semaphore slot
           mark_baking()                        ← bake_service
           MaterialService.extract_source_textures / fetch_image_bytes
           texture_baker.bake_option_textures   ← pure
           storage_service.upload_configurator_texture
           complete_bake() / fail_bake()        ← bake_service

Why one bake at a time per worker: a 4K texture decodes to tens of megabytes and
a multi-material part holds several at once. Running many in parallel is the
fastest way to OOM the pod, which is the same reason the colour-variant baker
uses a single-slot semaphore.

Why ``asyncio.create_task`` rather than FastAPI ``BackgroundTasks``: the seam
``bake_service`` calls is a plain coroutine with no request object in scope, and
a service has no business reaching for one. Both mechanisms are in-process and
both are lost on restart — which is exactly what the stale-bake sweep exists to
clean up.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from typing import Optional

from app.core.db import new_session
from app.database.configurator_repo import configurator_repository as repo
from app.schemas.configurator import StoredRecipe
from app.services.configurator import texture_baker
from app.services.configurator.bake_service import BakeService, StoredTexture
from app.services.configurator.material_service import material_service
from app.services.configurator.part_service import PartService
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

# One Configurator bake at a time per worker process.
_BAKE_SEMAPHORE = asyncio.Semaphore(1)

# Strong references to scheduled tasks. Without this the event loop is free to
# garbage-collect a task that nothing else holds, and the bake vanishes silently.
_scheduled: set[asyncio.Task] = set()

# Seller-facing fallback. The real exception goes to the log, never to the API.
GENERIC_FAILURE = "The texture bake failed. Please try again."
UNREADABLE_MODEL = (
    "This product's 3D model is stored in a location this service cannot read. "
    "Re-upload the model, then bake again."
)
NO_MODEL = "This product has no 3D model to bake from."
STALE_GLB = (
    "This part was created against an earlier version of the product's 3D "
    "model. Re-map its materials, then bake again."
)


class BakeFailure(Exception):
    """Carries a message that is safe to show a seller."""


# --------------------------------------------------------------------------- #
# The seam bake_service calls
# --------------------------------------------------------------------------- #
async def enqueue(option_id: uuid.UUID) -> None:
    """Schedule a bake and return. Never runs the bake inline.

    The caller is still inside a request with an open database session; the bake
    opens its own, because this one is closed the moment the response is sent.
    """
    task = asyncio.create_task(run_bake(option_id))
    _scheduled.add(task)
    task.add_done_callback(_scheduled.discard)
    logger.info("Bake scheduled for option %s", option_id)


def pending_task_count() -> int:
    """How many bakes this worker currently has scheduled or running."""
    return len(_scheduled)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
async def run_bake(option_id: uuid.UUID) -> None:
    """Execute one bake end to end. Never raises.

    The semaphore is taken OUTSIDE the session so a queued bake does not hold a
    database connection while it waits its turn.
    """
    async with _BAKE_SEMAPHORE:
        async with new_session() as db:
            await _run_bake_with_session(db, option_id)


async def _run_bake_with_session(db, option_id: uuid.UUID) -> None:
    recipe_hash: Optional[str] = None
    uploaded: list[str] = []

    try:
        # Load first, so recipe_hash is known before any check can fail. A
        # validation failure must still be RECORDED against the row — otherwise
        # it stays `pending` with no error and the sweep retries it forever.
        option = await repo.get_option_for_bake(db, option_id)
        if option is None:
            logger.warning("Bake requested for missing option %s", option_id)
            return
        recipe_hash = option.recipe_hash

        plan = await _build_plan(db, option)

        # Claim the row. bake_service re-checks the hash and refuses if another
        # worker got here first or the recipe moved on.
        if not await BakeService.mark_baking(db, option_id, recipe_hash):
            return

        # ---- no database work from here until completion -------------------
        sources = await _gather_sources(plan)

        baked = await asyncio.to_thread(
            texture_baker.bake_option_textures,
            sources=sources,
            recipe=plan.recipe,
        )

        stored: list[StoredTexture] = []
        for item in baked:
            url, blob_url = await _upload(plan, item)
            uploaded.append(url)
            stored.append(StoredTexture(texture=item, url=url, blob_url=blob_url))
        # ---- back to the database -----------------------------------------

        completion = await BakeService.complete_bake(
            db, option_id, recipe_hash, stored
        )

        if not completion.applied:
            # Superseded mid-bake. bake_service stored nothing and left the state
            # to the newer bake; these blobs are now unreferenced.
            await _purge(uploaded, reason="superseded bake")
            return

        await _purge(completion.orphaned_urls, reason="previous recipe")

    except BakeFailure as exc:
        await _record_failure(db, option_id, recipe_hash, str(exc))
        await _purge(uploaded, reason="failed bake")
    except Exception:  # noqa: BLE001 - a runner that raises leaves rows stuck
        logger.exception("Unexpected error baking option %s", option_id)
        await _record_failure(db, option_id, recipe_hash, GENERIC_FAILURE)
        await _purge(uploaded, reason="failed bake")


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
class _Plan:
    """Everything the expensive phase needs, read before any of it starts."""

    __slots__ = (
        "option_id",
        "product_id",
        "glb_version",
        "model_url",
        "material_indices",
        "recipe",
        "recipe_hash",
    )

    def __init__(
        self,
        option_id: uuid.UUID,
        product_id: uuid.UUID,
        glb_version: str,
        model_url: str,
        material_indices: list[int],
        recipe: StoredRecipe,
        recipe_hash: str,
    ):
        self.option_id = option_id
        self.product_id = product_id
        self.glb_version = glb_version
        self.model_url = model_url
        self.material_indices = material_indices
        self.recipe = recipe
        self.recipe_hash = recipe_hash


async def _build_plan(db, option) -> _Plan:
    """Read the rest of the state the bake needs. Raises BakeFailure if it cannot.

    Ownership is NOT re-checked here: it was established in
    ``BakeService.request_bake`` before this option was ever enqueued, and by now
    there is no authenticated user to check against.
    """
    option_id = option.id
    part = option.part
    if part is None:
        raise BakeFailure(GENERIC_FAILURE)

    try:
        recipe = StoredRecipe(**(option.recipe or {}))
    except Exception as exc:  # noqa: BLE001 - a bad stored recipe cannot be baked
        logger.error("Option %s has an unbakeable recipe: %s", option_id, exc)
        raise BakeFailure(
            "This option's colour recipe is no longer valid. Edit the option and "
            "save it again."
        ) from exc

    current = await PartService.current_glb_version(db, part.product_id, part.variant_id)
    if current is None:
        raise BakeFailure(NO_MODEL)
    if part.glb_version != current:
        raise BakeFailure(STALE_GLB)

    # The part's own model: the original GLB, or its model variant's (ADR-014).
    asset = await repo.get_model_mesh_asset(db, part.product_id, part.variant_id)
    if asset is None or not asset.image:
        raise BakeFailure(NO_MODEL)

    return _Plan(
        option_id=option.id,
        product_id=part.product_id,
        glb_version=part.glb_version,
        model_url=asset.image,
        material_indices=list(part.material_indices or []),
        recipe=recipe,
        recipe_hash=option.recipe_hash,
    )


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
async def _gather_sources(plan: _Plan) -> dict[int, Optional[bytes]]:
    """``material_index -> source bytes`` for every material the part owns.

    For an ``image`` recipe the seller's upload IS the appearance, so the GLB's
    own texture is never consulted — the same uploaded bytes go to every material
    in the part (data-model.md §6).
    """
    if plan.recipe.is_image:
        data = await _fetch_uploaded_image(plan.recipe.image_url)
        return {index: data for index in plan.material_indices}

    try:
        extracted = await material_service.extract_source_textures(plan.model_url)
    except RuntimeError as exc:
        raise BakeFailure(_model_failure_message(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not read GLB for option %s", plan.option_id)
        raise BakeFailure(
            "This product's 3D model could not be read. Re-upload it, then bake "
            "again."
        ) from exc

    # Absent index -> None -> texture_baker produces no file for that material,
    # which is the documented outcome for an untextured or factor material.
    return {
        index: (extracted[index][0] if index in extracted else None)
        for index in plan.material_indices
    }


async def _fetch_uploaded_image(image_url: Optional[str]) -> bytes:
    if not image_url:
        raise BakeFailure(
            "This option's uploaded image is missing. Edit the option and upload "
            "it again."
        )
    try:
        return await material_service.fetch_image_bytes(image_url)
    except RuntimeError as exc:
        raise BakeFailure(
            "The uploaded image for this option could not be read. Upload it "
            "again, then bake."
        ) from exc


def _model_failure_message(exc: RuntimeError) -> str:
    """Translate a storage resolution failure into something a seller can act on.

    The cross-account case is the one worth naming separately: it is not a
    transient error and re-baking will never fix it.
    """
    detail = str(exc)
    if "credentials" in detail or "cannot be read" in detail:
        logger.error("Canonical GLB is on an unreadable storage account: %s", detail)
        return UNREADABLE_MODEL
    logger.error("Canonical GLB could not be fetched: %s", detail)
    return "This product's 3D model could not be downloaded. Please try again."


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
async def _upload(
    plan: _Plan, item: texture_baker.BakedTexture
) -> tuple[str, Optional[str]]:
    """Store one baked texture. Returns (cdn_url, blob_url)."""

    def _work() -> tuple[str, str]:
        return storage_service.upload_configurator_texture(
            product_id=str(plan.product_id),
            glb_version=plan.glb_version,
            option_id=str(plan.option_id),
            material_index=item.material_index,
            recipe_hash=plan.recipe_hash,
            content_type=item.content_type,
            stream=io.BytesIO(item.data),
        )

    try:
        return await asyncio.to_thread(_work)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Could not store texture for option %s material %s",
            plan.option_id,
            item.material_index,
        )
        raise BakeFailure(
            "The generated texture could not be saved. Please try again."
        ) from exc


# --------------------------------------------------------------------------- #
# Cleanup and failure
# --------------------------------------------------------------------------- #
async def _purge(urls: list[str], *, reason: str) -> None:
    """Delete blobs nothing references any more. Best effort, never raises.

    A blob we cannot delete must not turn a completed bake into a failed one —
    the worst case is storage cost, and the URL is logged so it can be reclaimed.
    """
    for url in urls:
        if not url:
            continue
        try:
            deleted = await asyncio.to_thread(
                storage_service.delete_blob_by_cdn_url, url
            )
        except Exception:  # noqa: BLE001 - cleanup is best effort
            logger.warning("Could not delete %s blob %s", reason, url, exc_info=True)
            continue
        if not deleted:
            logger.warning("Orphaned %s blob was not deleted: %s", reason, url)


async def _record_failure(
    db, option_id: uuid.UUID, recipe_hash: Optional[str], message: str
) -> None:
    """Mark the bake failed, so no normal path can leave a row in `baking`."""
    if recipe_hash is None:
        # The option could not be loaded at all, so there is no row to mark.
        return
    try:
        await BakeService.fail_bake(db, option_id, recipe_hash, message)
    except Exception:  # noqa: BLE001 - last resort; the sweep is the backstop
        logger.exception(
            "Could not record bake failure for option %s; the stale-bake sweep "
            "will recover it",
            option_id,
        )
