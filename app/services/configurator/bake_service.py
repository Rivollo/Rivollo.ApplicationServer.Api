"""Bake orchestration and state transitions. No pixels, no I/O.

This is the contract the rest of the application holds onto while the *execution*
backend changes underneath it (ADR-007). Everything expensive — downloading the
GLB, recolouring, uploading — belongs to the runner. What lives here is the part
that must be correct regardless of how the work is scheduled: who may ask for a
bake, whether one is already in flight, which result is still wanted by the time
it arrives, and what the row looks like afterwards.

Division of labour with the runner:

    bake_service   decides, authorises, records state
    bake_runner    fetches, bakes (texture_baker), uploads (storage_service)

That split is why ``complete_bake`` takes ``StoredTexture`` rather than
``BakedTexture``: the runner has already uploaded, so it is the only party that
knows the CDN URL. This service never touches storage, and never deletes a blob —
it reports orphans and lets the caller reclaim them.

``recipe_hash`` is the identity of a bake throughout. A result whose hash no
longer matches the option is discarded rather than stored: the recipe changed
while the bake was running, a newer bake is already queued for it, and writing
the old pixels would show the seller a colour they have stopped asking for.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.configurator_repo import configurator_repository as repo
from app.models.configurator import PartOption, PartOptionTexture
from app.services.configurator.option_service import OptionService
from app.services.configurator.part_service import PartService
from app.services.configurator.texture_baker import BakedTexture

logger = logging.getLogger(__name__)

# The four states tbl_part_options.ck_options_bake_status permits.
PENDING = "pending"
BAKING = "baking"
COMPLETED = "completed"
FAILED = "failed"

# States from which a new bake must NOT be started, because one already is.
IN_FLIGHT = frozenset({PENDING, BAKING})

# bake_error is seller-facing and the column is TEXT; the colour-variant service
# truncates to the same width (variant_bake_service._mark_failed).
MAX_BAKE_ERROR_CHARS = 500

INTERRUPTED_BAKE_ERROR = "Bake interrupted; please retry."

NO_MODEL_DETAIL = (
    "This product has no 3D model yet. Upload a model before baking."
)
STALE_GLB_DETAIL = (
    "This part was created against an earlier version of the product's 3D "
    "model. Re-map its materials before baking."
)


# --------------------------------------------------------------------------- #
# Data carried across the seam
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StoredTexture:
    """A baked texture the runner has already uploaded.

    ``BakedTexture`` alone is not enough to persist a row: it carries pixels and
    dimensions but no URL, and this service does not upload. The runner pairs
    each baked texture with where it put it.
    """

    texture: BakedTexture
    url: str
    blob_url: Optional[str] = None


@dataclass(frozen=True)
class BakeTicket:
    """What the route needs to answer a bake request.

    ``already_current`` true means nothing was enqueued because the stored result
    already matches the option's recipe — the caller should still report success.
    """

    option_id: uuid.UUID
    bake_status: str
    recipe_hash: str
    already_current: bool
    enqueued: bool


@dataclass(frozen=True)
class BakeStatusView:
    """Current bake state plus derived progress."""

    option_id: uuid.UUID
    bake_status: str
    bake_error: Optional[str]
    bake_started_at: Optional[datetime]
    bake_completed_at: Optional[datetime]
    bake_attempts: int
    recipe_hash: str
    textures_total: int
    textures_done: int
    textures: list[PartOptionTexture] = field(default_factory=list)


@dataclass(frozen=True)
class BakeCompletion:
    """Outcome of handing a finished bake back to the service.

    ``applied`` false means the result was discarded as superseded — not an
    error. ``orphaned_urls`` are blobs the service unlinked from the database
    and that the CALLER should delete; this service does not touch storage.
    """

    applied: bool
    orphaned_urls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryReport:
    """What one stale-bake sweep did."""

    examined: int = 0
    requeued: list[uuid.UUID] = field(default_factory=list)
    exhausted: list[uuid.UUID] = field(default_factory=list)
    skipped: list[uuid.UUID] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        return bool(self.requeued or self.exhausted)


class BakeNotAvailable(RuntimeError):
    """No execution backend is wired up."""


# --------------------------------------------------------------------------- #
# The runner seam (ADR-007)
# --------------------------------------------------------------------------- #
async def _enqueue_via_runner(option_id: uuid.UUID) -> None:
    """Default seam: hand the option to ``bake_runner.enqueue``.

    Imported lazily and by name so this module stays importable before the runner
    exists, and so swapping BackgroundTasks for a queue later touches one file.
    Tests and DI replace ``enqueue_bake`` wholesale.
    """
    try:
        from app.services.configurator.bake_runner import enqueue  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - until the runner lands
        raise BakeNotAvailable(
            "No bake runner is installed; app.services.configurator.bake_runner "
            "does not exist yet."
        ) from exc
    await enqueue(option_id)


# Module-level so it can be replaced without a class or a container.
enqueue_bake = _enqueue_via_runner


class BakeService:
    """Orchestration only. Expensive work lives in the runner."""

    # ------------------------------------------------------------------ #
    # Request
    # ------------------------------------------------------------------ #
    @staticmethod
    async def request_bake(
        db: AsyncSession,
        option_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        force: bool = False,
    ) -> BakeTicket:
        """Authorise and record a bake request. Returns immediately.

        Nothing expensive happens here — no GLB download, no pixel work. The
        glb_version check uses the cheap indexed lookup rather than inspecting
        the model, precisely so the request path never waits on storage.

        ``force=True`` re-bakes a result that is already current. It deliberately
        does NOT bypass the in-flight guard (that would run two bakes for one
        option) and does NOT bypass the stale-GLB check (that would bake against
        materials the seller has not re-mapped). It exists for one case the
        api-spec names: the blob was deleted out of band and the database still
        believes it is there.
        """
        option = await OptionService.get_option(db, option_id, user_id)
        part = option.part
        if part is None:
            # get_owned_option selectinloads the part; absence means the row lost
            # its parent, which the FK makes impossible.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Option is not attached to a part.",
            )

        if not option.isactive:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A hidden option cannot be baked. Publish it first.",
            )
        if not part.isactive:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This option's part is hidden. Publish the part first.",
            )

        await BakeService._require_current_glb_version(db, part)

        recipe_hash = option.recipe_hash

        # Already in flight — pending or baking. Return the existing ticket
        # rather than starting a second bake, even under force.
        #
        # NOTE: a row stuck in `baking` after a replica restart stays stuck here.
        # Recovering it needs the stale-bake sweep (ADR-007), which is not built
        # yet; until it is, this branch is deliberately conservative.
        if option.bake_status in IN_FLIGHT:
            return BakeTicket(
                option_id=option.id,
                bake_status=option.bake_status,
                recipe_hash=recipe_hash,
                already_current=False,
                enqueued=False,
            )

        if (
            option.bake_status == COMPLETED
            and not force
            and await BakeService._stored_result_is_current(db, option)
        ):
            return BakeTicket(
                option_id=option.id,
                bake_status=COMPLETED,
                recipe_hash=recipe_hash,
                already_current=True,
                enqueued=False,
            )

        await BakeService._mark_pending(db, option)
        enqueued = await BakeService._enqueue(option.id)

        return BakeTicket(
            option_id=option.id,
            bake_status=PENDING,
            recipe_hash=recipe_hash,
            already_current=False,
            enqueued=enqueued,
        )

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    @staticmethod
    async def get_bake_status(
        db: AsyncSession,
        option_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> BakeStatusView:
        """Current state plus derived progress. Reads only.

        ``textures_total`` is the part's material count — an UPPER BOUND that
        api-spec §8 explicitly permits, because knowing the exact figure means
        inspecting the GLB to see which materials have a base-colour image at
        all. ``textures_done`` is exact: rows at the option's current hash.
        """
        option = await OptionService.get_option(db, option_id, user_id)
        part = option.part

        textures_done = await repo.count_current_textures(
            db, option.id, option.recipe_hash
        )
        textures_total = len(part.material_indices or []) if part is not None else 0

        return BakeStatusView(
            option_id=option.id,
            bake_status=option.bake_status,
            bake_error=option.bake_error,
            bake_started_at=option.bake_started_at,
            bake_completed_at=option.bake_completed_at,
            bake_attempts=option.bake_attempts,
            recipe_hash=option.recipe_hash,
            textures_total=textures_total,
            textures_done=textures_done,
            # Stale rows are filtered out — never served as the wrong colour.
            textures=OptionService.current_textures(option),
        )

    # ------------------------------------------------------------------ #
    # Runner-facing transitions — no user, no ownership
    # ------------------------------------------------------------------ #
    @staticmethod
    async def mark_baking(
        db: AsyncSession,
        option_id: uuid.UUID,
        recipe_hash: str,
    ) -> bool:
        """Claim an option for baking. False means "do not bake".

        Called by the runner when it picks the work up. ``bake_started_at`` is
        written in the SAME transaction that sets ``baking``, which is what makes
        a future stale-bake sweep possible at all — the colour-variant service's
        omission of exactly this field is why its rows can stick forever.

        ``bake_attempts`` increments here rather than at enqueue time: this is
        the point a real attempt begins, which is what baking.md §4.5 means by
        "per real attempt". An enqueue that never runs should not burn a retry.
        """
        option = await repo.get_option_for_bake(db, option_id)
        if option is None:
            logger.warning("Bake claimed for missing option %s", option_id)
            return False
        if option.recipe_hash != recipe_hash:
            logger.info(
                "Declining to bake superseded recipe for option %s", option_id
            )
            return False
        if option.bake_status == BAKING:
            logger.info("Option %s is already baking", option_id)
            return False

        option.bake_status = BAKING
        option.bake_started_at = datetime.now(timezone.utc)
        option.bake_completed_at = None
        option.bake_error = None
        option.bake_attempts = (option.bake_attempts or 0) + 1
        await db.commit()
        return True

    @staticmethod
    async def complete_bake(
        db: AsyncSession,
        option_id: uuid.UUID,
        recipe_hash: str,
        stored_textures: Sequence[StoredTexture],
    ) -> BakeCompletion:
        """Persist a finished bake and mark the option completed.

        Discards the result when the option's recipe moved on while the bake was
        running: a newer bake is already queued for the new recipe, and storing
        these pixels would serve a colour the seller has abandoned.

        An empty ``stored_textures`` is valid, not a failure — an option whose
        every material resolves to ``factor`` produces no files at all
        (data-model.md §6).
        """
        option = await repo.get_option_for_bake(db, option_id)
        if option is None:
            logger.warning("Completion for missing option %s", option_id)
            return BakeCompletion(applied=False)

        if option.recipe_hash != recipe_hash:
            logger.info(
                "Discarding superseded bake result for option %s", option_id
            )
            return BakeCompletion(applied=False)

        for item in stored_textures:
            BakeService._upsert_texture(db, option, item)

        orphaned = await BakeService._unlink_stale_textures(db, option)

        option.bake_status = COMPLETED
        option.bake_error = None
        option.bake_completed_at = datetime.now(timezone.utc)

        # Deliberately NOT made the part's default. A part with no default shows
        # the model's Original appearance; only an explicit PATCH picks a
        # starting option (api-spec §7.4).

        await db.commit()

        logger.info(
            "Bake completed for option %s: %d texture(s), %d orphaned",
            option_id,
            len(stored_textures),
            len(orphaned),
        )
        return BakeCompletion(applied=True, orphaned_urls=orphaned)

    @staticmethod
    async def fail_bake(
        db: AsyncSession,
        option_id: uuid.UUID,
        recipe_hash: str,
        message: str,
    ) -> bool:
        """Record a failed bake. False means the row was already superseded.

        Writes no texture rows: a half-written set is worse than none, because
        the staleness rule would serve the half that matched. ``recipe_hash`` is
        left untouched so a retry targets the same identity, and so the seller's
        recipe is not silently rewritten by a failure.
        """
        option = await repo.get_option_for_bake(db, option_id)
        if option is None:
            logger.warning("Failure recorded for missing option %s", option_id)
            return False

        if option.recipe_hash != recipe_hash:
            # The recipe changed; a newer bake owns this row now. Marking it
            # failed would report an error about work nobody is waiting for.
            logger.info(
                "Not recording failure against superseded recipe for option %s",
                option_id,
            )
            return False

        option.bake_status = FAILED
        option.bake_error = (message or "Bake failed.")[:MAX_BAKE_ERROR_CHARS]
        option.bake_completed_at = None
        await db.commit()
        logger.error("Bake failed for option %s: %s", option_id, option.bake_error)
        return True

    # ------------------------------------------------------------------ #
    # Stale-bake recovery
    # ------------------------------------------------------------------ #
    @staticmethod
    async def recover_stale_bakes(db: AsyncSession) -> RecoveryReport:
        """Reclaim bakes a recycled worker abandoned. Never raises.

        A bake commits ``baking`` and then spends tens of seconds on network and
        CPU. Kill the replica in that window and the row sits in ``baking``
        forever, which the UI renders as a permanent spinner. This is what turns
        that into progress.

        Two outcomes per row, and which one depends on ``bake_attempts``:

          * under the automatic cap -> back to ``pending`` and re-enqueued;
          * at or over it -> marked ``failed`` with a retryable message, because
            automatically retrying a deterministic failure forever burns CPU and
            never succeeds (baking.md §4.5). An explicit seller-requested bake is
            NOT subject to the cap.

        Safe to run concurrently on several replicas: the claim is one guarded
        UPDATE, so only one sweep can move any given row.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=settings.CONFIGURATOR_BAKE_STALE_AFTER_SECONDS
        )
        try:
            stale = await repo.get_stale_baking_options(
                db, cutoff, settings.CONFIGURATOR_BAKE_SWEEP_BATCH
            )
        except Exception:  # noqa: BLE001 - a sweep must never take the app down
            logger.exception("Stale-bake sweep could not read candidates")
            return RecoveryReport()

        requeued: list[uuid.UUID] = []
        exhausted: list[uuid.UUID] = []
        skipped: list[uuid.UUID] = []

        for option in stale:
            try:
                if (option.bake_attempts or 0) >= (
                    settings.CONFIGURATOR_MAX_AUTOMATIC_BAKE_ATTEMPTS
                ):
                    if await BakeService.fail_bake(
                        db, option.id, option.recipe_hash, INTERRUPTED_BAKE_ERROR
                    ):
                        exhausted.append(option.id)
                        logger.warning(
                            "Option %s exhausted automatic bake attempts (%s); "
                            "marked failed for the seller to retry",
                            option.id,
                            option.bake_attempts,
                        )
                    else:
                        skipped.append(option.id)
                    continue

                claimed = await repo.claim_stale_bake(
                    db, option.id, option.recipe_hash, cutoff
                )
                if not claimed:
                    # Another replica got there first, or the recipe changed and a
                    # newer bake owns the row.
                    skipped.append(option.id)
                    continue
                await db.commit()

                await BakeService._enqueue(option.id)
                requeued.append(option.id)
                logger.info(
                    "Recovered stale bake for option %s (attempt %s)",
                    option.id,
                    (option.bake_attempts or 0) + 1,
                )
            except Exception:  # noqa: BLE001 - one bad row must not stop the sweep
                logger.exception("Stale-bake recovery failed for option %s", option.id)
                skipped.append(option.id)

        return RecoveryReport(
            examined=len(stale),
            requeued=requeued,
            exhausted=exhausted,
            skipped=skipped,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _require_current_glb_version(db: AsyncSession, part) -> str:
        """Cheap glb_version check — one indexed lookup, no GLB download."""
        current = await PartService.current_glb_version(db, part.product_id, part.variant_id)
        if current is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=NO_MODEL_DETAIL
            )
        if part.glb_version != current:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=STALE_GLB_DETAIL
            )
        return current

    @staticmethod
    async def _stored_result_is_current(
        db: AsyncSession, option: PartOption
    ) -> bool:
        """True when every stored texture was baked from the current recipe.

        Zero rows counts as current: an all-``factor`` option legitimately has
        no textures, and re-baking it forever would be absurd.
        """
        for texture in await repo.get_textures_for_option(db, option.id):
            if texture.recipe_hash != option.recipe_hash:
                return False
        return True

    @staticmethod
    async def _mark_pending(db: AsyncSession, option: PartOption) -> None:
        option.bake_status = PENDING
        option.bake_error = None
        option.bake_started_at = None
        option.bake_completed_at = None
        await db.commit()

    @staticmethod
    async def _enqueue(option_id: uuid.UUID) -> bool:
        """Hand the option to the runner. False when none is installed.

        A missing runner is not fatal: the row is already `pending`, which is
        durable intent that a runner or sweep can act on later. Failing the
        request instead would leave the seller unable to record the request at
        all.
        """
        try:
            await enqueue_bake(option_id)
            return True
        except BakeNotAvailable:
            logger.warning(
                "Option %s is pending but no bake runner is installed; it will "
                "not be processed until one is.",
                option_id,
            )
            return False

    @staticmethod
    def _upsert_texture(
        db: AsyncSession, option: PartOption, item: StoredTexture
    ) -> None:
        """One row per (option, material_index) — uq_option_texture_material."""
        baked = item.texture
        existing = next(
            (
                t
                for t in (option.textures or [])
                if t.material_index == baked.material_index
            ),
            None,
        )
        if existing is None:
            repo.add(
                db,
                PartOptionTexture(
                    option_id=option.id,
                    material_index=baked.material_index,
                    url=item.url,
                    blob_url=item.blob_url,
                    content_type=baked.content_type,
                    width=baked.width,
                    height=baked.height,
                    size_bytes=baked.size_bytes,
                    recipe_hash=option.recipe_hash,
                ),
            )
            return

        existing.url = item.url
        existing.blob_url = item.blob_url
        existing.content_type = baked.content_type
        existing.width = baked.width
        existing.height = baked.height
        existing.size_bytes = baked.size_bytes
        existing.recipe_hash = option.recipe_hash

    @staticmethod
    async def _unlink_stale_textures(
        db: AsyncSession, option: PartOption
    ) -> list[str]:
        """Delete rows left over from a previous recipe. Returns their URLs.

        The blobs are NOT deleted here — this service does not touch storage.
        Their paths are content-addressed by the old recipe_hash, so nothing will
        ever point at them again; the caller reclaims them.
        """
        orphaned: list[str] = []
        for texture in await repo.get_textures_for_option(db, option.id):
            if texture.recipe_hash == option.recipe_hash:
                continue
            if texture.url:
                orphaned.append(texture.url)
            await repo.delete(db, texture)
        return orphaned


bake_service = BakeService()
