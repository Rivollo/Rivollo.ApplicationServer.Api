"""Layout from photo: generate a model variant from one image (ADR-015).

    request   ownership · image is the seller's own upload · plan + credits
              -> row 'queued' -> credits charged -> generation_runner.enqueue()
    runner    'generating' -> fal -> candidate GLB uploaded -> 'ready' | 'failed'
    accept    'ready' -> ModelVariantService.create_variant(candidate bytes,
              source photo as thumbnail) -> 'accepted' + accepted_variant_id
    discard   -> 'discarded', candidate blob deleted

A candidate is never a model variant row. The variant is created by the
unchanged upload pipeline (ownership, Draco + re-check, unmapped asset row,
blob path, USDZ request), so every ADR-014 rule holds for generated variants
exactly as for uploaded ones.

Ownership: every seller-facing call resolves the generation through its own
product's ``created_by`` and answers 404, never 403 (ADR-008). The runner-side
methods (mark_generating, complete, fail, recover_stale) take no user: they run
after the request is gone, on a row whose ownership was checked when it was
created.

Transactions: this service owns them. The repository never commits.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.configurator_repo import configurator_repository as repo
from app.models.configurator import (
    GENERATION_STATUSES,
    ModelVariantGeneration,
    ProductModelVariant,
)
from app.services.configurator import generation_runner
from app.services.configurator.model_variant_service import (
    GLB_CONTENT_TYPE,
    PRODUCT_NOT_FOUND,
    CreatedVariant,
    ModelVariantService,
    UploadedFile,
)
from app.services.configurator.recipe import validate_image_url_ownership
from app.services.generation_estimate_service import generation_estimate_service
from app.services.generation_gate import authorize_generation, charge_generation
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

GENERATION_NOT_FOUND = "Generation not found"

INTERRUPTED = "Generation was interrupted. Please try again."
NOT_STARTED = "Generation could not be started. Please try again."

# Layout tiles are small; a phone photo is not.
THUMBNAIL_MAX_SIDE_PX = 1024
THUMBNAIL_JPEG_QUALITY = 85


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RequestedGeneration:
    generation: ModelVariantGeneration
    estimate: Optional[dict[str, Any]]


@dataclass(frozen=True)
class AcceptedGeneration:
    generation: ModelVariantGeneration
    # The variant the candidate became. None only when an earlier accept made
    # it and it has since been deleted.
    variant: Optional[ProductModelVariant]
    glb_url: Optional[str]
    created: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class SweepReport:
    interrupted: list[uuid.UUID] = field(default_factory=list)
    expired: list[uuid.UUID] = field(default_factory=list)

    @property
    def acted(self) -> bool:
        return bool(self.interrupted or self.expired)


class ModelVariantGenerationService:
    """Business rules for generating model variants from photos."""

    # ------------------------------------------------------------------ #
    # Request
    # ------------------------------------------------------------------ #
    @staticmethod
    async def request(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        name: str,
        image_url: str,
        model_key: Optional[str] = None,
        client_ref: Optional[str] = None,
        auto_accept: bool = False,
    ) -> RequestedGeneration:
        """Queue one generation and charge for it. Returns before any work starts."""
        ModelVariantService.require_enabled()
        clean_name = ModelVariantService._validate_name(name)

        product = await repo.get_owned_product(db, product_id, user_id)
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)

        ModelVariantGenerationService._validate_source_image(image_url, user_id)
        clean_ref = (client_ref or "").strip() or None

        spec = await authorize_generation(db, user_id, model_key)

        now = _utcnow()
        generation = ModelVariantGeneration(
            id=uuid.uuid4(),
            product_id=product_id,
            name=clean_name,
            source_image_url=image_url,
            model_key=spec.key,
            credit_cost=spec.credit_cost,
            status="queued",
            auto_accept=bool(auto_accept),
            client_ref=clean_ref,
            created_by=user_id,
            created_date=now,
        )
        repo.add(db, generation)
        await db.commit()

        # Charged only once the row exists, so a failed write never costs
        # credits. If charging fails the work is not started: the row is failed
        # rather than left queued for the sweep.
        try:
            await charge_generation(db, user_id, spec.credit_cost)
        except Exception:
            logger.exception("Could not charge for generation %s", generation.id)
            await db.rollback()
            await ModelVariantGenerationService.fail(db, generation.id, NOT_STARTED)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=NOT_STARTED
            )

        await generation_runner.enqueue(generation.id, spec)
        logger.info(
            "Generation %s queued for product %s (model=%s, cost=%d)",
            generation.id, product_id, spec.key, spec.credit_cost,
        )

        estimate: Optional[dict[str, Any]] = None
        try:
            estimate = (
                await generation_estimate_service.estimate(
                    db, spec.key, spec.baseline_estimate_seconds
                )
            ).to_payload()
        except Exception:  # noqa: BLE001 - an ETA is a nicety, never a failure
            logger.warning("Could not estimate generation %s", generation.id, exc_info=True)

        return RequestedGeneration(generation=generation, estimate=estimate)

    @staticmethod
    def _validate_source_image(image_url: Optional[str], user_id: uuid.UUID) -> None:
        """The image must be the seller's own upload. Never fetched to decide."""
        try:
            validate_image_url_ownership(image_url, user_id)
        except HTTPException as exc:
            if exc.status_code == status.HTTP_400_BAD_REQUEST:
                # Same rule as ADR-013; the field here is image_url, not recipe.image_url.
                detail = str(exc.detail).replace("recipe.image_url", "image_url")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
            raise

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    @staticmethod
    async def list_for_product(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        status_filter: Optional[str] = None,
        client_ref: Optional[str] = None,
    ) -> list[ModelVariantGeneration]:
        ModelVariantService.require_enabled()
        if status_filter is not None and status_filter not in GENERATION_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"status must be one of: {', '.join(GENERATION_STATUSES)}",
            )
        product = await repo.get_owned_product(db, product_id, user_id)
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)
        return await repo.list_generations(
            db, product_id, status=status_filter, client_ref=client_ref
        )

    @staticmethod
    async def get(
        db: AsyncSession, generation_id: uuid.UUID, user_id: uuid.UUID
    ) -> ModelVariantGeneration:
        ModelVariantService.require_enabled()
        generation = await repo.get_owned_generation(db, generation_id, user_id)
        if generation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=GENERATION_NOT_FOUND)
        return generation

    # ------------------------------------------------------------------ #
    # Discard
    # ------------------------------------------------------------------ #
    @staticmethod
    async def discard(
        db: AsyncSession, generation_id: uuid.UUID, user_id: uuid.UUID
    ) -> ModelVariantGeneration:
        """Throw a candidate away. Idempotent.

        Allowed while queued or generating too: the runner sees 'discarded' when
        it finishes and deletes what it made. Credits are not refunded.
        """
        ModelVariantService.require_enabled()
        generation = await repo.get_owned_generation(db, generation_id, user_id, for_update=True)
        if generation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=GENERATION_NOT_FOUND)
        if generation.status == "accepted":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This generation is already a model variant. Delete the variant instead.",
            )
        if generation.status == "discarded":
            return generation

        blobs = [generation.candidate_glb_url]
        now = _utcnow()
        generation.status = "discarded"
        generation.candidate_glb_url = None
        generation.candidate_glb_blob_url = None
        generation.updated_by = user_id
        generation.updated_date = now
        await db.commit()

        await ModelVariantGenerationService._purge_blobs(blobs)
        return generation

    # ------------------------------------------------------------------ #
    # Accept
    # ------------------------------------------------------------------ #
    @staticmethod
    async def accept(
        db: AsyncSession,
        generation_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        name: Optional[str] = None,
    ) -> AcceptedGeneration:
        """Turn a ready candidate into a model variant. Idempotent.

        Order matters, and is what makes a retry or a concurrent second accept
        unable to create a second variant:

          1. download the candidate and build the thumbnail (network, no lock)
          2. lock the generation row, re-check it is still 'ready'
          3. mark it 'accepted' and call create_variant, whose own commit
             persists both the variant and that status together
          4. record accepted_variant_id, then delete the candidate blob
        """
        ModelVariantService.require_enabled()
        generation = await repo.get_owned_generation(db, generation_id, user_id)
        if generation is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=GENERATION_NOT_FOUND)
        if generation.status == "accepted":
            return await ModelVariantGenerationService._already_accepted(db, generation, user_id)
        ModelVariantGenerationService._require_ready(generation)

        clean_name = ModelVariantService._validate_name(name if name is not None else generation.name)
        candidate_url = generation.candidate_glb_url
        glb_bytes = await ModelVariantGenerationService._download_candidate(candidate_url)
        thumbnail = await ModelVariantGenerationService._thumbnail_from_source(
            generation.source_image_url
        )

        locked = await repo.get_owned_generation(db, generation_id, user_id, for_update=True)
        if locked is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=GENERATION_NOT_FOUND)
        if locked.status == "accepted":
            await db.rollback()
            return await ModelVariantGenerationService._already_accepted(db, locked, user_id)
        ModelVariantGenerationService._require_ready(locked)

        now = _utcnow()
        locked.status = "accepted"
        locked.updated_by = user_id
        locked.updated_date = now
        try:
            created: CreatedVariant = await ModelVariantService.create_variant(
                db,
                locked.product_id,
                user_id,
                name=clean_name,
                glb=UploadedFile(filename="model.glb", content_type=GLB_CONTENT_TYPE, data=glb_bytes),
                thumbnail=thumbnail,
            )
        except HTTPException as exc:
            await db.rollback()
            # Put the in-memory row back too, rather than relying on the
            # rollback's attribute expiry: it is still a ready candidate.
            locked.status = "ready"
            if exc.status_code == status.HTTP_400_BAD_REQUEST:
                # The candidate itself is unusable (it failed the glTF parse).
                # Retrying cannot help, so say so on the row.
                await ModelVariantGenerationService._mark_unusable(db, generation_id, str(exc.detail))
            raise
        except Exception:
            await db.rollback()
            locked.status = "ready"
            raise

        locked.accepted_variant_id = created.variant.id
        locked.candidate_glb_url = None
        locked.candidate_glb_blob_url = None
        await db.commit()
        await ModelVariantGenerationService._purge_blobs([candidate_url])

        logger.info(
            "Generation %s accepted as model variant %s for product %s",
            generation_id, created.variant.id, locked.product_id,
        )
        return AcceptedGeneration(
            generation=locked,
            variant=created.variant,
            glb_url=created.glb_url,
            created=True,
            warnings=list(created.warnings),
        )

    @staticmethod
    def _require_ready(generation: ModelVariantGeneration) -> None:
        if generation.status != "ready":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Only a ready candidate can be accepted; this one is {generation.status}.",
            )

    @staticmethod
    async def _already_accepted(
        db: AsyncSession, generation: ModelVariantGeneration, user_id: uuid.UUID
    ) -> AcceptedGeneration:
        variant = None
        glb_url = None
        if generation.accepted_variant_id is not None:
            variant = await repo.get_owned_model_variant(db, generation.accepted_variant_id, user_id)
        if variant is not None and variant.glb_asset_id is not None:
            assets = await repo.get_assets_by_ids(db, [variant.glb_asset_id])
            asset = assets.get(variant.glb_asset_id)
            glb_url = asset.image if asset is not None else None
        return AcceptedGeneration(generation=generation, variant=variant, glb_url=glb_url, created=False)

    @staticmethod
    async def _mark_unusable(db: AsyncSession, generation_id: uuid.UUID, detail: str) -> None:
        try:
            row = await repo.get_generation(db, generation_id, for_update=True)
            if row is not None and row.status == "ready":
                row.status = "failed"
                row.error = f"The generated model could not be used: {detail}"[:500]
                row.completed_at = _utcnow()
                await db.commit()
        except Exception:  # noqa: BLE001 - the original error is what the caller sees
            logger.warning("Could not mark generation %s unusable", generation_id, exc_info=True)
            await db.rollback()

    @staticmethod
    async def _download_candidate(url: Optional[str]) -> bytes:
        if not url:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This candidate has no model file. Generate it again.",
            )
        try:
            content, _type, _name = await asyncio.to_thread(
                storage_service.download_upload_blob_bytes, url
            )
            return content
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not read candidate GLB %s", url)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The generated model could not be read. Please try again.",
            ) from exc

    @staticmethod
    async def _thumbnail_from_source(image_url: Optional[str]) -> Optional[UploadedFile]:
        """The source photo, shrunk to a JPEG layout-tile thumbnail. None on any problem.

        A missing thumbnail never blocks accepting a model: the seller can set
        one later with PUT /configurator/model-variants/{id}/thumbnail.
        """
        if not image_url:
            return None

        def _work() -> Optional[bytes]:
            content, _type, _name = storage_service.download_upload_blob_bytes(image_url)
            if not content or len(content) > settings.GENERATION_MAX_SOURCE_IMAGE_BYTES:
                return None
            with Image.open(io.BytesIO(content)) as image:
                image = image.convert("RGB")
                image.thumbnail((THUMBNAIL_MAX_SIDE_PX, THUMBNAIL_MAX_SIDE_PX))
                out = io.BytesIO()
                image.save(out, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY)
                return out.getvalue()

        try:
            data = await asyncio.to_thread(_work)
        except Exception:  # noqa: BLE001
            logger.warning("Could not build a thumbnail from %s", image_url, exc_info=True)
            return None
        if not data:
            return None
        return UploadedFile(filename="thumbnail.jpg", content_type="image/jpeg", data=data)

    # ------------------------------------------------------------------ #
    # Runner-side transitions (no user; ownership was checked at request)
    # ------------------------------------------------------------------ #
    @staticmethod
    async def mark_generating(
        db: AsyncSession, generation_id: uuid.UUID
    ) -> Optional[ModelVariantGeneration]:
        """Claim a queued row. None if it is gone or no longer queued."""
        generation = await repo.get_generation(db, generation_id, for_update=True)
        if generation is None or generation.status != "queued":
            await db.rollback()
            return None
        generation.status = "generating"
        generation.started_at = _utcnow()
        await db.commit()
        return generation

    @staticmethod
    async def complete(
        db: AsyncSession,
        generation_id: uuid.UUID,
        *,
        glb_url: str,
        glb_blob_url: Optional[str],
        size_bytes: int,
    ) -> bool:
        """Record a finished candidate. False when the row moved on (discarded, swept)."""
        generation = await repo.get_generation(db, generation_id, for_update=True)
        if generation is None or generation.status != "generating":
            await db.rollback()
            return False
        generation.status = "ready"
        generation.completed_at = _utcnow()
        generation.candidate_glb_url = glb_url
        generation.candidate_glb_blob_url = glb_blob_url
        generation.candidate_size_bytes = size_bytes
        generation.error = None
        await db.commit()
        return True

    @staticmethod
    async def fail(db: AsyncSession, generation_id: uuid.UUID, message: str) -> None:
        """Mark an in-flight row failed. Never raises; the sweep is the backstop."""
        try:
            generation = await repo.get_generation(db, generation_id, for_update=True)
            if generation is None or generation.status not in ("queued", "generating"):
                await db.rollback()
                return
            generation.status = "failed"
            generation.error = message[:500]
            generation.completed_at = _utcnow()
            await db.commit()
        except Exception:  # noqa: BLE001
            logger.exception("Could not record failure of generation %s", generation_id)
            try:
                await db.rollback()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    async def record_auto_accept_failure(
        db: AsyncSession, generation_id: uuid.UUID, detail: str
    ) -> None:
        """Leave the candidate 'ready' for a manual accept, with a note why."""
        try:
            generation = await repo.get_generation(db, generation_id, for_update=True)
            if generation is not None and generation.status == "ready":
                generation.error = f"Could not add it automatically: {detail}"[:500]
                await db.commit()
            else:
                await db.rollback()
        except Exception:  # noqa: BLE001
            logger.warning("Could not note auto-accept failure on %s", generation_id, exc_info=True)

    # ------------------------------------------------------------------ #
    # Sweep (startup + periodic, wired in app/main.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    async def recover_stale(db: AsyncSession, now: Optional[datetime] = None) -> SweepReport:
        """Fail generations that died with their replica; expire old candidates.

        No retry: the work was paid for, and re-running it silently could charge
        the fal account twice for one seller request. The seller retries.
        """
        now = now or _utcnow()
        report = SweepReport()

        stale_cutoff = now - timedelta(seconds=settings.GENERATION_STALE_AFTER_SECONDS)
        for generation in await repo.get_stale_generations(
            db, stale_cutoff, settings.GENERATION_SWEEP_BATCH
        ):
            generation.status = "failed"
            generation.error = INTERRUPTED
            generation.completed_at = now
            report.interrupted.append(generation.id)

        expiry_cutoff = now - timedelta(days=settings.GENERATION_CANDIDATE_TTL_DAYS)
        blobs: list[Optional[str]] = []
        for generation in await repo.get_expired_candidates(
            db, expiry_cutoff, settings.GENERATION_SWEEP_BATCH
        ):
            blobs.append(generation.candidate_glb_url)
            generation.status = "discarded"
            generation.candidate_glb_url = None
            generation.candidate_glb_blob_url = None
            generation.updated_date = now
            report.expired.append(generation.id)

        await db.commit()
        await ModelVariantGenerationService._purge_blobs(blobs)
        return report

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _purge_blobs(urls: list[Optional[str]]) -> None:
        for url in urls:
            if not url:
                continue
            try:
                await asyncio.to_thread(storage_service.delete_blob_by_cdn_url, url)
            except Exception:  # noqa: BLE001 - best effort; the user-prefix purge catches leftovers
                logger.warning("Could not delete candidate blob %s", url, exc_info=True)


model_variant_generation_service = ModelVariantGenerationService()
