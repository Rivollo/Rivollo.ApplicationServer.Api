"""In-process execution of model-variant generations (ADR-015, ADR-007 pattern).

The runner owns the WORKFLOW; ModelVariantGenerationService owns every
database state transition.

    enqueue(generation_id, spec)       schedule, return immediately
      └─ run(generation_id, spec)       own session, own semaphore slot
           mark_generating()                          ← service
           fal_queue_client.generate_3d(source image) ← unchanged fal client
           inspect_glb(bytes)                          ← reject unreadable output
           storage_service.upload_model_variant_file   ← candidate blob
           complete() / fail()                         ← service
           accept()  (only when auto_accept)           ← service

``spec`` is passed in memory, as the product pipeline does
(ProductService._run_fal_3d_generation_background), so a registry edit between
request and run cannot change what the seller paid for. After a restart the
spec is gone with the task; the stale-generation sweep fails the row rather
than silently re-running paid work.

Like bake_runner, this uses asyncio.create_task rather than FastAPI
BackgroundTasks: the seam is a plain coroutine with no request in scope. Both
are lost on restart, which is what the sweep is for. Keep execution behind
enqueue() so it can move to a worker later without touching the API.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid

from app.core.config import settings
from app.core.db import new_session
from app.integrations.fal.queue_client import fal_queue_client
from app.integrations.fal.registry import FalModelSpec
from app.services.configurator.glb_inspection import InvalidGlbError, inspect_glb
from app.services.generation_estimate_service import generation_estimate_service
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

GENERIC_FAILURE = "The 3D model could not be generated from this image. Please try again."
UNREADABLE_OUTPUT = "The generated model could not be read. Please try again or use another image."
STORAGE_FAILURE = "The generated model could not be saved. Please try again."

_SEMAPHORE = asyncio.Semaphore(max(1, settings.GENERATION_CONCURRENCY))

# Strong references: an unreferenced task can be garbage-collected mid-run.
_scheduled: set[asyncio.Task] = set()


class GenerationFailure(Exception):
    """Carries a message that is safe to show a seller."""


async def enqueue(generation_id: uuid.UUID, spec: FalModelSpec) -> None:
    """Schedule a generation and return. Never runs it inline."""
    task = asyncio.create_task(run(generation_id, spec))
    _scheduled.add(task)
    task.add_done_callback(_scheduled.discard)
    logger.info("Generation %s scheduled (model=%s)", generation_id, spec.key)


def pending_task_count() -> int:
    return len(_scheduled)


async def run(generation_id: uuid.UUID, spec: FalModelSpec) -> None:
    """Execute one generation end to end. Never raises.

    The semaphore is taken outside the session so a queued run does not hold a
    database connection while it waits.
    """
    async with _SEMAPHORE:
        async with new_session() as db:
            await run_with_session(db, generation_id, spec)


async def run_with_session(db, generation_id: uuid.UUID, spec: FalModelSpec) -> None:
    from app.services.configurator.model_variant_generation_service import (
        ModelVariantGenerationService as service,
    )

    generation = await service.mark_generating(db, generation_id)
    if generation is None:
        logger.info("Generation %s no longer queued; skipping", generation_id)
        return

    started = time.perf_counter()
    uploaded_url = None
    try:
        if generation.created_by is None:
            raise GenerationFailure(GENERIC_FAILURE)

        response = await fal_queue_client.generate_3d(
            spec=spec,
            product_id=generation.product_id,
            image_url=generation.source_image_url,
        )
        if not response.success or not response.glb_bytes:
            logger.error(
                "fal %s failed for generation %s (request_id=%s): %s",
                spec.key, generation_id, response.request_id, response.error,
            )
            raise GenerationFailure(GENERIC_FAILURE)
        glb_bytes = response.glb_bytes

        try:
            await asyncio.to_thread(inspect_glb, glb_bytes)
        except InvalidGlbError as exc:
            logger.error("Generation %s produced an unreadable GLB: %s", generation_id, exc)
            raise GenerationFailure(UNREADABLE_OUTPUT) from exc

        try:
            uploaded_url, blob_url = await asyncio.to_thread(
                storage_service.upload_model_variant_file,
                user_id=str(generation.created_by),
                product_id=str(generation.product_id),
                variant_id=str(generation.id),
                filename="candidate.glb",
                content_type=response.glb_content_type or "model/gltf-binary",
                stream=io.BytesIO(glb_bytes),
            )
        except Exception as exc:
            logger.exception("Could not store candidate for generation %s", generation_id)
            raise GenerationFailure(STORAGE_FAILURE) from exc

        applied = await service.complete(
            db, generation_id, glb_url=uploaded_url, glb_blob_url=blob_url, size_bytes=len(glb_bytes)
        )
        if not applied:
            # Discarded (or swept) while fal was working: nothing references
            # the blob we just wrote.
            logger.info("Generation %s moved on while running; dropping its output", generation_id)
            await _purge(uploaded_url)
            return

        await _record_duration(db, spec.key, time.perf_counter() - started, succeeded=True)
        logger.info("Generation %s ready (%d bytes)", generation_id, len(glb_bytes))

        if generation.auto_accept:
            await _auto_accept(db, generation_id, generation.created_by)

    except GenerationFailure as exc:
        await service.fail(db, generation_id, str(exc))
        await _record_duration(db, spec.key, time.perf_counter() - started, succeeded=False)
        await _purge(uploaded_url)
    except Exception:  # noqa: BLE001 - a runner that raises leaves rows stuck
        logger.exception("Unexpected error in generation %s", generation_id)
        await service.fail(db, generation_id, GENERIC_FAILURE)
        await _purge(uploaded_url)


async def _auto_accept(db, generation_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Accept as the requesting seller. On failure the candidate stays 'ready'."""
    from fastapi import HTTPException

    from app.services.configurator.model_variant_generation_service import (
        ModelVariantGenerationService as service,
    )

    try:
        await service.accept(db, generation_id, user_id)
    except HTTPException as exc:
        logger.warning("Auto-accept of generation %s refused: %s", generation_id, exc.detail)
        await service.record_auto_accept_failure(db, generation_id, str(exc.detail))
    except Exception:  # noqa: BLE001
        logger.exception("Auto-accept of generation %s failed", generation_id)
        await service.record_auto_accept_failure(
            db, generation_id, "an unexpected error occurred. Accept it manually."
        )


async def _record_duration(db, model_key: str, seconds: float, *, succeeded: bool) -> None:
    """Feed the ETA median, exactly as the product pipeline does. Never raises."""
    try:
        await generation_estimate_service.record(db, model_key, seconds, succeeded=succeeded)
    except Exception:  # noqa: BLE001
        logger.warning("Could not record generation duration", exc_info=True)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def _purge(url) -> None:
    if not url:
        return
    try:
        await asyncio.to_thread(storage_service.delete_blob_by_cdn_url, url)
    except Exception:  # noqa: BLE001
        logger.warning("Could not delete orphaned candidate blob %s", url, exc_info=True)
