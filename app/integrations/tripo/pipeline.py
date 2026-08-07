"""Two-stage Tripo pipeline: segmented geometry, then texture.

A mesh that is both segmented AND textured cannot be produced in one Tripo call
— ``generate_parts`` refuses to run alongside texturing. So this runs:

    stage 1  /generation/image-to-model   generate_parts=true, texture=false
    stage 2  /models/texture              input = stage 1's task_id

The stages are chained by task id, so the intermediate mesh never passes
through our storage.

Progress from both stages is mapped onto a single 0-100 scale for the caller.
Geometry is the slower half, so it owns the larger share; a naive 50/50 split
would make the bar stall at the midpoint.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from app.integrations.tripo import tasks
from app.integrations.tripo.client import TripoError, tripo_client

logger = logging.getLogger(__name__)

# Share of the overall progress bar owned by each stage.
_GEOMETRY_SHARE = 65
_TEXTURE_SHARE = 100 - _GEOMETRY_SHARE

# (overall_progress_0_to_100, human_stage_label) -> awaitable
PipelineProgressCallback = Callable[[int, str], Awaitable[None]]


@dataclass
class PartsGenerationResult:
    """Outcome of a full segmented-and-textured generation."""

    glb_bytes: bytes
    content_type: str
    model_url: str
    geometry_task_id: str
    texture_task_id: str
    # Tripo bills per task; both are recorded so the real cost is knowable.
    geometry_credits: Optional[int] = None
    texture_credits: Optional[int] = None
    duration_seconds: float = 0.0

    @property
    def total_credits(self) -> Optional[int]:
        if self.geometry_credits is None and self.texture_credits is None:
            return None
        return (self.geometry_credits or 0) + (self.texture_credits or 0)


class TripoPartsPipeline:
    """Runs the geometry -> texture sequence as one logical operation."""

    @staticmethod
    async def generate(
        *,
        image_url: str,
        on_progress: Optional[PipelineProgressCallback] = None,
        geometry_task_id: Optional[str] = None,
    ) -> PartsGenerationResult:
        """Produce a segmented, textured GLB from a product photo.

        ``geometry_task_id`` resumes a run whose geometry already succeeded but
        whose texturing failed. Stage 1 is the expensive half and is already
        paid for at that point, so retrying should not repeat it.

        Raises :class:`TripoError` on any failure, with the task id attached so
        the caller can persist it and retry stage 2 later.
        """
        started = time.perf_counter()

        async def report(overall: int, label: str) -> None:
            if on_progress:
                await on_progress(max(0, min(100, overall)), label)

        # ---- Stage 1: segmented geometry --------------------------------
        geometry_credits: Optional[int] = None

        if geometry_task_id:
            logger.info(
                "Tripo parts: reusing existing geometry task %s", geometry_task_id
            )
            await report(_GEOMETRY_SHARE, "Geometry ready")
        else:
            await report(0, "Starting geometry")
            geometry_task_id = await tripo_client.submit(
                tasks.IMAGE_TO_MODEL_PATH,
                tasks.build_parts_geometry_request(image_url),
            )

            async def geometry_progress(progress: int, _status: str) -> None:
                await report(
                    int(progress * _GEOMETRY_SHARE / 100), "Generating parts"
                )

            geometry = await tripo_client.wait(
                geometry_task_id, on_progress=geometry_progress
            )
            geometry_credits = geometry.credits_consumed
            logger.info(
                "Tripo parts: geometry done  task_id=%s  credits=%s",
                geometry_task_id, geometry_credits,
            )

        # ---- Stage 2: texture -------------------------------------------
        await report(_GEOMETRY_SHARE, "Texturing parts")

        texture_task_id = await tripo_client.submit(
            tasks.TEXTURE_PATH,
            tasks.build_texture_request(geometry_task_id, image_url),
        )

        async def texture_progress(progress: int, _status: str) -> None:
            await report(
                _GEOMETRY_SHARE + int(progress * _TEXTURE_SHARE / 100),
                "Texturing parts",
            )

        texture = await tripo_client.wait(
            texture_task_id, on_progress=texture_progress
        )

        model_url = texture.model_url
        if not model_url:
            raise TripoError(
                "Tripo texture task succeeded but returned no model_url",
                texture_task_id,
            )

        # ---- Download -----------------------------------------------------
        glb_bytes, content_type = await tripo_client.download(model_url)
        await report(100, "Complete")

        elapsed = time.perf_counter() - started
        logger.info(
            "Tripo parts complete  geometry=%s  texture=%s  %.1fs  %.1f MB",
            geometry_task_id, texture_task_id, elapsed, len(glb_bytes) / 1e6,
        )

        return PartsGenerationResult(
            glb_bytes=glb_bytes,
            content_type=content_type,
            model_url=model_url,
            geometry_task_id=geometry_task_id,
            texture_task_id=texture_task_id,
            geometry_credits=geometry_credits,
            texture_credits=texture.credits_consumed,
            duration_seconds=elapsed,
        )


tripo_parts_pipeline = TripoPartsPipeline()
