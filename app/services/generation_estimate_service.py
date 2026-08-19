"""Estimated generation time for image-to-3D models.

**fal.ai does not provide an ETA.** Verified against the live queue API: the
status payload contains only ``status``, ``queue_position``, ``logs`` and
``metrics``, and ``metrics`` stays empty (``{}``) for the whole run — it is
populated with ``inference_time`` only *after* completion, which is a
retrospective measurement, not a prediction.

So the estimate is derived from our own history: record how long each finished
generation actually took, and predict the next one as the **median of that
model's recent runs**.

Why the median rather than the mean: generation times are right-skewed. One run
stuck behind a long fal queue would pull a mean upward and make every subsequent
estimate pessimistic. The median ignores those outliers.

Why end-to-end wall-clock rather than fal's ``inference_time``: the seller waits
for fal's queue *plus* generation *plus* our download, Draco compression and
Azure upload. Only the full duration predicts what they experience.
"""

from __future__ import annotations

import logging
from statistics import median
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ModelGenerationStat

logger = logging.getLogger(__name__)

# How many recent runs feed the median. Large enough to be stable, small enough
# to track a genuine shift in fal's performance within a day or so.
SAMPLE_SIZE = 20
# Below this, one unusual run would swing the estimate wildly — keep the seed.
MIN_SAMPLES = 3
# Used when a provider supplies no baseline and the fal registry has no entry.
DEFAULT_BASELINE_SECONDS = 300


class GenerationEstimate:
    """An ETA plus enough context for the UI to phrase it honestly."""

    def __init__(self, seconds: int, sample_count: int, model_key: str) -> None:
        self.seconds = seconds
        self.sample_count = sample_count
        self.model_key = model_key

    @property
    def is_measured(self) -> bool:
        """True when this came from real runs rather than the registry seed."""
        return self.sample_count >= MIN_SAMPLES

    @property
    def display(self) -> str:
        """Human phrasing: '3 min 20 sec', '45 sec'."""
        minutes, seconds = divmod(max(0, self.seconds), 60)
        if minutes and seconds:
            return f"{minutes} min {seconds} sec"
        if minutes:
            return f"{minutes} min"
        return f"{seconds} sec"

    def to_payload(self) -> dict:
        """Shaped for the portal's existing estimate contract.

        Reuses the ``gpu`` envelope the SAM path already returns so the portal's
        `normalizeGpuEstimate` + countdown work unchanged.
        """
        return {
            "estimated_time": self.display,
            "estimated_seconds": self.seconds,
            "gpu_status": "warm",  # fal is managed — there is no cold start
            "message": f"Generating your 3D model. This usually takes about {self.display}.",
            "model": self.model_key,
            "is_measured": self.is_measured,
            "sample_count": self.sample_count,
        }


class GenerationEstimateService:
    """Records real generation durations and predicts the next one."""

    @staticmethod
    async def record(
        db: AsyncSession,
        model_key: str,
        duration_seconds: float,
        succeeded: bool = True,
    ) -> None:
        """Store one finished generation.

        Never raises: a stats failure must not fail a generation that otherwise
        succeeded — the worst case is a slightly staler estimate.
        """
        try:
            db.add(
                ModelGenerationStat(
                    model_key=model_key,
                    duration_seconds=max(0, int(round(duration_seconds))),
                    succeeded=succeeded,
                )
            )
            await db.commit()
            logger.info(
                "Recorded %s generation: %.1fs (succeeded=%s)",
                model_key, duration_seconds, succeeded,
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the flow
            logger.exception("Could not record generation stat for %s", model_key)
            try:
                await db.rollback()
            except Exception:
                pass

    @staticmethod
    async def _baseline_for(db: AsyncSession, model_key: str) -> int:
        """Seed estimate for a model with too little history.

        Resolved lazily and defensively: the model registry is one provider
        among several now, so a key it does not know (the Tripo parts
        pipeline, say) must fall back rather than raise — and now that the
        registry is database-backed, the model could also have been
        deactivated since its last run, which must fall back the same way.
        Uses ``get_model_spec_any`` (not ``get_model_spec``): this is a
        historical/ETA lookup, not a new-generation-request one, so an
        inactive model must still resolve — see the two-lookup-mode note in
        ``app/integrations/fal/registry.py``.
        """
        try:
            from app.integrations.fal import get_model_spec_any

            spec = await get_model_spec_any(db, model_key)
            return spec.baseline_estimate_seconds if spec else DEFAULT_BASELINE_SECONDS
        except Exception:
            return DEFAULT_BASELINE_SECONDS

    @staticmethod
    async def estimate(
        db: AsyncSession,
        model_key: str,
        baseline_seconds: Optional[int] = None,
    ) -> GenerationEstimate:
        """Predicted duration for the next run of ``model_key``.

        ``baseline_seconds`` is used until enough real runs exist. Callers
        outside the fal registry (other providers) should pass their own.
        """
        baseline = (
            baseline_seconds
            if baseline_seconds is not None
            else await GenerationEstimateService._baseline_for(db, model_key)
        )

        try:
            result = await db.execute(
                select(ModelGenerationStat.duration_seconds)
                .where(
                    ModelGenerationStat.model_key == model_key,
                    ModelGenerationStat.succeeded.is_(True),
                )
                .order_by(ModelGenerationStat.created_at.desc())
                .limit(SAMPLE_SIZE)
            )
            samples = [row for row in result.scalars().all()]
        except Exception:  # noqa: BLE001 - fall back to the seed on any DB issue
            logger.exception("Could not read generation stats for %s", model_key)
            samples = []

        if len(samples) < MIN_SAMPLES:
            return GenerationEstimate(
                seconds=baseline,
                sample_count=len(samples),
                model_key=model_key,
            )

        return GenerationEstimate(
            seconds=int(round(median(samples))),
            sample_count=len(samples),
            model_key=model_key,
        )


generation_estimate_service = GenerationEstimateService()
