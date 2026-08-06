import logging
import os
from datetime import datetime, timezone
from typing import Any, NoReturn, Optional

from fastapi import HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.segmentation import (
    Sam2AutoSegmentRequest,
    Sam2AutoSegmentResponse,
    Sam2ImageSegmentRequest,
    Sam2ImageSegmentResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upstream (fal.ai) rate limit
#
# fal.ai rate-limits per API KEY, not per caller, so the budget that actually
# matters is global: every seller's segmentation is spent from the same key.
# This counter therefore has no user dimension and guards every fal call, so
# the point/box and auto-segment models draw from one shared budget.
#
# This is a different concern from the per-user limiters in app.api.routes.ai,
# which exist for fairness (one seller must not spend everyone's quota). Both
# are needed: a per-user cap alone cannot bound total upstream traffic, and a
# global cap alone lets a single user starve the rest.
#
# KNOWN LIMITATION: per-process, same as the per-user limiters. With N replicas
# the key can still see FAL_SEGMENT_RATE_LIMIT_PER_MINUTE x N calls per minute.
# A Redis INCR/EXPIRE on the same minute-bucket key would make it exact; the
# call site (_fal_rate_limiter.check) would not change.
# ---------------------------------------------------------------------------

class _FalCallRateLimiter:
    """Process-wide, per-minute cap on outbound fal.ai calls."""

    def __init__(self) -> None:
        self._bucket: Optional[str] = None
        self._count = 0

    def check(self) -> None:
        """Raise HTTP 429 if this minute's shared fal.ai budget is spent.

        A single bucket is kept rather than a dict — there is no per-user key to
        accumulate, so nothing can grow unboundedly and no eviction is needed.
        """
        now = datetime.now(timezone.utc)
        bucket = now.strftime("%Y-%m-%dT%H:%M")
        if bucket != self._bucket:
            self._bucket = bucket
            self._count = 0

        limit = settings.FAL_SEGMENT_RATE_LIMIT_PER_MINUTE
        self._count += 1

        if self._count > limit:
            retry_after = max(1, 60 - now.second)
            logger.warning(
                "fal.ai segmentation budget exhausted (count=%d, limit=%d, bucket=%s)",
                self._count, limit, bucket,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Image segmentation is busy: the shared limit of {limit} "
                    f"upstream calls per minute is used up. "
                    f"Please retry in {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )


_fal_rate_limiter = _FalCallRateLimiter()


class ImageSegmentationService:
    """Server-side wrapper for fal.ai SAM2 image segmentation."""

    _IMAGE_SEGMENT_MODEL_ID = "fal-ai/sam2/image"
    _AUTO_SEGMENT_MODEL_ID = "fal-ai/sam2/auto-segment"

    async def segment_with_sam2(
        self,
        payload: Sam2ImageSegmentRequest,
    ) -> Sam2ImageSegmentResponse:
        result = await self._call_fal_model(
            model_id=self._IMAGE_SEGMENT_MODEL_ID,
            arguments=payload.model_dump(mode="json", exclude_none=True),
            failure_message="SAM2 image segmentation failed",
        )
        data = self._extract_result_data(result)
        return Sam2ImageSegmentResponse.model_validate(data)

    async def auto_segment_with_sam2(
        self,
        payload: Sam2AutoSegmentRequest,
    ) -> Sam2AutoSegmentResponse:
        result = await self._call_fal_model(
            model_id=self._AUTO_SEGMENT_MODEL_ID,
            arguments=payload.model_dump(mode="json", exclude_none=True),
            failure_message="SAM2 auto segmentation failed",
        )
        data = self._extract_result_data(result)
        return Sam2AutoSegmentResponse.model_validate(data)

    async def _call_fal_model(
        self,
        model_id: str,
        arguments: dict[str, Any],
        failure_message: str,
    ) -> Any:
        if not settings.FAL_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="FAL_KEY is not configured",
            )

        os.environ["FAL_KEY"] = settings.FAL_KEY

        _fal_rate_limiter.check()

        try:
            return await run_in_threadpool(self._subscribe, model_id, arguments)
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="fal-client is not installed. Run `uv sync` after updating dependencies.",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            self._raise_upstream_error(exc, failure_message)

    @staticmethod
    def _raise_upstream_error(exc: Exception, failure_message: str) -> NoReturn:
        """Translate a fal.ai failure into the closest HTTP status.

        fal's own 429 is forwarded as a 429 carrying its Retry-After rather than
        being flattened into 502, so a caller can tell "back off and retry" apart
        from "upstream is broken". status_code/response_headers are read by
        duck-typing instead of catching fal_client.FalClientHTTPError, because
        fal_client is imported lazily to keep it an optional dependency.
        """
        if getattr(exc, "status_code", None) == status.HTTP_429_TOO_MANY_REQUESTS:
            headers = getattr(exc, "response_headers", None) or {}
            retry_after = next(
                (
                    str(headers[k])
                    for k in ("retry-after", "Retry-After")
                    if headers.get(k)
                ),
                "60",
            )
            logger.warning("fal.ai returned 429 for %s; retry after %ss", failure_message, retry_after)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"{failure_message}: upstream rate limit reached. Please retry shortly.",
                headers={"Retry-After": retry_after},
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{failure_message}: {str(exc)}",
        ) from exc

    @staticmethod
    def _subscribe(model_id: str, arguments: dict[str, Any]) -> Any:
        import fal_client

        return fal_client.subscribe(
            model_id,
            arguments=arguments,
            with_logs=False,
        )

    @staticmethod
    def _extract_result_data(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result.get("data", result)
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="SAM2 image segmentation returned an invalid response",
        )


image_segmentation_service = ImageSegmentationService()
