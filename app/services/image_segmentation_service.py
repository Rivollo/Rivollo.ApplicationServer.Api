import os
from typing import Any

from fastapi import HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.segmentation import (
    Sam2AutoSegmentRequest,
    Sam2AutoSegmentResponse,
    Sam2ImageSegmentRequest,
    Sam2ImageSegmentResponse,
)


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

        try:
            return await run_in_threadpool(self._subscribe, model_id, arguments)
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="fal-client is not installed. Run `uv sync` after updating dependencies.",
            ) from exc
        except Exception as exc:
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
