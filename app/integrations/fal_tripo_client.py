"""Async HTTP client for the fal.ai Tripo H3.1 image-to-3D API.

This mirrors the structure of ``threed_model_client`` but targets fal.ai's
async, queue-based API instead of the self-hosted SAM 3D service. The flow is:

    1. POST to the queue submit endpoint -> {request_id, status_url, response_url}
    2. Poll ``status_url`` (GET) until status == "COMPLETED"
       (IN_QUEUE -> IN_PROGRESS -> COMPLETED)
    3. GET ``response_url`` for the result JSON
    4. Download the GLB referenced by ``result.model_urls.glb.url``
       (falling back to ``result.model_mesh.url``)

The ``status_url`` / ``response_url`` returned by the submit response are used
verbatim — never constructed by hand.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Model endpoint ID for the fal.ai queue API.
_MODEL_ENDPOINT_ID = "tripo3d/h3.1/image-to-3d"
_SUBMIT_URL = f"https://queue.fal.run/{_MODEL_ENDPOINT_ID}"

# Submit / status / result requests are quick — generation happens server-side.
_REQUEST_TIMEOUT = httpx.Timeout(timeout=60.0, connect=10.0)
# GLB download can be large; give it a long read window.
_DOWNLOAD_TIMEOUT = httpx.Timeout(timeout=600.0, connect=10.0)

# Poll loop: exponential backoff capped, total wait bounded.
_POLL_INITIAL_BACKOFF = 2.0
_POLL_MAX_BACKOFF = 15.0
_POLL_MAX_WAIT_SECONDS = 600.0


class FalGenerateResponse:
    """Parsed result of a full fal.ai Tripo generate-and-download cycle."""

    def __init__(
        self,
        success: bool,
        glb_bytes: Optional[bytes] = None,
        glb_content_type: Optional[str] = None,
        glb_source_url: Optional[str] = None,
        request_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.success = success
        self.glb_bytes = glb_bytes
        self.glb_content_type = glb_content_type
        self.glb_source_url = glb_source_url
        self.request_id = request_id
        self.error = error

    def __repr__(self) -> str:  # pragma: no cover
        size = len(self.glb_bytes) if self.glb_bytes else 0
        return (
            f"FalGenerateResponse(success={self.success}, "
            f"request_id={self.request_id!r}, glb_bytes={size}, "
            f"glb_source_url={self.glb_source_url!r})"
        )


class FalTripoClient:
    """Client for the fal.ai Tripo H3.1 image-to-3D queue API."""

    @staticmethod
    def _auth_headers() -> dict:
        key = (settings.FAL_KEY or "").strip()
        if not key:
            raise RuntimeError(
                "FAL_KEY is not configured. Set FAL_KEY in your environment "
                "(Container App secret) to call the fal.ai Tripo API."
            )
        return {"Authorization": f"Key {key}"}

    @staticmethod
    def _build_request_body(image_url: str) -> dict:
        return {
            "image_url": image_url,
            "texture": True,
            "pbr": True,
            "texture_quality": "detailed",
            "geometry_quality": "detailed",
            "texture_alignment": "original_image",
            "orientation": "align_image",
        }

    @staticmethod
    def _extract_glb_url(result: dict) -> Optional[str]:
        """Pull the GLB download URL out of the fal result JSON.

        Prefers ``result.model_urls.glb.url`` and falls back to
        ``result.model_mesh.url`` when the former is absent.
        """
        model_urls = result.get("model_urls") or {}
        glb = model_urls.get("glb") or {}
        glb_url = glb.get("url") if isinstance(glb, dict) else None
        if glb_url:
            return glb_url

        model_mesh = result.get("model_mesh") or {}
        if isinstance(model_mesh, dict):
            return model_mesh.get("url")
        return None

    @staticmethod
    async def generate_3d(
        *,
        product_id: uuid.UUID,
        image_url: str,
    ) -> FalGenerateResponse:
        """Run the full fal.ai Tripo submit -> poll -> result -> download cycle.

        ``image_url`` must be a publicly reachable URL (the blob's public/SAS
        URL produced by the existing upload flow).

        Returns a :class:`FalGenerateResponse`; ``success`` is False with a
        populated ``error`` on any failure (never raises for expected errors).
        """
        if not image_url:
            logger.error("fal Tripo: missing image_url  product_id=%s", product_id)
            return FalGenerateResponse(
                success=False, error="image_url is required for fal.ai generation"
            )

        try:
            headers = FalTripoClient._auth_headers()
        except RuntimeError as exc:
            logger.error("fal Tripo: %s  product_id=%s", exc, product_id)
            return FalGenerateResponse(success=False, error=str(exc))
        body = FalTripoClient._build_request_body(image_url)

        # ---- 1. Submit -----------------------------------------------------
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                submit_resp = await client.post(_SUBMIT_URL, json=body, headers=headers)
        except httpx.RequestError as exc:
            msg = f"fal Tripo submit request failed: {exc}"
            logger.error("%s  product_id=%s", msg, product_id)
            return FalGenerateResponse(success=False, error=msg)

        if submit_resp.status_code != 200:
            error_text = submit_resp.text[:500]
            logger.error(
                "fal Tripo submit returned non-200: status=%s  product_id=%s  body=%s",
                submit_resp.status_code, product_id, error_text,
            )
            return FalGenerateResponse(
                success=False,
                error=f"fal submit returned status {submit_resp.status_code}: {error_text}",
            )

        submit_data: dict = submit_resp.json()
        request_id = submit_data.get("request_id")
        status_url = submit_data.get("status_url")
        response_url = submit_data.get("response_url")

        if not status_url or not response_url:
            logger.error(
                "fal Tripo submit missing status/response URL  product_id=%s  request_id=%s  body=%s",
                product_id, request_id, submit_data,
            )
            return FalGenerateResponse(
                success=False,
                request_id=request_id,
                error="fal submit response missing status_url/response_url",
            )

        logger.info(
            "fal Tripo submitted  product_id=%s  request_id=%s",
            product_id, request_id,
        )

        # ---- 2. Poll status_url until COMPLETED ----------------------------
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _POLL_MAX_WAIT_SECONDS
        backoff = _POLL_INITIAL_BACKOFF

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                while True:
                    status_resp = await client.get(status_url, headers=headers)
                    if status_resp.status_code != 200:
                        logger.warning(
                            "fal Tripo status poll non-200: status=%s  request_id=%s",
                            status_resp.status_code, request_id,
                        )
                    else:
                        status_payload: dict = status_resp.json()
                        current_status = status_payload.get("status")
                        logger.info(
                            "fal Tripo status=%s  request_id=%s",
                            current_status, request_id,
                        )
                        if current_status == "COMPLETED":
                            break
                        if current_status in {"FAILED", "ERROR", "CANCELLED"}:
                            err = status_payload.get("error") or current_status
                            logger.error(
                                "fal Tripo generation failed: status=%s  request_id=%s  detail=%s",
                                current_status, request_id, err,
                            )
                            return FalGenerateResponse(
                                success=False,
                                request_id=request_id,
                                error=f"fal generation {current_status}: {err}",
                            )

                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        msg = (
                            f"fal generation did not complete within "
                            f"{_POLL_MAX_WAIT_SECONDS:.0f}s"
                        )
                        logger.error("%s  request_id=%s", msg, request_id)
                        return FalGenerateResponse(
                            success=False, request_id=request_id, error=msg
                        )

                    await asyncio.sleep(min(backoff, remaining))
                    backoff = min(backoff * 2, _POLL_MAX_BACKOFF)

                # ---- 3. Fetch result ----------------------------------------
                result_resp = await client.get(response_url, headers=headers)
        except httpx.RequestError as exc:
            msg = f"fal Tripo poll/result request failed: {exc}"
            logger.error("%s  request_id=%s", msg, request_id)
            return FalGenerateResponse(success=False, request_id=request_id, error=msg)

        if result_resp.status_code != 200:
            error_text = result_resp.text[:500]
            logger.error(
                "fal Tripo result returned non-200: status=%s  request_id=%s  body=%s",
                result_resp.status_code, request_id, error_text,
            )
            return FalGenerateResponse(
                success=False,
                request_id=request_id,
                error=f"fal result returned status {result_resp.status_code}: {error_text}",
            )

        result: dict = result_resp.json()
        glb_url = FalTripoClient._extract_glb_url(result)
        if not glb_url:
            logger.error(
                "fal Tripo result missing GLB url  request_id=%s  body=%s",
                request_id, result,
            )
            return FalGenerateResponse(
                success=False,
                request_id=request_id,
                error="fal result contained no GLB URL (model_urls.glb / model_mesh)",
            )

        logger.info(
            "fal Tripo completed  request_id=%s  glb_url=%s",
            request_id, glb_url,
        )

        # ---- 4. Download the GLB -------------------------------------------
        try:
            async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT) as client:
                glb_resp = await client.get(glb_url)
        except httpx.RequestError as exc:
            msg = f"fal Tripo GLB download failed: {exc}"
            logger.error("%s  request_id=%s", msg, request_id)
            return FalGenerateResponse(success=False, request_id=request_id, error=msg)

        if glb_resp.status_code != 200 or not glb_resp.content:
            logger.error(
                "fal Tripo GLB download returned status=%s  request_id=%s",
                glb_resp.status_code, request_id,
            )
            return FalGenerateResponse(
                success=False,
                request_id=request_id,
                error=f"fal GLB download returned status {glb_resp.status_code}",
            )

        content_type = glb_resp.headers.get("content-type") or "model/gltf-binary"
        logger.info(
            "fal Tripo GLB downloaded  request_id=%s  bytes=%d",
            request_id, len(glb_resp.content),
        )

        return FalGenerateResponse(
            success=True,
            glb_bytes=glb_resp.content,
            glb_content_type=content_type,
            glb_source_url=glb_url,
            request_id=request_id,
        )


# Module-level singleton-style alias for convenience
fal_tripo_client = FalTripoClient()
