"""Async HTTP client for fal.ai's queue-based image-to-3D APIs.

Every fal image-to-3D model speaks the same protocol, so this module implements
it once and takes the per-model differences from a :class:`FalModelSpec`:

    1. POST to the queue submit endpoint -> {request_id, status_url, response_url}
    2. Poll ``status_url`` (GET) until status == "COMPLETED"
       (IN_QUEUE -> IN_PROGRESS -> COMPLETED)
    3. GET ``response_url`` for the result JSON
    4. Download the GLB, located by the spec's ``extract_glb_url``

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
from app.integrations.fal.registry import FalModelSpec

logger = logging.getLogger(__name__)

# Submit / status / result requests are quick — generation happens server-side.
_REQUEST_TIMEOUT = httpx.Timeout(timeout=60.0, connect=10.0)
# GLB download can be large; give it a long read window.
_DOWNLOAD_TIMEOUT = httpx.Timeout(timeout=600.0, connect=10.0)

# Poll loop: exponential backoff capped, total wait bounded.
_POLL_INITIAL_BACKOFF = 2.0
_POLL_MAX_BACKOFF = 15.0
_POLL_MAX_WAIT_SECONDS = 600.0


class FalGenerateResponse:
    """Parsed result of a full fal.ai generate-and-download cycle."""

    def __init__(
        self,
        success: bool,
        glb_bytes: Optional[bytes] = None,
        glb_content_type: Optional[str] = None,
        glb_source_url: Optional[str] = None,
        request_id: Optional[str] = None,
        model_key: Optional[str] = None,
        error: Optional[str] = None,
        usdz_bytes: Optional[bytes] = None,
        usdz_content_type: Optional[str] = None,
        usdz_source_url: Optional[str] = None,
    ) -> None:
        self.success = success
        self.glb_bytes = glb_bytes
        self.glb_content_type = glb_content_type
        self.glb_source_url = glb_source_url
        self.request_id = request_id
        self.model_key = model_key
        self.error = error
        # Populated only for models that export USDZ themselves (Meshy). When
        # present the caller stores it directly instead of running the Azure
        # GLB->USDZ conversion job.
        self.usdz_bytes = usdz_bytes
        self.usdz_content_type = usdz_content_type
        self.usdz_source_url = usdz_source_url

    def __repr__(self) -> str:  # pragma: no cover
        size = len(self.glb_bytes) if self.glb_bytes else 0
        return (
            f"FalGenerateResponse(success={self.success}, "
            f"model={self.model_key!r}, request_id={self.request_id!r}, "
            f"glb_bytes={size}, glb_source_url={self.glb_source_url!r})"
        )


class FalQueueClient:
    """Client for any fal.ai queue-based image-to-3D model."""

    @staticmethod
    def _auth_headers() -> dict:
        key = (settings.FAL_KEY or "").strip()
        if not key:
            raise RuntimeError(
                "FAL_KEY is not configured. Set FAL_KEY in your environment "
                "(Container App secret) to call the fal.ai API."
            )
        return {"Authorization": f"Key {key}"}

    @staticmethod
    async def generate_3d(
        *,
        spec: FalModelSpec,
        product_id: uuid.UUID,
        image_url: str,
    ) -> FalGenerateResponse:
        """Run the full submit -> poll -> result -> download cycle for one model.

        ``image_url`` must be a publicly reachable URL (the blob's public/SAS
        URL produced by the existing upload flow).

        Returns a :class:`FalGenerateResponse`; ``success`` is False with a
        populated ``error`` on any failure (never raises for expected errors).
        """
        model = spec.key

        if not image_url:
            logger.error("fal %s: missing image_url  product_id=%s", model, product_id)
            return FalGenerateResponse(
                success=False,
                model_key=model,
                error="image_url is required for fal.ai generation",
            )

        try:
            headers = FalQueueClient._auth_headers()
        except RuntimeError as exc:
            logger.error("fal %s: %s  product_id=%s", model, exc, product_id)
            return FalGenerateResponse(success=False, model_key=model, error=str(exc))

        body = spec.build_body(image_url)

        # ---- 1. Submit -----------------------------------------------------
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                submit_resp = await client.post(
                    spec.submit_url, json=body, headers=headers
                )
        except httpx.RequestError as exc:
            msg = f"fal {model} submit request failed: {exc}"
            logger.error("%s  product_id=%s", msg, product_id)
            return FalGenerateResponse(success=False, model_key=model, error=msg)

        if submit_resp.status_code != 200:
            error_text = submit_resp.text[:500]
            logger.error(
                "fal %s submit returned non-200: status=%s  product_id=%s  body=%s",
                model, submit_resp.status_code, product_id, error_text,
            )
            return FalGenerateResponse(
                success=False,
                model_key=model,
                error=f"fal submit returned status {submit_resp.status_code}: {error_text}",
            )

        submit_data: dict = submit_resp.json()
        request_id = submit_data.get("request_id")
        status_url = submit_data.get("status_url")
        response_url = submit_data.get("response_url")

        if not status_url or not response_url:
            logger.error(
                "fal %s submit missing status/response URL  product_id=%s  request_id=%s  body=%s",
                model, product_id, request_id, submit_data,
            )
            return FalGenerateResponse(
                success=False,
                model_key=model,
                request_id=request_id,
                error="fal submit response missing status_url/response_url",
            )

        logger.info(
            "fal %s submitted  product_id=%s  request_id=%s",
            model, product_id, request_id,
        )

        # ---- 2. Poll status_url until COMPLETED ----------------------------
        loop = asyncio.get_event_loop()
        # Per-model ceiling: Meshy documents 5-10 minutes, which would race the
        # shared default and time out a run that was about to succeed.
        max_wait = spec.max_wait_seconds
        deadline = loop.time() + max_wait
        backoff = _POLL_INITIAL_BACKOFF

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                while True:
                    status_resp = await client.get(status_url, headers=headers)
                    # fal answers 202 while the request is still queued or
                    # running and 200 once it completes. Both carry the same
                    # JSON body — treating 202 as an error would mean we never
                    # read `status` during the wait and so could not notice a
                    # FAILED result until the overall timeout expired.
                    if status_resp.status_code not in (200, 202):
                        logger.warning(
                            "fal %s status poll unexpected HTTP %s  request_id=%s",
                            model, status_resp.status_code, request_id,
                        )
                    else:
                        status_payload: dict = status_resp.json()
                        current_status = status_payload.get("status")
                        logger.info(
                            "fal %s status=%s  request_id=%s",
                            model, current_status, request_id,
                        )
                        if current_status == "COMPLETED":
                            break
                        if current_status in {"FAILED", "ERROR", "CANCELLED"}:
                            err = status_payload.get("error") or current_status
                            logger.error(
                                "fal %s generation failed: status=%s  request_id=%s  detail=%s",
                                model, current_status, request_id, err,
                            )
                            return FalGenerateResponse(
                                success=False,
                                model_key=model,
                                request_id=request_id,
                                error=f"fal generation {current_status}: {err}",
                            )

                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        msg = (
                            f"fal generation did not complete within "
                            f"{max_wait:.0f}s"
                        )
                        logger.error("%s  model=%s  request_id=%s", msg, model, request_id)
                        return FalGenerateResponse(
                            success=False,
                            model_key=model,
                            request_id=request_id,
                            error=msg,
                        )

                    await asyncio.sleep(min(backoff, remaining))
                    backoff = min(backoff * 2, _POLL_MAX_BACKOFF)

                # ---- 3. Fetch result ----------------------------------------
                result_resp = await client.get(response_url, headers=headers)
        except httpx.RequestError as exc:
            msg = f"fal {model} poll/result request failed: {exc}"
            logger.error("%s  request_id=%s", msg, request_id)
            return FalGenerateResponse(
                success=False, model_key=model, request_id=request_id, error=msg
            )

        if result_resp.status_code != 200:
            error_text = result_resp.text[:500]
            logger.error(
                "fal %s result returned non-200: status=%s  request_id=%s  body=%s",
                model, result_resp.status_code, request_id, error_text,
            )
            return FalGenerateResponse(
                success=False,
                model_key=model,
                request_id=request_id,
                error=f"fal result returned status {result_resp.status_code}: {error_text}",
            )

        result: dict = result_resp.json()
        glb_url = spec.extract_glb_url(result)
        if not glb_url:
            logger.error(
                "fal %s result missing GLB url  request_id=%s  body=%s",
                model, request_id, result,
            )
            return FalGenerateResponse(
                success=False,
                model_key=model,
                request_id=request_id,
                error=f"fal {model} result contained no GLB URL",
            )

        logger.info(
            "fal %s completed  request_id=%s  glb_url=%s",
            model, request_id, glb_url,
        )

        # ---- 4. Download the GLB -------------------------------------------
        try:
            async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT) as client:
                glb_resp = await client.get(glb_url)
        except httpx.RequestError as exc:
            msg = f"fal {model} GLB download failed: {exc}"
            logger.error("%s  request_id=%s", msg, request_id)
            return FalGenerateResponse(
                success=False, model_key=model, request_id=request_id, error=msg
            )

        if glb_resp.status_code != 200 or not glb_resp.content:
            logger.error(
                "fal %s GLB download returned status=%s  request_id=%s",
                model, glb_resp.status_code, request_id,
            )
            return FalGenerateResponse(
                success=False,
                model_key=model,
                request_id=request_id,
                error=f"fal GLB download returned status {glb_resp.status_code}",
            )

        content_type = glb_resp.headers.get("content-type") or "model/gltf-binary"
        logger.info(
            "fal %s GLB downloaded  request_id=%s  bytes=%d",
            model, request_id, len(glb_resp.content),
        )

        # ---- 5. Download the vendor's USDZ, when it exports one ------------
        # Best-effort: a missing or failed USDZ must not fail a successful GLB
        # generation. The caller falls back to the Azure conversion job.
        usdz_bytes: Optional[bytes] = None
        usdz_content_type: Optional[str] = None
        usdz_url: Optional[str] = None

        if spec.extract_usdz_url is not None:
            usdz_url = spec.extract_usdz_url(result)
            if usdz_url:
                try:
                    async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT) as client:
                        usdz_resp = await client.get(usdz_url)
                    if usdz_resp.status_code == 200 and usdz_resp.content:
                        usdz_bytes = usdz_resp.content
                        usdz_content_type = (
                            usdz_resp.headers.get("content-type")
                            or "model/vnd.usdz+zip"
                        )
                        logger.info(
                            "fal %s USDZ downloaded  request_id=%s  bytes=%d",
                            model, request_id, len(usdz_bytes),
                        )
                    else:
                        logger.warning(
                            "fal %s USDZ download returned status=%s  request_id=%s",
                            model, usdz_resp.status_code, request_id,
                        )
                except httpx.RequestError as exc:
                    logger.warning(
                        "fal %s USDZ download failed (non-fatal): %s  request_id=%s",
                        model, exc, request_id,
                    )
            else:
                logger.info(
                    "fal %s returned no USDZ url  request_id=%s", model, request_id
                )

        return FalGenerateResponse(
            success=True,
            glb_bytes=glb_resp.content,
            glb_content_type=content_type,
            glb_source_url=glb_url,
            request_id=request_id,
            model_key=model,
            usdz_bytes=usdz_bytes,
            usdz_content_type=usdz_content_type,
            usdz_source_url=usdz_url if usdz_bytes else None,
        )


fal_queue_client = FalQueueClient()
