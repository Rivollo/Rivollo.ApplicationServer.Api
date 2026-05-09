"""Async HTTP client for the 3D model generation API."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Long timeout — 3D generation can take several minutes
_TIMEOUT = httpx.Timeout(timeout=600.0, connect=10.0)

# Readiness probe: short per-request timeout, total wait capped at 20 min
_READINESS_HEALTH_PATH = "/health"
_READINESS_REQUEST_TIMEOUT = httpx.Timeout(timeout=5.0, connect=5.0)
_READINESS_MAX_WAIT_SECONDS = 1200.0
_READINESS_INITIAL_BACKOFF = 2.0
_READINESS_MAX_BACKOFF = 30.0

# Warm/cold check: longer timeout so a slow-responding but live GPU isn't
# misclassified as cold. Two attempts to handle transient network blips.
_WARMTH_CHECK_TIMEOUT = httpx.Timeout(timeout=15.0, connect=10.0)
_WARMTH_CHECK_ATTEMPTS = 2


class ThreeDGenerateResponse:
    """Parsed response from the generate-3d endpoint."""

    def __init__(
        self,
        success: bool,
        voxel_url: Optional[str] = None,
        ply_url: Optional[str] = None,
        glb_url: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.success = success
        self.voxel_url = voxel_url
        self.ply_url = ply_url
        self.glb_url = glb_url
        self.error = error

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ThreeDGenerateResponse(success={self.success}, "
            f"voxel={self.voxel_url!r}, ply={self.ply_url!r}, glb={self.glb_url!r})"
        )


class ThreeDModelClient:
    """Client for the external 3D model generation service."""

    @staticmethod
    def _base_url() -> str:
        url = settings.MODEL_3D_API_BASE_URL.rstrip("/")
        if not url:
            raise RuntimeError(
                "MODEL_3D_API_BASE_URL is not configured. "
                "Set 3D_MODEL_API_BASE_URL in your .env file."
            )
        return url

    @staticmethod
    async def wait_until_ready(
        *,
        product_id: Optional[uuid.UUID] = None,
        max_wait_seconds: float = _READINESS_MAX_WAIT_SECONDS,
    ) -> bool:
        """
        Poll <base>/health until it returns 200 or max_wait_seconds elapses.

        Used to absorb cold-start of the 3D VM — callers should invoke this
        from a background task before calling generate_3d, so the create-product
        response is never blocked.

        Returns True once the service responds 200, False on timeout.
        """
        import asyncio  # local import keeps module import light

        endpoint = f"{ThreeDModelClient._base_url()}{_READINESS_HEALTH_PATH}"
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max_wait_seconds
        backoff = _READINESS_INITIAL_BACKOFF
        attempt = 0

        async with httpx.AsyncClient(timeout=_READINESS_REQUEST_TIMEOUT) as client:
            while True:
                attempt += 1
                try:
                    resp = await client.get(endpoint)
                    if resp.status_code == 200:
                        logger.info(
                            "3D service ready  product_id=%s  attempt=%d  endpoint=%s",
                            product_id, attempt, endpoint,
                        )
                        return True
                    logger.info(
                        "3D readiness probe non-200: status=%s  attempt=%d  product_id=%s",
                        resp.status_code, attempt, product_id,
                    )
                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    logger.info(
                        "3D readiness probe failed: %s  attempt=%d  product_id=%s",
                        exc, attempt, product_id,
                    )

                remaining = deadline - loop.time()
                if remaining <= 0:
                    logger.error(
                        "3D service did not become ready within %.0fs  product_id=%s",
                        max_wait_seconds, product_id,
                    )
                    return False

                sleep_for = min(backoff, remaining)
                await asyncio.sleep(sleep_for)
                backoff = min(backoff * 2, _READINESS_MAX_BACKOFF)

    @staticmethod
    async def quick_warmth_check() -> bool:
        """
        Probes /health up to _WARMTH_CHECK_ATTEMPTS times with a generous
        timeout. Returns True only if the service responds 200.

        Uses a longer timeout than the readiness poll (_WARMTH_CHECK_TIMEOUT)
        so a healthy-but-slow GPU health endpoint isn't misclassified as cold.
        Two attempts guard against transient network blips.
        """
        import asyncio  # local import keeps module import light

        endpoint = f"{ThreeDModelClient._base_url()}{_READINESS_HEALTH_PATH}"
        for attempt in range(1, _WARMTH_CHECK_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=_WARMTH_CHECK_TIMEOUT) as client:
                    resp = await client.get(endpoint)
                    if resp.status_code == 200:
                        logger.info(
                            "Warmth check: GPU warm (attempt=%d  endpoint=%s)",
                            attempt, endpoint,
                        )
                        return True
                    logger.info(
                        "Warmth check: non-200 status=%s attempt=%d",
                        resp.status_code, attempt,
                    )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                logger.info(
                    "Warmth check: probe failed attempt=%d  exc=%s", attempt, exc
                )
            if attempt < _WARMTH_CHECK_ATTEMPTS:
                await asyncio.sleep(2)

        logger.info("Warmth check: GPU cold after %d attempt(s)", _WARMTH_CHECK_ATTEMPTS)
        return False

    @staticmethod
    async def generate_3d(
        *,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        blob_url: str,
        mask_blob_url: str,
        target_format: str,
        asset_id: int,
        mesh_asset_id: int,
        name: str,
    ) -> ThreeDGenerateResponse:
        """
        Call <base>/generate-3d and return parsed response.

        Expected request payload::

            {
                "product_id": "<uuid>",
                "user_id": "<uuid>",
                "blob_url": "<image url>",
                "mask_blob_url": "<mask url>",
                "target_format": "glb",
                "asset_id": 1,
                "mesh_asset_id": 9,
                "name": "product name"
            }

        Expected success response::

            {
                "success": true,
                "files": {
                    "voxel": "<voxel url>",
                    "ply":   "<ply url>",
                    "glb":   "<glb url>"
                }
            }
        """
        payload = {
            "product_id": str(product_id),
            "user_id": str(user_id),
            "blob_url": blob_url,
            "mask_blob_url": mask_blob_url,
            "target_format": target_format,
            "asset_id": asset_id,
            "mesh_asset_id": mesh_asset_id,
            "name": name,
        }

        endpoint = f"{ThreeDModelClient._base_url()}/generate-3d"
        logger.info(
            "Calling 3D generation API: endpoint=%s  product_id=%s",
            endpoint,
            product_id,
        )

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(endpoint, json=payload)

            logger.info(
                "3D generation API responded: status=%s  product_id=%s",
                response.status_code,
                product_id,
            )

            if response.status_code != 200:
                error_text = response.text[:500]
                logger.error(
                    "3D generation API returned non-200: status=%s  body=%s",
                    response.status_code,
                    error_text,
                )
                return ThreeDGenerateResponse(
                    success=False,
                    error=f"API returned status {response.status_code}: {error_text}",
                )

            data: dict = response.json()
            success: bool = bool(data.get("success", False))

            if not success:
                error_msg = data.get("error") or data.get("detail") or "Unknown error from 3D API"
                logger.error("3D generation API returned success=false: %s", error_msg)
                return ThreeDGenerateResponse(success=False, error=error_msg)

            files: dict = data.get("files", {})
            voxel_url = files.get("voxel")
            ply_url = files.get("ply")
            glb_url = files.get("glb")

            logger.info(
                "3D generation succeeded: product_id=%s  voxel=%s  ply=%s  glb=%s",
                product_id,
                voxel_url,
                ply_url,
                glb_url,
            )

            return ThreeDGenerateResponse(
                success=True,
                voxel_url=voxel_url,
                ply_url=ply_url,
                glb_url=glb_url,
            )

        except httpx.TimeoutException as exc:
            msg = f"3D generation API timed out after {_TIMEOUT.read}s"
            logger.error("%s  product_id=%s  exc=%s", msg, product_id, exc)
            return ThreeDGenerateResponse(success=False, error=msg)

        except httpx.RequestError as exc:
            msg = f"3D generation API request failed: {exc}"
            logger.error("%s  product_id=%s", msg, product_id)
            return ThreeDGenerateResponse(success=False, error=msg)

        except Exception as exc:
            logger.exception(
                "Unexpected error calling 3D generation API  product_id=%s", product_id
            )
            return ThreeDGenerateResponse(success=False, error=str(exc))


# Module-level singleton-style alias for convenience
threed_model_client = ThreeDModelClient()
