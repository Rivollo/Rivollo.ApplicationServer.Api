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
