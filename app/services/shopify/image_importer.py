"""Copy a Shopify image into the seller's own Rivollo uploads.

The ONLY outbound fetch in the Shopify module, so it is locked down:

  * the URL must be https on exactly cdn.shopify.com (no port, no credentials),
    and callers additionally require it to be one of the URLs the merchant
    synced for that product (an allow-list from our own rows)
  * redirects are NOT followed, so the host check cannot be bypassed
  * the body is streamed with a hard size cap and a timeout
  * the bytes must decode as JPEG / PNG / WebP (Pillow), whatever the header says

The copy lands at users/{user_id}/uploads/{id}/shopify-<hash>.<ext> through the
existing storage_service.upload_file_content, so it passes
validate_image_url_ownership exactly like an upload through POST /uploads/content.
Copying (rather than pointing fal or a thumbnail at Shopify's CDN) means a
merchant editing their Shopify gallery can never break a Rivollo product.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import uuid

import httpx
from fastapi import HTTPException, status
from PIL import Image

from app.core.config import settings
from app.schemas.shopify import validate_shopify_image_url
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

_FORMATS = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"), "WEBP": ("webp", "image/webp")}
UNREADABLE = "The Shopify image could not be downloaded. Check it still exists and try again."


class ShopifyImageImporter:
    @staticmethod
    async def fetch(url: str) -> bytes:
        """Download a Shopify CDN image with every guard applied. Raises 400/502."""
        try:
            validate_shopify_image_url(url)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        limit = settings.SHOPIFY_IMAGE_MAX_BYTES
        try:
            async with httpx.AsyncClient(
                timeout=settings.SHOPIFY_IMAGE_TIMEOUT_SECONDS, follow_redirects=False
            ) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        logger.warning("Shopify image fetch %s -> HTTP %s", url, response.status_code)
                        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=UNREADABLE)
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > limit:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="The Shopify image is too large.",
                        )
                    chunks, total = [], 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > limit:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail="The Shopify image is too large.",
                            )
                        chunks.append(chunk)
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            logger.warning("Shopify image fetch failed for %s: %s", url, exc)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=UNREADABLE) from exc
        return b"".join(chunks)

    @staticmethod
    def identify(data: bytes) -> tuple[str, str]:
        """(extension, content type) from the decoded bytes. Raises 400 if not an image."""
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                fmt = (image.format or "").upper()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="The Shopify file is not a readable image."
            ) from exc
        if fmt not in _FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The Shopify image must be a JPEG, PNG or WebP.",
            )
        return _FORMATS[fmt]

    @staticmethod
    async def copy_to_uploads(user_id: uuid.UUID, url: str) -> str:
        """Fetch, verify and store. Returns the new Rivollo CDN URL."""
        data = await ShopifyImageImporter.fetch(url)
        extension, content_type = ShopifyImageImporter.identify(data)
        digest = hashlib.sha256(data).hexdigest()[:16]
        try:
            cdn_url, _blob_url = await asyncio.to_thread(
                storage_service.upload_file_content,
                str(user_id),
                f"shopify-{digest}.{extension}",
                content_type,
                io.BytesIO(data),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not store imported Shopify image for user %s", user_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The image could not be saved. Please try again.",
            ) from exc
        return cdn_url


image_importer = ShopifyImageImporter()
