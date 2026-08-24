import io
import logging
import os
import uuid
from typing import Optional

import httpx
from azure.storage.blob import ContentSettings
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Product, ProductAsset, ProductAssetMapping
from app.services.storage import storage_service
from app.services.image_moderation_service import image_moderation_service

logger = logging.getLogger(__name__)


class BackgroundRemovalService:
    """Background removal via external API + blob + DB persistence."""

    def __init__(self) -> None:
        # External rembg HTTP API endpoint
        self._external_url = "http://74.225.198.112:8000/remove"

    def _upload_to_blob(self, file_stream: io.BytesIO, filename: str) -> str:
        """Upload a stream to blob storage under the dev container and return URL."""
        client = storage_service._get_blob_service_client()  # type: ignore[attr-defined]
        container = "dev"
        container_client = client.get_container_client(container)
        if not container_client.exists():
            container_client.create_container()

        blob_client = client.get_blob_client(container=container, blob=filename)
        blob_client.upload_blob(
            file_stream,
            overwrite=True,
            content_settings=ContentSettings(content_type="image/png"),
        )
        return blob_client.url

    async def process(
        self,
        db: AsyncSession,
        file: UploadFile,
        product_id: str,
    ) -> dict:
        """Call external background-remove API, upload to blob, save DB, and return response data."""
        allowed_types = {"image/png", "image/jpeg", "image/webp"}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Allowed: image/png, image/jpeg, image/webp",
            )

        try:
            prod_uuid = uuid.UUID(product_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid productId format. Expected UUID string.",
            )

        product = await db.get(Product, prod_uuid)
        if not product:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found.")

        # Read original image bytes
        try:
            content = await file.read()
            if not content:
                raise ValueError("Empty file")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read image: {str(exc)}",
            )

        # Unauthenticated route, so no strike can be recorded — but the image is
        # still refused. Attribute to the product owner where one exists.
        await image_moderation_service.screen(
            content,
            user_id=getattr(product, "created_by", None),
            source="products.remove_background",
            filename=file.filename,
            content_type=file.content_type,
        )

        # Call external rembg HTTP API
        try:
            params = {
                "model": "birefnet-general",
                "max_size": 2048,
                "alpha_matting": "false",
                "alpha_matting_foreground_threshold": 240,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 10,
                "post_process_mask": "true",
            }

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self._external_url,
                    params=params,
                    files={
                        "file": (
                            file.filename or "upload.jpg",
                            content,
                            file.content_type or "image/jpeg",
                        )
                    },
                )
        except httpx.RequestError as exc:
            logger.exception("Error calling external background removal API")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to call background removal service: {exc}",
            )

        if response.status_code != 200:
            logger.error(
                "Background removal service returned %s: %s",
                response.status_code,
                response.text[:200],
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Background removal service error {response.status_code}: "
                    f"{response.text[:200]}"
                ),
            )

        output_bytes = response.content
        if not output_bytes:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Background removal service returned empty content",
            )

        buffer = io.BytesIO(output_bytes)
        buffer.seek(0)
        file_size = buffer.getbuffer().nbytes

        file_token = uuid.uuid4().hex
        filename = f"{product_id}/{file_token}.png"
        try:
            blob_url = self._upload_to_blob(buffer, filename)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Blob upload failed: {str(exc)}",
            )

        try:
            asset_id_to_use = 11

            product_asset = ProductAsset(
                asset_id=asset_id_to_use,
                image=blob_url,
                size_bytes=file_size,
                created_by=None,
            )
            db.add(product_asset)
            await db.flush()

            product_asset_mapping = ProductAssetMapping(
                name=file.filename or "Processed Image",
                productid=prod_uuid,
                product_asset_id=product_asset.id,
                isactive=True,
                created_by=None,
            )
            db.add(product_asset_mapping)

            await db.commit()
        except Exception as exc:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database save failed: {str(exc)}",
            )

        return {
            "asset_id": asset_id_to_use,
            "blob_url": blob_url,
            "message": "Background removed and saved successfully",
            "image_size_bytes": file_size,
        }


background_removal_service = BackgroundRemovalService()

