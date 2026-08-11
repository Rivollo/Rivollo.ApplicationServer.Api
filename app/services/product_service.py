"""Product service for handling product creation with image storage."""

import io
import os
import time
import uuid
from typing import BinaryIO, Optional

import httpx
import logging
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, insert
from datetime import datetime

from app.core.config import settings
from app.models.models import Product, ProductAsset, ProductAssetMapping, ProductStatus, Background, Job
from app.services.storage import storage_service
from app.services.glb_compression_service import CompressedMeshPackage, glb_compression_service
from app.services.gpu_status_service import gpu_status_service
from app.integrations.threed_model_client import threed_model_client
from app.integrations.fal_sam3_client import fal_sam3_client
from app.integrations.fal import DEFAULT_MODEL_KEY, fal_queue_client, get_model_spec
from app.integrations.service_bus_publisher import ServiceBusPublisher
from app.database.products_repo import ProductRepository
from app.database.subscription_repo import SubscriptionRepository
from app.schemas.products import ProductWithPrimaryAsset, ProductsByUserResponse
from app.services.notification_service import NotificationService
from app.services.generation_estimate_service import generation_estimate_service
from app.integrations.tripo import TripoError, tripo_parts_pipeline


# Asset id under which a USDZ is stored, matching the GLB->USDZ conversion job
# and the viewer's lookup (Rivollo.Viewer.Api ProductAssetsRepository: asset 9
# is the GLB, asset 11 the USDZ).
USDZ_ASSET_ID = 11

# Stats/estimate key for the two-stage Tripo pipeline. Deliberately distinct
# from the single-stage fal "tripo" so their durations never mix.
TRIPO_PARTS_MODEL_KEY = "tripo-parts"


def _compress_mesh_for_storage(
    glb_bytes: bytes, *, product_id, context: str = ""
) -> tuple[bytes, Optional["CompressedMeshPackage"]]:
    """Draco-compress a generated GLB, and build the glTF package alongside it.

    Returns ``(glb_to_upload, gltf_package_or_None)``. Never raises: every
    generation path treats compression as an optimisation, so any failure
    degrades to the next-best artifact rather than losing a product the seller
    already paid to generate.

    The degradation ladder is deliberate:
        package  ->  GLB-only  ->  original uncompressed bytes
    A glTF-specific failure therefore still leaves asset id 9 with a properly
    Draco-compressed GLB, exactly as before this feature existed.
    """
    logger = logging.getLogger(__name__)

    if not settings.ENABLE_DRACO_COMPRESSION:
        return glb_bytes, None

    if settings.ENABLE_GLTF_DRACO_PACKAGE:
        try:
            package = glb_compression_service.compress_package(glb_bytes)
            return package.glb, package
        except Exception:
            logger.warning(
                "Draco glTF package build failed for product %s%s — "
                "falling back to GLB-only compression",
                product_id, context, exc_info=True,
            )

    try:
        return glb_compression_service.compress(glb_bytes), None
    except Exception:
        logger.warning(
            "Draco compression failed for product %s%s — uploading original GLB",
            product_id, context, exc_info=True,
        )
        return glb_bytes, None


async def _persist_gltf_draco_asset(
    db: AsyncSession,
    *,
    package: Optional["CompressedMeshPackage"],
    user_id,
    product_id,
    name: str,
) -> Optional[str]:
    """Upload the glTF package and map it to the product. Never raises.

    MUST be called only after the GLB asset has been committed. It runs in its
    own transaction so that a failure here — bad blob credentials, a missing
    tbl_asset row because the seed script has not been applied yet — rolls back
    only these rows and leaves the product READY with its GLB, which is still
    what every current client reads.

    Returns the CDN URL of the .gltf entry file, or None if nothing was stored.
    """
    if package is None:
        return None

    logger = logging.getLogger(__name__)
    try:
        gltf_url, _gltf_blob_url, prefix = storage_service.upload_gltf_package(
            user_id=str(user_id),
            product_id=str(product_id),
            files=package.gltf_files,
            entry=package.gltf_entry,
        )

        gltf_asset = ProductAsset(
            asset_id=settings.GLTF_DRACO_ASSET_ID,
            image=gltf_url,
            size_bytes=package.gltf_total_bytes,
            created_by=user_id,
        )
        db.add(gltf_asset)
        await db.flush()

        db.add(
            ProductAssetMapping(
                name=name,
                productid=product_id,
                product_asset_id=gltf_asset.id,
                isactive=True,
                created_by=user_id,
            )
        )
        await db.commit()

        logger.info(
            "Draco glTF package stored for product %s: prefix=%s  files=%d  url=%s",
            product_id, prefix, len(package.gltf_files), gltf_url,
        )
        return gltf_url
    except Exception:
        logger.warning(
            "Failed to store Draco glTF package for product %s — "
            "product keeps its GLB (asset %s) and stays READY",
            product_id, settings.GLTF_DRACO_ASSET_ID, exc_info=True,
        )
        try:
            await db.rollback()
        except Exception:
            pass
        return None

_MAX_REMOTE_MASK_IMAGE_BYTES = 25 * 1024 * 1024
_ALLOWED_REMOTE_MASK_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class ProductService:
    """Service for product operations."""

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL-friendly slug."""
        import re

        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "-", text)
        return text[:100]

    @staticmethod
    async def _generate_unique_slug(
        db: AsyncSession, base_slug: str, exclude_id: Optional[uuid.UUID] = None
    ) -> str:
        """Generate a unique slug for a product."""
        pattern = f"{base_slug}%"
        res = await db.execute(
            select(Product.slug, Product.id).where(Product.slug.like(pattern))
        )
        rows = res.all()
        existing = {slug for slug, pid in rows if exclude_id is None or pid != exclude_id}
        if base_slug not in existing:
            return base_slug
        i = 2
        while True:
            cand = f"{base_slug}-{i}"
            if cand not in existing:
                return cand
            i += 1

    @staticmethod
    async def _download_remote_mask_image(source_url: str) -> tuple[io.BytesIO, str, str, int]:
        """Download a remote mask image and return stream, filename, content type, and size."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            ) as client:
                response = await client.get(source_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("Failed to download mask image URL") from exc

        content = response.content
        content_size = len(content)
        if content_size == 0:
            raise RuntimeError("Downloaded mask image is empty")
        if content_size > _MAX_REMOTE_MASK_IMAGE_BYTES:
            raise RuntimeError("Mask image exceeds the 25 MB limit")

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in _ALLOWED_REMOTE_MASK_CONTENT_TYPES:
            raise RuntimeError("Mask image URL must return a supported image content type")

        extension = _CONTENT_TYPE_EXTENSIONS[content_type]
        url_extension = os.path.splitext(httpx.URL(source_url).path)[1].lower()
        if url_extension in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            extension = ".jpg" if url_extension == ".jpeg" else url_extension

        return io.BytesIO(content), f"mask{extension}", content_type, content_size

    @staticmethod
    async def create_product_with_image(
        db: AsyncSession,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        name: str,
        asset_id: int,
        mask_asset_id: int,
        target_format: str,
        mesh_asset_id: int,
        image_stream: BinaryIO,
        image_filename: str,
        image_content_type: Optional[str] = None,
        image_size_bytes: Optional[int] = None,

        #  NEW MASK PARAMETERS
        mask_stream: Optional[BinaryIO] = None,
        mask_filename: Optional[str] = None,
        mask_content_type: Optional[str] = None,
        mask_size_bytes: Optional[int] = None,
        quality: Optional[str] = None,
        with_mesh_postprocess: Optional[bool] = None,
        with_texture_baking: Optional[bool] = None,
        use_vertex_color: Optional[bool] = None,
        simplify: Optional[float] = None,
        fill_holes: Optional[bool] = None,
        texture_size: Optional[int] = None,
    ) -> tuple[Product, str, Optional[str], Optional[str], dict]:
        """Create a product and publish processing request to Service Bus.""" 
             
        logger = logging.getLogger(__name__)

        # -------------------------------
        # 1. Create product (DB)
        # -------------------------------
        base_slug = ProductService._slugify(name)
        slug = await ProductService._generate_unique_slug(db, base_slug)

        product = Product(
            name=name,
            slug=slug,
            status=ProductStatus.DRAFT,
            created_by=user_id,
        )

        db.add(product)
        await db.flush()

        product_id = str(product.id)
        user_id_str = str(user_id)

        # -------------------------------
        # 2. Upload original image
        # -------------------------------
        try:
            image_stream.seek(0)
            cdn_url, blob_url = storage_service.upload_product_image(
                user_id=user_id_str,
                product_id=product_id,
                filename=image_filename,
                content_type=image_content_type,
                stream=image_stream,
            )
            logger.info("Image uploaded: cdn=%s blob=%s", cdn_url, blob_url)
        except Exception as e:
            logger.exception("Image upload failed")
            raise RuntimeError("Failed to upload product image") from e

        # -------------------------------
        # 3. Upload MASK image (NEW)
        # -------------------------------
        mask_cdn_url: Optional[str] = None
        mask_blob_url: Optional[str] = None

        if mask_stream and mask_filename:
            try:
                mask_stream.seek(0)
                mask_cdn_url, mask_blob_url = storage_service.upload_product_image(
                    user_id=user_id_str,
                    product_id=product_id,
                    filename=f"mask_{mask_filename}",
                    content_type=mask_content_type,
                    stream=mask_stream,
                )
                logger.info("Mask uploaded: cdn=%s blob=%s", mask_cdn_url, mask_blob_url)
            except Exception as e:
                logger.exception("Mask upload failed")
                raise RuntimeError("Failed to upload mask image") from e

        # -------------------------------
        # 4. Create ProductAsset + Mapping
        # -------------------------------
        try:
            product_asset = ProductAsset(
                asset_id=1,
                image=blob_url,
                size_bytes=image_size_bytes,
                created_by=user_id,
            )
            db.add(product_asset)
            await db.flush()
            await db.refresh(product_asset)

            product_asset_mapping = ProductAssetMapping(
                name=name,
                productid=product.id,
                product_asset_id=product_asset.id,
                isactive=True,
                created_by=user_id,
            )
            db.add(product_asset_mapping)

            # -------------------------------
            # 5. Create ProductAsset (MASK) - NEW
            # -------------------------------
            if mask_blob_url:
                mask_asset = ProductAsset(
                    asset_id=mask_asset_id,
                    image=mask_blob_url,
                    size_bytes=mask_size_bytes,
                    created_by=user_id,
                )
                db.add(mask_asset)
                await db.flush()
                await db.refresh(mask_asset)

                mask_asset_mapping = ProductAssetMapping(
                    name=f"{name} mask",
                    productid=product.id,
                    product_asset_id=mask_asset.id,
                    isactive=True,
                    created_by=user_id,
                )
                db.add(mask_asset_mapping)

        except Exception as e:
            logger.exception("Failed to create asset records")
            # await db.rollback()
            raise RuntimeError("Failed to create asset entries") from e

        # -------------------------------
        # 6. Commit with status = DRAFT
        # The background task will probe the 3D service /health, then flip to
        # PROCESSING via generate_3d_and_finalize once the service is reachable.
        # -------------------------------
        product.status = ProductStatus.DRAFT
        logger.info("Product %s committed as DRAFT pending 3D readiness probe", product.id)
        await db.commit()

        gpu_status = await gpu_status_service.get_status(
            db=db,
            touch_activation=True,
        )

        try:
            await db.refresh(product)
        except Exception:
            pass

        # -------------------------------
        # 7. Fire-and-forget 3D generation — returns immediately
        # -------------------------------
        logger.info("Scheduling background 3D generation for product %s", product.id)
        background_tasks.add_task(
            ProductService._run_3d_generation_background,
            user_id=user_id,
            product_id=product.id,
            asset_id=asset_id,
            mesh_asset_id=mesh_asset_id,
            name=name,
            target_format=target_format,
            blob_url=blob_url,
            mask_blob_url=mask_blob_url,
            quality=quality,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            use_vertex_color=use_vertex_color,
            simplify=simplify,
            fill_holes=fill_holes,
            texture_size=texture_size,
        )

        return product, cdn_url, mask_cdn_url, None, gpu_status

    @staticmethod
    async def create_product_with_image_urls(
        db: AsyncSession,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        name: str,
        mask_asset_id: int,
        mesh_asset_id: int,
        image_url: str,
        mask_image_url: str,
    ) -> tuple[Product, str, str]:
        """Create a product from existing original/mask image URLs.

        The 3D model is generated by fal.ai's SAM-3 (``fal-ai/sam-3/3d-objects``)
        in a background task, so this returns as soon as the product and its
        image/mask assets are stored. Returns ``(product, image_url, mask_cdn_url)``;
        the product is DRAFT here and flips to QUEUE -> PROCESSING -> READY as the
        background task progresses.
        """
        logger = logging.getLogger(__name__)

        base_slug = ProductService._slugify(name)
        slug = await ProductService._generate_unique_slug(db, base_slug)

        product = Product(
            name=name,
            slug=slug,
            status=ProductStatus.DRAFT,
            created_by=user_id,
        )

        db.add(product)
        await db.flush()

        product_id = str(product.id)
        user_id_str = str(user_id)
        try:
            mask_stream, mask_filename, mask_content_type, mask_size_bytes = (
                await ProductService._download_remote_mask_image(mask_image_url)
            )
            mask_stream.seek(0)
            mask_cdn_url, mask_blob_url = storage_service.upload_product_image(
                user_id=user_id_str,
                product_id=product_id,
                filename=mask_filename,
                content_type=mask_content_type,
                stream=mask_stream,
            )
            logger.info("Remote mask copied to Azure: source=%s blob=%s", mask_image_url, mask_blob_url)
        except Exception as exc:
            logger.exception("Failed to copy remote mask image to Azure")
            await db.rollback()
            raise RuntimeError("Failed to store mask image in Azure storage") from exc

        try:
            product_asset = ProductAsset(
                asset_id=1,
                image=image_url,
                created_by=user_id,
            )
            db.add(product_asset)
            await db.flush()
            await db.refresh(product_asset)

            db.add(
                ProductAssetMapping(
                    name=name,
                    productid=product.id,
                    product_asset_id=product_asset.id,
                    isactive=True,
                    created_by=user_id,
                )
            )

            mask_asset = ProductAsset(
                asset_id=mask_asset_id,
                image=mask_blob_url,
                size_bytes=mask_size_bytes,
                created_by=user_id,
            )
            db.add(mask_asset)
            await db.flush()
            await db.refresh(mask_asset)

            db.add(
                ProductAssetMapping(
                    name=f"{name} mask",
                    productid=product.id,
                    product_asset_id=mask_asset.id,
                    isactive=True,
                    created_by=user_id,
                )
            )
        except Exception as exc:
            logger.exception("Failed to create product asset records from URLs")
            await db.rollback()
            raise RuntimeError("Failed to create asset entries") from exc

        product.status = ProductStatus.DRAFT
        logger.info("Product %s committed as DRAFT pending 3D generation", product.id)
        await db.commit()

        try:
            await db.refresh(product)
        except Exception:
            pass

        logger.info("Scheduling background fal SAM-3 generation for product %s", product.id)
        background_tasks.add_task(
            ProductService._run_sam3_fal_generation_background,
            user_id=user_id,
            product_id=product.id,
            mesh_asset_id=mesh_asset_id,
            name=name,
            blob_url=image_url,
            mask_blob_url=mask_blob_url,
        )

        return product, image_url, mask_cdn_url

    @staticmethod
    async def create_product_from_glb(
        db: AsyncSession,
        user_id: uuid.UUID,
        name: str,
        mesh_asset_id: int,
        glb_stream: BinaryIO,
        glb_size_bytes: Optional[int] = None,
        glb_content_type: Optional[str] = None,
        image_stream: Optional[BinaryIO] = None,
        image_filename: Optional[str] = None,
        image_content_type: Optional[str] = None,
        image_size_bytes: Optional[int] = None,
    ) -> tuple[Product, str, Optional[str]]:
        """Create a product directly from an uploaded GLB file (no AI generation).

        The user supplies a ready ``.glb``; we store it under the mesh asset
        (``asset_id == mesh_asset_id``, i.e. asset 9) exactly like the AI paths do
        in :meth:`generate_3d_and_finalize_fal`, and optionally store a thumbnail
        image under asset 1. Because the mesh already exists, this runs
        synchronously and the product lands in READY immediately.

        Returns ``(product, glb_url, image_url)`` where ``image_url`` is None when
        no thumbnail was supplied.
        """
        logger = logging.getLogger(__name__)

        # -------------------------------
        # 1. Create product (DRAFT until assets are stored)
        # -------------------------------
        base_slug = ProductService._slugify(name)
        slug = await ProductService._generate_unique_slug(db, base_slug)

        product = Product(
            name=name,
            slug=slug,
            status=ProductStatus.DRAFT,
            created_by=user_id,
        )
        db.add(product)
        await db.flush()

        product_id = str(product.id)
        user_id_str = str(user_id)

        # -------------------------------
        # 2. Upload the GLB to Azure storage
        # -------------------------------
        try:
            glb_stream.seek(0)
            # Use the CDN URL returned by storage — it already includes the
            # "/<container>/" segment (e.g. "/dev/"), matching the fal GLB path
            # (generate_3d_and_finalize_fal). Do NOT hand-build the URL: that
            # dropped the container segment and produced broken links.
            glb_url, glb_blob_url = storage_service.upload_product_image(
                user_id=user_id_str,
                product_id=product_id,
                filename="model.glb",
                content_type=glb_content_type or "model/gltf-binary",
                stream=glb_stream,
            )
            logger.info(
                "Direct GLB uploaded: product=%s blob=%s url=%s",
                product_id, glb_blob_url, glb_url,
            )
        except Exception as exc:
            logger.exception("Direct GLB upload failed for product %s", product_id)
            await db.rollback()
            raise RuntimeError("Failed to upload GLB file") from exc

        # -------------------------------
        # 3. Optional thumbnail image (asset 1)
        # -------------------------------
        image_url: Optional[str] = None
        if image_stream and image_filename:
            try:
                image_stream.seek(0)
                # Keep the CDN URL returned by storage verbatim — it already
                # includes the "/<container>/" segment (e.g. "/dev/"), matching
                # the GLB asset above and the fal GLB path.
                image_url, image_blob_url = storage_service.upload_product_image(
                    user_id=user_id_str,
                    product_id=product_id,
                    filename=image_filename,
                    content_type=image_content_type,
                    stream=image_stream,
                )
                logger.info(
                    "Direct GLB thumbnail uploaded: product=%s blob=%s url=%s",
                    product_id, image_blob_url, image_url,
                )
            except Exception as exc:
                logger.exception("Thumbnail upload failed for product %s", product_id)
                await db.rollback()
                raise RuntimeError("Failed to upload thumbnail image") from exc

        # -------------------------------
        # 4. Persist asset rows + mappings, mark READY
        # -------------------------------
        try:
            if image_url:
                image_asset = ProductAsset(
                    asset_id=1,
                    image=image_url,
                    size_bytes=image_size_bytes,
                    created_by=user_id,
                )
                db.add(image_asset)
                await db.flush()
                db.add(
                    ProductAssetMapping(
                        name=name,
                        productid=product.id,
                        product_asset_id=image_asset.id,
                        isactive=True,
                        created_by=user_id,
                    )
                )

            glb_asset = ProductAsset(
                asset_id=mesh_asset_id,
                image=glb_url,
                size_bytes=glb_size_bytes,
                created_by=user_id,
            )
            db.add(glb_asset)
            await db.flush()
            db.add(
                ProductAssetMapping(
                    name=name,
                    productid=product.id,
                    product_asset_id=glb_asset.id,
                    isactive=True,
                    created_by=user_id,
                )
            )

            product.status = ProductStatus.READY
            await db.commit()
            await db.refresh(product)
            logger.info("Product %s → READY (direct GLB) glb_url=%s", product_id, glb_url)
        except Exception as exc:
            logger.exception("Failed to persist direct-GLB assets for product %s", product_id)
            await db.rollback()
            raise RuntimeError("Failed to save product assets") from exc

        # Best-effort ready notification (never blocks the response)
        try:
            await NotificationService.create_and_push_notification(
                db=db,
                user_id=user_id,
                notification_type="product.ready",
                title="Product Ready",
                body=f"Your product '{name}' is ready.",
                data={
                    "product_id": product_id,
                    "product_name": name,
                    "status": ProductStatus.READY.value,
                    "glb_url": glb_url,
                },
            )
        except Exception:
            logger.warning(
                "Failed to send product-ready notification for product %s",
                product_id, exc_info=True,
            )
            try:
                await db.rollback()
            except Exception:
                pass

        return product, glb_url, image_url

    @staticmethod
    async def create_product_with_fal_image_urls(
        db: AsyncSession,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        name: str,
        asset_id: int,
        mesh_asset_id: int,
        image_url: str,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> tuple[Product, str, Optional[str]]:
        """Create a product from an uploaded image URL, generating 3D via fal.ai.

        Shares the same internals as :meth:`create_product_with_image_urls`
        (product record, original image asset, background 3D generation) but fal
        needs only the image — there is no mask to download/store and none of the
        SAM tuning options or GPU-warmth probe apply.

        Returns ``(product, image_url, glb_url)``; ``glb_url`` is None because
        generation runs in a background task.
        """
        logger = logging.getLogger(__name__)

        base_slug = ProductService._slugify(name)
        slug = await ProductService._generate_unique_slug(db, base_slug)

        product = Product(
            name=name,
            slug=slug,
            status=ProductStatus.DRAFT,
            created_by=user_id,
        )

        db.add(product)
        await db.flush()

        try:
            product_asset = ProductAsset(
                asset_id=1,
                image=image_url,
                created_by=user_id,
            )
            db.add(product_asset)
            await db.flush()
            await db.refresh(product_asset)

            db.add(
                ProductAssetMapping(
                    name=name,
                    productid=product.id,
                    product_asset_id=product_asset.id,
                    isactive=True,
                    created_by=user_id,
                )
            )
        except Exception as exc:
            logger.exception("Failed to create product asset records from URL")
            await db.rollback()
            raise RuntimeError("Failed to create asset entries") from exc

        product.status = ProductStatus.DRAFT
        logger.info("Product %s committed as DRAFT pending fal 3D generation", product.id)
        await db.commit()

        try:
            await db.refresh(product)
        except Exception:
            pass

        logger.info(
            "Scheduling background fal 3D generation for product %s (model=%s)",
            product.id, model_key,
        )
        background_tasks.add_task(
            ProductService._run_fal_3d_generation_background,
            user_id=user_id,
            product_id=product.id,
            mesh_asset_id=mesh_asset_id,
            name=name,
            blob_url=image_url,
            model_key=model_key,
        )

        return product, image_url, None

    @staticmethod
    async def generate_3d_and_finalize(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        asset_id: int,
        mesh_asset_id: int,
        name: str,
        target_format: str,
        blob_url: str,
        mask_blob_url: str,
        quality: Optional[str] = None,
        with_mesh_postprocess: Optional[bool] = None,
        with_texture_baking: Optional[bool] = None,
        use_vertex_color: Optional[bool] = None,
        simplify: Optional[float] = None,
        fill_holes: Optional[bool] = None,
        texture_size: Optional[int] = None,
    ) -> str:
        """
        Synchronously call the 3D generation API and persist the GLB asset.

        Sets the product status to PROCESSING before calling the API, then
        READY on success. Raises RuntimeError on failure so the caller can
        surface a clean HTTP 500.

        Returns:
            glb_url: The URL of the generated GLB file.
        """
        logger = logging.getLogger(__name__)

        # 1. Mark product as PROCESSING so callers / UI can show in-progress state
        product = await db.get(Product, product_id)
        if not product:
            raise RuntimeError(f"Product {product_id} not found")

        product.status = ProductStatus.PROCESSING
        await db.commit()
        logger.info("Product %s status → PROCESSING", product_id)

        # 2. Call the 3D generation API (blocks until response arrives)
        response = await threed_model_client.generate_3d(
            product_id=product_id,
            user_id=user_id,
            blob_url=blob_url,
            mask_blob_url=mask_blob_url,
            target_format=target_format,
            asset_id=asset_id,
            mesh_asset_id=mesh_asset_id,
            name=name,
            quality=quality,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            use_vertex_color=use_vertex_color,
            simplify=simplify,
            fill_holes=fill_holes,
            texture_size=texture_size,
        )

        # Re-fetch in case the session state drifted during the long await
        product = await db.get(Product, product_id)
        if not product:
            raise RuntimeError(f"Product {product_id} disappeared during 3D generation")

        # 3. Handle failure
        if not response.success:
            error_msg = response.error or "Unknown error from 3D generation API"
            logger.error("3D generation failed for product %s: %s", product_id, error_msg)
            product.status = ProductStatus.DRAFT
            await db.commit()
            raise RuntimeError(f"3D generation failed: {error_msg}")

        glb_url: Optional[str] = response.glb_url
        if not glb_url:
            logger.error("3D API returned success but no GLB URL for product %s", product_id)
            product.status = ProductStatus.DRAFT
            await db.commit()
            raise RuntimeError("3D generation succeeded but returned no GLB URL")

        # 4. Persist only the GLB asset + mapping
        try:
            glb_asset = ProductAsset(
                asset_id=mesh_asset_id,
                image=glb_url,
                created_by=user_id,
            )
            db.add(glb_asset)
            await db.flush()

            glb_mapping = ProductAssetMapping(
                name=name,
                productid=product_id,
                product_asset_id=glb_asset.id,
                isactive=True,
                created_by=user_id,
            )
            db.add(glb_mapping)

            # 5. Mark as READY
            product.status = ProductStatus.READY
            await db.commit()
            logger.info("Product %s → READY  glb_url=%s", product_id, glb_url)

            try:
                await NotificationService.create_and_push_notification(
                    db=db,
                    user_id=user_id,
                    notification_type="product.ready",
                    title="Product Ready",
                    body=f"Your product '{name}' is ready.",
                    data={
                        "product_id": str(product_id),
                        "product_name": name,
                        "status": ProductStatus.READY.value,
                        "glb_url": glb_url,
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to send product-ready notification for product %s",
                    product_id,
                    exc_info=True,
                )
                try:
                    await db.rollback()
                except Exception:
                    pass

        except Exception as exc:
            logger.exception("Failed to persist GLB asset for product %s", product_id)
            await db.rollback()
            try:
                product = await db.get(Product, product_id)
                if product:
                    product.status = ProductStatus.DRAFT
                    await db.commit()
            except Exception:
                pass
            raise RuntimeError("Failed to save GLB asset to database") from exc

        return glb_url

    @staticmethod
    async def _run_3d_generation_background(
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        asset_id: int,
        mesh_asset_id: int,
        name: str,
        target_format: str,
        blob_url: str,
        mask_blob_url: str,
        quality: Optional[str] = None,
        with_mesh_postprocess: Optional[bool] = None,
        with_texture_baking: Optional[bool] = None,
        use_vertex_color: Optional[bool] = None,
        simplify: Optional[float] = None,
        fill_holes: Optional[bool] = None,
        texture_size: Optional[int] = None,
    ) -> None:
        """Open a fresh DB session and run generate_3d_and_finalize in the background."""
        from app.core.db import new_session
        from app.api.websocket.broadcaster import broadcaster
        logger = logging.getLogger(__name__)
        try:
            # Flip status to QUEUE immediately so the WebSocket and REST clients
            # see that work has started (fires pg_notify to any connected browsers).
            async with new_session() as db:
                product = await db.get(Product, product_id)
                if product:
                    product.status = ProductStatus.QUEUE
                    await db.commit()

            product_id_str = str(product_id)

            # Fast probe to determine warm vs cold-start.
            # Broadcast the estimate to connected WebSocket clients right away
            # so the browser can show a meaningful wait time before the full
            # polling loop begins.
            async with new_session() as db:
                gpu_status = await gpu_status_service.get_status(
                    db=db,
                    touch_activation=True,
                )

            is_warm = gpu_status["gpu_status"] == "warm"
            estimated_time = gpu_status["estimated_time"]
            gpu_message = gpu_status["message"]

            await broadcaster.broadcast_to_product(
                product_id_str,
                {
                    "new_status": ProductStatus.QUEUE.value,
                    "estimated_time": estimated_time,
                    "gpu_status": gpu_status["gpu_status"],
                    "message": gpu_message,
                },
            )
            logger.info(
                "GPU warmth check — product=%s  warm=%s  estimated_time=%s",
                product_id_str, is_warm, estimated_time,
            )

            # If warm the service already responded 200; skip the full polling loop.
            if not is_warm:
                ready = await threed_model_client.wait_until_ready(product_id=product_id)
                if ready:
                    await broadcaster.broadcast_to_product(
                        product_id_str,
                        {
                            "new_status": ProductStatus.QUEUE.value,
                            "estimated_time": 20,
                            "gpu_status": "warm",
                            "message": "GPU is now ready — your 3D model generation is starting.",
                        },
                    )
            else:
                ready = True

            if not ready:
                async with new_session() as db:
                    product = await db.get(Product, product_id)
                    if product:
                        product.status = ProductStatus.DRAFT
                        await db.commit()
                logger.error(
                    "3D service unreachable — product %s left in DRAFT for retry",
                    product_id,
                )
                return

            async with new_session() as db:
                await ProductService.generate_3d_and_finalize(
                    db=db,
                    user_id=user_id,
                    product_id=product_id,
                    asset_id=asset_id,
                    mesh_asset_id=mesh_asset_id,
                    name=name,
                    target_format=target_format,
                    blob_url=blob_url,
                    mask_blob_url=mask_blob_url,
                    quality=quality,
                    with_mesh_postprocess=with_mesh_postprocess,
                    with_texture_baking=with_texture_baking,
                    use_vertex_color=use_vertex_color,
                    simplify=simplify,
                    fill_holes=fill_holes,
                    texture_size=texture_size,
                )
        except Exception:
            logger.exception("Background 3D generation failed for product %s", product_id)

    @staticmethod
    async def generate_3d_and_finalize_fal(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        mesh_asset_id: int,
        name: str,
        blob_url: str,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> str:
        """
        Generate a GLB via a fal.ai image-to-3D model and persist it.

        Mirrors :meth:`generate_3d_and_finalize`; the only difference is the
        model-invocation step calls fal.ai instead of the SAM 3D service. fal
        takes only the public image URL (no mask, no SAM tuning params), and the
        GLB it returns is downloaded and re-uploaded to Azure so it lands in our
        internal DB format identically to the SAM path.

        ``model_key`` selects which fal model runs (see
        :mod:`app.integrations.fal.registry`). Everything after generation —
        Draco compression, upload, asset mapping, USDZ trigger — is identical
        regardless of which model produced the mesh.

        Returns:
            glb_url: The URL of the stored GLB file.
        """
        logger = logging.getLogger(__name__)
        spec = get_model_spec(model_key)

        # 1. Mark product as PROCESSING so callers / UI can show in-progress state
        product = await db.get(Product, product_id)
        if not product:
            raise RuntimeError(f"Product {product_id} not found")

        product.status = ProductStatus.PROCESSING
        await db.commit()
        logger.info("Product %s status → PROCESSING (fal/%s)", product_id, spec.key)

        # Start the clock here, not at the fal call: the seller waits for the
        # whole pipeline — queue, generation, download, compression, upload —
        # and that full duration is what feeds the ETA for the next run.
        started_at = time.perf_counter()

        # 2. Call the fal.ai API (submit → poll → result → download GLB).
        #    The image must be a publicly reachable URL — pass the blob URL.
        response = await fal_queue_client.generate_3d(
            spec=spec,
            product_id=product_id,
            image_url=blob_url,
        )

        # Re-fetch in case the session state drifted during the long await
        product = await db.get(Product, product_id)
        if not product:
            raise RuntimeError(f"Product {product_id} disappeared during 3D generation")

        # 3. Handle failure
        if not response.success:
            error_msg = response.error or f"Unknown error from fal.ai {spec.label} API"
            logger.error(
                "fal %s 3D generation failed for product %s (request_id=%s): %s",
                spec.key, product_id, response.request_id, error_msg,
            )
            product.status = ProductStatus.DRAFT
            await db.commit()
            # Recorded but excluded from the median — a fast failure must not
            # make the next estimate optimistic.
            await generation_estimate_service.record(
                db, spec.key, time.perf_counter() - started_at, succeeded=False
            )
            raise RuntimeError(f"3D generation failed: {error_msg}")

        if not response.glb_bytes:
            logger.error(
                "fal API succeeded but returned no GLB bytes for product %s (request_id=%s)",
                product_id, response.request_id,
            )
            product.status = ProductStatus.DRAFT
            await db.commit()
            raise RuntimeError("3D generation succeeded but returned no GLB data")

        # 3a. Draco-compress the mesh geometry before it goes anywhere near
        #     Azure, and build the Draco glTF package from the SAME pass.
        #     Non-fatal: on any failure the original GLB is uploaded.
        glb_bytes, gltf_package = _compress_mesh_for_storage(
            response.glb_bytes,
            product_id=product_id,
            context=f" (request_id={response.request_id})",
        )

        # 3b. Re-upload the downloaded GLB to Azure so it lands in our storage
        #     domain exactly like the SAM path's hosted glb_url.
        try:
            glb_stream = io.BytesIO(glb_bytes)
            glb_stream.seek(0)
            # Store the GLB the same way as the product image/mask: keep the
            # CDN URL returned by storage_service (built by _cdn_url) verbatim.
            glb_url, glb_blob_url = storage_service.upload_product_image(
                user_id=str(user_id),
                product_id=str(product_id),
                filename="model.glb",
                content_type=response.glb_content_type or "model/gltf-binary",
                stream=glb_stream,
            )
            logger.info(
                "fal GLB stored in Azure: product=%s  request_id=%s  blob=%s  url=%s",
                product_id, response.request_id, glb_blob_url, glb_url,
            )
        except Exception as exc:
            logger.exception(
                "Failed to store fal GLB in Azure for product %s (request_id=%s)",
                product_id, response.request_id,
            )
            product.status = ProductStatus.DRAFT
            await db.commit()
            raise RuntimeError("Failed to store generated GLB in Azure storage") from exc

        # 4. Persist only the GLB asset + mapping
        try:
            glb_asset = ProductAsset(
                asset_id=mesh_asset_id,
                image=glb_url,
                created_by=user_id,
            )
            db.add(glb_asset)
            await db.flush()

            glb_mapping = ProductAssetMapping(
                name=name,
                productid=product_id,
                product_asset_id=glb_asset.id,
                isactive=True,
                created_by=user_id,
            )
            db.add(glb_mapping)

            # 5. Mark as READY
            product.status = ProductStatus.READY
            await db.commit()
            elapsed = time.perf_counter() - started_at
            logger.info(
                "Product %s → READY  model=%s  elapsed=%.1fs  glb_url=%s",
                product_id, spec.key, elapsed, glb_url,
            )

            # 5a. Additive: the Draco glTF package for the viewer. Runs only
            #     after the product is committed READY, so a failure here can
            #     never cost the seller the GLB they waited for.
            await _persist_gltf_draco_asset(
                db, package=gltf_package, user_id=user_id,
                product_id=product_id, name=name,
            )

            # Feed the measured duration back so the next seller sees an ETA
            # based on what this model actually takes right now.
            await generation_estimate_service.record(db, spec.key, elapsed)

            try:
                await NotificationService.create_and_push_notification(
                    db=db,
                    user_id=user_id,
                    notification_type="product.ready",
                    title="Product Ready",
                    body=f"Your product '{name}' is ready.",
                    data={
                        "product_id": str(product_id),
                        "product_name": name,
                        "status": ProductStatus.READY.value,
                        "glb_url": glb_url,
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to send product-ready notification for product %s",
                    product_id,
                    exc_info=True,
                )
                try:
                    await db.rollback()
                except Exception:
                    pass

        except Exception as exc:
            logger.exception("Failed to persist GLB asset for product %s", product_id)
            await db.rollback()
            try:
                product = await db.get(Product, product_id)
                if product:
                    product.status = ProductStatus.DRAFT
                    await db.commit()
            except Exception:
                pass
            raise RuntimeError("Failed to save GLB asset to database") from exc

        # 6. USDZ for iOS Quick Look.
        #
        #    Two ways to obtain one:
        #      * the generator exported it itself (Meshy) — store those bytes
        #        directly; the vendor's own export beats a converted one on
        #        fidelity and costs no extra job
        #      * otherwise fire the Azure Container Apps Job, which downloads
        #        the GLB via the Azure Blob SDK (so it needs the DIRECT blob
        #        URL, NOT the CDN one), uploads the USDZ and writes its URL to
        #        the DB itself — we neither wait for nor persist that result
        #
        #    Either route is a paid-plan feature, so the entitlement check wraps
        #    BOTH: a vendor-supplied USDZ must not slip past the gate just
        #    because it arrived for free. The product is already READY with its
        #    GLB, so a non-eligible plan simply gets no USDZ.
        #    /createProductFal gates on plan too, but this runs in a background
        #    task long after that check, so it verifies the plan itself.
        try:
            from app.services.licensing_service import LicensingService

            if await LicensingService.can_create_usdz(db, user_id):
                stored_usdz = False
                if response.usdz_bytes:
                    try:
                        stored_usdz = await ProductService._store_vendor_usdz(
                            db=db,
                            user_id=user_id,
                            product_id=product_id,
                            name=name,
                            usdz_bytes=response.usdz_bytes,
                            content_type=response.usdz_content_type,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to store %s USDZ for product %s — falling back "
                            "to the conversion job (non-fatal)",
                            spec.key, product_id, exc_info=True,
                        )

                if not stored_usdz:
                    from app.services.usdz_trigger_service import usdz_trigger_service
                    await usdz_trigger_service.trigger_conversion(
                        glb_blob_url=glb_blob_url,
                        product_id=str(product_id),
                        user_id=str(user_id),
                        product_name=name,
                    )
            else:
                logger.info(
                    "USDZ skipped for product %s — user %s is not on a "
                    "USDZ-eligible plan. GLB remains available.",
                    product_id,
                    user_id,
                )
        except Exception:
            logger.warning(
                "Failed to schedule USDZ conversion for product %s (non-fatal)",
                product_id,
                exc_info=True,
            )

        return glb_url

    @staticmethod
    async def _store_vendor_usdz(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        name: str,
        usdz_bytes: bytes,
        content_type: Optional[str],
    ) -> bool:
        """Persist a USDZ the generator produced itself. Returns True on success.

        Lands in exactly the same shape the conversion job would have written,
        so the viewer's Apple-device lookup finds it with no special-casing.
        """
        logger = logging.getLogger(__name__)

        stream = io.BytesIO(usdz_bytes)
        stream.seek(0)
        usdz_url, _ = storage_service.upload_product_image(
            user_id=str(user_id),
            product_id=str(product_id),
            filename="model.usdz",
            content_type=content_type or "model/vnd.usdz+zip",
            stream=stream,
        )

        usdz_asset = ProductAsset(
            asset_id=USDZ_ASSET_ID,
            image=usdz_url,
            created_by=user_id,
        )
        db.add(usdz_asset)
        await db.flush()

        db.add(
            ProductAssetMapping(
                name=name,
                productid=product_id,
                product_asset_id=usdz_asset.id,
                isactive=True,
                created_by=user_id,
            )
        )
        await db.commit()

        logger.info(
            "Stored vendor USDZ for product %s (%d bytes) — skipping conversion job",
            product_id, len(usdz_bytes),
        )
        return True

    @staticmethod
    async def _run_fal_3d_generation_background(
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        mesh_asset_id: int,
        name: str,
        blob_url: str,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> None:
        """Open a fresh DB session and run generate_3d_and_finalize_fal in the background.

        Mirrors :meth:`_run_3d_generation_background`. fal.ai is a managed,
        queue-based service, so there is no self-hosted GPU to warm up — the
        SAM-VM readiness/warmth probe is intentionally omitted and fal's own
        poll loop handles the wait. The QUEUE-status flip and WebSocket
        broadcast are kept so the UI behaves identically.
        """
        from app.core.db import new_session
        from app.api.websocket.broadcaster import broadcaster
        logger = logging.getLogger(__name__)
        try:
            # Flip status to QUEUE immediately so the WebSocket and REST clients
            # see that work has started (fires pg_notify to any connected browsers).
            async with new_session() as db:
                product = await db.get(Product, product_id)
                if product:
                    product.status = ProductStatus.QUEUE
                    await db.commit()

            product_id_str = str(product_id)

            await broadcaster.broadcast_to_product(
                product_id_str,
                {
                    "new_status": ProductStatus.QUEUE.value,
                    "message": "Your 3D model generation is starting.",
                },
            )

            async with new_session() as db:
                await ProductService.generate_3d_and_finalize_fal(
                    db=db,
                    user_id=user_id,
                    product_id=product_id,
                    mesh_asset_id=mesh_asset_id,
                    name=name,
                    blob_url=blob_url,
                    model_key=model_key,
                )
        except Exception:
            logger.exception(
                "Background fal 3D generation failed for product %s (model=%s)",
                product_id, model_key,
            )

    @staticmethod
    async def generate_3d_and_finalize_sam3_fal(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        mesh_asset_id: int,
        name: str,
        blob_url: str,
        mask_blob_url: Optional[str] = None,
    ) -> str:
        """
        Generate a GLB via fal.ai SAM-3 (``fal-ai/sam-3/3d-objects``) and persist it.

        Mirrors :meth:`generate_3d_and_finalize_fal`. Unlike Tripo, SAM-3 accepts a
        mask, so the caller's mask blob URL is passed through as ``mask_urls``.
        Both URLs must be publicly reachable — fal fetches them server-side.

        Returns:
            glb_url: The URL of the stored GLB file.
        """
        logger = logging.getLogger(__name__)

        # 1. Mark product as PROCESSING so callers / UI can show in-progress state
        product = await db.get(Product, product_id)
        if not product:
            raise RuntimeError(f"Product {product_id} not found")

        product.status = ProductStatus.PROCESSING
        await db.commit()
        logger.info("Product %s status → PROCESSING (fal SAM-3)", product_id)

        # 2. Call fal SAM-3 (submit → poll → result → download GLB).
        #    export_textured_glb is left at the client's default (True): fal
        #    defaults it to False, which returns bare geometry with no texture.
        response = await fal_sam3_client.generate_3d(
            product_id=product_id,
            image_url=blob_url,
            mask_url=mask_blob_url,
        )

        # Re-fetch in case the session state drifted during the long await
        product = await db.get(Product, product_id)
        if not product:
            raise RuntimeError(f"Product {product_id} disappeared during 3D generation")

        # 3. Handle failure
        if not response.success:
            error_msg = response.error or "Unknown error from fal.ai SAM-3 API"
            logger.error(
                "fal SAM-3 generation failed for product %s (request_id=%s): %s",
                product_id, response.request_id, error_msg,
            )
            product.status = ProductStatus.DRAFT
            await db.commit()
            raise RuntimeError(f"3D generation failed: {error_msg}")

        if not response.glb_bytes:
            logger.error(
                "fal SAM-3 succeeded but returned no GLB bytes for product %s (request_id=%s)",
                product_id, response.request_id,
            )
            product.status = ProductStatus.DRAFT
            await db.commit()
            raise RuntimeError("3D generation succeeded but returned no GLB data")

        # 3a. Draco-compress the mesh geometry before it goes anywhere near
        #     Azure, and build the Draco glTF package from the SAME pass.
        #     Non-fatal: on any failure the original GLB is uploaded.
        glb_bytes, gltf_package = _compress_mesh_for_storage(
            response.glb_bytes,
            product_id=product_id,
            context=f" (request_id={response.request_id})",
        )

        # 3b. Re-upload the downloaded GLB to Azure so it lands in our storage
        #     domain exactly like the other generation paths.
        try:
            glb_stream = io.BytesIO(glb_bytes)
            glb_stream.seek(0)
            glb_url, glb_blob_url = storage_service.upload_product_image(
                user_id=str(user_id),
                product_id=str(product_id),
                filename="model.glb",
                content_type=response.glb_content_type or "model/gltf-binary",
                stream=glb_stream,
            )
            logger.info(
                "fal SAM-3 GLB stored in Azure: product=%s  request_id=%s  blob=%s  url=%s",
                product_id, response.request_id, glb_blob_url, glb_url,
            )
        except Exception as exc:
            logger.exception(
                "Failed to store fal SAM-3 GLB in Azure for product %s (request_id=%s)",
                product_id, response.request_id,
            )
            product.status = ProductStatus.DRAFT
            await db.commit()
            raise RuntimeError("Failed to store generated GLB in Azure storage") from exc

        # 4. Persist only the GLB asset + mapping
        try:
            glb_asset = ProductAsset(
                asset_id=mesh_asset_id,
                image=glb_url,
                created_by=user_id,
            )
            db.add(glb_asset)
            await db.flush()

            glb_mapping = ProductAssetMapping(
                name=name,
                productid=product_id,
                product_asset_id=glb_asset.id,
                isactive=True,
                created_by=user_id,
            )
            db.add(glb_mapping)

            # 5. Mark as READY
            product.status = ProductStatus.READY
            await db.commit()
            logger.info("Product %s → READY (fal SAM-3)  glb_url=%s", product_id, glb_url)

            # Additive: the Draco glTF package for the viewer. Runs only after
            # the product is committed READY — see _persist_gltf_draco_asset.
            await _persist_gltf_draco_asset(
                db, package=gltf_package, user_id=user_id,
                product_id=product_id, name=name,
            )

            try:
                await NotificationService.create_and_push_notification(
                    db=db,
                    user_id=user_id,
                    notification_type="product.ready",
                    title="Product Ready",
                    body=f"Your product '{name}' is ready.",
                    data={
                        "product_id": str(product_id),
                        "product_name": name,
                        "status": ProductStatus.READY.value,
                        "glb_url": glb_url,
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to send product-ready notification for product %s",
                    product_id,
                    exc_info=True,
                )
                try:
                    await db.rollback()
                except Exception:
                    pass

        except Exception as exc:
            logger.exception("Failed to persist GLB asset for product %s", product_id)
            await db.rollback()
            try:
                product = await db.get(Product, product_id)
                if product:
                    product.status = ProductStatus.DRAFT
                    await db.commit()
            except Exception:
                pass
            raise RuntimeError("Failed to save GLB asset to database") from exc

        # 6. Fire-and-forget: trigger the GLB -> USDZ converter (Azure Container
        #    Apps Job), which needs the DIRECT blob URL — it downloads via the
        #    Azure Blob SDK. USDZ is a paid-plan feature; a non-eligible plan
        #    keeps the GLB and simply skips the trigger.
        try:
            from app.services.licensing_service import LicensingService

            if await LicensingService.can_create_usdz(db, user_id):
                from app.services.usdz_trigger_service import usdz_trigger_service
                await usdz_trigger_service.trigger_conversion(
                    glb_blob_url=glb_blob_url,
                    product_id=str(product_id),
                    user_id=str(user_id),
                    product_name=name,
                )
            else:
                logger.info(
                    "USDZ conversion skipped for product %s — user %s is not on a "
                    "USDZ-eligible plan. GLB remains available.",
                    product_id,
                    user_id,
                )
        except Exception:
            logger.warning(
                "Failed to schedule USDZ conversion for product %s (non-fatal)",
                product_id,
                exc_info=True,
            )

        return glb_url

    @staticmethod
    async def _run_sam3_fal_generation_background(
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        mesh_asset_id: int,
        name: str,
        blob_url: str,
        mask_blob_url: Optional[str] = None,
    ) -> None:
        """Open a fresh DB session and run generate_3d_and_finalize_sam3_fal.

        Mirrors :meth:`_run_fal_3d_generation_background`: fal is a managed,
        queue-based service, so the self-hosted SAM-VM readiness/warmth probe is
        intentionally omitted and fal's own poll loop handles the wait.
        """
        from app.core.db import new_session
        logger = logging.getLogger(__name__)
        try:
            # Flip status to QUEUE immediately so the WebSocket and REST clients
            # see that work has started (fires pg_notify to any connected browsers).
            async with new_session() as db:
                product = await db.get(Product, product_id)
                if product:
                    product.status = ProductStatus.QUEUE
                    await db.commit()

            async with new_session() as db:
                await ProductService.generate_3d_and_finalize_sam3_fal(
                    db=db,
                    user_id=user_id,
                    product_id=product_id,
                    mesh_asset_id=mesh_asset_id,
                    name=name,
                    blob_url=blob_url,
                    mask_blob_url=mask_blob_url,
                )
        except Exception:
            logger.exception(
                "Background fal SAM-3 generation failed for product %s", product_id
            )

    # ------------------------------------------------------------------ #
    # Tripo parts pipeline — segmented + textured mesh
    #
    # Kept entirely separate from the fal paths above. It speaks a different
    # protocol, bills against a different account and runs two chained tasks,
    # so sharing code with them would couple things that fail differently.
    # ------------------------------------------------------------------ #

    @staticmethod
    async def create_product_with_parts_image_urls(
        db: AsyncSession,
        background_tasks: BackgroundTasks,
        user_id: uuid.UUID,
        name: str,
        asset_id: int,
        mesh_asset_id: int,
        image_url: str,
    ) -> tuple[Product, str, Optional[str]]:
        """Create a product whose 3D model is generated as separate parts.

        Mirrors the shape of the fal creation methods: the product row and its
        image asset commit immediately, generation is scheduled in the
        background, and ``glb_url`` is None because nothing exists yet.
        """
        logger = logging.getLogger(__name__)

        base_slug = ProductService._slugify(name)
        slug = await ProductService._generate_unique_slug(db, base_slug)

        product = Product(
            name=name,
            slug=slug,
            status=ProductStatus.DRAFT,
            created_by=user_id,
        )
        db.add(product)
        await db.flush()

        try:
            product_asset = ProductAsset(
                asset_id=1,
                image=image_url,
                created_by=user_id,
            )
            db.add(product_asset)
            await db.flush()
            await db.refresh(product_asset)

            db.add(
                ProductAssetMapping(
                    name=name,
                    productid=product.id,
                    product_asset_id=product_asset.id,
                    isactive=True,
                    created_by=user_id,
                )
            )
        except Exception as exc:
            await db.rollback()
            raise RuntimeError("Failed to store the product image") from exc

        await db.commit()

        try:
            await db.refresh(product)
        except Exception:
            pass

        logger.info(
            "Scheduling background Tripo parts generation for product %s", product.id
        )
        background_tasks.add_task(
            ProductService._run_parts_generation_background,
            user_id=user_id,
            product_id=product.id,
            mesh_asset_id=mesh_asset_id,
            name=name,
            image_url=image_url,
        )

        return product, image_url, None

    @staticmethod
    async def generate_3d_with_parts_and_finalize(
        db: AsyncSession,
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        mesh_asset_id: int,
        name: str,
        image_url: str,
        resume_geometry_task_id: Optional[str] = None,
    ) -> str:
        """Run the two-stage Tripo pipeline and persist the resulting GLB.

        Progress is broadcast from Tripo's own ``progress`` field rather than
        estimated — the main operational advantage of this provider over fal.

        ``resume_geometry_task_id`` skips stage 1 when a previous attempt got
        that far: geometry is the expensive half and is already paid for.
        """
        logger = logging.getLogger(__name__)
        from app.api.websocket.broadcaster import broadcaster

        product = await db.get(Product, product_id)
        if not product:
            raise RuntimeError(f"Product {product_id} not found")

        product.status = ProductStatus.PROCESSING
        await db.commit()
        logger.info("Product %s status -> PROCESSING (tripo-parts)", product_id)

        # Written BEFORE generation starts so a crash mid-run still leaves a
        # trace that can be resumed.
        job = Job(
            product_id=product_id,
            status=ProductStatus.PROCESSING.value,
            engine=TRIPO_PARTS_MODEL_KEY,
            stage="geometry",
            created_by=user_id,
        )
        db.add(job)
        await db.commit()
        job_id = job.id

        started_at = time.perf_counter()

        # Tripo reports progress continuously; logging every change would emit
        # ~100 lines per run. Log at 10% milestones so a long generation visibly
        # advances in the console without drowning it.
        last_logged_decile = -1

        async def on_progress(percent: int, label: str) -> None:
            nonlocal last_logged_decile
            decile = percent // 10
            if decile != last_logged_decile:
                last_logged_decile = decile
                logger.info(
                    "Tripo parts %3d%%  %s  (product %s)", percent, label, product_id
                )

            # Fire-and-forget: a dropped socket must never fail generation.
            try:
                await broadcaster.broadcast_to_product(
                    str(product_id),
                    {
                        "new_status": ProductStatus.PROCESSING.value,
                        "message": label,
                        "progress": percent,
                    },
                )
            except Exception:
                logger.debug(
                    "Progress broadcast failed for %s", product_id, exc_info=True
                )

        try:
            result = await tripo_parts_pipeline.generate(
                image_url=image_url,
                on_progress=on_progress,
                geometry_task_id=resume_geometry_task_id,
            )
        except TripoError as exc:
            elapsed = time.perf_counter() - started_at
            logger.error(
                "Tripo parts generation failed for product %s (task=%s): %s",
                product_id, exc.task_id, exc,
            )
            await db.rollback()
            failed_product = await db.get(Product, product_id)
            if failed_product:
                failed_product.status = ProductStatus.DRAFT
            failed_job = await db.get(Job, job_id)
            if failed_job:
                failed_job.status = "failed"
                # Persist whichever task id we reached so a retry can resume.
                if exc.task_id:
                    failed_job.external_task_id = exc.task_id
            await db.commit()
            await generation_estimate_service.record(
                db, TRIPO_PARTS_MODEL_KEY, elapsed, succeeded=False
            )
            raise RuntimeError(f"3D generation failed: {exc}") from exc

        done_job = await db.get(Job, job_id)
        if done_job:
            # The GEOMETRY id is the resumable half, so that is what we keep.
            done_job.external_task_id = result.geometry_task_id
            done_job.stage = "complete"
            done_job.provider_credits = result.total_credits
            await db.commit()

        glb_bytes, gltf_package = _compress_mesh_for_storage(
            result.glb_bytes, product_id=product_id, context=" (tripo-parts)"
        )

        try:
            stream = io.BytesIO(glb_bytes)
            stream.seek(0)
            glb_url, glb_blob_url = storage_service.upload_product_image(
                user_id=str(user_id),
                product_id=str(product_id),
                filename="model.glb",
                content_type=result.content_type or "model/gltf-binary",
                stream=stream,
            )
        except Exception as exc:
            logger.exception("Failed to store parts GLB for product %s", product_id)
            failed_product = await db.get(Product, product_id)
            if failed_product:
                failed_product.status = ProductStatus.DRAFT
                await db.commit()
            raise RuntimeError("Failed to store generated GLB in Azure storage") from exc

        try:
            glb_asset = ProductAsset(
                asset_id=mesh_asset_id,
                image=glb_url,
                created_by=user_id,
            )
            db.add(glb_asset)
            await db.flush()

            db.add(
                ProductAssetMapping(
                    name=name,
                    productid=product_id,
                    product_asset_id=glb_asset.id,
                    isactive=True,
                    created_by=user_id,
                )
            )

            ready_product = await db.get(Product, product_id)
            if ready_product:
                ready_product.status = ProductStatus.READY
            await db.commit()
        except Exception as exc:
            await db.rollback()
            failed_product = await db.get(Product, product_id)
            if failed_product:
                failed_product.status = ProductStatus.DRAFT
                await db.commit()
            raise RuntimeError("Failed to save GLB asset to database") from exc

        elapsed = time.perf_counter() - started_at
        logger.info(
            "Product %s -> READY (tripo-parts)  %.1fs  credits=%s  glb=%s",
            product_id, elapsed, result.total_credits, glb_url,
        )

        # Additive: the Draco glTF package for the viewer. Runs only after the
        # product is committed READY — see _persist_gltf_draco_asset.
        await _persist_gltf_draco_asset(
            db, package=gltf_package, user_id=user_id,
            product_id=product_id, name=name,
        )

        await generation_estimate_service.record(db, TRIPO_PARTS_MODEL_KEY, elapsed)

        try:
            await NotificationService.create_and_push_notification(
                db=db,
                user_id=user_id,
                notification_type="product.ready",
                title="Product Ready",
                body=f"Your product '{name}' is ready.",
                data={
                    "product_id": str(product_id),
                    "product_name": name,
                    "status": ProductStatus.READY.value,
                    "glb_url": glb_url,
                },
            )
        except Exception:
            logger.warning(
                "Failed to send product-ready notification for product %s",
                product_id, exc_info=True,
            )
            try:
                await db.rollback()
            except Exception:
                pass

        # USDZ, on the same paid-plan terms as every other creation path.
        try:
            from app.services.licensing_service import LicensingService

            if await LicensingService.can_create_usdz(db, user_id):
                from app.services.usdz_trigger_service import usdz_trigger_service

                await usdz_trigger_service.trigger_conversion(
                    glb_blob_url=glb_blob_url,
                    product_id=str(product_id),
                    user_id=str(user_id),
                    product_name=name,
                )
            else:
                logger.info(
                    "USDZ skipped for product %s — user %s is not on a "
                    "USDZ-eligible plan.",
                    product_id, user_id,
                )
        except Exception:
            logger.warning(
                "Failed to schedule USDZ conversion for product %s (non-fatal)",
                product_id, exc_info=True,
            )

        return glb_url

    @staticmethod
    async def _run_parts_generation_background(
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        mesh_asset_id: int,
        name: str,
        image_url: str,
    ) -> None:
        """Open a fresh session and run the parts pipeline in the background."""
        from app.core.db import new_session
        from app.api.websocket.broadcaster import broadcaster

        logger = logging.getLogger(__name__)
        try:
            async with new_session() as db:
                queued = await db.get(Product, product_id)
                if queued:
                    queued.status = ProductStatus.QUEUE
                    await db.commit()

            await broadcaster.broadcast_to_product(
                str(product_id),
                {
                    "new_status": ProductStatus.QUEUE.value,
                    "message": "Your 3D model generation is starting.",
                },
            )

            async with new_session() as db:
                await ProductService.generate_3d_with_parts_and_finalize(
                    db=db,
                    user_id=user_id,
                    product_id=product_id,
                    mesh_asset_id=mesh_asset_id,
                    name=name,
                    image_url=image_url,
                )
        except Exception:
            logger.exception(
                "Background Tripo parts generation failed for product %s", product_id
            )

    @staticmethod
    async def update_product_background_image(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        image_stream: BinaryIO,
        image_filename: str,
        image_content_type: Optional[str] = None,
    ) -> tuple[int, str]:
        """Upload background image and create Background record.
        
        Args:
            db: Database session
            product_id: Product ID
            user_id: User ID
            image_stream: Binary stream of the image
            image_filename: Image filename
            image_content_type: Content type of the image
            
        Returns:
            Tuple of (background_id, blob_url)
        """
        logger = logging.getLogger(__name__)
        
        # -------------------------------
        # 1. Upload background image
        # -------------------------------
        try:
            image_stream.seek(0)
            cdn_url, _blob_url = storage_service.upload_background_image(
                user_id=str(user_id),
                product_id=str(product_id),
                filename=image_filename,
                content_type=image_content_type,
                stream=image_stream,
            )
            logger.info("Background image uploaded: %s", cdn_url)
        except Exception as e:
            logger.exception("Background image upload failed")
            raise RuntimeError("Failed to upload background image") from e
        
        # -------------------------------
        # 2. Get next available Background ID
        # -------------------------------
        try:
            result = await db.execute(select(func.max(Background.id)))
            max_id = result.scalar()
            next_id = (max_id or 0) + 1
            logger.info("Next Background ID: %s", next_id)
        except Exception as e:
            logger.exception("Failed to get next Background ID")
            raise RuntimeError("Failed to get next Background ID") from e
        
        # -------------------------------
        # 3. Insert new Background record
        # -------------------------------
        try:
            stmt = insert(Background).values(
                id=next_id,
                background_type_id=2,  # image type
                name=f"Background Image {image_filename}",
                description="Uploaded background image",
                image=cdn_url,  # CDN URL stored in DB — served directly to clients
                isactive=True,
                created_by=user_id,
                created_date=datetime.utcnow(),
            )
            await db.execute(stmt)
            await db.commit()
            logger.info("Background record created with ID: %s", next_id)
        except Exception as e:
            logger.exception("Failed to create Background record")
            await db.rollback()
            raise RuntimeError("Failed to create Background record") from e

        return next_id, cdn_url

    @staticmethod
    async def upload_original_product_image(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        image_stream: BinaryIO,
        image_filename: str,
        image_content_type: Optional[str] = None,
        image_size_bytes: Optional[int] = None,
    ) -> str:
        """Upload or replace the product's original image asset (asset_id = 1)."""
        logger = logging.getLogger(__name__)

        product = await db.get(Product, product_id)
        if not product or product.created_by != user_id:
            raise ValueError("Product not found")

        try:
            image_stream.seek(0)
            cdn_url, blob_url = storage_service.upload_product_image(
                user_id=str(user_id),
                product_id=str(product_id),
                filename=image_filename,
                content_type=image_content_type,
                stream=image_stream,
            )
            logger.info("Original product image uploaded: product=%s cdn=%s", product_id, cdn_url)
        except Exception as exc:
            logger.exception("Original product image upload failed for product %s", product_id)
            raise RuntimeError("Failed to upload original product image") from exc

        try:
            stmt = (
                select(ProductAsset, ProductAssetMapping)
                .join(ProductAssetMapping, ProductAsset.id == ProductAssetMapping.product_asset_id)
                .where(ProductAssetMapping.productid == product_id)
                .where(ProductAssetMapping.isactive == True)
                .where(ProductAsset.asset_id == 1)
                .order_by(ProductAssetMapping.created_date.desc())
            )
            rows = (await db.execute(stmt)).all()

            if rows:
                primary_asset, primary_mapping = rows[0]
                primary_asset.image = blob_url
                primary_asset.size_bytes = image_size_bytes
                primary_asset.updated_by = user_id
                primary_asset.updated_date = datetime.utcnow()
                primary_mapping.updated_by = user_id
                primary_mapping.updated_date = datetime.utcnow()

                for duplicate_asset, duplicate_mapping in rows[1:]:
                    duplicate_mapping.isactive = False
                    duplicate_mapping.updated_by = user_id
                    duplicate_mapping.updated_date = datetime.utcnow()
            else:
                primary_asset = ProductAsset(
                    asset_id=1,
                    image=blob_url,
                    size_bytes=image_size_bytes,
                    created_by=user_id,
                )
                db.add(primary_asset)
                await db.flush()

                db.add(
                    ProductAssetMapping(
                        name=product.name,
                        productid=product.id,
                        product_asset_id=primary_asset.id,
                        isactive=True,
                        created_by=user_id,
                    )
                )

            product.updated_by = user_id
            product.updated_date = datetime.utcnow()
            await db.commit()
        except Exception as exc:
            logger.exception("Failed to save original image asset for product %s", product_id)
            await db.rollback()
            raise RuntimeError("Failed to save original product image") from exc

        return cdn_url

    @staticmethod
    async def update_product_background_color(
        db: AsyncSession,
        user_id: uuid.UUID,
        color_value: str,
    ) -> int:
        """Get or create Background record for a color value.
        
        Args:
            db: Database session
            user_id: User ID
            color_value: Color value (e.g., hex color)
            
        Returns:
            Background ID
        """
        logger = logging.getLogger(__name__)
        
        # -------------------------------
        # 1. Query for existing Background
        # -------------------------------
        try:
            stmt = select(Background).where(
                Background.background_type_id == 1,  # color type
                Background.image == color_value,
                Background.isactive == True,
            )
            result = await db.execute(stmt)
            existing_background = result.scalar_one_or_none()
            
            if existing_background:
                logger.info("Found existing color background with ID: %s", existing_background.id)
                return existing_background.id
        except Exception as e:
            logger.exception("Failed to query for existing Background")
            raise RuntimeError("Failed to query for existing Background") from e
        
        # -------------------------------
        # 2. Create new Background if not found
        # -------------------------------
        try:
            # Get next available Background ID
            result = await db.execute(select(func.max(Background.id)))
            max_id = result.scalar()
            next_id = (max_id or 0) + 1
            logger.info("Next Background ID: %s", next_id)
            
            # Insert new Background record
            stmt = insert(Background).values(
                id=next_id,
                background_type_id=1,  # color type
                name=f"Background Color {color_value}",
                description="Color background",
                image=color_value,
                isactive=True,
                created_by=user_id,
                created_date=datetime.utcnow(),
            )
            await db.execute(stmt)
            await db.commit()
            logger.info("Background color record created with ID: %s", next_id)
            
            return next_id
        except Exception as e:
            logger.exception("Failed to create Background color record")
            await db.rollback()
            raise RuntimeError("Failed to create Background color record") from e

    @staticmethod
    async def get_products_for_current_user(
        db: AsyncSession, user_id: uuid.UUID
    ) -> dict:


        products = await ProductRepository.get_products_by_user_id(db, user_id)
        
        # Fetch public_ids for all published products in bulk
        published_product_ids = [
            product.id for product in products 
            if product.status.value == "published"
        ]
        public_id_map = await ProductRepository.get_public_ids_for_products(
            db, published_product_ids
        )
        
        items: list[ProductWithPrimaryAsset] = []

        product_ids = [p.id for p in products]
        asset_map = await ProductRepository.get_primary_assets_for_products(db, product_ids)

        for product in products:
            asset_data = asset_map.get(product.id)
            
            image = None
            asset_type = None
            asset_type_id = None
            
            if asset_data:
                image, asset_type, asset_type_id = asset_data

            # Get public_id if product is published and has enabled publish link
            public_id = None
            if product.status.value == "published":
                public_id = public_id_map.get(product.id)

            items.append(
                ProductWithPrimaryAsset(
                    id=str(product.id),
                    name=product.name,
                    status=product.status.value,
                    image=image,
                    asset_type=asset_type,
                    asset_type_id=asset_type_id,
                    description=product.description,
                    price=float(product.price) if product.price is not None else None,
                    currency_type=str(product.currency_type) if product.currency_type is not None else None,
                    background_type=str(product.background_type) if product.background_type is not None else None,
                    created_at=product.created_at,
                    updated_at=product.updated_at,
                    public_id=public_id,
                )
            )

        return ProductsByUserResponse(items=items).model_dump()


product_service = ProductService()
