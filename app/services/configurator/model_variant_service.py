"""Model Variant business rules (ADR-014).

A model variant is an EXTRA shape of a product — "3 Seater", "Corner" — with
its own GLB. The product's original model (the GLB mapped to it) stays the
product's model and its permanent default; it has no row here.

Creating a variant is purely additive. It writes one tbl_product_assets row for
the served GLB — with NO tbl_product_asset_mapping row, so no existing reader of
"the product's model" can see it — and one tbl_product_model_variants row.
Nothing existing is read for writing, updated or re-pointed.

Upload pipeline, in order:

  1. feature flag, name, ownership (404, never 403 — ADR-008)
  2. file checks: extension, empty, size, and a real glTF 2.0 parse
  3. Draco compression via the existing glb_compression_service (Node,
     gltf-transform), off the event loop; skipped when the upload is already
     Draco-compressed
  4. the compressed GLB is re-inspected: material and mesh names and their
     order must be identical, because parts attach to material indices. Any
     mismatch or failure keeps the original bytes and logs a warning
  5. blobs first (no transaction open across network I/O), then one commit;
     a failed commit deletes the blobs it just wrote

Transactions: this service owns them. Repositories never commit.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.configurator_repo import (
    MESH_ASSET_ID,
    USDZ_ASSET_ID,
    configurator_repository as repo,
)
from app.models.configurator import ProductModelVariant
from app.models.models import ProductAsset
from app.schemas.configurator import MODEL_VARIANT_NAME_MAX
from app.services.configurator.glb_inspection import GlbSummary, InvalidGlbError, inspect_glb
from app.services.glb_compression_service import glb_compression_service
from app.services.storage import storage_service
from app.services.usdz_trigger_service import usdz_trigger_service

logger = logging.getLogger(__name__)

PRODUCT_NOT_FOUND = "Product not found"
VARIANT_NOT_FOUND = "Model variant not found"
FEATURE_DISABLED = "Not found"

# The path token and display name of the product's original model, which has
# no row of its own.
ORIGINAL_TOKEN = "original"
ORIGINAL_NAME = "Default"
THUMBNAIL_ASSET_ID = 1

GLB_CONTENT_TYPE = "model/gltf-binary"
THUMBNAIL_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}

# Furniture is expected to measure between these, in metres. Outside them the
# upload is kept but flagged — usually a units problem (centimetres exported as
# metres, or the reverse).
PLAUSIBLE_MIN_SIDE_M = 0.1
PLAUSIBLE_MAX_SIDE_M = 10.0


@dataclass(frozen=True)
class UploadedFile:
    """A file already read from the request. The route only parses; rules live here."""

    filename: str
    content_type: Optional[str]
    data: bytes


@dataclass
class CreatedVariant:
    variant: ProductModelVariant
    glb_url: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelEntry:
    """One row of the Variants list: the original model or an extra variant.

    The original model has no database row; ``variant`` is None for it and its
    URLs come from the product's mapped assets, exactly as the rest of the app
    reads them.
    """

    product_id: uuid.UUID
    name: str
    is_original: bool
    glb_url: Optional[str]
    usdz_url: Optional[str]
    thumbnail_url: Optional[str]
    order_index: int
    variant: Optional[ProductModelVariant] = None


@dataclass(frozen=True)
class _Compression:
    served: bytes
    status: str  # 'compressed' | 'fallback_original'
    error: Optional[str]
    warning: Optional[str]


class ModelVariantService:
    """Business rules for Model Variants."""

    @staticmethod
    def require_enabled() -> None:
        """404 while the feature is off, so the routes look absent."""
        if not settings.ENABLE_MODEL_VARIANTS:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=FEATURE_DISABLED)

    # ------------------------------------------------------------------ #
    # Path token: "original" or a variant id
    # ------------------------------------------------------------------ #
    @staticmethod
    async def resolve_model(
        db: AsyncSession,
        product_id: uuid.UUID,
        token: str,
        user_id: uuid.UUID,
    ) -> Optional[uuid.UUID]:
        """``original`` -> None (the original model); a variant id -> that id.

        Ownership of the product is checked here, and a variant must be a live
        variant of THIS product. Everything else is 404, never 403 (ADR-008).
        """
        ModelVariantService.require_enabled()
        product = await repo.get_owned_product(db, product_id, user_id)
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)
        if token == ORIGINAL_TOKEN:
            return None
        try:
            variant_id = uuid.UUID(token)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=VARIANT_NOT_FOUND)
        variant = await repo.get_owned_model_variant(db, variant_id, user_id)
        if variant is None or variant.product_id != product_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=VARIANT_NOT_FOUND)
        return variant_id

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    @staticmethod
    async def list_models(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> list[ModelEntry]:
        """The original model first, then the live extra variants in order.

        The original is read from the product's mapped assets (GLB 9, USDZ 11,
        thumbnail 1) — the same rows every existing reader uses — so it always
        matches what the product list and /assets show.
        """
        ModelVariantService.require_enabled()
        product = await repo.get_owned_product(db, product_id, user_id)
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)

        glb = await repo.get_product_mesh_asset(db, product_id)
        usdz = await repo.get_product_asset(db, product_id, USDZ_ASSET_ID)
        thumbnail = await repo.get_product_asset(db, product_id, THUMBNAIL_ASSET_ID)
        entries = [
            ModelEntry(
                product_id=product_id,
                name=ORIGINAL_NAME,
                is_original=True,
                glb_url=glb.image if glb is not None else None,
                usdz_url=usdz.image if usdz is not None else None,
                thumbnail_url=thumbnail.image if thumbnail is not None else None,
                order_index=0,
            )
        ]

        variants = await repo.get_model_variants(db, product_id)
        assets = await repo.get_assets_by_ids(
            db, [a for v in variants for a in (v.glb_asset_id, v.usdz_asset_id)]
        )
        for variant in variants:
            glb_row = assets.get(variant.glb_asset_id)
            usdz_row = assets.get(variant.usdz_asset_id)
            entries.append(
                ModelEntry(
                    product_id=product_id,
                    name=variant.name,
                    is_original=False,
                    glb_url=glb_row.image if glb_row is not None else None,
                    usdz_url=usdz_row.image if usdz_row is not None else None,
                    thumbnail_url=variant.thumbnail_url,
                    order_index=variant.order_index,
                    variant=variant,
                )
            )
        return entries

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    @staticmethod
    async def rename(
        db: AsyncSession,
        variant_id: uuid.UUID,
        user_id: uuid.UUID,
        name: str,
    ) -> ProductModelVariant:
        variant = await ModelVariantService._require_owned_variant(db, variant_id, user_id)
        variant.name = ModelVariantService._validate_name(name)
        variant.updated_by = user_id
        variant.updated_date = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(variant)
        return variant

    @staticmethod
    async def reorder(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        variant_ids: list[uuid.UUID],
    ) -> None:
        """Set the display order of the EXTRA variants; the original stays first.

        The list must name every live variant of the product exactly once, so a
        stale client cannot silently drop one out of the order.
        """
        ModelVariantService.require_enabled()
        product = await repo.get_owned_product(db, product_id, user_id, for_update=True)
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)

        variants = {v.id: v for v in await repo.get_model_variants(db, product_id)}
        if len(set(variant_ids)) != len(variant_ids) or set(variant_ids) != set(variants):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="variant_ids must list every variant of this product exactly once.",
            )
        now = datetime.now(timezone.utc)
        for position, vid in enumerate(variant_ids, start=1):
            variant = variants[vid]
            if variant.order_index != position:
                variant.order_index = position
                variant.updated_by = user_id
                variant.updated_date = now
        await db.commit()

    @staticmethod
    async def set_thumbnail(
        db: AsyncSession,
        variant_id: uuid.UUID,
        user_id: uuid.UUID,
        thumbnail: UploadedFile,
    ) -> ProductModelVariant:
        """Store the editor's ``toBlob()`` capture; the previous file is deleted."""
        variant = await ModelVariantService._require_owned_variant(db, variant_id, user_id)
        extension = ModelVariantService._validate_thumbnail(thumbnail)
        if extension is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The thumbnail file is empty.")

        try:
            url, blob_url = await ModelVariantService._upload(
                user_id, variant.product_id, variant.id, f"thumbnail.{extension}",
                thumbnail.content_type, thumbnail.data,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Thumbnail upload failed for model variant %s", variant_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The thumbnail could not be stored. Please try again.",
            ) from exc

        previous = variant.thumbnail_url
        variant.thumbnail_url = url
        variant.thumbnail_blob_url = blob_url
        variant.updated_by = user_id
        variant.updated_date = datetime.now(timezone.utc)
        try:
            await db.commit()
        except Exception:
            await db.rollback()
            await ModelVariantService._purge_blobs([url])
            raise
        await db.refresh(variant)
        if previous:
            await ModelVariantService._purge_blobs([previous])
        return variant

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #
    @staticmethod
    async def delete(
        db: AsyncSession,
        variant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Soft delete (``isactive = false``). Its parts become unreachable.

        Blobs and rows are kept, so a mistaken delete can be restored by hand;
        the account purge removes both when the account goes. The original
        model is not a row and cannot be deleted here.
        """
        variant = await ModelVariantService._require_owned_variant(db, variant_id, user_id)
        variant.isactive = False
        variant.updated_by = user_id
        variant.updated_date = datetime.now(timezone.utc)
        await db.commit()

    @staticmethod
    async def _require_owned_variant(
        db: AsyncSession,
        variant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ProductModelVariant:
        ModelVariantService.require_enabled()
        variant = await repo.get_owned_model_variant(db, variant_id, user_id)
        if variant is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=VARIANT_NOT_FOUND)
        return variant

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #
    @staticmethod
    async def create_variant(
        db: AsyncSession,
        product_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        name: str,
        glb: UploadedFile,
        thumbnail: Optional[UploadedFile] = None,
    ) -> CreatedVariant:
        ModelVariantService.require_enabled()
        clean_name = ModelVariantService._validate_name(name)

        product = await repo.get_owned_product(db, product_id, user_id)
        if product is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=PRODUCT_NOT_FOUND)

        ModelVariantService._validate_glb_file(glb)
        thumb_ext = ModelVariantService._validate_thumbnail(thumbnail)

        original_bytes = glb.data
        try:
            original = await asyncio.to_thread(inspect_glb, original_bytes)
        except InvalidGlbError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        compression = await ModelVariantService._compress(original_bytes, original, product_id)
        warnings = [w for w in (compression.warning,) if w]
        warnings.extend(ModelVariantService._dimension_warnings(original))

        variant_id = uuid.uuid4()
        uploaded: list[str] = []
        try:
            glb_url, glb_blob_url = await ModelVariantService._upload(
                user_id, product_id, variant_id, "model.glb", GLB_CONTENT_TYPE, compression.served
            )
            uploaded.append(glb_url)

            original_url: Optional[str] = None
            original_blob_url: Optional[str] = None
            if compression.served is not original_bytes:
                # Keep the untouched upload for re-processing. When the served
                # file IS the original, there is nothing extra to keep, and the
                # columns stay NULL rather than naming the same blob twice.
                original_url, original_blob_url = await ModelVariantService._upload(
                    user_id, product_id, variant_id, "original.glb", GLB_CONTENT_TYPE, original_bytes
                )
                uploaded.append(original_url)

            thumb_url: Optional[str] = None
            thumb_blob_url: Optional[str] = None
            if thumbnail is not None and thumb_ext is not None:
                thumb_url, thumb_blob_url = await ModelVariantService._upload(
                    user_id, product_id, variant_id, f"thumbnail.{thumb_ext}",
                    thumbnail.content_type, thumbnail.data,
                )
                uploaded.append(thumb_url)
        except Exception as exc:  # noqa: BLE001 - reported as one storage failure
            logger.exception("Model variant upload failed for product %s", product_id)
            await ModelVariantService._purge_blobs(uploaded)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The model could not be stored. Please try again.",
            ) from exc

        try:
            asset = ProductAsset(
                id=uuid.uuid4(),
                asset_id=MESH_ASSET_ID,
                image=glb_url,
                size_bytes=len(compression.served),
                # Always set: this row is deliberately unmapped, and the account
                # purge finds unmapped assets through created_by (ADR-014).
                created_by=user_id,
            )
            variant = ProductModelVariant(
                id=variant_id,
                product_id=product_id,
                name=clean_name,
                glb_asset_id=asset.id,
                thumbnail_url=thumb_url,
                thumbnail_blob_url=thumb_blob_url,
                original_glb_url=original_url,
                original_glb_blob_url=original_blob_url,
                original_size_bytes=len(original_bytes),
                compressed_size_bytes=(
                    len(compression.served) if compression.status == "compressed" else None
                ),
                compression_status=compression.status,
                compression_error=compression.error,
                width_m=original.width_m,
                depth_m=original.depth_m,
                height_m=original.height_m,
                order_index=await repo.get_next_model_variant_order_index(db, product_id),
                isactive=True,
                created_by=user_id,
            )
            # No ProductAssetMapping, on purpose: see the module docstring.
            repo.add(db, asset)
            # The asset row must exist before the variant's FK names it. The
            # unit of work cannot work that out: glb_asset_id is a plain column
            # with no relationship(), so without this flush the variant can be
            # inserted first and fk_model_variants_glb_asset fails.
            await repo.flush(db)
            repo.add(db, variant)
            await db.commit()
            await db.refresh(variant)
        except Exception:
            logger.exception("Model variant rows failed to save for product %s", product_id)
            await db.rollback()
            await ModelVariantService._purge_blobs(uploaded)
            raise

        logger.info(
            "Model variant %s created for product %s: status=%s original=%d served=%d bytes",
            variant.id, product_id, compression.status, len(original_bytes), len(compression.served),
        )
        await ModelVariantService._request_usdz(
            variant, user_id, original_blob_url or glb_blob_url, clean_name
        )
        return CreatedVariant(variant=variant, glb_url=glb_url, warnings=warnings)

    @staticmethod
    async def _request_usdz(
        variant: ProductModelVariant,
        user_id: uuid.UUID,
        source_blob_url: str,
        name: str,
    ) -> None:
        """Fire-and-forget USDZ conversion for iOS AR. Never fails the upload.

        The converter job writes the USDZ into the variant's folder and sets
        ``usdz_asset_id`` — it creates no product mapping, so the product's own
        AR model is untouched. Converts from the uncompressed upload when one
        was kept, which spares Blender a Draco decode.

        Off unless ENABLE_VARIANT_USDZ is set: the deployed converter image
        rejects --model-variant-id until it is updated, and every upload would
        otherwise start a job run.
        """
        if not settings.ENABLE_VARIANT_USDZ:
            logger.info(
                "Skipping USDZ conversion for model variant %s: ENABLE_VARIANT_USDZ is off",
                variant.id,
            )
            return
        try:
            await usdz_trigger_service.trigger_conversion(
                glb_blob_url=source_blob_url,
                product_id=str(variant.product_id),
                user_id=str(user_id),
                product_name=name,
                output_blob_name="model.usdz",
                model_variant_id=str(variant.id),
            )
        except Exception:  # noqa: BLE001 - AR is optional; the variant is saved
            logger.warning(
                "Could not request USDZ conversion for model variant %s", variant.id, exc_info=True
            )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_name(name: str) -> str:
        clean = (name or "").strip()
        if not clean:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A variant name is required.")
        if len(clean) > MODEL_VARIANT_NAME_MAX:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The variant name must be at most {MODEL_VARIANT_NAME_MAX} characters.",
            )
        return clean

    @staticmethod
    def _validate_glb_file(glb: UploadedFile) -> None:
        if not glb.filename or not glb.filename.lower().endswith(".glb"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A .glb file is required.")
        if not glb.data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded GLB file is empty.")
        if len(glb.data) > settings.MAX_VARIANT_GLB_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"The GLB is larger than {settings.MAX_VARIANT_GLB_BYTES // (1024 * 1024)} MB.",
            )

    @staticmethod
    def _validate_thumbnail(thumbnail: Optional[UploadedFile]) -> Optional[str]:
        """The file extension to store it under, or None when there is none."""
        if thumbnail is None or not thumbnail.data:
            return None
        extension = THUMBNAIL_EXTENSIONS.get((thumbnail.content_type or "").lower())
        if extension is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The thumbnail must be a PNG, JPEG or WebP image.",
            )
        if len(thumbnail.data) > settings.MAX_VARIANT_THUMBNAIL_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="The thumbnail image is too large.",
            )
        return extension

    @staticmethod
    def _dimension_warnings(summary: GlbSummary) -> list[str]:
        largest = summary.largest_side_m
        if largest is None:
            return ["The model's size could not be measured; dimensions are not stored."]
        if largest < PLAUSIBLE_MIN_SIDE_M or largest > PLAUSIBLE_MAX_SIDE_M:
            return [
                f"The model's largest side is {largest:.2f} m, which is unusual for "
                "furniture. Check that it was exported in metres."
            ]
        return []

    # ------------------------------------------------------------------ #
    # Compression
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _compress(data: bytes, original: GlbSummary, product_id: uuid.UUID) -> _Compression:
        """Draco-compress, verifying names survived. Never raises."""
        if original.draco_compressed:
            # Re-encoding would decode and quantise the geometry a second time.
            return _Compression(served=data, status="compressed", error=None, warning=None)

        if not settings.ENABLE_DRACO_COMPRESSION:
            return ModelVariantService._fallback(
                data, product_id, "Draco compression is disabled in this environment."
            )

        try:
            compressed = await asyncio.to_thread(glb_compression_service.compress, data)
            result = await asyncio.to_thread(inspect_glb, compressed)
        except Exception as exc:  # noqa: BLE001 - every failure falls back
            return ModelVariantService._fallback(data, product_id, f"Draco compression failed: {exc}")

        if not result.same_names_as(original):
            return ModelVariantService._fallback(
                data, product_id, "Draco compression changed material or mesh names; kept the original."
            )
        if not result.draco_compressed:
            return ModelVariantService._fallback(
                data, product_id, "The compressed file is not Draco-encoded; kept the original."
            )
        return _Compression(served=compressed, status="compressed", error=None, warning=None)

    @staticmethod
    def _fallback(data: bytes, product_id: uuid.UUID, reason: str) -> _Compression:
        logger.warning(
            "Model variant for product %s stored uncompressed: %s", product_id, reason
        )
        return _Compression(
            served=data,
            status="fallback_original",
            error=reason[:500],
            warning="The model was stored without Draco compression, so it may load more slowly.",
        )

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _upload(
        user_id: uuid.UUID,
        product_id: uuid.UUID,
        variant_id: uuid.UUID,
        filename: str,
        content_type: Optional[str],
        data: bytes,
    ) -> tuple[str, str]:
        return await asyncio.to_thread(
            storage_service.upload_model_variant_file,
            user_id=str(user_id),
            product_id=str(product_id),
            variant_id=str(variant_id),
            filename=filename,
            content_type=content_type,
            stream=io.BytesIO(data),
        )

    @staticmethod
    async def _purge_blobs(urls: list[str]) -> None:
        for url in urls:
            try:
                await asyncio.to_thread(storage_service.delete_blob_by_cdn_url, url)
            except Exception:  # noqa: BLE001 - best effort; the prefix sweep catches leftovers
                logger.warning("Could not delete orphaned model variant blob", exc_info=True)


model_variant_service = ModelVariantService()
