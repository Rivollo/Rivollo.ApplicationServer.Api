"""Variant bake service — turns a colourway recipe into a real GLB on blob storage.

Pipeline for one variant:

    1. skip if a ready asset already exists at this config hash   (idempotent)
    2. download the product's source GLB to a temp dir
    3. recolour it with the variant's overrides                   (pure engine)
    4. upload the baked GLB, content-addressed by config hash
    5. record the URL on tbl_variant_assets, mark the variant ready
    6. always clean the temp dir

Why this runs in the background: a 15 MB recolour is seconds-to-a-minute of CPU
and I/O. Blocking an HTTP request on it would tie up a worker and time out the
client. The route returns immediately with ``bake_status='pending'`` and the UI
polls until it flips to ``ready``.

Why the blocking work is pushed to a thread: pygltflib/Pillow/Azure are all
synchronous. Running them directly in the event loop would stall every other
request on the same worker.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import new_session
from app.database.color_variant_repo import color_variant_repository as repo
from app.models.models import ProductColorVariant, VariantAsset
from app.services.color import BAKER_VERSION, glb_recolor
from app.services.color.glb_recolor import RecolorOverride
from app.services.model_cache import model_cache
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

# Only one bake at a time per API worker. These are memory-heavy (a 15 MB GLB
# expands to several hundred MB of decoded textures); running many in parallel
# is the fastest way to OOM the pod.
_BAKE_SEMAPHORE = asyncio.Semaphore(1)

GLB_MIME = "model/gltf-binary"


class VariantBakeService:
    """Bakes colourway recipes into real GLB files."""

    # ---------- Hashing ----------

    @staticmethod
    def compute_config_hash(
        source_model_url: Optional[str],
        overrides: list[dict[str, Any]],
    ) -> str:
        """Stable fingerprint of everything that affects the baked bytes.

        Includes the source model's URL so that re-uploading the product's model
        invalidates every variant automatically (a new upload lands at a new
        blob path), and the baker version so that improving the engine does the
        same.
        """
        canonical = json.dumps(
            [
                {
                    "material_index": o.get("material_index"),
                    "color": (o.get("color") or "").upper(),
                    "method": o.get("method", "auto"),
                    "brightness": round(float(o.get("brightness", 1.0)), 4),
                }
                for o in sorted(overrides, key=lambda o: o.get("material_index", 0))
            ],
            separators=(",", ":"),
            sort_keys=True,
        )
        payload = f"{source_model_url}|{canonical}|v{BAKER_VERSION}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # ---------- Source model ----------

    @staticmethod
    def _fetch_source(model_url: str) -> Path:
        """Local path to the product's source GLB, downloading only on a miss.

        Product meshes routinely exceed 60 MB, which is ~35s over the CDN. Every
        colourway for a product bakes from the same file, so caching it turns
        the second and subsequent bakes from "wait a minute" into "a few
        seconds". Must be called from a worker thread — it blocks.
        """
        cached = model_cache.get(model_url)
        if cached is not None:
            logger.debug("Source model cache hit for %s", model_url)
            return cached

        content, _, _ = storage_service.download_upload_blob_bytes(model_url)
        stored = model_cache.put(model_url, content)
        if stored is not None:
            return stored

        # Caching unavailable (read-only disk, full volume) — fall back to a
        # throwaway file. The caller cleans it up with its own temp dir.
        fallback = Path(tempfile.mkdtemp(prefix="rivollo-src-")) / "source.glb"
        fallback.write_bytes(content)
        return fallback

    # ---------- Inspection ----------

    @staticmethod
    async def inspect_model(model_url: str) -> list[dict[str, Any]]:
        """Report the colourable parts of a product's GLB."""

        def _work() -> list[dict[str, Any]]:
            src = VariantBakeService._fetch_source(model_url)
            return [
                {
                    "material_index": p.material_index,
                    "name": p.name,
                    "mesh_names": p.mesh_names,
                    "has_base_color_texture": p.has_base_color_texture,
                    "average_color": p.average_color,
                    "suggested_method": p.suggested_method,
                    "group_id": p.group_id,
                    "center": p.center,
                }
                for p in glb_recolor.inspect(src)
            ]

        return await asyncio.to_thread(_work)

    @staticmethod
    async def resolve_methods(
        model_url: str,
        overrides: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replace every ``method: "auto"`` with the concrete method for that part.

        Resolved once, at save time, so the browser preview and the baked file
        can never diverge because two heuristics disagreed.
        """
        if not any(o.get("method", "auto") == "auto" for o in overrides):
            return overrides

        try:
            parts = await VariantBakeService.inspect_model(model_url)
        except Exception as exc:  # noqa: BLE001 - inspection is best-effort
            logger.warning("Could not inspect model to resolve methods: %s", exc)
            return overrides

        suggested = {p["material_index"]: p["suggested_method"] for p in parts}
        resolved: list[dict[str, Any]] = []
        for override in overrides:
            item = dict(override)
            if item.get("method", "auto") == "auto":
                item["method"] = suggested.get(item.get("material_index"), "factor")
            resolved.append(item)
        return resolved

    # ---------- Baking ----------

    @staticmethod
    def _bake_bytes(src: Path, overrides: list[dict[str, Any]]) -> bytes:
        """Pure CPU work: source GLB + recipe -> baked GLB bytes.

        The source is read from the shared cache and never modified; only the
        output lives in a throwaway directory.
        """
        with tempfile.TemporaryDirectory(prefix="rivollo-bake-") as tmp:
            out = Path(tmp) / "baked.glb"
            glb_recolor.recolor(
                src,
                [
                    RecolorOverride(
                        material_index=int(o["material_index"]),
                        color=o["color"],
                        method=o.get("method", "auto"),
                        brightness=float(o.get("brightness", 1.0)),
                    )
                    for o in overrides
                ],
                out,
            )
            return out.read_bytes()
            # TemporaryDirectory removes the output on exit, including on error.

    @staticmethod
    async def bake_variant(variant_id: uuid.UUID) -> None:
        """Bake one variant end to end, in its own database session.

        Runs detached from the request that triggered it, so it opens a fresh
        session rather than borrowing the (already closed) request session.
        Never raises — failures are recorded on the variant for the UI to show.
        """
        async with _BAKE_SEMAPHORE:
            async with new_session() as db:
                try:
                    await VariantBakeService._bake_with_session(db, variant_id)
                except Exception as exc:  # noqa: BLE001 - background task guard
                    logger.exception("Bake failed for variant %s", variant_id)
                    await VariantBakeService._mark_failed(db, variant_id, str(exc))

    @staticmethod
    async def _bake_with_session(db: AsyncSession, variant_id: uuid.UUID) -> None:
        variant = await repo.get_variant_by_id(db, variant_id)
        if variant is None:
            logger.warning("Bake requested for missing variant %s", variant_id)
            return

        # The original is the untouched product model — there is nothing to bake.
        if variant.is_original or not variant.overrides:
            variant.bake_status = "ready"
            variant.bake_error = None
            variant.baked_at = datetime.now(timezone.utc)
            await db.commit()
            return

        # Already baked at this exact recipe -> nothing to do. This is what makes
        # a seller nudging the same colour repeatedly cost one bake, not ten.
        existing = await repo.get_asset_for_format(db, variant.id, "glb")
        if existing is not None and existing.config_hash == variant.config_hash:
            variant.bake_status = "ready"
            variant.bake_error = None
            variant.baked_at = variant.baked_at or datetime.now(timezone.utc)
            await db.commit()
            return

        source_url = await repo.get_product_model_url(db, variant.product_id)
        if not source_url:
            await VariantBakeService._mark_failed(
                db, variant_id, "Product has no source 3D model to recolour"
            )
            return

        variant.bake_status = "baking"
        variant.bake_error = None
        await db.commit()

        product_id = str(variant.product_id)
        config_hash = variant.config_hash
        overrides = list(variant.overrides or [])
        stale_url = existing.url if existing is not None else None

        def _work() -> tuple[str, str, int]:
            src = VariantBakeService._fetch_source(source_url)
            baked = VariantBakeService._bake_bytes(src, overrides)
            cdn_url, blob_url = storage_service.upload_variant_model(
                product_id=product_id,
                config_hash=config_hash,
                extension="glb",
                content_type=GLB_MIME,
                stream=io.BytesIO(baked),
            )
            return cdn_url, blob_url, len(baked)

        cdn_url, blob_url, size = await asyncio.to_thread(_work)

        # Re-read: the variant may have been edited while we were baking.
        variant = await repo.get_variant_by_id(db, variant_id)
        if variant is None:
            return
        if variant.config_hash != config_hash:
            # Superseded mid-bake. Drop this result; the newer recipe has its own
            # bake queued and will overwrite the row when it lands.
            logger.info("Discarding superseded bake for variant %s", variant_id)
            return

        record = await repo.get_asset_for_format(db, variant.id, "glb")
        if record is None:
            record = VariantAsset(variant_id=variant.id, format="glb")
            repo.add(db, record)
        record.url = cdn_url
        record.blob_url = blob_url
        record.size_bytes = size
        record.config_hash = config_hash

        variant.bake_status = "ready"
        variant.bake_error = None
        variant.baked_at = datetime.now(timezone.utc)

        await db.commit()

        # Purge the superseded file only after the replacement is committed, so a
        # viewer mid-request never hits a 404.
        if stale_url and stale_url != cdn_url:
            await asyncio.to_thread(storage_service.delete_blob_by_cdn_url, stale_url)

        logger.info(
            "Baked variant %s (%s) -> %s (%.1f KB)",
            variant_id, variant.name, cdn_url, size / 1024,
        )

    @staticmethod
    async def _mark_failed(db: AsyncSession, variant_id: uuid.UUID, message: str) -> None:
        try:
            await db.rollback()
            variant = await repo.get_variant_by_id(db, variant_id)
            if variant is not None:
                variant.bake_status = "failed"
                variant.bake_error = message[:500]
                await db.commit()
        except Exception:  # noqa: BLE001 - never let error handling raise
            logger.exception("Could not record bake failure for %s", variant_id)

    # ---------- Purge ----------

    @staticmethod
    async def purge_variant_assets(variant: ProductColorVariant) -> None:
        """Delete a variant's baked blobs. Called before the row is removed."""
        for asset in list(variant.assets or []):
            if asset.url:
                await asyncio.to_thread(storage_service.delete_blob_by_cdn_url, asset.url)


variant_bake_service = VariantBakeService()
