"""Resolves a product's current GLB into the facts the Configurator validates against.

Answers three questions the part and option services need and cannot answer for
themselves:

    * what is this product's current ``glb_version``  (ADR-006)
    * how many glTF materials does that model have    (index range validation)
    * what method would ``auto`` resolve to per material

Reuses the PURE colour engine (``app.services.color.glb_recolor``) and the
shared ``model_cache``, not ``variant_bake_service``. Those are shared
infrastructure that architecture.md section 4 lists as reusable; the bake
service is colour-variant domain code, and importing it would couple this
domain to the feature the Configurator replaces.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.configurator_repo import configurator_repository as repo
from app.services.color import glb_recolor
from app.services.model_cache import model_cache
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

NO_MODEL_DETAIL = (
    "This product has no 3D model yet. Upload a model before configuring parts."
)


@dataclass(frozen=True)
class MeshContext:
    """Everything the Configurator needs to know about a product's current GLB."""

    glb_version: str
    model_url: str
    material_count: int
    # material_index -> "factor" | "luminance" | "remap", for resolving "auto".
    suggested_methods: dict[int, str]

    def is_valid_index(self, material_index: int) -> bool:
        return 0 <= material_index < self.material_count


class MaterialService:
    """Reads the product's GLB. Never writes anything."""

    @staticmethod
    def build_glb_version(asset_id: uuid.UUID) -> str:
        """Phase-1 glb_version: the mesh asset's row id, prefixed.

        Prefix-discriminated so ADR-006 can move to ``sha256:<hex>`` by data
        migration rather than schema change. ADR-006 is still Needs
        Verification — nothing may depend on what the prefix MEANS, only that
        two different models produce two different strings.
        """
        return f"asset:{asset_id}"

    @staticmethod
    async def get_mesh_context(db: AsyncSession, product_id: uuid.UUID) -> MeshContext:
        """Resolve and inspect the product's current GLB.

        Caller must already have established ownership of ``product_id`` — this
        does not check it, and calling it first would let an attacker probe
        whether another seller's product has a model.

        Downloads and parses the GLB on a cache miss, so it is slow (typical
        meshes are 40-80 MB). ``model_cache`` makes the second and subsequent
        calls for the same model cheap.
        """
        asset = await repo.get_product_mesh_asset(db, product_id)
        if asset is None or not asset.image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=NO_MODEL_DETAIL
            )

        model_url = asset.image
        glb_version = MaterialService.build_glb_version(asset.id)

        try:
            parts = await asyncio.to_thread(MaterialService._inspect, model_url)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a gateway failure
            logger.exception("Could not inspect GLB for product %s", product_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The product's 3D model could not be read.",
            ) from exc

        return MeshContext(
            glb_version=glb_version,
            model_url=model_url,
            material_count=len(parts),
            suggested_methods={p.material_index: p.suggested_method for p in parts},
        )

    @staticmethod
    async def list_materials(
        db: AsyncSession,
        product_id: uuid.UUID,
    ) -> tuple[str, str, list[dict]]:
        """(glb_version, model_url, materials) for the Part Editor.

        Caller must already have established ownership. Returns plain dicts so
        the route layer owns serialisation; ``assigned_part_id`` is filled in by
        the caller, which is the layer that knows about parts.
        """
        asset = await repo.get_product_mesh_asset(db, product_id)
        if asset is None or not asset.image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=NO_MODEL_DETAIL
            )

        model_url = asset.image
        try:
            parts = await asyncio.to_thread(MaterialService._inspect, model_url)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a gateway failure
            logger.exception("Could not inspect GLB for product %s", product_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The product's 3D model could not be read.",
            ) from exc

        materials = [
            {
                "material_index": p.material_index,
                "name": p.name,
                "mesh_names": list(p.mesh_names),
                "has_base_color_texture": p.has_base_color_texture,
                "average_color": p.average_color,
                "suggested_method": p.suggested_method,
                # Renamed from the engine's group_id — a recomputed heuristic,
                # never a Part identity (ADR-004).
                "similarity_group_hint": p.group_id,
                "center": p.center,
            }
            for p in parts
        ]
        return MaterialService.build_glb_version(asset.id), model_url, materials

    @staticmethod
    async def extract_source_textures(model_url: str) -> dict[int, tuple[bytes, str]]:
        """``material_index -> (encoded bytes, mime)`` from a product's GLB.

        The one public way to obtain source pixels for baking. Wraps the fetch
        (cache-aware) and ``glb_recolor.extract_base_color_images`` in a single
        worker-thread call, so no caller needs this module's private helpers or a
        second GLB parse of its own.

        Materials absent from the result have no usable source image — an
        external ``uri``, no baseColorTexture, or an unreadable bufferView. The
        caller maps absence to ``None``.

        Deliberately does NOT translate failures into HTTPException: the bake
        runner is not serving a request and needs the underlying error to decide
        what to tell the seller.
        """

        def _work() -> dict[int, tuple[bytes, str]]:
            return glb_recolor.extract_base_color_images(
                MaterialService._fetch_source(model_url)
            )

        return await asyncio.to_thread(_work)

    @staticmethod
    async def fetch_image_bytes(image_url: str) -> bytes:
        """Download a seller-uploaded image. Blocking work, off the event loop."""

        def _work() -> bytes:
            content, _content_type, _filename = storage_service.download_upload_blob_bytes(
                image_url
            )
            return content

        return await asyncio.to_thread(_work)

    # ------------------------------------------------------------------ #
    # Blocking work — must run in a worker thread
    # ------------------------------------------------------------------ #
    @staticmethod
    def _inspect(model_url: str) -> list[glb_recolor.PartInfo]:
        return glb_recolor.inspect(MaterialService._fetch_source(model_url))

    @staticmethod
    def _fetch_source(model_url: str) -> Path:
        """Local path to the product's GLB, downloading only on a cache miss.

        The cache is keyed by source URL, so a re-uploaded model — which lands
        at a new blob path — naturally misses rather than serving stale
        geometry. Entries live in the system temp directory and are disposable.
        """
        cached = model_cache.get(model_url)
        if cached is not None:
            return cached

        content, _, _ = storage_service.download_upload_blob_bytes(model_url)
        stored = model_cache.put(model_url, content)
        if stored is not None:
            return stored

        # Caching unavailable (read-only disk, full volume). Fall back to a
        # throwaway file rather than failing the request.
        fallback = Path(tempfile.mkdtemp(prefix="rivollo-cfg-")) / "source.glb"
        fallback.write_bytes(content)
        return fallback


material_service = MaterialService()
