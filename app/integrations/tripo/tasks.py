"""Request builders for the Tripo tasks we use.

Kept separate from the client so the client stays a pure protocol layer: it
knows about Bearer auth and task polling, not about what a "part" is. Adding
another Tripo task means adding a builder here and nothing else.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.config import settings

# Endpoints, relative to TRIPO_API_BASE_URL.
IMAGE_TO_MODEL_PATH = "/generation/image-to-model"
TEXTURE_PATH = "/models/texture"


def build_parts_geometry_request(image_url: str) -> dict[str, Any]:
    """Stage 1 — segmented geometry, deliberately untextured.

    ``generate_parts`` is mutually exclusive with texturing: Tripo's docs state
    it is "not compatible with texture=true, pbr=true, or quad=true. To use
    this, set all three to false." Sending any of them alongside it makes the
    task fail, so all three are pinned to False here rather than left to a
    default that might change.

    Texture is applied afterwards by :func:`build_texture_request`, which is the
    only way to get a mesh that is BOTH segmented and textured.
    """
    return {
        "input": image_url,
        # generate_parts requires >= v3.0; we use the newest for geometry.
        "model": settings.TRIPO_GEOMETRY_MODEL,
        "generate_parts": True,
        # The three that must be false for parts to be produced at all.
        "texture": False,
        "pbr": False,
        "quad": False,
        # Web/mobile guidance from Tripo's own docs is 10k-50k faces. Left
        # generous here because decimating before segmentation risks merging
        # parts that we specifically asked to keep separate.
        "face_limit": 150000,
        # UV unwrapping is handled during the texture stage, so skipping it here
        # is faster and produces a smaller intermediate file — exactly what the
        # `export_uv` docs describe.
        "export_uv": False,
    }


def build_texture_request(
    geometry_task_id: str,
    image_url: str,
    *,
    part_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Stage 2 — texture the segmented mesh produced by stage 1.

    ``input`` takes the stage-1 task id directly, so the mesh never round-trips
    through our storage.

    The original photo is re-supplied as ``texture_prompt.image`` because Tripo
    "strongly recommends" it when texturing from an existing task id — without
    it, texturing leans on the model's priors and drifts away from the actual
    product.

    ``part_names`` is left unset by default, which Tripo documents as "all parts
    are textured". We have no endpoint that reports part names, so selective
    texturing stays opt-in for a caller that has obtained them some other way.
    """
    body: dict[str, Any] = {
        "input": geometry_task_id,
        # Tripo pairs the v3.0 texture model with v3.0 AND v3.1 geometry; there
        # is no v3.1 texture model.
        "model": settings.TRIPO_TEXTURE_MODEL,
        "texture_prompt": {"image": image_url},
        "pbr": True,
        "texture_quality": "detailed",
        "texture_alignment": "original_image",
        # Bakes advanced material effects into the base textures. Required for
        # broad viewer compatibility — model-viewer will not reproduce the
        # unbaked effects.
        "bake": True,
    }
    if part_names:
        body["part_names"] = part_names
    return body
