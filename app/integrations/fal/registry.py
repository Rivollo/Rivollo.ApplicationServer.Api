"""Registry of fal.ai image-to-3D models.

Adding a model is a single entry in ``FAL_MODELS`` — no route, service, billing
or frontend change. That is the point of this file: everything that differs
between models lives here, and everything they share lives in
``queue_client.py``.

Each spec owns the three things fal models genuinely disagree about:

  * ``endpoint_id``      which queue endpoint to POST to
  * ``build_body``       the request payload (field names and options differ)
  * ``extract_glb_url``  where the GLB lands in the result JSON

Credit cost lives here too, because models are not priced the same and the
quota check and the deduction must both read the same number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class FalModelSpec:
    """Everything that makes one fal image-to-3D model different from another."""

    # Stable identifier used by the API and stored against generations.
    key: str
    # Shown to sellers in the portal dropdown.
    label: str
    description: str
    # fal queue endpoint, e.g. "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d".
    endpoint_id: str
    # AI credits charged per generation.
    credit_cost: int
    # Seed ETA in seconds, used only until this model has enough real runs
    # recorded to compute a median. fal provides no estimate of its own, so
    # this is a starting point that measured history replaces.
    baseline_estimate_seconds: int
    # image_url -> request payload
    build_body: Callable[[str], dict]
    # result JSON -> GLB download URL (None when absent)
    extract_glb_url: Callable[[dict], Optional[str]]
    # result JSON -> USDZ download URL, for models that emit one themselves.
    # When set and it yields a URL, we store that file directly and skip the
    # Azure GLB->USDZ conversion job entirely — the vendor's own export is both
    # faster and higher fidelity than a converted one.
    extract_usdz_url: Optional[Callable[[dict], Optional[str]]] = None
    # Upper bound on the fal poll loop. Most models finish inside the default;
    # Meshy documents 5-10 minutes, which would race a 600s ceiling.
    max_wait_seconds: float = 600.0

    @property
    def submit_url(self) -> str:
        return f"https://queue.fal.run/{self.endpoint_id}"

    @property
    def provides_usdz(self) -> bool:
        return self.extract_usdz_url is not None


# --------------------------------------------------------------------------- #
# Result extractors
# --------------------------------------------------------------------------- #
def _file_url(node: object) -> Optional[str]:
    """fal wraps files as {url, content_type, file_name, file_size}."""
    if isinstance(node, dict):
        url = node.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def _extract_tripo_glb(result: dict) -> Optional[str]:
    """Tripo returns model_urls.glb, older responses only model_mesh."""
    model_urls = result.get("model_urls") or {}
    if isinstance(model_urls, dict):
        url = _file_url(model_urls.get("glb"))
        if url:
            return url
    return _file_url(result.get("model_mesh"))


def _extract_hunyuan_glb(result: dict) -> Optional[str]:
    """Hunyuan returns model_glb directly, and also mirrors it in model_urls.glb."""
    url = _file_url(result.get("model_glb"))
    if url:
        return url
    model_urls = result.get("model_urls") or {}
    if isinstance(model_urls, dict):
        return _file_url(model_urls.get("glb"))
    return None


def _extract_trellis_glb(result: dict) -> Optional[str]:
    """Trellis returns a single model_glb — no model_urls block."""
    return _file_url(result.get("model_glb"))


def _extract_meshy_glb(result: dict) -> Optional[str]:
    """Meshy returns model_glb, mirrored in model_urls.glb."""
    url = _file_url(result.get("model_glb"))
    if url:
        return url
    model_urls = result.get("model_urls") or {}
    if isinstance(model_urls, dict):
        return _file_url(model_urls.get("glb"))
    return None


def _extract_meshy_usdz(result: dict) -> Optional[str]:
    """Meshy exports USDZ alongside the GLB, under model_urls.usdz.

    Note the entries in ``model_urls`` can be present but null (Meshy returns
    ``"blend": null``), so this must tolerate a missing or empty value rather
    than assume the key implies a file.
    """
    model_urls = result.get("model_urls") or {}
    if isinstance(model_urls, dict):
        return _file_url(model_urls.get("usdz"))
    return None


# --------------------------------------------------------------------------- #
# Request builders
# --------------------------------------------------------------------------- #
def _build_tripo_body(image_url: str) -> dict:
    """Tripo H3.1 request, tuned for a web product viewer.

    Note ``quad`` is deliberately never set: the docs warn that quad topology
    makes Tripo return **FBX** bytes instead of GLB, which would break every
    downstream step (viewer, colour configurator, USDZ conversion).
    """
    return {
        "image_url": image_url,
        "texture": True,
        "pbr": True,
        "texture_quality": "detailed",
        "geometry_quality": "detailed",
        "texture_alignment": "original_image",
        "orientation": "align_image",
        # Cap the mesh. Left unset, Tripo "adaptively determines the count" and
        # on a real product photo that meant 2,000,000 triangles / 1,035,799
        # vertices — a 60.8 MB file whose geometry alone was ~56 MB. Two million
        # triangles also drop the viewer's frame rate far enough that
        # model-viewer's adaptive renderer lowers resolution while the user
        # drags, which is why such models look blurry while rotating and snap
        # sharp when released.
        #
        # 50k keeps Tripo in line with Trellis (~36k verts) and Meshy (~29k),
        # which render smoothly. Surface detail comes from the textures, which
        # stay at "detailed".
        "face_limit": 50000,
    }


def _build_hunyuan_body(image_url: str) -> dict:
    return {
        "input_image_url": image_url,
        # "Normal" produces a textured model. "Geometry" would return an
        # untextured white mesh, which the configurator cannot recolour.
        "generate_type": "Normal",
        # PBR is always on: the colour configurator recolours the base-colour
        # map while preserving normal/roughness/metallic detail, and without
        # PBR maps a recoloured part looks like flat paint.
        "enable_pbr": True,
        # fal's default. Range 40,000-1,500,000.
        "face_count": 500000,
    }


def _build_trellis_body(image_url: str) -> dict:
    """Trellis 2 request, tuned for a web product viewer.

    The guidance/sampling parameters are pinned to fal's documented defaults
    rather than omitted. Trellis exposes ~25 knobs and silently changing one of
    their defaults would change every model we generate; pinning keeps output
    reproducible and makes any future change to our own settings explicit.

    Two settings deliberately depart from the API defaults — see the comments on
    ``decimation_target`` and ``texture_size``. Both were validated against a
    real product photo, not assumed.
    """
    return {
        "image_url": image_url,
        # Highest available structure detail. 512 / 1024 / 1536.
        "resolution": 1536,
        # Stage 1 — sparse structure
        "ss_guidance_strength": 7.5,
        "ss_guidance_rescale": 0.7,
        "ss_guidance_interval_start": 0.6,
        "ss_guidance_interval_end": 1,
        "ss_sampling_steps": 12,
        "ss_rescale_t": 5,
        # Stage 2 — shape refinement
        "shape_slat_guidance_strength": 7.5,
        "shape_slat_guidance_rescale": 0.5,
        "shape_slat_guidance_interval_start": 0.6,
        "shape_slat_guidance_interval_end": 1,
        "shape_slat_sampling_steps": 12,
        "shape_slat_rescale_t": 3,
        # Stage 3 — texture
        "tex_slat_guidance_strength": 1,
        "tex_slat_guidance_rescale": 0,
        "tex_slat_guidance_interval_start": 0.6,
        "tex_slat_guidance_interval_end": 0.9,
        "tex_slat_sampling_steps": 12,
        "tex_slat_rescale_t": 3,
        # ---- Mesh output ----
        # 50k vertices, NOT the API default of 500k. fal's own docs say "500k is
        # good for most uses, reduce to 20k-50k for web/mobile" — and 500k does
        # not merely produce a heavy file, it makes the remesh/UV-unwrap stage
        # fail outright: a real product photo returned HTTP 500 after 404s with
        # every sampling stage already complete. The same image at 50k succeeds.
        "decimation_target": 50000,
        # 4096, above the API default of 2048. With geometry decimated for the
        # web, fine surface detail has to come from the texture rather than from
        # triangle count — so spend the budget here. Measured at 4.5 MB total,
        # which is still far lighter than the 500k-vertex output ever was.
        "texture_size": 4096,
        # Clean topology for downstream use. Adds time but the mesh is small
        # enough now that it completes comfortably.
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0,
        "uv_unwrap_angle_threshold_deg": 90,
        "uv_unwrap_refine_iterations": 0,
        "uv_unwrap_global_iterations": 1,
        "uv_unwrap_smooth_strength": 1,
    }


def _build_meshy_body(image_url: str) -> dict:
    """Meshy-6 Preview request, tuned for a product viewer.

    Rigging and animation are deliberately OFF. fal's own example enables them,
    but Meshy's docs say rigging targets "humanoid characters with clearly
    defined limbs" — a chair or a shoe has none, so it adds minutes of work and
    returns nulls (fal's own sample output shows rig_task_id: null). Turn them
    on only if Rivollo ever sells character models.
    """
    return {
        "image_url": image_url,
        # "standard" = regular high-detail mesh. "lowpoly" would discard the
        # detail we are paying Meshy for.
        "model_type": "standard",
        # Triangles for detailed geometry; quads matter for animation, not for
        # a static product render.
        "topology": "triangle",
        # Meshy's default, and already web-appropriate.
        "target_polycount": 30000,
        "symmetry_mode": "auto",
        "should_remesh": True,
        "should_texture": True,
        # Metallic / roughness / normal maps in addition to base colour. Without
        # this the colour configurator has no surface detail to preserve when it
        # recolours, and recoloured parts read as flat paint.
        "enable_pbr": True,
        "enable_rigging": False,
        "enable_animation": False,
        "enable_safety_checker": True,
    }


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_KEY = "tripo"

FAL_MODELS: dict[str, FalModelSpec] = {
    "tripo": FalModelSpec(
        key="tripo",
        label="Tripo",
        # Kept to a short tag, not a sentence — it renders inline next to the
        # time and credit cost in the model picker.
        description="Fast and reliable",
        endpoint_id="tripo3d/h3.1/image-to-3d",
        credit_cost=10,
        # Measured at 165s end-to-end on a real product photo with face_limit
        # applied. Seeded slightly above so the countdown does not hit zero
        # before the model lands; the measured median replaces it after 3 runs.
        baseline_estimate_seconds=180,
        build_body=_build_tripo_body,
        extract_glb_url=_extract_tripo_glb,
    ),
    "hunyuan": FalModelSpec(
        key="hunyuan",
        label="Hunyuan",
        description="Full PBR textures",
        endpoint_id="fal-ai/hunyuan-3d/v3.1/pro/image-to-3d",
        credit_cost=10,
        # Measured end-to-end at ~202s (submit → GLB downloaded) on a live run.
        baseline_estimate_seconds=240,
        build_body=_build_hunyuan_body,
        extract_glb_url=_extract_hunyuan_glb,
    ),
    "trellis": FalModelSpec(
        key="trellis",
        label="Trellis",
        description="Sharpest detail",
        endpoint_id="fal-ai/trellis-2",
        credit_cost=10,
        # Seed from live runs on a real product photo: 241s and 57s for the
        # same input on different runners — fal's queue variance is wide, so
        # this sits between them until the measured median takes over.
        baseline_estimate_seconds=180,
        build_body=_build_trellis_body,
        extract_glb_url=_extract_trellis_glb,
    ),
    "meshy": FalModelSpec(
        key="meshy",
        label="Meshy",
        description="Best overall quality",
        endpoint_id="fal-ai/meshy/v6/image-to-3d",
        credit_cost=20,
        # Meshy documents 5-10 minutes; a live run on a real product photo took
        # 204s. Seed between the two rather than trusting either alone — the
        # measured median replaces this after three runs.
        baseline_estimate_seconds=360,
        # ...which also means the default 600s poll ceiling would race the
        # model itself. Give it room for queue time on top of generation.
        max_wait_seconds=1200.0,
        build_body=_build_meshy_body,
        extract_glb_url=_extract_meshy_glb,
        # Meshy exports USDZ itself — no Azure conversion job needed.
        extract_usdz_url=_extract_meshy_usdz,
    ),
}


def list_model_specs() -> list[FalModelSpec]:
    """All selectable models, in registry (display) order."""
    return list(FAL_MODELS.values())


def get_model_spec(key: Optional[str]) -> FalModelSpec:
    """Look up a model by key.

    An unknown key raises rather than silently falling back to the default —
    a seller must never be charged for a model they did not choose.
    """
    resolved = (key or DEFAULT_MODEL_KEY).strip().lower()
    spec = FAL_MODELS.get(resolved)
    if spec is None:
        valid = ", ".join(FAL_MODELS)
        raise ValueError(f"Unknown 3D model '{key}'. Valid models: {valid}")
    return spec
