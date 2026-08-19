"""Registry of image-to-3D generation models — database-backed.

Each row in ``tbl_mstr_3d_models`` (see ``app/models/model_registry.py`` and
``sql/create_3d_model_registry.sql``) is one selectable model. Adding,
repricing, reordering, or disabling a model is a database change — no route,
service, billing, or frontend deploy.

This is possible because every model this app calls today speaks fal.ai's
queue protocol (submit -> poll -> result -> download) and differs from every
other model only in its request payload and where the result JSON hides the
download URL — genuinely static data, not logic. ``build_body`` is a template
merge; ``extract_glb_url``/``extract_usdz_url`` are an ordered "try this
JSON path, then that one" search. Both are implemented once, generically,
below, driven entirely by each row's ``provider_config`` — no per-model
Python function exists anymore.

Cached in-process with a short TTL — the same pattern
``app/services/pricing_service.py`` already uses for the same kind of
problem (configuration edited directly in the database, read on nearly every
request). See ``MODEL_REGISTRY_CACHE_TTL_SECONDS`` below.

Two lookup functions, deliberately not one, because they answer different
questions:

  * :func:`get_model_spec`     — "what should a NEW generation request use?"
    Never resolves an inactive model; an unknown or deactivated key raises.
  * :func:`get_model_spec_any` — "what WAS this key, historically?" Used only
    for ETA baselines of past runs, where a model that has since been
    deactivated must still resolve. Returns ``None`` rather than raising.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_registry import Model3DConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FalModelSpec:
    """One selectable image-to-3D model, resolved from a database row.

    ``build_body``/``extract_glb_url``/``extract_usdz_url`` are ordinary
    methods here (not per-instance closures), generic across every model —
    see the module docstring. Callers (``queue_client.py``,
    ``product_service.py``) use this exactly as before; only where the data
    comes from changed.
    """

    key: str
    provider: str
    label: str
    description: str
    endpoint_id: str
    credit_cost: int
    baseline_estimate_seconds: int
    free_plan_eligible: bool
    is_default: bool
    max_wait_seconds: float
    image_url_field: str
    request_body_template: dict
    glb_url_paths: tuple[str, ...]
    usdz_url_paths: tuple[str, ...]

    @property
    def submit_url(self) -> str:
        return f"https://queue.fal.run/{self.endpoint_id}"

    @property
    def provides_usdz(self) -> bool:
        return len(self.usdz_url_paths) > 0

    def build_body(self, image_url: str) -> dict:
        """image_url -> request payload.

        Every current model's request is this static template with one field
        substituted — see ``provider_config.request_body_template`` on the
        row this spec came from.
        """
        return {**self.request_body_template, self.image_url_field: image_url}

    def extract_glb_url(self, result: dict) -> Optional[str]:
        """result JSON -> GLB download URL (None when absent)."""
        return _first_matching_url(result, self.glb_url_paths)

    def extract_usdz_url(self, result: dict) -> Optional[str]:
        """result JSON -> vendor-supplied USDZ download URL (None when this
        model doesn't export one, i.e. usdz_url_paths is empty)."""
        return _first_matching_url(result, self.usdz_url_paths)


# --------------------------------------------------------------------------- #
# Generic path resolution — replaces every per-model extractor function.
# --------------------------------------------------------------------------- #
def _resolve_path(node: Any, segments: list[str]) -> Any:
    """Walk one dot-path's segments through a JSON-like structure.

    A segment suffixed ``[]`` means "this key holds a list — resolve the
    REST of the path against each item in turn, and return the first item
    whose resolution is truthy." That is the one list construct any current
    model's extractor needs (SAM 3D's ``individual_glbs`` fallback — try
    each entry's ``.url`` in order, use the first non-empty one); everything
    else is plain dict traversal. Deliberately not full JSONPath — this one
    rule is what the real data needs. Never raises: an unexpected shape
    (wrong type, missing key) just resolves to ``None``.
    """
    if not segments:
        return node
    segment, *rest = segments
    if segment.endswith("[]"):
        key = segment[:-2]
        items = node.get(key) if isinstance(node, dict) else None
        if not isinstance(items, list):
            return None
        for item in items:
            value = _resolve_path(item, rest)
            if value:
                return value
        return None
    if not isinstance(node, dict):
        return None
    return _resolve_path(node.get(segment), rest)


def _first_matching_url(result: dict, paths: tuple[str, ...]) -> Optional[str]:
    """Try each dot-path in order; return the first that resolves to a
    non-empty string. fal wraps files as ``{url, content_type, ...}``, so
    every current path ends in ``.url``."""
    for path in paths:
        value = _resolve_path(result, path.split("."))
        if isinstance(value, str) and value:
            return value
    return None


def _row_to_spec(row: Model3DConfig) -> FalModelSpec:
    config = row.provider_config or {}
    return FalModelSpec(
        key=row.key,
        provider=row.provider,
        label=row.label,
        description=row.description,
        endpoint_id=row.endpoint_id,
        credit_cost=row.credit_cost,
        baseline_estimate_seconds=row.baseline_estimate_seconds,
        free_plan_eligible=row.free_plan_eligible,
        is_default=row.is_default,
        max_wait_seconds=float(row.max_wait_seconds),
        image_url_field=config.get("image_url_field") or "image_url",
        request_body_template=dict(config.get("request_body_template") or {}),
        glb_url_paths=tuple(config.get("glb_url_paths") or ()),
        usdz_url_paths=tuple(config.get("usdz_url_paths") or ()),
    )


# --------------------------------------------------------------------------- #
# Cache
#
# tbl_mstr_3d_models is configuration — it changes when someone edits the
# database, not per request — so resolving it fresh on every product
# creation / model-picker call spent a DB round trip to produce an answer
# that is the same 99.9% of the time. Mirrors PRICING_CACHE_TTL_SECONDS in
# app/services/pricing_service.py exactly, for the same reason: there is no
# explicit invalidation because nothing in this app writes a row — rows are
# edited directly against the database, so a change appears within the TTL
# rather than immediately. Set to 0 to disable caching entirely.
# --------------------------------------------------------------------------- #
MODEL_REGISTRY_CACHE_TTL_SECONDS = int(os.getenv("MODEL_REGISTRY_CACHE_TTL_SECONDS", "60"))

_registry_cache: Optional[tuple[float, list[FalModelSpec]]] = None
_registry_cache_lock = asyncio.Lock()


def clear_model_registry_cache() -> None:
    """Drop the cached model list.

    Nothing in the request path calls this — it exists so tests don't leak
    state between cases, and so a future admin write path has an obvious hook.
    """
    global _registry_cache
    _registry_cache = None


async def _load_active_specs(db: AsyncSession) -> list[FalModelSpec]:
    result = await db.execute(
        select(Model3DConfig)
        .where(Model3DConfig.isactive.is_(True))
        .order_by(Model3DConfig.order_index)
    )
    return [_row_to_spec(row) for row in result.scalars().all()]


async def list_model_specs(db: AsyncSession) -> list[FalModelSpec]:
    """All selectable models, in registry (display) order.

    Filters to ``isactive`` rows only — this is the "new create request"
    lookup mode. An inactive model never appears here, so it can never be
    offered in the picker or charged for on a new generation. Cached; see
    the module-level comment above.
    """
    global _registry_cache

    if MODEL_REGISTRY_CACHE_TTL_SECONDS <= 0:
        return await _load_active_specs(db)

    cached = _registry_cache
    if cached and time.monotonic() < cached[0]:
        return cached[1]

    async with _registry_cache_lock:
        # Re-check under the lock. A request that queued here while another
        # rebuilt the cache should use that result, not immediately rebuild.
        cached = _registry_cache
        if cached and time.monotonic() < cached[0]:
            return cached[1]

        specs = await _load_active_specs(db)
        # Never cache an empty result — an unmigrated or unreachable
        # database would otherwise leave the model picker blank for a full
        # TTL after the real problem was fixed.
        if specs:
            _registry_cache = (time.monotonic() + MODEL_REGISTRY_CACHE_TTL_SECONDS, specs)
        return specs


async def get_model_spec(db: AsyncSession, key: Optional[str]) -> FalModelSpec:
    """Resolve one model for a NEW generation request.

    An unknown OR inactive key raises rather than silently falling back — a
    seller must never be charged for a model they did not choose, and must
    never be able to select a model that has been turned off. ``key=None``
    resolves to whichever active row has ``is_default`` set (enforced unique
    in the database — see the partial index on ``tbl_mstr_3d_models``).
    """
    specs = await list_model_specs(db)

    if key is None:
        default = next((s for s in specs if s.is_default), None)
        if default is None:
            # Configuration error (no default row, or the only default is
            # inactive) — surfaces as a 500 via the route's generic handler,
            # which is correct: this is an operator mistake, not a bad request.
            raise ValueError("No default 3D model is configured.")
        return default

    resolved_key = key.strip().lower()
    spec = next((s for s in specs if s.key == resolved_key), None)
    if spec is None:
        valid = ", ".join(sorted(s.key for s in specs))
        raise ValueError(f"Unknown 3D model '{key}'. Valid models: {valid}")
    return spec


async def get_model_spec_any(db: AsyncSession, key: str) -> Optional[FalModelSpec]:
    """Resolve a model by key regardless of ``isactive``.

    For ETA baselines and other historical lookups (a
    ``ModelGenerationStat`` row can reference a model that has since been
    deactivated, and computing its baseline must still work). Deliberately
    NOT used for new generation requests — see :func:`get_model_spec`.
    Bypasses the cache (this path is rare — every current caller already
    passes an explicit baseline, see ``generation_estimate_service.py``) and
    returns ``None`` rather than raising when the key never existed, since
    callers here want a graceful fallback, not a 400/500.
    """
    result = await db.execute(
        select(Model3DConfig).where(Model3DConfig.key == key.strip().lower())
    )
    row = result.scalar_one_or_none()
    return _row_to_spec(row) if row else None
