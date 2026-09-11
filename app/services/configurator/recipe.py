"""Recipe helpers — hashing and the image_url ownership check.

Pure functions plus one settings-dependent validator. No database, no session.
Split out from the services because ``compute_recipe_hash`` is needed by the
bake service too (Phase 4), and a hash that two modules compute differently is
the kind of bug that only shows up as a texture nobody can invalidate.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.configurator import StoredRecipe

# The Configurator's OWN baker version, not app.services.color.BAKER_VERSION.
#
# That constant ("2") versions the colour-variant whole-GLB baker. The
# Configurator bakes textures, a different implementation with different
# output, so it needs a version it can bump independently. Bump this whenever a
# change would make the same recipe produce different texture bytes — every
# recipe_hash changes, which invalidates baked textures and causes a re-bake.
CONFIGURATOR_BAKER_VERSION = "1"

# What a seller may point an image recipe at.
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def compute_recipe_hash(recipe: StoredRecipe, glb_version: str) -> str:
    """Stable fingerprint of everything that affects the baked texture bytes.

    Folds in ``glb_version`` so that re-uploading the product's model
    invalidates every option automatically, and the baker version so that
    improving the engine does the same.

    For an image recipe, ``color`` and ``brightness`` are deliberately excluded:
    api-spec.md marks them "ignored" for that method, so letting them into the
    hash would make an irrelevant edit trigger a pointless re-bake.
    """
    if recipe.is_image:
        payload: dict[str, Any] = {
            "version": recipe.version,
            "method": recipe.method,
            "image_url": recipe.image_url,
        }
    else:
        payload = {
            "version": recipe.version,
            "method": recipe.method,
            "color": recipe.color,
            "brightness": round(float(recipe.brightness), 4),
            "overrides": [
                {
                    "material_index": o.material_index,
                    "method": o.method,
                    "color": o.color,
                }
                for o in sorted(recipe.overrides, key=lambda o: o.material_index)
            ],
        }

    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    digest = f"{glb_version}|{canonical}|v{CONFIGURATOR_BAKER_VERSION}"
    return hashlib.sha256(digest.encode("utf-8")).hexdigest()[:32]


def uploads_namespace_prefix(user_id: Any) -> str:
    """The CDN prefix under which this seller's own uploads live.

    Mirrors ``storage_service.upload_file_content``, which writes to
    ``{container}/users/{user_id}/uploads/{upload_id}/{filename}`` and returns
    ``{CDN_BASE_URL}/{container}/{blob_path}``.
    """
    base = (settings.CDN_BASE_URL or "").rstrip("/")
    container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
    return f"{base}/{container}/users/{user_id}/uploads/"


def validate_image_url_ownership(image_url: Optional[str], user_id: Any) -> str:
    """Reject any image_url that is not this seller's own upload. Raises 400.

    🔴 Security-critical (ADR-013). The bake service dereferences this URL
    server-side, so an unvalidated value is an SSRF vector; and a URL under
    another seller's prefix is a cross-tenant hotlink.

    The check is a prefix comparison against a URL we constructed the shape of.
    It deliberately does NOT fetch the URL to decide whether it is acceptable —
    that would be the very request we are trying to prevent.
    """
    detail = (
        "recipe.image_url must be a file you uploaded via POST /uploads/content."
    )

    if not image_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    if not settings.CDN_BASE_URL:
        # Fail closed. Without a configured CDN base there is no prefix to
        # anchor the check against, and accepting an arbitrary URL here is
        # exactly the vulnerability this function exists to prevent.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage is not configured; cannot validate uploaded images.",
        )

    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    # Reject traversal before normalising, so "..%2F" cannot climb out of the
    # seller's prefix after the prefix check has already passed.
    decoded_path = unquote(parsed.path)
    if ".." in decoded_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    prefix = uploads_namespace_prefix(user_id)
    if not image_url.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    extension = os.path.splitext(decoded_path)[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "recipe.image_url must point at an image "
                f"({', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))})."
            ),
        )

    return image_url
