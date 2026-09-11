"""Pure Configurator texture baking: source pixels + recipe -> texture bytes.

This is the whole of the Configurator's "bake" arithmetic, and deliberately
nothing else. It does not touch the database, Azure, HTTP, or FastAPI; it does
not read or write a GLB. The original GLB stays canonical (ADR-002) and the
Configurator stores textures rather than per-option models (ADR-003), so this
layer's entire job is: given the bytes of a source image and a validated
``StoredRecipe``, return the bytes to store.

Fetching the source — out of the GLB for a recolour, or from the seller's upload
for ``method="image"`` — belongs to the caller. Keeping I/O out means this module
is testable against a handful of in-memory images, which is what makes the
recolour maths verifiable at all.

Two returns matter and are easy to confuse:

    BakedTexture  a file to store for this material
    None          no file for this material, legitimately

``None`` is not a failure. A ``factor`` material and a material with no
base-colour image both produce no texture — the viewer colours them through
``baseColorFactor`` instead. docs/configurator/baking.md §3.2 and
data-model.md §6 both state an option may end up with zero texture rows.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from app.schemas.configurator import StoredRecipe
from app.services.color import texture

logger = logging.getLogger(__name__)

# Methods that repaint the GLB's own artwork.
RECOLOUR_METHODS = frozenset({"factor", "luminance", "remap"})
# The method that substitutes a seller-supplied image wholesale.
IMAGE_METHOD = "image"


class TextureBakeError(ValueError):
    """The recipe could not be turned into a texture."""


@dataclass(frozen=True)
class BakedTexture:
    """One texture to store, for one glTF material index."""

    material_index: int
    data: bytes
    content_type: str
    width: int
    height: int

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class Treatment:
    """How one material is treated by a recipe, after overrides are applied."""

    method: str
    color: Optional[str]
    brightness: float

    @property
    def produces_a_texture(self) -> bool:
        """``factor`` changes a material parameter, not an image."""
        return self.method != "factor"


def effective_treatment(recipe: StoredRecipe, material_index: int) -> Treatment:
    """Resolve the recipe down to what this one material actually gets.

    A per-material ``override`` wins over the recipe's top-level method and
    colour. Overrides carry no ``brightness`` of their own, so they inherit the
    recipe's — brightness is a part-level slider in the editor, and resetting it
    to 1.0 for an overridden material would silently drop the seller's setting.

    ``image`` recipes cannot carry overrides (the schema rejects them), so the
    treatment is uniform across the part.
    """
    if recipe.method == IMAGE_METHOD:
        return Treatment(method=IMAGE_METHOD, color=None, brightness=1.0)

    for override in recipe.overrides:
        if override.material_index == material_index:
            return Treatment(
                method=override.method,
                color=override.color,
                brightness=recipe.brightness,
            )

    return Treatment(
        method=recipe.method,
        color=recipe.color,
        brightness=recipe.brightness,
    )


def bake_material_texture(
    *,
    material_index: int,
    source_image: Optional[bytes],
    recipe: StoredRecipe,
) -> Optional[BakedTexture]:
    """Bake one material's texture, or return None when there is none to bake.

    ``source_image`` is the material's base-colour image for a recolour, or the
    seller's uploaded file for ``method="image"``. Pass ``None`` when the
    material has no base-colour image.

    Returns ``None`` — not an error — when:
      * the treatment resolves to ``factor``; or
      * a pixel method was asked for but the material has no source image.

    The second case mirrors ``glb_recolor.recolor``, which downgrades
    ``luminance``/``remap`` to ``factor`` when the material has no image to
    repaint. Same rule, same outcome: no file.

    Raises ``TextureBakeError`` when a texture *should* have been produced and
    could not be.
    """
    treatment = effective_treatment(recipe, material_index)

    if treatment.method == IMAGE_METHOD:
        if source_image is None:
            # The uploaded file is the ONLY source for an image recipe. Missing
            # it is a real failure, unlike a missing GLB texture.
            raise TextureBakeError(
                f"Material {material_index}: recipe.method is 'image' but no "
                "uploaded image was supplied."
            )
        return _normalise_uploaded_image(material_index, source_image)

    if treatment.method not in RECOLOUR_METHODS:
        raise TextureBakeError(
            f"Material {material_index}: unsupported recipe method "
            f"{treatment.method!r}."
        )

    if not treatment.produces_a_texture:
        # factor: the colour is applied as baseColorFactor, no file.
        return None

    if source_image is None:
        # No artwork to repaint — the engine's documented downgrade to factor.
        logger.debug(
            "Material %s has no base-colour image; %s produces no texture.",
            material_index,
            treatment.method,
        )
        return None

    if not treatment.color:
        raise TextureBakeError(
            f"Material {material_index}: method {treatment.method!r} needs a "
            "colour and the recipe has none."
        )

    try:
        data, content_type = texture.recolor_texture(
            source_image,
            treatment.method,
            treatment.color,
            treatment.brightness,
        )
    except texture.UnsupportedTextureMethod as exc:
        raise TextureBakeError(f"Material {material_index}: {exc}") from exc
    except ValueError as exc:
        raise TextureBakeError(
            f"Material {material_index}: source texture could not be recoloured: {exc}"
        ) from exc

    width, height = texture.image_dimensions(data)
    return BakedTexture(
        material_index=material_index,
        data=data,
        content_type=content_type,
        width=width,
        height=height,
    )


def bake_option_textures(
    *,
    sources: Mapping[int, Optional[bytes]],
    recipe: StoredRecipe,
) -> list[BakedTexture]:
    """Bake every material of a part. Materials that yield no file are omitted.

    ``sources`` maps material index -> source bytes (``None`` when the material
    has no base-colour image). For an ``image`` recipe the caller supplies the
    same uploaded bytes for every index, because the upload replaces the base
    colour of every material in the part (data-model.md §6).

    The returned list is ordered by material index, so the same recipe over the
    same sources always produces the same sequence.
    """
    baked: list[BakedTexture] = []
    for material_index in sorted(sources):
        result = bake_material_texture(
            material_index=material_index,
            source_image=sources[material_index],
            recipe=recipe,
        )
        if result is not None:
            baked.append(result)
    return baked


# --------------------------------------------------------------------------- #
# Uploaded images
# --------------------------------------------------------------------------- #
def _normalise_uploaded_image(material_index: int, source_image: bytes) -> BakedTexture:
    """Re-encode a seller's upload into a storable texture.

    Re-encoding is NOT optional. ``recipe.ALLOWED_IMAGE_EXTENSIONS`` accepts
    ``.webp``, but ``ck_option_texture_mime`` on tbl_part_option_textures permits
    only ``image/png`` and ``image/jpeg`` — so a WEBP upload stored verbatim
    would violate the constraint. Routing every upload through the same encoder
    as the recolour path also means one format policy for all Configurator
    textures.

    The image is otherwise left alone: no resize, no colour adjustment. The
    recipe's ``color`` and ``brightness`` are ignored for ``image``, which is
    what api-spec.md §7 specifies and what ``compute_recipe_hash`` already
    assumes by excluding them from the hash.
    """
    try:
        pil = texture.decode_image(source_image)
    except ValueError as exc:
        raise TextureBakeError(
            f"Material {material_index}: uploaded image could not be decoded: {exc}"
        ) from exc

    data, content_type = texture.encode_texture(
        pil, has_alpha=texture.source_has_alpha(pil)
    )
    width, height = texture.image_dimensions(data)
    return BakedTexture(
        material_index=material_index,
        data=data,
        content_type=content_type,
        width=width,
        height=height,
    )
