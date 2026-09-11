"""Texture-level recolouring — the pixel primitive, with no glTF container.

Lifted verbatim from ``glb_recolor._recolor_pixels`` so that the colour-variant
baker (which rewrites a whole GLB) and the Configurator's texture baker (which
does not) share ONE implementation of the maths. Two copies of a recolour
algorithm diverge, and this one already has a suspected divergence against the
browser preview — a third copy is the last thing it needs.

Nothing here touches glTF, the database, storage, or HTTP. Bytes in, bytes out.

`factor` is deliberately NOT implemented here. In glTF it is not a pixel
operation at all: it sets ``pbrMetallicRoughness.baseColorFactor`` and leaves the
image untouched (``glb_recolor.recolor``). The Configurator's documented
behaviour for a factor material is that **no texture file is produced**
(docs/configurator/baking.md §3.2, data-model.md §6). Implementing a pixel-space
"factor" here would be inventing an algorithm the backend has never had, so
``recolor_texture`` rejects it instead — see ``UnsupportedTextureMethod``.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
from PIL import Image as PILImage

from app.services.color import colors

# Methods that actually repaint pixels.
PixelMethod = Literal["luminance", "remap"]

PNG_MIME = "image/png"
JPEG_MIME = "image/jpeg"

# Rec. 709 luminance weights (perceptual brightness).
_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class UnsupportedTextureMethod(ValueError):
    """A method with no texture-level implementation was requested."""


# --------------------------------------------------------------------------- #
# Decode / encode
# --------------------------------------------------------------------------- #
def decode_image(image_bytes: bytes) -> PILImage.Image:
    """Decode texture bytes into a fully-loaded PIL image.

    Raises ``UnsupportedTextureMethod``'s sibling ``ValueError`` on anything
    Pillow cannot open, so callers get one exception type to handle.
    """
    if not image_bytes:
        raise ValueError("Texture image is empty.")
    try:
        pil = PILImage.open(io.BytesIO(image_bytes))
        pil.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure is one failure
        raise ValueError(f"Texture image could not be decoded: {exc}") from exc
    return pil


def image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """``(width, height)`` of encoded image bytes, from the header only.

    Pillow reads dimensions without decoding pixels, so this is cheap enough to
    call on a freshly encoded texture just to record what was stored.
    """
    try:
        with PILImage.open(io.BytesIO(image_bytes)) as pil:
            return pil.size
    except Exception as exc:  # noqa: BLE001 - any header failure is one failure
        raise ValueError(f"Texture image has no readable dimensions: {exc}") from exc


def source_has_alpha(pil: PILImage.Image) -> bool:
    """Whether the SOURCE carried alpha — this decides the output format."""
    return pil.mode in ("RGBA", "LA", "P") and "A" in pil.getbands()


def encode_texture(image: PILImage.Image, *, has_alpha: bool) -> tuple[bytes, str]:
    """Encode a texture using the project's existing format policy.

    PNG when the source had alpha (it must be preserved), JPEG otherwise. Both
    are the only values ``ck_option_texture_mime`` permits, so every path that
    produces a Configurator texture has to come through here.
    """
    buf = io.BytesIO()
    if has_alpha:
        # compress_level=1 rather than optimize=True: on a 2K texture the
        # exhaustive filter search costs seconds for a few percent of size, and
        # this file is transferred once to a CDN that serves it compressed.
        image.save(buf, format="PNG", compress_level=1)
        return buf.getvalue(), PNG_MIME
    image.convert("RGB").save(buf, format="JPEG", quality=92, subsampling=0)
    return buf.getvalue(), JPEG_MIME


# --------------------------------------------------------------------------- #
# The pixel primitive
# --------------------------------------------------------------------------- #
def recolor_pixels(
    pil: PILImage.Image,
    hex_color: str,
    remap: bool,
) -> tuple[bytes, str]:
    """Repaint an image to ``hex_color`` while keeping its surface detail.

    Moved here from ``glb_recolor._recolor_pixels``; the body is unchanged, and
    a test asserts both entry points produce byte-identical output.

    ``remap`` selects the near-black treatment. Note the dark/bright points are
    the **2nd and 98th percentiles**, not the absolute darkest and brightest
    pixels. That is the backend's long-standing behaviour and is deliberately
    preserved here — the browser preview
    (Rivollo.Web.Portal/lib/utils/textureRecolor.ts) uses absolute min/max, and
    reconciling the two is an open decision (ADR-005), not this module's call.
    """
    target = np.array(colors.hex_to_rgb(hex_color), dtype=np.float32) / 255.0

    has_alpha = source_has_alpha(pil)
    rgba = pil.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3:]

    lum = rgb @ _REC709  # HxW brightness in 0..1

    # The key idea: we don't multiply the target by the RAW brightness (that
    # makes dark textures come out dark/muddy — the "merging" bug). Instead we
    # build a "detail" map centred on 1.0 that captures only the *relative*
    # light/dark variation (weave, shadows, stitching). Multiplying the target
    # by a detail map centred on 1.0 means the AVERAGE pixel equals the target
    # colour exactly, while highlights/shadows still read as highlights/shadows.
    if remap:
        # Near-black parts: spread the tiny brightness range out first, then
        # centre it so the target colour is fully visible.
        lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
        if hi - lo < 1e-4:
            hi = lo + 1e-4
        norm = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)  # 0..1
        detail = 0.55 + 0.9 * norm  # centred a bit under 1, range ~0.55..1.45
    else:
        mean = float(lum.mean())
        if mean < 1e-3:
            mean = 1e-3
        detail = np.clip(lum / mean, 0.35, 1.8)  # centred on 1.0

    recolored = np.clip(detail[..., None] * target[None, None, :], 0.0, 1.0)
    out = np.concatenate([recolored, alpha], axis=-1)
    out_img = PILImage.fromarray((out * 255).astype(np.uint8), mode="RGBA")

    return encode_texture(out_img, has_alpha=has_alpha)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def recolor_texture(
    image_bytes: bytes,
    method: str,
    color: str,
    brightness: float = 1.0,
) -> tuple[bytes, str]:
    """Recolour encoded texture bytes. Returns ``(bytes, mime)``.

    ``brightness`` is folded into the target colour first, exactly as
    ``glb_recolor.recolor`` does before calling the pixel primitive, so the two
    bakers cannot drift on the slider's meaning.

    Raises ``UnsupportedTextureMethod`` for ``factor`` (not a pixel operation —
    see the module docstring) and for anything unrecognised.
    """
    if method == "factor":
        raise UnsupportedTextureMethod(
            "'factor' is not a texture operation: it sets the material's "
            "baseColorFactor and leaves the image untouched, so no texture file "
            "is produced. Use colors.hex_to_linear_factor for that path."
        )
    if method not in ("luminance", "remap"):
        raise UnsupportedTextureMethod(
            f"Unsupported recolour method {method!r}. "
            "Expected 'luminance' or 'remap'."
        )

    effective_color = colors.adjust_brightness_hex(color, brightness)
    return recolor_pixels(
        decode_image(image_bytes), effective_color, remap=(method == "remap")
    )
