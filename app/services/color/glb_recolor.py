"""Deterministic GLB recolour engine.

Design goals (the whole point of the feature):
  * Change ONLY the base colour of a part.
  * NEVER touch geometry, UVs, normal maps, or roughness/metallic maps — so mesh
    quality and surface detail are preserved byte-for-byte.
  * Keep the texture's internal light/dark variation (weave, stitching, shadows)
    so a recoloured part still looks like real material, not flat paint.

Two methods, chosen per part by its average albedo:

  factor     -> For near-white / neutral parts (soles, white leather) and for
                parts that have no base-colour image at all. We only change the 4
                floats of ``baseColorFactor`` (sRGB->linear). The image bytes stay
                identical. ``white x colour = colour`` and shading survives.

  luminance  -> For already-coloured parts (a blue denim upper). We read the base
                image, compute each pixel's brightness, and repaint it as
                ``brightness x target_colour``. Hue changes, detail stays. We write
                the repainted image as a NEW image/texture and point the material at
                it, leaving the original (and any part sharing it) untouched.

  remap      -> Variant of luminance for very dark parts (laces): we percentile-
                stretch the brightness first so the target colour is actually visible.

The GLB writer relocates every existing bufferView into a fresh 4-byte-aligned
binary blob and appends the new recoloured images at the end. Accessors reference
bufferViews by index (not absolute offset), so geometry stays valid.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from pygltflib import GLTF2, BufferView, Image, Texture, TextureInfo

from app.services.color import colors

# --- thresholds for automatic method selection -----------------------------
_NEAR_WHITE_BRIGHTNESS = 0.82   # >= this and low saturation -> neutral -> factor
_NEAR_WHITE_SATURATION = 0.18
_NEAR_BLACK_BRIGHTNESS = 0.16   # <= this -> very dark -> remap

# Two materials whose average colours are within this RGB distance are treated
# as the SAME visual part (AI mesh tools like Tripo split one part into many
# materials). Materials sharing a base-colour image are always grouped.
_GROUP_COLOR_DISTANCE = 42.0

_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Data returned to the API layer
# --------------------------------------------------------------------------- #
@dataclass
class PartInfo:
    material_index: int
    name: str
    mesh_names: list[str]
    has_base_color_texture: bool
    average_color: str          # sRGB hex, for the UI swatch
    suggested_method: str       # "factor" | "luminance" | "remap"
    group_id: int = 0           # materials that form one visual part share this
    center: list[float] | None = None  # world-space centre, for the selection pin


@dataclass
class RecolorOverride:
    material_index: int
    color: str                  # sRGB hex from the picker
    method: str = "auto"        # "auto" | "factor" | "luminance" | "remap"
    brightness: float = 1.0     # HSL-lightness multiplier (1.0 = unchanged)


@dataclass
class _LuminanceJob:
    material_index: int
    encoded: bytes
    mime: str
    sampler: int | None
    alpha: float


# --------------------------------------------------------------------------- #
# Low-level GLB image access
# --------------------------------------------------------------------------- #
def _bufferview_bytes(gltf: GLTF2, blob: bytes, bv_index: int) -> bytes:
    bv = gltf.bufferViews[bv_index]
    start = bv.byteOffset or 0
    return blob[start : start + bv.byteLength]


# Texture extensions that relocate the image index out of `texture.source`.
# Trellis emits EXT_texture_webp; Basis/KTX2 pipelines use the other two. A
# texture using any of these has `source = None`, so reading only `source`
# would make a fully textured part look untextured — and the recolour would
# silently fall back to a flat colour, destroying the surface detail.
_TEXTURE_SOURCE_EXTENSIONS = (
    "EXT_texture_webp",
    "KHR_texture_basisu",
    "EXT_texture_avif",
)


def _texture_source(tex) -> int | None:
    """Image index for a texture, honouring the source-relocating extensions."""
    if tex.source is not None:
        return tex.source
    extensions = getattr(tex, "extensions", None) or {}
    for name in _TEXTURE_SOURCE_EXTENSIONS:
        node = extensions.get(name)
        if isinstance(node, dict) and node.get("source") is not None:
            return node["source"]
    return None


def _base_color_image_index(gltf: GLTF2, material_index: int) -> int | None:
    """material -> pbr.baseColorTexture -> image index."""
    if material_index >= len(gltf.materials):
        return None
    mat = gltf.materials[material_index]
    pbr = mat.pbrMetallicRoughness
    if pbr is None or pbr.baseColorTexture is None:
        return None
    tex = gltf.textures[pbr.baseColorTexture.index]
    return _texture_source(tex)


def _base_color_texture_and_sampler(gltf: GLTF2, material_index: int) -> tuple[int | None, int | None]:
    mat = gltf.materials[material_index]
    pbr = mat.pbrMetallicRoughness
    if pbr is None or pbr.baseColorTexture is None:
        return None, None
    tex = gltf.textures[pbr.baseColorTexture.index]
    return _texture_source(tex), tex.sampler


def _load_pil_from_image(gltf: GLTF2, blob: bytes, image_index: int) -> tuple[PILImage.Image, str]:
    img = gltf.images[image_index]
    if img.bufferView is None:
        raise ValueError("Image is not embedded in the GLB buffer (external/data-uri not supported).")
    data = _bufferview_bytes(gltf, blob, img.bufferView)
    pil = PILImage.open(io.BytesIO(data))
    pil.load()
    mime = img.mimeType or ("image/png" if pil.format == "PNG" else "image/jpeg")
    return pil, mime


# --------------------------------------------------------------------------- #
# Method selection
# --------------------------------------------------------------------------- #
def _brightness_saturation(rgb: colors.Rgb) -> tuple[float, float]:
    r, g, b = (c / 255 for c in rgb)
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx, mn = max(r, g, b), min(r, g, b)
    saturation = 0.0 if mx == 0 else (mx - mn) / mx
    return brightness, saturation


def _suggest_method(avg_rgb: colors.Rgb, has_texture: bool) -> str:
    if not has_texture:
        return "factor"
    brightness, saturation = _brightness_saturation(avg_rgb)
    if brightness >= _NEAR_WHITE_BRIGHTNESS and saturation <= _NEAR_WHITE_SATURATION:
        return "factor"
    if brightness <= _NEAR_BLACK_BRIGHTNESS:
        return "remap"
    return "luminance"


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def inspect(path: Path) -> list[PartInfo]:
    gltf = GLTF2().load(str(path))
    blob = gltf.binary_blob() or b""

    # Map material index -> list of mesh/node names that use it (best effort).
    mesh_names: dict[int, list[str]] = {}
    for mesh in gltf.meshes or []:
        for prim in mesh.primitives:
            if prim.material is not None:
                mesh_names.setdefault(prim.material, []).append(mesh.name or "mesh")

    parts: list[PartInfo] = []
    avg_rgbs: list[colors.Rgb] = []
    img_sources: list[int | None] = []
    for i, mat in enumerate(gltf.materials or []):
        img_index = _base_color_image_index(gltf, i)
        has_texture = img_index is not None
        if has_texture:
            try:
                pil, _ = _load_pil_from_image(gltf, blob, img_index)
                small = pil.convert("RGB").resize((16, 16))
                arr = np.asarray(small, dtype=np.float32).reshape(-1, 3)
                avg = tuple(int(round(c)) for c in arr.mean(axis=0))  # type: ignore[assignment]
            except Exception:
                has_texture = False
                img_index = None
                avg = (255, 255, 255)
        else:
            factor = mat.pbrMetallicRoughness.baseColorFactor if mat.pbrMetallicRoughness else None
            avg = colors.hex_to_rgb(colors.linear_factor_to_hex(factor))

        avg_rgbs.append(avg)  # type: ignore[arg-type]
        img_sources.append(img_index)
        parts.append(
            PartInfo(
                material_index=i,
                name=mat.name or f"Material {i}",
                mesh_names=sorted(set(mesh_names.get(i, []))),
                has_base_color_texture=has_texture,
                average_color=colors.rgb_to_hex(avg),  # type: ignore[arg-type]
                suggested_method=_suggest_method(avg, has_texture),  # type: ignore[arg-type]
            )
        )

    group_ids = _compute_groups(avg_rgbs, img_sources)
    centers = _material_centers(gltf)
    for part, gid in zip(parts, group_ids):
        part.group_id = gid
        part.center = centers.get(part.material_index)
    return parts


# --------------------------------------------------------------------------- #
# World-space centres — so the UI can drop a selection pin on the right spot
# --------------------------------------------------------------------------- #
def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return m


def _node_local_matrix(node) -> np.ndarray:
    if node.matrix:
        # glTF matrices are column-major; reshape+transpose -> row-major.
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T
    m = np.eye(4, dtype=np.float64)
    if node.translation:
        m[:3, 3] = node.translation
    if node.rotation:
        m = m @ _quat_to_matrix(*node.rotation)
    if node.scale:
        s = np.eye(4, dtype=np.float64)
        s[0, 0], s[1, 1], s[2, 2] = node.scale
        m = m @ s
    return m


def _material_centers(gltf: GLTF2) -> dict[int, list[float]]:
    """Average world-space centre of the geometry using each material."""
    nodes = gltf.nodes or []
    meshes = gltf.meshes or []
    accessors = gltf.accessors or []

    # roots: prefer the active scene, else nodes that are nobody's child
    child_ids = {c for n in nodes for c in (n.children or [])}
    if gltf.scenes:
        roots = gltf.scenes[gltf.scene or 0].nodes or []
    else:
        roots = [i for i in range(len(nodes)) if i not in child_ids]

    acc: dict[int, list[np.ndarray]] = {}

    def visit(idx: int, parent: np.ndarray) -> None:
        if idx >= len(nodes):
            return
        node = nodes[idx]
        world = parent @ _node_local_matrix(node)
        if node.mesh is not None and node.mesh < len(meshes):
            for prim in meshes[node.mesh].primitives:
                if prim.material is None or prim.attributes.POSITION is None:
                    continue
                a = accessors[prim.attributes.POSITION]
                if not a.min or not a.max:
                    continue
                local = np.array(
                    [(a.min[i] + a.max[i]) / 2 for i in range(3)] + [1.0],
                    dtype=np.float64,
                )
                p = (world @ local)[:3]
                acc.setdefault(prim.material, []).append(p)
        for ch in node.children or []:
            visit(ch, world)

    for r in roots:
        visit(r, np.eye(4, dtype=np.float64))

    return {
        mat: [float(v) for v in np.mean(pts, axis=0)]
        for mat, pts in acc.items()
        if pts
    }


# --------------------------------------------------------------------------- #
# Grouping — collapse the many materials of one visual part into one selection
# --------------------------------------------------------------------------- #
def _compute_groups(
    avg_rgbs: list[colors.Rgb], img_sources: list[int | None]
) -> list[int]:
    """Union materials that are the same visual part.

    Signals (either one unions two materials):
      * they reference the SAME base-colour image source, or
      * their average colours are within ``_GROUP_COLOR_DISTANCE``.
    Returns a dense group id (0..k-1) per material, ordered by first appearance.
    """
    n = len(avg_rgbs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # union by shared image source
    by_source: dict[int, int] = {}
    for i, src in enumerate(img_sources):
        if src is None:
            continue
        if src in by_source:
            union(by_source[src], i)
        else:
            by_source[src] = i

    # union by colour proximity
    for i in range(n):
        ri, gi, bi = avg_rgbs[i]
        for j in range(i + 1, n):
            rj, gj, bj = avg_rgbs[j]
            dist = ((ri - rj) ** 2 + (gi - gj) ** 2 + (bi - bj) ** 2) ** 0.5
            if dist <= _GROUP_COLOR_DISTANCE:
                union(i, j)

    # densify root ids to 0..k-1 in first-seen order
    remap: dict[int, int] = {}
    result: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap)
        result.append(remap[root])
    return result


# --------------------------------------------------------------------------- #
# Pixel recolour (luminance / remap)
# --------------------------------------------------------------------------- #
def _recolor_pixels(pil: PILImage.Image, hex_color: str, remap: bool) -> tuple[bytes, str]:
    target = np.array(colors.hex_to_rgb(hex_color), dtype=np.float32) / 255.0

    has_alpha = pil.mode in ("RGBA", "LA", "P") and "A" in pil.getbands()
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

    buf = io.BytesIO()
    if has_alpha:
        # compress_level=1 rather than optimize=True: on a 2K texture the
        # exhaustive filter search costs seconds for a few percent of size, and
        # this file is transferred once to a CDN that serves it compressed.
        out_img.save(buf, format="PNG", compress_level=1)
        mime = "image/png"
    else:
        out_img.convert("RGB").save(buf, format="JPEG", quality=92, subsampling=0)
        mime = "image/jpeg"
    return buf.getvalue(), mime


# --------------------------------------------------------------------------- #
# Baking
# --------------------------------------------------------------------------- #
def recolor(src: Path, overrides: list[RecolorOverride], out: Path) -> Path:
    gltf = GLTF2().load(str(src))
    blob = gltf.binary_blob() or b""

    # inspect() re-parses the file AND decodes every base-colour image to compute
    # averages — on a 12-material mesh that is most of a second for information
    # we only need in order to resolve "auto". Callers that already know the
    # method (the portal always does, so preview and bake cannot diverge) skip
    # it entirely; texture presence is a cheap index lookup, no decode required.
    needs_suggestions = any(ov.method == "auto" for ov in overrides)
    parts = (
        {p.material_index: p for p in inspect(src)} if needs_suggestions else {}
    )
    luminance_jobs: list[_LuminanceJob] = []

    for ov in overrides:
        if ov.material_index >= len(gltf.materials):
            continue
        mat = gltf.materials[ov.material_index]
        if mat.pbrMetallicRoughness is None:
            continue

        method = ov.method
        if method == "auto":
            part = parts.get(ov.material_index)
            method = part.suggested_method if part else "factor"
        # No image to repaint -> must fall back to a factor multiply.
        if method in ("luminance", "remap"):
            if _base_color_image_index(gltf, ov.material_index) is None:
                method = "factor"

        alpha = 1.0
        if mat.pbrMetallicRoughness.baseColorFactor:
            alpha = mat.pbrMetallicRoughness.baseColorFactor[3]

        # Apply the brightness slider to the target colour once, up front, so
        # both the factor and pixel paths stay in sync.
        effective_color = colors.adjust_brightness_hex(ov.color, ov.brightness)

        if method == "factor":
            mat.pbrMetallicRoughness.baseColorFactor = colors.hex_to_linear_factor(effective_color, alpha)
            continue

        # luminance / remap -> repaint the source image into a new one
        img_index = _base_color_image_index(gltf, ov.material_index)
        pil, _ = _load_pil_from_image(gltf, blob, img_index)  # type: ignore[arg-type]
        encoded, mime = _recolor_pixels(pil, effective_color, remap=(method == "remap"))
        _, sampler = _base_color_texture_and_sampler(gltf, ov.material_index)
        luminance_jobs.append(
            _LuminanceJob(ov.material_index, encoded, mime, sampler, alpha)
        )

    _rebuild_and_write(gltf, blob, luminance_jobs, out)
    return out


def _align4(blob: bytearray) -> None:
    pad = (-len(blob)) % 4
    if pad:
        blob.extend(b"\x00" * pad)


def _rebuild_and_write(
    gltf: GLTF2, blob: bytes, jobs: list[_LuminanceJob], out: Path
) -> None:
    new_blob = bytearray()

    # 1. Relocate every existing bufferView verbatim (geometry, normals, RM,
    #    and untouched base-colour images all stay byte-identical in content).
    for bv in gltf.bufferViews:
        data = _bufferview_bytes_from(blob, bv)
        _align4(new_blob)
        bv.byteOffset = len(new_blob)
        # byteLength unchanged
        new_blob.extend(data)

    # 2. Append each recoloured image as a NEW bufferView + image + texture, and
    #    repoint the material at it. Sharing-safe: originals are left alone.
    for job in jobs:
        _align4(new_blob)
        bv = BufferView(buffer=0, byteOffset=len(new_blob), byteLength=len(job.encoded))
        gltf.bufferViews.append(bv)
        bv_index = len(gltf.bufferViews) - 1
        new_blob.extend(job.encoded)

        gltf.images.append(Image(mimeType=job.mime, bufferView=bv_index))
        img_index = len(gltf.images) - 1

        gltf.textures.append(Texture(source=img_index, sampler=job.sampler))
        tex_index = len(gltf.textures) - 1

        pbr = gltf.materials[job.material_index].pbrMetallicRoughness
        pbr.baseColorTexture = TextureInfo(index=tex_index)
        # neutral factor so the repainted texture shows its true colour
        pbr.baseColorFactor = [1.0, 1.0, 1.0, job.alpha]

    _align4(new_blob)
    if not gltf.buffers:
        raise ValueError("GLB has no buffer to write into.")
    gltf.buffers[0].byteLength = len(new_blob)
    gltf.buffers[0].uri = None  # keep it a binary GLB buffer
    gltf.set_binary_blob(bytes(new_blob))
    gltf.save(str(out))


def _bufferview_bytes_from(blob: bytes, bv: BufferView) -> bytes:
    start = bv.byteOffset or 0
    return blob[start : start + bv.byteLength]
