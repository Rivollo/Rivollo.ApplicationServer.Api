"""Read what the Configurator needs from a GLB's JSON chunk: names and size.

Pure and synchronous — bytes in, a summary out, no I/O — so callers run it with
``asyncio.to_thread`` and tests drive it with GLBs built in memory.

Only the JSON chunk is consulted. Material names and order, mesh names and
order, and POSITION accessor bounds all live there, so this works identically
on a Draco-compressed GLB: KHR_draco_mesh_compression keeps the accessor
``min`` / ``max`` in the JSON even though the vertex data is encoded.

Why names matter: configurator parts attach to glTF material indices, and the
Draco step must not rename or reorder materials or meshes. Comparing two
summaries is how the upload path proves it did not (ADR-014).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pygltflib import GLTF2

_GLB_MAGIC = b"glTF"
_DRACO_EXTENSION = "KHR_draco_mesh_compression"

# Divisors for KHR_mesh_quantization normalized integer positions.
_NORMALIZED_MAX = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}


class InvalidGlbError(ValueError):
    """The bytes are not a usable binary glTF 2.0 model."""


@dataclass(frozen=True)
class GlbSummary:
    material_names: tuple[Optional[str], ...]
    mesh_names: tuple[Optional[str], ...]
    draco_compressed: bool
    # Axis-aligned bounding box of the default scene in glTF units (metres),
    # glTF being Y-up: width = X, height = Y, depth = Z. None when no mesh
    # primitive carried POSITION bounds.
    width_m: Optional[float]
    height_m: Optional[float]
    depth_m: Optional[float]

    def same_names_as(self, other: "GlbSummary") -> bool:
        """Material and mesh names identical, in the same order."""
        return (
            self.material_names == other.material_names
            and self.mesh_names == other.mesh_names
        )

    @property
    def largest_side_m(self) -> Optional[float]:
        sides = [s for s in (self.width_m, self.height_m, self.depth_m) if s is not None]
        return max(sides) if sides else None


def inspect_glb(data: bytes) -> GlbSummary:
    """Summarise a GLB. Raises InvalidGlbError if it is not one."""
    if len(data) < 12:
        raise InvalidGlbError("The file is too small to be a GLB.")
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != _GLB_MAGIC:
        raise InvalidGlbError("The file is not a binary glTF (.glb) model.")
    if version != 2:
        raise InvalidGlbError(f"Only glTF 2.0 is supported (this file is version {version}).")

    try:
        gltf = GLTF2.load_from_bytes(data)
    except Exception as exc:  # noqa: BLE001 - any parse failure is "not a GLB"
        raise InvalidGlbError("The GLB could not be parsed.") from exc
    # pygltflib returns None, rather than raising, for some malformed chunks.
    if gltf is None:
        raise InvalidGlbError("The GLB could not be parsed.")

    if not gltf.meshes:
        raise InvalidGlbError("The GLB contains no meshes.")

    width, height, depth = _scene_extent(gltf)
    return GlbSummary(
        material_names=tuple(m.name for m in (gltf.materials or [])),
        mesh_names=tuple(m.name for m in gltf.meshes),
        draco_compressed=_DRACO_EXTENSION in (gltf.extensionsUsed or []),
        width_m=width,
        height_m=height,
        depth_m=depth,
    )


# --------------------------------------------------------------------------- #
# Bounding box
# --------------------------------------------------------------------------- #
def _scene_extent(gltf: GLTF2) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """World-space extent of every mesh in the default scene."""
    lo = np.full(3, math.inf)
    hi = np.full(3, -math.inf)

    for node_index in _root_nodes(gltf):
        _visit(gltf, node_index, np.identity(4), lo, hi, depth=0)

    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return None, None, None
    size = hi - lo
    return float(size[0]), float(size[1]), float(size[2])


def _root_nodes(gltf: GLTF2) -> list[int]:
    if gltf.scenes:
        scene_index = gltf.scene if gltf.scene is not None else 0
        if 0 <= scene_index < len(gltf.scenes):
            return list(gltf.scenes[scene_index].nodes or [])
    # No scene: treat every node that is nobody's child as a root.
    children = {c for n in (gltf.nodes or []) for c in (n.children or [])}
    return [i for i in range(len(gltf.nodes or [])) if i not in children]


def _visit(
    gltf: GLTF2,
    node_index: int,
    parent: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    depth: int,
) -> None:
    # glTF forbids cycles; the depth cap stops a malformed file looping forever.
    if depth > 256 or not (0 <= node_index < len(gltf.nodes or [])):
        return
    node = gltf.nodes[node_index]
    world = parent @ _local_matrix(node)

    if node.mesh is not None and 0 <= node.mesh < len(gltf.meshes):
        for primitive in gltf.meshes[node.mesh].primitives or []:
            bounds = _position_bounds(gltf, primitive)
            if bounds is None:
                continue
            for corner in _corners(*bounds):
                point = world @ np.append(corner, 1.0)
                lo[:] = np.minimum(lo, point[:3])
                hi[:] = np.maximum(hi, point[:3])

    for child in node.children or []:
        _visit(gltf, child, world, lo, hi, depth=depth + 1)


def _local_matrix(node) -> np.ndarray:
    if node.matrix and len(node.matrix) == 16:
        # glTF matrices are column-major.
        return np.array(node.matrix, dtype=float).reshape(4, 4).T

    t = np.identity(4)
    if node.translation:
        t[:3, 3] = node.translation
    r = np.identity(4)
    if node.rotation:
        r[:3, :3] = _quaternion_matrix(node.rotation)
    s = np.identity(4)
    if node.scale:
        s[0, 0], s[1, 1], s[2, 2] = node.scale
    return t @ r @ s


def _quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _position_bounds(gltf: GLTF2, primitive) -> Optional[tuple[np.ndarray, np.ndarray]]:
    attributes = primitive.attributes
    index = getattr(attributes, "POSITION", None) if attributes is not None else None
    if index is None or not (0 <= index < len(gltf.accessors or [])):
        return None
    accessor = gltf.accessors[index]
    if not accessor.min or not accessor.max or len(accessor.min) < 3 or len(accessor.max) < 3:
        return None
    lo = np.array(accessor.min[:3], dtype=float)
    hi = np.array(accessor.max[:3], dtype=float)
    if accessor.normalized and accessor.componentType in _NORMALIZED_MAX:
        divisor = _NORMALIZED_MAX[accessor.componentType]
        lo, hi = np.maximum(lo / divisor, -1.0), np.maximum(hi / divisor, -1.0)
    return lo, hi


def _corners(lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    return [
        np.array([x, y, z])
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
