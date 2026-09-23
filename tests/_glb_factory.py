"""Build small, valid GLBs in memory for tests — no fixture files to drift.

Each mesh is a box of the given size, standing on the floor (y from 0 to
height) and centred on x/z, so the expected bounding box is easy to state.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pygltflib as g


def make_glb(
    *,
    materials: Sequence[Optional[str]] = ("Seat", "Legs"),
    meshes: Sequence[Optional[str]] = ("Sofa",),
    size: tuple[float, float, float] = (2.0, 0.8, 0.9),
    node_scale: Optional[list[float]] = None,
    draco_flag: bool = False,
    with_bounds: bool = True,
) -> bytes:
    width, height, depth = size
    points = np.array(
        [
            [x, y, z]
            for x in (-width / 2, width / 2)
            for y in (0.0, height)
            for z in (-depth / 2, depth / 2)
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 1, 3, 2, 4, 6, 5, 5, 6, 7], dtype=np.uint16)
    blob = points.tobytes() + indices.tobytes()

    gltf = g.GLTF2(
        scene=0,
        scenes=[g.Scene(nodes=list(range(len(meshes))))],
        nodes=[g.Node(mesh=i, scale=node_scale) for i in range(len(meshes))],
        meshes=[
            g.Mesh(
                name=name,
                primitives=[
                    g.Primitive(
                        attributes=g.Attributes(POSITION=0),
                        indices=1,
                        material=(i % len(materials)) if materials else None,
                    )
                ],
            )
            for i, name in enumerate(meshes)
        ],
        materials=[g.Material(name=name) for name in materials],
        accessors=[
            g.Accessor(
                bufferView=0,
                componentType=g.FLOAT,
                count=len(points),
                type=g.VEC3,
                min=points.min(axis=0).tolist() if with_bounds else None,
                max=points.max(axis=0).tolist() if with_bounds else None,
            ),
            g.Accessor(
                bufferView=1, componentType=g.UNSIGNED_SHORT, count=len(indices), type=g.SCALAR
            ),
        ],
        bufferViews=[
            g.BufferView(buffer=0, byteOffset=0, byteLength=len(points.tobytes())),
            g.BufferView(
                buffer=0, byteOffset=len(points.tobytes()), byteLength=len(indices.tobytes())
            ),
        ],
        buffers=[g.Buffer(byteLength=len(blob))],
        extensionsUsed=["KHR_draco_mesh_compression"] if draco_flag else [],
    )
    gltf.set_binary_blob(blob)
    return b"".join(gltf.save_to_bytes())
