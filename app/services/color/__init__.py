"""Colour engine — pure, DB-free, storage-free.

Everything in this package takes bytes/paths in and returns bytes/paths out.
It never touches the database, Azure, or HTTP. That is deliberate: it keeps the
recolour maths unit-testable against a fixture GLB, and lets the persistence
layer (``color_variant_service``) and the storage layer (``variant_bake_service``)
evolve independently.

  colors       sRGB <-> linear conversion + HSL brightness adjustment
  glb_recolor  inspect() a GLB's parts, and recolor() it into a new GLB

BAKER_VERSION is folded into every variant's config_hash. Bump it whenever a
change here would produce different output for the same input — that invalidates
previously baked files and causes them to be regenerated on next use.
"""

from app.services.color import colors, glb_recolor
from app.services.color.glb_recolor import PartInfo, RecolorOverride, inspect, recolor

# v2: textures behind EXT_texture_webp / KHR_texture_basisu are now found and
#     repainted. Previously such parts looked untextured and fell back to a flat
#     colour, so the same recipe can now produce different (correct) bytes —
#     which must invalidate anything baked under v1.
BAKER_VERSION = "2"

__all__ = [
    "BAKER_VERSION",
    "colors",
    "glb_recolor",
    "PartInfo",
    "RecolorOverride",
    "inspect",
    "recolor",
]
