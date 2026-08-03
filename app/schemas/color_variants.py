"""Color configurator schemas.

A colourway is stored as a small JSON recipe — a list of per-material colour
overrides — and baked into a real GLB. These models define the contract for both
halves: what the client sends to describe a colour, and what the API returns
once the file exists.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Matches #RGB and #RRGGBB, with or without the leading hash.
_HEX_RE = re.compile(r"^#?(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")

RecolorMethod = Literal["auto", "factor", "luminance", "remap"]
BakeStatus = Literal["pending", "baking", "ready", "failed"]


def _normalize_hex(value: str) -> str:
    """'22c55e' / '#2c5' -> '#22C55E'. Raises ValueError on anything else."""
    raw = value.strip()
    if not _HEX_RE.match(raw):
        raise ValueError(f"Invalid hex colour: {value!r}. Expected #RRGGBB.")
    digits = raw.lstrip("#")
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return f"#{digits.upper()}"


# --------------------------------------------------------------------------- #
# Material inspection — what the model is made of
# --------------------------------------------------------------------------- #
class MaterialPart(BaseModel):
    """One colourable part discovered on the product's GLB.

    ``material_index`` is the join key used everywhere: it is the glTF material
    array index, which is also the index model-viewer exposes as
    ``model.materials[i]`` in the browser. Names are for humans only — glTF
    material names are optional and frequently duplicated by exporters.
    """

    material_index: int
    name: str
    mesh_names: list[str] = Field(default_factory=list)
    has_base_color_texture: bool
    average_color: str = Field(description="sRGB hex of the part's current colour")
    suggested_method: RecolorMethod
    group_id: int = Field(0, description="Materials forming one visual part share this")
    center: Optional[list[float]] = Field(None, description="World-space centre, for a selection pin")


# --------------------------------------------------------------------------- #
# The recipe
# --------------------------------------------------------------------------- #
class MaterialOverride(BaseModel):
    """A single part's colour choice inside a colourway."""

    material_index: int = Field(..., ge=0, description="glTF material index — the join key")
    material_name: Optional[str] = Field(None, description="Human label; not used for matching")
    color: str = Field(..., description="Target colour as sRGB hex")
    method: RecolorMethod = Field(
        "auto",
        description=(
            "How the colour is applied. 'auto' picks per part from its albedo; "
            "resolved to a concrete method at save time so the browser preview "
            "and the baked file can never disagree."
        ),
    )
    brightness: float = Field(1.0, ge=0.1, le=2.0, description="HSL lightness multiplier")

    @field_validator("color")
    @classmethod
    def _validate_color(cls, v: str) -> str:
        return _normalize_hex(v)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class ColorVariantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    swatch_hex: Optional[str] = Field(
        None,
        description="Swatch dot colour. Defaults to the first override's colour.",
    )
    overrides: list[MaterialOverride] = Field(default_factory=list, max_length=64)
    is_default: bool = False
    isactive: bool = True

    @field_validator("swatch_hex")
    @classmethod
    def _validate_swatch(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_hex(v) if v else v


class ColorVariantUpdate(BaseModel):
    """All fields optional — only what is sent gets changed.

    Changing ``overrides`` changes the config hash, which marks the variant
    stale and triggers a re-bake.
    """

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    swatch_hex: Optional[str] = None
    overrides: Optional[list[MaterialOverride]] = Field(None, max_length=64)
    is_default: Optional[bool] = None
    isactive: Optional[bool] = None

    @field_validator("swatch_hex")
    @classmethod
    def _validate_swatch(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_hex(v) if v else v


class ColorVariantReorder(BaseModel):
    variant_ids: list[str] = Field(..., min_length=1, description="Variant ids in display order")


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class VariantAssetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    format: str
    url: str
    size_bytes: Optional[int] = None


class ColorVariantResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    product_id: str
    name: str
    slug: str
    swatch_hex: str
    is_default: bool
    is_original: bool
    isactive: bool
    order_index: int
    overrides: list[MaterialOverride]
    bake_status: BakeStatus | str
    bake_error: Optional[str] = None
    baked_at: Optional[datetime] = None
    # The baked colourway GLB. Null while the bake is pending or failed — for
    # the original variant this is the untouched product model.
    model_url: Optional[str] = None
    assets: list[VariantAssetResponse] = Field(default_factory=list)
    created_at: Optional[datetime] = None
