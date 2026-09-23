"""Product Configurator API contracts.

Two shapes of appearance live in ONE option model, discriminated by
``recipe.method``: a generated recolour (``factor`` / ``luminance`` / ``remap``)
and a seller-uploaded texture (``image``). There is deliberately no
``option_type`` / ``source_type`` field — ``method`` is already the value that
dispatches to different code paths, and a second discriminator could contradict
the first. See ADR-013.

Seller and shopper contracts are separate classes, not one class with fields
omitted. An omission-based approach leaks the next field somebody adds; the
shopper must never see ``recipe``, ``bake_*``, ``glb_version`` or audit columns.

Reference: docs/configurator/api-spec.md, docs/configurator/data-model.md.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Matches #RGB and #RRGGBB, with or without the leading hash.
_HEX_RE = re.compile(r"^#?(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")

# How a colour is applied. "auto" is accepted on input and resolved to a
# concrete method server-side at save time, so the preview the seller approved
# and the file the backend bakes come from the same rule — it is NEVER stored.
RecipeMethod = Literal["auto", "factor", "luminance", "remap", "image"]

# What may actually reach the database.
StoredRecipeMethod = Literal["factor", "luminance", "remap", "image"]

BakeStatus = Literal["pending", "baking", "completed", "failed"]

MaterialType = Literal["fabric", "wood", "metal", "leather", "plastic"]

RECIPE_VERSION = 1

MAX_MATERIAL_INDICES_PER_PART = 64
MAX_OPTIONS_PER_PART = 32

# Methods that recolour existing pixels and therefore need a target colour.
_RECOLOUR_METHODS = {"auto", "factor", "luminance", "remap"}


def normalize_hex(value: str) -> str:
    """'22c55e' / '#2c5' -> '#22C55E'. Raises ValueError on anything else.

    Deliberately duplicated from ``app.schemas.color_variants._normalize_hex``
    rather than imported: that name is private to another domain, and the
    planned fix (architecture.md section 9) is to extract the colour helpers to
    a shared module, which means editing colour-variant code that is explicitly
    out of scope right now. Eight lines of pure hex parsing is the cheaper debt.
    """
    raw = value.strip()
    if not _HEX_RE.match(raw):
        raise ValueError(f"Invalid hex colour: {value!r}. Expected #RRGGBB.")
    digits = raw.lstrip("#")
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return f"#{digits.upper()}"


# --------------------------------------------------------------------------- #
# Recipe — how an option's appearance is produced
# --------------------------------------------------------------------------- #
class RecipeOverride(BaseModel):
    """A per-material deviation inside a recolour recipe.

    For parts whose materials should not all take the same treatment. That the
    index actually belongs to the parent part is a service-layer check — this
    model cannot see the part.
    """

    material_index: int = Field(..., ge=0)
    method: RecipeMethod
    color: str

    @field_validator("color")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_hex(v)

    @model_validator(mode="after")
    def _reject_image_override(self) -> "RecipeOverride":
        if self.method == "image":
            raise ValueError(
                "method 'image' is not valid inside overrides — an uploaded "
                "texture replaces the whole part's base colour."
            )
        return self


class Recipe(BaseModel):
    """The appearance specification. Stored whole as JSONB.

    Field applicability by method (api-spec.md section 7):

        factor / luminance / remap   color required, brightness honoured,
                                     image_url rejected
        image                        image_url required, color and brightness
                                     ignored, overrides rejected
    """

    version: int = RECIPE_VERSION
    method: RecipeMethod
    color: Optional[str] = None
    brightness: float = Field(1.0, ge=0.1, le=2.0)
    image_url: Optional[str] = None
    overrides: list[RecipeOverride] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if v != RECIPE_VERSION:
            raise ValueError(f"Unsupported recipe version {v}. Expected {RECIPE_VERSION}.")
        return v

    @field_validator("color")
    @classmethod
    def _normalize(cls, v: Optional[str]) -> Optional[str]:
        return normalize_hex(v) if v is not None else None

    @model_validator(mode="after")
    def _method_specific_rules(self) -> "Recipe":
        if self.method == "image":
            if not self.image_url:
                raise ValueError("recipe.image_url is required when method is 'image'.")
            if self.overrides:
                raise ValueError(
                    "recipe.overrides is not valid when method is 'image' — an "
                    "uploaded texture replaces the base colour of every material "
                    "in the part."
                )
        else:
            if self.image_url is not None:
                raise ValueError(
                    f"recipe.image_url is only valid when method is 'image', not "
                    f"{self.method!r}."
                )
            if not self.color:
                raise ValueError(f"recipe.color is required when method is {self.method!r}.")
        return self

    @property
    def is_image(self) -> bool:
        return self.method == "image"


class StoredRecipe(Recipe):
    """A recipe as it may be PERSISTED — ``auto`` already resolved.

    ``auto`` is resolved server-side at save time. The existing colour-variant
    implementation swallows an inspection failure and stores ``auto`` anyway
    (variant_bake_service.py:152-155); this type is what makes that impossible
    here — the write fails instead.
    """

    method: StoredRecipeMethod

    @model_validator(mode="after")
    def _reject_unresolved_overrides(self) -> "StoredRecipe":
        for override in self.overrides:
            if override.method == "auto":
                raise ValueError(
                    f"Unresolved method 'auto' for material_index "
                    f"{override.material_index}; resolve before persisting."
                )
        return self


# --------------------------------------------------------------------------- #
# Parts — requests
# --------------------------------------------------------------------------- #
class ProductPartCreate(BaseModel):
    """`slug` and `glb_version` are set server-side and are not accepted here."""

    name: str = Field(..., min_length=1, max_length=100)
    material_indices: list[int] = Field(
        ..., min_length=1, max_length=MAX_MATERIAL_INDICES_PER_PART
    )
    material_type: Optional[MaterialType] = None
    order_index: Optional[int] = Field(None, ge=0)
    shopper_selectable: bool = True

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("name cannot be blank.")
        return stripped

    @field_validator("material_indices")
    @classmethod
    def _non_negative_and_unique(cls, v: list[int]) -> list[int]:
        if any(i < 0 for i in v):
            raise ValueError("material_indices must all be >= 0.")
        if len(set(v)) != len(v):
            raise ValueError("material_indices must not contain duplicates.")
        return v


class ProductPartUpdate(BaseModel):
    """All fields optional — only what is sent is changed."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    material_indices: Optional[list[int]] = Field(
        None, min_length=1, max_length=MAX_MATERIAL_INDICES_PER_PART
    )
    material_type: Optional[MaterialType] = None
    order_index: Optional[int] = Field(None, ge=0)
    shopper_selectable: Optional[bool] = None
    isactive: Optional[bool] = None

    _strip = field_validator("name")(ProductPartCreate._strip.__func__)  # type: ignore[attr-defined]
    _non_negative_and_unique = field_validator("material_indices")(
        ProductPartCreate._non_negative_and_unique.__func__  # type: ignore[attr-defined]
    )



# --------------------------------------------------------------------------- #
# Options — requests
# --------------------------------------------------------------------------- #
class PartOptionCreate(BaseModel):
    """`swatch_hex` is required for image options — see the validator.

    For a recolour it defaults server-side to ``recipe.color``; an image recipe
    has no colour to default from, and deriving an average from the uploaded
    file is work the seller can do better.
    """

    name: str = Field(..., min_length=1, max_length=100)
    swatch_hex: Optional[str] = None
    recipe: Recipe
    order_index: Optional[int] = Field(None, ge=0)

    # No `set_as_default`. `is_default` is only valid on an isactive+completed
    # option and a new one is always `pending`, so there is nothing a create-time
    # flag could legitimately set — and no column records a deferred intent.
    # Nor does a bake assign one: a part with no default starts on the model's
    # Original appearance until the seller PATCHes a choice (api-spec 7.4).

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("name cannot be blank.")
        return stripped

    @field_validator("swatch_hex")
    @classmethod
    def _normalize(cls, v: Optional[str]) -> Optional[str]:
        return normalize_hex(v) if v is not None else None

    @model_validator(mode="after")
    def _swatch_required_for_image(self) -> "PartOptionCreate":
        if self.recipe.is_image and not self.swatch_hex:
            raise ValueError(
                "swatch_hex is required when recipe.method is 'image' — there is "
                "no recipe.color to default from."
            )
        return self

    def resolved_swatch_hex(self) -> str:
        """The swatch to store. Safe to call only after validation."""
        if self.swatch_hex:
            return self.swatch_hex
        assert self.recipe.color is not None  # guaranteed by Recipe's validator
        return self.recipe.color


class PartOptionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    swatch_hex: Optional[str] = None
    recipe: Optional[Recipe] = None
    order_index: Optional[int] = Field(None, ge=0)
    isactive: Optional[bool] = None
    set_as_default: Optional[bool] = None

    _strip = field_validator("name")(PartOptionCreate._strip.__func__)  # type: ignore[attr-defined]
    _normalize = field_validator("swatch_hex")(
        PartOptionCreate._normalize.__func__  # type: ignore[attr-defined]
    )

    @model_validator(mode="after")
    def _swatch_required_when_switching_to_image(self) -> "PartOptionUpdate":
        # Only enforceable when the recipe is part of THIS request; switching an
        # existing option to an image recipe without supplying a swatch is
        # otherwise caught in the service, which can see the stored row.
        if self.recipe is not None and self.recipe.is_image and self.swatch_hex is None:
            raise ValueError(
                "swatch_hex must be supplied when changing recipe.method to 'image'."
            )
        return self



# --------------------------------------------------------------------------- #
# Seller responses
# --------------------------------------------------------------------------- #
class PartOptionTextureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    material_index: int
    url: str
    content_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None


class PartOptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    part_id: uuid.UUID
    name: str
    slug: str
    swatch_hex: str
    recipe: dict[str, Any]
    recipe_hash: str
    order_index: int
    is_default: bool
    isactive: bool
    bake_status: BakeStatus
    bake_error: Optional[str] = None
    bake_started_at: Optional[datetime] = None
    bake_completed_at: Optional[datetime] = None
    bake_attempts: int
    textures: list[PartOptionTextureResponse] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class ProductPartResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    product_id: uuid.UUID
    # None = the product's original model; otherwise its model variant (ADR-014).
    variant_id: Optional[uuid.UUID] = None
    name: str
    slug: str
    material_indices: list[int]
    material_type: Optional[str] = None
    order_index: int
    shopper_selectable: bool
    isactive: bool
    glb_version: str
    # Computed, never stored — see ADR-011 and ADR-006 respectively.
    default_option_id: Optional[uuid.UUID] = None
    glb_stale: bool = False
    options: list[PartOptionResponse] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# --------------------------------------------------------------------------- #
# Shopper responses — a SEPARATE contract, not the seller schema with holes
# --------------------------------------------------------------------------- #
class PublicOptionTexture(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    material_index: int
    url: str
    content_type: str


class PublicPartOption(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    swatch_hex: str
    order_index: int
    textures: list[PublicOptionTexture] = Field(default_factory=list)


class PublicProductPart(BaseModel):
    """What the viewer needs, and nothing else.

    Never carries recipe (including image_url, which points into the seller's
    own upload namespace), bake state, glb_version, blob_url, is_default,
    isactive, or audit columns. `default_option_id` IS exposed — the viewer must
    know which option to load first — but the raw flag it is derived from is not.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    material_indices: list[int]
    order_index: int
    default_option_id: Optional[uuid.UUID] = None
    options: list[PublicPartOption] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Materials — what the Part Editor is built from
# --------------------------------------------------------------------------- #
class MaterialResponse(BaseModel):
    """One colourable material on the product's current GLB.

    Two fields api-spec.md §5 lists are deliberately ABSENT until their open
    questions close, rather than shipped as guesses:

      * ``base_color_texture_url`` — needs source-texture extraction (Q3)
      * ``triangle_count``        — unverified under Draco compression (Q4)
    """

    material_index: int
    name: str
    mesh_names: list[str] = Field(default_factory=list)
    has_base_color_texture: bool
    average_color: str
    suggested_method: str
    # Renamed from the engine's `group_id` so no client mistakes it for a Part
    # identity: it is recomputed per call and unstable across a re-upload.
    similarity_group_hint: int = 0
    center: Optional[list[float]] = None
    # Which part already claims this index, if any.
    assigned_part_id: Optional[uuid.UUID] = None
    eligible_for_part: bool = True


class MaterialsResponse(BaseModel):
    glb_version: str
    model_url: str
    material_count: int
    materials: list[MaterialResponse] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Part update — names the options whose bakes it invalidated
# --------------------------------------------------------------------------- #
class ProductPartUpdateResponse(ProductPartResponse):
    invalidated_option_ids: list[uuid.UUID] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Bake status — the poll target
# --------------------------------------------------------------------------- #
class BakeProgress(BaseModel):
    """Best-effort. ``textures_total`` may over-count; drive completion off
    ``bake_status``, never off ``textures_done == textures_total``."""

    textures_total: int
    textures_done: int


class BakeStatusResponse(BaseModel):
    option_id: uuid.UUID
    bake_status: BakeStatus
    bake_error: Optional[str] = None
    bake_started_at: Optional[datetime] = None
    bake_completed_at: Optional[datetime] = None
    bake_attempts: int
    recipe_hash: str
    progress: Optional[BakeProgress] = None
    textures: list[PartOptionTextureResponse] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Shopper payload envelope
# --------------------------------------------------------------------------- #
class PublicModelVariant(BaseModel):
    """One shape of the product for the shopper (ADR-014).

    ``id`` is "original" for the product's own model. A separate shopper
    contract: never carries compression state, original-upload URLs, blob URLs
    or audit columns.
    """

    id: str
    name: str
    glb_url: str
    usdz_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    is_default: bool
    order_index: int
    # Bounding box in metres for true-to-scale display; None when unmeasured
    # (always, for the original model).
    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    height_m: Optional[float] = None
    parts: list[PublicProductPart] = Field(default_factory=list)


class PublicConfiguratorResponse(BaseModel):
    product_id: uuid.UUID
    product_name: str
    # Always the ORIGINAL model — unchanged for clients that predate variants.
    model_url: Optional[str] = None
    ar_model_url: Optional[str] = None
    parts: list[PublicProductPart] = Field(default_factory=list)
    # Present only when the product has extra model variants; the route drops
    # the key otherwise, so a single-model product's payload is unchanged.
    variants: Optional[list[PublicModelVariant]] = None


# --------------------------------------------------------------------------- #
# Model variants (ADR-014) — seller contracts
#
# An extra variant is one more shape of the product, with its own GLB. The
# product's original model is not a row; list endpoints present it with
# ``is_original = true``. Shopper payloads get their own schema (step e) and
# never carry the compression or original-upload fields below.
# --------------------------------------------------------------------------- #
CompressionStatus = Literal["compressed", "fallback_original"]

MODEL_VARIANT_NAME_MAX = 100


class ModelVariantResponse(BaseModel):
    # "original" for the product's original model, which has no row; otherwise
    # the variant's UUID. The same token addresses both in the variant-scoped
    # parts and materials routes.
    id: str
    product_id: uuid.UUID
    name: str
    glb_url: Optional[str]
    usdz_url: Optional[str] = None
    thumbnail_url: Optional[str]
    order_index: int
    is_original: bool = False
    isactive: bool = True
    # The fields below are None for the original model: it was not uploaded
    # through this pipeline.
    compression_status: Optional[CompressionStatus] = None
    compression_error: Optional[str] = None
    original_size_bytes: Optional[int] = None
    compressed_size_bytes: Optional[int] = None
    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    height_m: Optional[float] = None
    created_at: Optional[datetime] = None


class ModelVariantCreateResponse(ModelVariantResponse):
    # Advisory only — the upload succeeded. E.g. implausible dimensions, or
    # compression falling back to the original file.
    warnings: list[str] = Field(default_factory=list)


class ModelVariantUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=MODEL_VARIANT_NAME_MAX)


class ModelVariantReorder(BaseModel):
    # Every live extra variant of the product, in the new order. The original
    # model is always first and is not listed.
    variant_ids: list[uuid.UUID] = Field(default_factory=list, max_length=200)
