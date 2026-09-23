"""ORM models for the Product Configurator domain.

Four tables: a Model Variant (one GLB — one shape — of a product), the
seller-named Parts owning glTF material indices of that variant's GLB, the
Options a shopper can pick for a part, and the baked texture each Option
produces per material. See docs/configurator/data-model.md.

Lives in its own module rather than in ``models/models.py``, following
plan.py, subscription.py and login_otp.py, so the domain's schema is one file
and the shared module stays out of its change set. Registered with the
SQLAlchemy metadata through ``app/models/__init__.py``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TIMESTAMP

from app.models.base import Base
from app.models.models import AuditMixin, Product, UUIDMixin


# --------------------------------------------------------------------------- #
# Product Configurator
#
# Four tables: the extra Model Variants of a product (one GLB — one shape —
# each; the product's original model has no row), the seller-named Parts
# owning glTF material indices of one GLB, the Options a shopper can pick for
# a part, and the baked texture each Option produces per material. See
# docs/configurator/data-model.md and ADR-014.
#
# Two things NOT here, both on purpose:
#   * No `default_option_id` on the part. The default lives on the option as
#     `is_default`, guaranteed by a partial unique index — a circular FK bought
#     nothing (ADR-011).
#   * No ForeignKey from created_by / updated_by to tbl_users. Assertion 9 of
#     the ACCOUNT_PURGE_JOB_HANDOFF.md schema contract aborts every production
#     purge run on an unrecognised FK to tbl_users, and AuditMixin already
#     declares these as plain UUIDs for that reason (ADR-010). The one FK to
#     tbl_products IS kept — the purge deletes products before their owner and
#     needs the cascade — and requires that job to be updated before deploy.
#     tbl_product_model_variants.product_id is a second such FK (ADR-014).
# --------------------------------------------------------------------------- #
class ProductModelVariant(UUIDMixin, AuditMixin, Base):
    """An EXTRA shape of a product — "3 Seater", "Corner" — and its GLB.

    The product's original model (the GLB mapped to it) is the permanent
    default and has no row here; parts with ``variant_id IS NULL`` belong to it.
    Parts of an extra shape carry its id, because material indices refer to one
    GLB and differ between GLBs.

    ``glb_asset_id`` points at a ``tbl_product_assets`` row (asset 9) that has
    NO ``tbl_product_asset_mapping`` row: every reader of "the product's model"
    takes the newest mapped asset 9, so an unmapped row is invisible to all of
    them and nothing existing changes.

    Both asset FKs are ON DELETE SET NULL — the account purge deletes asset rows
    before products — which is the only reason they are nullable. See ADR-014.
    """

    __tablename__ = "tbl_product_model_variants"
    __table_args__ = (
        # Not partial: it also serves the product -> variant cascade.
        Index("ix_model_variants_product_order", "product_id", "order_index"),
        # Serve the ON DELETE SET NULL lookups when the purge deletes assets.
        Index("ix_model_variants_glb_asset", "glb_asset_id"),
        Index("ix_model_variants_usdz_asset", "usdz_asset_id"),
        CheckConstraint(
            "compression_status IN ('compressed', 'fallback_original')",
            name="ck_model_variants_compression_status",
        ),
        CheckConstraint(
            "(width_m IS NULL OR width_m >= 0)"
            " AND (depth_m IS NULL OR depth_m >= 0)"
            " AND (height_m IS NULL OR height_m >= 0)",
            name="ck_model_variants_dimensions",
        ),
    )

    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)

    # The GLB actually served (Draco-compressed unless compression fell back).
    # Its id is this variant's glb_version: "asset:<glb_asset_id>" (ADR-006).
    glb_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_product_assets.id", ondelete="SET NULL")
    )
    # Per-variant USDZ for iOS AR, written back by the converter job.
    usdz_asset_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_product_assets.id", ondelete="SET NULL")
    )

    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text)
    thumbnail_blob_url: Mapped[Optional[str]] = mapped_column(Text)

    # The seller's uncompressed upload, kept for re-processing. Never a
    # tbl_product_assets row, so no asset reader can ever serve it.
    original_glb_url: Mapped[Optional[str]] = mapped_column(Text)
    original_glb_blob_url: Mapped[Optional[str]] = mapped_column(Text)
    original_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    compressed_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    # 'compressed' | 'fallback_original' (compression failed or changed names).
    compression_status: Mapped[str] = mapped_column(Text, nullable=False)
    compression_error: Mapped[Optional[str]] = mapped_column(Text)

    # Axis-aligned bounding box in metres (glTF units, Y up): width = X,
    # height = Y, depth = Z. NULL when it could not be measured.
    width_m: Mapped[Optional[float]] = mapped_column(Float)
    depth_m: Mapped[Optional[float]] = mapped_column(Float)
    height_m: Mapped[Optional[float]] = mapped_column(Float)

    # The original model is implicitly first; extra variants start at 1.
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    @property
    def created_at(self) -> datetime:
        return self.created_date

    product: Mapped[Product] = relationship("Product")


class ProductPart(UUIDMixin, AuditMixin, Base):
    """A seller-defined configurable region of a product — "Seat", "Legs".

    ``material_indices`` maps the part onto the original GLB: these are glTF
    material array indices, which are also what the browser exposes as
    ``model.materials[i]``. The original mesh is never modified; a Part is a
    naming and grouping layer over it.

    Overlap (one material index claimed by two parts) is rejected in
    PartService under a row lock on the product, not by a database constraint —
    a trigger has no precedent in this migration chain and a normalised child
    table would add a second FK to tbl_products. See ADR-012.

    ``variant_id`` names the extra model variant whose GLB the material indices
    refer to; NULL means the product's original model (ADR-014). ``product_id``
    stays the ownership anchor and must equal the variant's product_id; the
    service guarantees that.
    """

    __tablename__ = "tbl_product_parts"
    __table_args__ = (
        Index("ix_parts_product_order", "product_id", "order_index"),
        Index("ix_parts_variant_order", "variant_id", "order_index"),
        # The shopper payload reads exactly this slice.
        Index(
            "ix_parts_product_active",
            "product_id",
            postgresql_where=text("isactive AND shopper_selectable"),
        ),
        # Unchanged by ADR-014: still per product. The service suffixes a slug
        # when two variants name a part the same.
        UniqueConstraint("product_id", "slug", name="uq_parts_product_slug"),
    )

    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    # NULL = the product's original model.
    variant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tbl_product_model_variants.id", ondelete="CASCADE"),
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)

    # [0, 3, 7] — glTF material indices. Validated against the product's actual
    # GLB in the service layer; no JSONB CHECK, matching every other JSONB
    # column in this project.
    material_indices: Mapped[list[int]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # Informational: drives editor defaults, never dispatch.
    material_type: Mapped[Optional[str]] = mapped_column(Text)

    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    # False = seller-only scaffolding, omitted from the public payload entirely.
    shopper_selectable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )

    # Identity of the GLB this part was authored against. Prefix-discriminated
    # ("asset:<uuid>" now, "sha256:<hex>" later) so the strategy can change
    # without a schema change — ADR-006 is still Needs Verification, so nothing
    # may depend on the prefix's meaning yet.
    glb_version: Mapped[str] = mapped_column(Text, nullable=False)

    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    @property
    def created_at(self) -> datetime:
        return self.created_date

    product: Mapped[Product] = relationship("Product")
    variant: Mapped[Optional[ProductModelVariant]] = relationship("ProductModelVariant")
    options: Mapped[list["PartOption"]] = relationship(
        "PartOption",
        back_populates="part",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class PartOption(UUIDMixin, AuditMixin, Base):
    """One shopper-selectable appearance for a Part.

    ``recipe`` covers BOTH kinds of option — a generated recolour and a
    seller-uploaded texture — discriminated by ``recipe["method"]``, which is
    already the field that dispatches to different code paths in the colour
    engine. There is deliberately no ``option_type`` / ``source_type`` column:
    it would be a second discriminator that could contradict the first
    (ADR-013).

        {"version": 1, "method": "luminance", "color": "#C0182B", "brightness": 1.0}
        {"version": 1, "method": "image", "image_url": "https://cdn/..."}
    """

    __tablename__ = "tbl_part_options"
    __table_args__ = (
        Index("ix_options_part_order", "part_id", "order_index"),
        # At most ONE default per part, enforced by the database rather than by
        # code — same shape as ux_color_variants_one_default.
        Index(
            "ux_part_options_one_default",
            "part_id",
            unique=True,
            postgresql_where=text("is_default"),
        ),
        # Sized to the in-flight bakes only, which makes the stale-bake sweep
        # (bake_status='baking' AND bake_started_at < now() - interval) free.
        Index(
            "ix_options_stale",
            "bake_started_at",
            postgresql_where=text("bake_status = 'baking'"),
        ),
        UniqueConstraint("part_id", "slug", name="uq_options_part_slug"),
        CheckConstraint("swatch_hex ~* '^#[0-9A-F]{6}$'", name="ck_options_swatch_hex"),
        CheckConstraint(
            "bake_status IN ('pending', 'baking', 'completed', 'failed')",
            name="ck_options_bake_status",
        ),
    )

    part_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tbl_product_parts.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    # The picker dot. Lets both portals render the swatch grid without
    # downloading a single byte of any model or texture.
    swatch_hex: Mapped[str] = mapped_column(Text, nullable=False)

    recipe: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # sha256(canonical recipe + glb_version + BAKER_VERSION). Changes exactly
    # when the baked bytes would, which is what makes re-baking idempotent and
    # what marks an existing texture stale.
    recipe_hash: Mapped[str] = mapped_column(Text, nullable=False)

    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    # The look loaded first for this part. `default_option_id` is computed from
    # this at the API layer; there is no such column.
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    bake_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'")
    )
    bake_error: Mapped[Optional[str]] = mapped_column(Text)
    # Set in the same transaction that sets bake_status='baking'. Without it a
    # replica recycled mid-bake leaves the row 'baking' forever, which is the
    # state the colour-variant feature is in today.
    bake_started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    bake_completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    bake_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    @property
    def created_at(self) -> datetime:
        return self.created_date

    part: Mapped[ProductPart] = relationship("ProductPart", back_populates="options")
    textures: Mapped[list["PartOptionTexture"]] = relationship(
        "PartOptionTexture",
        back_populates="option",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class PartOptionTexture(UUIDMixin, AuditMixin, Base):
    """A baked texture for one material index of one Option.

    Pure cache: every row can be deleted and regenerated from the parent
    option's ``recipe``. ``recipe_hash`` must equal the parent's — when it does
    not, the file was baked from a superseded recipe and must not be served.

    An option may legitimately have FEWER textures than its part has material
    indices, or none at all: a material with no base-colour image can only be
    treated by the `factor` method, which sets baseColorFactor and produces no
    file.
    """

    __tablename__ = "tbl_part_option_textures"
    __table_args__ = (
        Index("ix_option_textures_option", "option_id"),
        UniqueConstraint("option_id", "material_index", name="uq_option_texture_material"),
        CheckConstraint("material_index >= 0", name="ck_option_texture_index"),
        CheckConstraint(
            "content_type IN ('image/png', 'image/jpeg')", name="ck_option_texture_mime"
        ),
    )

    option_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tbl_part_options.id", ondelete="CASCADE"),
        nullable=False,
    )
    material_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # CDN-fronted public URL served to clients.
    url: Mapped[str] = mapped_column(Text, nullable=False)
    # Raw Azure blob URL — used internally for re-processing and cleanup.
    blob_url: Mapped[Optional[str]] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)

    recipe_hash: Mapped[str] = mapped_column(Text, nullable=False)

    @property
    def created_at(self) -> datetime:
        return self.created_date

    option: Mapped[PartOption] = relationship("PartOption", back_populates="textures")
