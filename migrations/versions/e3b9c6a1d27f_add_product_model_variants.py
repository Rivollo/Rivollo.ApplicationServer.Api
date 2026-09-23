"""add tbl_product_model_variants and an optional variant_id on configurator parts

Revision ID: e3b9c6a1d27f
Revises: c7a4e0d51b83
Create Date: 2026-09-21 00:00:00.000000

A product can have extra model variants: different SHAPES of the same product
("3 Seater", "Corner"), each its own GLB. The product's ORIGINAL model — the
GLB mapped to it in tbl_product_asset_mapping — stays the product's model and
its permanent default; it has no row here. See docs/configurator/decisions.md
ADR-014.

Purely additive, by design:

  1. Creates tbl_product_model_variants, holding the EXTRA variants only.
  2. Adds a NULLABLE tbl_product_parts.variant_id. NULL means "the product's
     original model", so every existing part keeps its meaning with no rewrite.

No backfill, no data written, no constraint dropped, and no change to
tbl_products, tbl_product_assets or tbl_product_asset_mapping. Part slugs stay
unique per product (uq_parts_product_slug is untouched); the service suffixes a
slug when two variants use the same part name.

Written by hand, like c7a4e0d51b83: migrations/env.py refuses to autogenerate
foreign keys, and the schema has drifted from the chain.

FOREIGN KEYS — dictated by the account purge job
------------------------------------------------
* product_id -> tbl_products ON DELETE CASCADE. A NEW FK to tbl_products, so
  ACCOUNT_PURGE_JOB_HANDOFF.md section 25 assertion 9 aborts the purge run
  until Rivollo.AccountPurge.Job allow-lists it (its decision D10). Deploy that
  job change in the same window as this migration (ADR-010, ADR-014).
* glb_asset_id / usdz_asset_id -> tbl_product_assets ON DELETE SET NULL, and
  therefore nullable. The purge deletes tbl_product_assets (its step 4) BEFORE
  tbl_products (its step 6); RESTRICT or NO ACTION here would fail step 4 for
  every seller with a variant, and CASCADE would let a stray asset delete wipe
  a variant together with its parts, options and baked textures.
* tbl_product_parts.variant_id -> tbl_product_model_variants ON DELETE CASCADE.
  Only a hard delete (purge) reaches it: the app soft-deletes variants.
* NO FK from created_by / updated_by to tbl_users — plain UUIDs, as AuditMixin
  and c7a4e0d51b83 already do.

A variant's GLB is a tbl_product_assets row with NO tbl_product_asset_mapping
row. Every reader of "the product's model" (this API, Rivollo.Viewer.Api, the
editor, the USDZ job) takes the newest active mapped asset 9, so an unmapped
row is invisible to all of them — which is what keeps this feature additive.

DEV NOTE: rivollo-dev-db has the three configurator tables (created from
docs/configurator/create_configurator_tables.sql) while alembic_version still
reads f61a03d7b8e4. Stamp it to c7a4e0d51b83 before upgrading, or c7a4e0d51b83
will fail with "relation already exists".
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "e3b9c6a1d27f"
down_revision: Union[str, Sequence[str], None] = "c7a4e0d51b83"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _audit_columns() -> list[sa.Column]:
    """The four AuditMixin columns, with NO FK to tbl_users. See the docstring."""
    return [
        sa.Column("created_by", postgresql.UUID(as_uuid=True)),
        sa.Column(
            "created_date",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True)),
        sa.Column("updated_date", sa.TIMESTAMP(timezone=True)),
    ]


def upgrade() -> None:
    """Create model variants and give parts an optional variant."""

    op.create_table(
        "tbl_product_model_variants",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        # The GLB actually served (Draco-compressed unless compression fell
        # back). Its id is the variant's glb_version ("asset:<uuid>", ADR-006).
        # Nullable only because of the SET NULL rule — see the docstring.
        sa.Column("glb_asset_id", postgresql.UUID(as_uuid=True)),
        # Per-variant USDZ for iOS AR, written back by the converter job.
        sa.Column("usdz_asset_id", postgresql.UUID(as_uuid=True)),
        # Captured in the editor with model-viewer toBlob(); derived and
        # re-capturable, so plain columns rather than an asset row.
        sa.Column("thumbnail_url", sa.Text()),
        sa.Column("thumbnail_blob_url", sa.Text()),
        # The seller's uncompressed upload, kept for re-processing. Columns
        # only — never a tbl_product_assets row, so no asset reader can serve it.
        sa.Column("original_glb_url", sa.Text()),
        sa.Column("original_glb_blob_url", sa.Text()),
        sa.Column("original_size_bytes", sa.BigInteger()),
        sa.Column("compressed_size_bytes", sa.BigInteger()),
        sa.Column("compression_status", sa.Text(), nullable=False),
        sa.Column("compression_error", sa.Text()),
        # Axis-aligned bounding box of the GLB, in metres (glTF units).
        sa.Column("width_m", sa.Float()),
        sa.Column("depth_m", sa.Float()),
        sa.Column("height_m", sa.Float()),
        # The original model is implicitly first; extra variants start at 1.
        sa.Column("order_index", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("isactive", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        *_audit_columns(),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["tbl_products.id"],
            name="fk_model_variants_product",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["glb_asset_id"],
            ["tbl_product_assets.id"],
            name="fk_model_variants_glb_asset",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["usdz_asset_id"],
            ["tbl_product_assets.id"],
            name="fk_model_variants_usdz_asset",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "compression_status IN ('compressed', 'fallback_original')",
            name="ck_model_variants_compression_status",
        ),
        sa.CheckConstraint(
            "(width_m IS NULL OR width_m >= 0)"
            " AND (depth_m IS NULL OR depth_m >= 0)"
            " AND (height_m IS NULL OR height_m >= 0)",
            name="ck_model_variants_dimensions",
        ),
    )
    # Full indexes, not partial: the purge's product -> variant CASCADE and the
    # asset -> variant SET NULL lookups ignore isactive. Unindexed, each asset
    # the purge deletes would scan this whole table.
    op.create_index(
        "ix_model_variants_product_order",
        "tbl_product_model_variants",
        ["product_id", "order_index"],
    )
    op.create_index(
        "ix_model_variants_glb_asset", "tbl_product_model_variants", ["glb_asset_id"]
    )
    op.create_index(
        "ix_model_variants_usdz_asset", "tbl_product_model_variants", ["usdz_asset_id"]
    )

    # NULL = the product's original model. Existing rows are left as they are.
    op.add_column(
        "tbl_product_parts",
        sa.Column("variant_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_parts_model_variant",
        "tbl_product_parts",
        "tbl_product_model_variants",
        ["variant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_parts_variant_order", "tbl_product_parts", ["variant_id", "order_index"]
    )


def downgrade() -> None:
    """Drop model variants.

    Parts that belonged to an extra variant lose their variant_id and read as
    parts of the original model; their glb_version names the variant's GLB, not
    the product's, so the pre-variant code already treats them as stale and
    hides them. Variant blobs and their unmapped tbl_product_assets rows are
    NOT deleted — they are orphaned.
    """
    op.drop_index("ix_parts_variant_order", table_name="tbl_product_parts")
    op.drop_constraint("fk_parts_model_variant", "tbl_product_parts", type_="foreignkey")
    op.drop_column("tbl_product_parts", "variant_id")

    op.drop_index("ix_model_variants_usdz_asset", table_name="tbl_product_model_variants")
    op.drop_index("ix_model_variants_glb_asset", table_name="tbl_product_model_variants")
    op.drop_index("ix_model_variants_product_order", table_name="tbl_product_model_variants")
    op.drop_table("tbl_product_model_variants")
