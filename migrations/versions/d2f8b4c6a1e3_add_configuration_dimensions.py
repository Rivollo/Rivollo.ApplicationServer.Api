"""add configuration dimensions (Capacity x Layout, ADR-017)

Revision ID: d2f8b4c6a1e3
Revises: c9e5a3b1d8f6
Create Date: 2026-10-08 00:00:00.000000

Named dimensions over a product's models (Capacity, Layout), their values, and
one value per dimension assigned to each model, so one published product can
offer Capacity x Layout and resolve the exact model. See
docs/configurator/decisions.md ADR-017.

Additive: three new tables, and one nullable JSONB column on
tbl_shopify_products (the dimension -> Shopify option mapping). No core table
is altered and no data is written: a product without dimensions is unchanged.
The Configurator becomes EIGHT tables (CLAUDE.md updated in the same change).

Written by hand: migrations/env.py refuses to autogenerate foreign keys.

FOREIGN KEYS — dictated by the account purge job
------------------------------------------------
* tbl_configuration_dimensions.product_id and
  tbl_model_configuration_values.product_id -> tbl_products ON DELETE CASCADE.
  Two NEW FKs to tbl_products: Rivollo.AccountPurge.Job must allow-list them
  before this ships (Q8; docs/account-purge-job-changes.md section 8).
* model_variant_id -> tbl_product_model_variants CASCADE; dimension_id and
  value_id -> the new tables CASCADE. None of these reach tbl_users or
  tbl_products, so the purge contract does not see them.
* NO FK from created_by / updated_by to tbl_users (ADR-010).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d2f8b4c6a1e3"
down_revision: Union[str, Sequence[str], None] = "c9e5a3b1d8f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CODE_CHECK = "code ~ '^[a-z][a-z0-9_]*$'"
# A value code may start with a digit ("3_seater"); a dimension code may not.
VALUE_CODE_CHECK = "code ~ '^[a-z0-9][a-z0-9_]*$'"


def _id() -> sa.Column:
    return sa.Column(
        "id",
        postgresql.UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=sa.text("gen_random_uuid()"),
    )


def _audit() -> list[sa.Column]:
    return [
        sa.Column("created_by", postgresql.UUID(as_uuid=True)),
        sa.Column("created_date", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True)),
        sa.Column("updated_date", sa.TIMESTAMP(timezone=True)),
    ]


def upgrade() -> None:
    op.create_table(
        "tbl_configuration_dimensions",
        _id(),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("display_type", sa.Text(), server_default=sa.text("'button'"), nullable=False),
        sa.Column("order_index", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_required", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        *_audit(),
        sa.ForeignKeyConstraint(
            ["product_id"], ["tbl_products.id"], name="fk_configuration_dimensions_product", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("product_id", "code", name="uq_configuration_dimensions_product_code"),
        sa.CheckConstraint(CODE_CHECK, name="ck_configuration_dimensions_code"),
        sa.CheckConstraint(
            "display_type IN ('button', 'image', 'swatch')", name="ck_configuration_dimensions_display_type"
        ),
        sa.CheckConstraint("order_index >= 0", name="ck_configuration_dimensions_order"),
    )
    op.create_index(
        "ix_configuration_dimensions_product_order",
        "tbl_configuration_dimensions",
        ["product_id", "order_index"],
    )

    op.create_table(
        "tbl_configuration_values",
        _id(),
        sa.Column("dimension_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("order_index", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("thumbnail_url", sa.Text()),
        sa.Column("isactive", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        *_audit(),
        sa.ForeignKeyConstraint(
            ["dimension_id"],
            ["tbl_configuration_dimensions.id"],
            name="fk_configuration_values_dimension",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("dimension_id", "code", name="uq_configuration_values_dimension_code"),
        sa.CheckConstraint(VALUE_CODE_CHECK, name="ck_configuration_values_code"),
        sa.CheckConstraint("order_index >= 0", name="ck_configuration_values_order"),
    )
    op.create_index(
        "ux_configuration_values_dimension_label",
        "tbl_configuration_values",
        ["dimension_id", sa.text("lower(label)")],
        unique=True,
    )
    op.create_index(
        "ux_configuration_values_one_default",
        "tbl_configuration_values",
        ["dimension_id"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )

    op.create_table(
        "tbl_model_configuration_values",
        _id(),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=False),
        # NULL = the product's original model, as tbl_product_parts.variant_id.
        sa.Column("model_variant_id", postgresql.UUID(as_uuid=True)),
        sa.Column("dimension_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("value_id", postgresql.UUID(as_uuid=True), nullable=False),
        *_audit(),
        sa.ForeignKeyConstraint(
            ["product_id"], ["tbl_products.id"], name="fk_model_configuration_values_product", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["model_variant_id"],
            ["tbl_product_model_variants.id"],
            name="fk_model_configuration_values_variant",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dimension_id"],
            ["tbl_configuration_dimensions.id"],
            name="fk_model_configuration_values_dimension",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["value_id"],
            ["tbl_configuration_values.id"],
            name="fk_model_configuration_values_value",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ux_model_configuration_values_variant_dimension",
        "tbl_model_configuration_values",
        ["model_variant_id", "dimension_id"],
        unique=True,
        postgresql_where=sa.text("model_variant_id IS NOT NULL"),
    )
    op.create_index(
        "ux_model_configuration_values_original_dimension",
        "tbl_model_configuration_values",
        ["product_id", "dimension_id"],
        unique=True,
        postgresql_where=sa.text("model_variant_id IS NULL"),
    )
    # Also serve the CASCADE lookups from products, dimensions and values.
    op.create_index("ix_model_configuration_values_product", "tbl_model_configuration_values", ["product_id"])
    op.create_index("ix_model_configuration_values_dimension", "tbl_model_configuration_values", ["dimension_id"])
    op.create_index("ix_model_configuration_values_value", "tbl_model_configuration_values", ["value_id"])

    op.add_column(
        "tbl_shopify_products",
        sa.Column("dimension_mapping", postgresql.JSONB(astext_type=sa.Text())),
    )


def downgrade() -> None:
    """Drop the configuration tables and the mapping column.

    Products return to their unconfigured payloads; models, parts and blobs are
    untouched.
    """
    op.drop_column("tbl_shopify_products", "dimension_mapping")
    for index in (
        "ix_model_configuration_values_value",
        "ix_model_configuration_values_dimension",
        "ix_model_configuration_values_product",
        "ux_model_configuration_values_original_dimension",
        "ux_model_configuration_values_variant_dimension",
    ):
        op.drop_index(index, table_name="tbl_model_configuration_values")
    op.drop_table("tbl_model_configuration_values")
    op.drop_index("ux_configuration_values_one_default", table_name="tbl_configuration_values")
    op.drop_index("ux_configuration_values_dimension_label", table_name="tbl_configuration_values")
    op.drop_table("tbl_configuration_values")
    op.drop_index("ix_configuration_dimensions_product_order", table_name="tbl_configuration_dimensions")
    op.drop_table("tbl_configuration_dimensions")
