"""add the Shopify integration tables (ADR-016)

Revision ID: c9e5a3b1d8f6
Revises: b7d3f1a2c9e4
Create Date: 2026-09-29 00:00:02.000000

Four tables for the isolated Shopify module (docs/shopify-integration/spec.md):
tbl_shopify_connections, tbl_shopify_products, tbl_shopify_product_variants,
tbl_shopify_layouts.

Purely additive: no existing table is altered and no data is written.

FOREIGN KEYS (all ON DELETE CASCADE)
------------------------------------
  tbl_shopify_connections.user_id          -> tbl_users
  tbl_shopify_connections.api_key_id       -> tbl_api_keys
  tbl_shopify_products.user_id             -> tbl_users
  tbl_shopify_products.rivollo_product_id  -> tbl_products
  tbl_shopify_product_variants.shopify_product_ref -> tbl_shopify_products
  tbl_shopify_layouts.shopify_product_ref          -> tbl_shopify_products
The purge deletes products (step 6), then the user (step 9); these cascades
remove every Shopify row. 🔴 DEPLOYMENT DEPENDENCY: Rivollo.AccountPurge.Job
must allow-list the three tbl_users / tbl_products FKs first, or its contract
check (A14) aborts every run — docs/account-purge-job-changes.md.
created_by / updated_by stay plain UUIDs (AuditMixin). No blob columns: images
copied from Shopify live under users/{user_id}/uploads/, already swept.

Written by hand: migrations/env.py refuses to autogenerate foreign keys.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c9e5a3b1d8f6"
down_revision: Union[str, Sequence[str], None] = "b7d3f1a2c9e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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
        sa.Column(
            "created_date", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True)),
        sa.Column("updated_date", sa.TIMESTAMP(timezone=True)),
    ]


def _jsonb(name: str, default: str) -> sa.Column:
    return sa.Column(
        name, postgresql.JSONB(), server_default=sa.text(f"'{default}'::jsonb"), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "tbl_shopify_connections",
        _id(),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("shop_domain", sa.Text(), nullable=False),
        sa.Column("isactive", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("connected_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("disconnected_at", sa.TIMESTAMP(timezone=True)),
        *_audit(),
        sa.ForeignKeyConstraint(
            ["user_id"], ["tbl_users.id"], name="fk_shopify_connections_user", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["api_key_id"], ["tbl_api_keys.id"], name="fk_shopify_connections_api_key", ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ux_shopify_connections_active_shop",
        "tbl_shopify_connections",
        ["shop_domain"],
        unique=True,
        postgresql_where=sa.text("isactive"),
    )
    op.create_index(
        "ux_shopify_connections_active_key",
        "tbl_shopify_connections",
        ["api_key_id"],
        unique=True,
        postgresql_where=sa.text("isactive"),
    )
    op.create_index("ix_shopify_connections_user", "tbl_shopify_connections", ["user_id"])
    # Full index for the api_key -> connection cascade (the unique one is partial).
    op.create_index("ix_shopify_connections_api_key", "tbl_shopify_connections", ["api_key_id"])

    op.create_table(
        "tbl_shopify_products",
        _id(),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("shop_domain", sa.Text(), nullable=False),
        sa.Column("shopify_product_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("handle", sa.Text(), nullable=False),
        sa.Column("description_html", sa.Text()),
        sa.Column("vendor", sa.Text()),
        sa.Column("product_type", sa.Text()),
        _jsonb("tags", "[]"),
        sa.Column("shopify_status", sa.Text(), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        _jsonb("images", "[]"),
        _jsonb("options", "[]"),
        _jsonb("option_roles", "{}"),
        sa.Column("rivollo_product_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("main_glb_requested_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("synced_at", sa.TIMESTAMP(timezone=True), nullable=False),
        *_audit(),
        sa.ForeignKeyConstraint(
            ["user_id"], ["tbl_users.id"], name="fk_shopify_products_user", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["rivollo_product_id"], ["tbl_products.id"], name="fk_shopify_products_product", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("shop_domain", "shopify_product_id", name="uq_shopify_products_shop_product"),
    )
    op.create_index("ix_shopify_products_user", "tbl_shopify_products", ["user_id"])
    op.create_index("ix_shopify_products_rivollo_product", "tbl_shopify_products", ["rivollo_product_id"])

    op.create_table(
        "tbl_shopify_product_variants",
        _id(),
        sa.Column("shopify_product_ref", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("shopify_variant_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("sku", sa.Text()),
        sa.Column("price", sa.Numeric(12, 2), nullable=False),
        sa.Column("compare_at_price", sa.Numeric(12, 2)),
        sa.Column("inventory_quantity", sa.Integer()),
        sa.Column("available", sa.Boolean(), nullable=False),
        _jsonb("image_urls", "[]"),
        _jsonb("options", "[]"),
        sa.Column("position", sa.Integer(), nullable=False),
        *_audit(),
        sa.ForeignKeyConstraint(
            ["shopify_product_ref"],
            ["tbl_shopify_products.id"],
            name="fk_shopify_variants_product",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "shopify_product_ref", "shopify_variant_id", name="uq_shopify_variants_product_variant"
        ),
    )

    op.create_table(
        "tbl_shopify_layouts",
        _id(),
        sa.Column("shopify_product_ref", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("option_value", sa.Text(), nullable=False),
        sa.Column("is_original", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        *_audit(),
        sa.ForeignKeyConstraint(
            ["shopify_product_ref"],
            ["tbl_shopify_products.id"],
            name="fk_shopify_layouts_product",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("shopify_product_ref", "option_value", name="uq_shopify_layouts_value"),
    )
    op.create_index(
        "ux_shopify_layouts_one_original",
        "tbl_shopify_layouts",
        ["shopify_product_ref"],
        unique=True,
        postgresql_where=sa.text("is_original"),
    )


def downgrade() -> None:
    """Drop the Shopify tables. Rivollo products created by sync are kept."""
    op.drop_index("ux_shopify_layouts_one_original", table_name="tbl_shopify_layouts")
    op.drop_table("tbl_shopify_layouts")
    op.drop_table("tbl_shopify_product_variants")
    op.drop_index("ix_shopify_products_rivollo_product", table_name="tbl_shopify_products")
    op.drop_index("ix_shopify_products_user", table_name="tbl_shopify_products")
    op.drop_table("tbl_shopify_products")
    op.drop_index("ix_shopify_connections_api_key", table_name="tbl_shopify_connections")
    op.drop_index("ix_shopify_connections_user", table_name="tbl_shopify_connections")
    op.drop_index("ux_shopify_connections_active_key", table_name="tbl_shopify_connections")
    op.drop_index("ux_shopify_connections_active_shop", table_name="tbl_shopify_connections")
    op.drop_table("tbl_shopify_connections")
