"""add tbl_model_variant_generations (layout from photo, ADR-015)

Revision ID: b7d3f1a2c9e4
Revises: a5c1e9d4b7f2
Create Date: 2026-09-29 00:00:01.000000

One row per attempt to generate a model variant from a photo. The generated GLB
is a private candidate until the seller accepts it, and acceptance creates an
ordinary tbl_product_model_variants row through the existing upload pipeline.
See docs/configurator/decisions.md ADR-015.

Purely additive: one new table and its indexes. No existing table is altered
and no data is written. The Configurator becomes FIVE tables (CLAUDE.md,
data-model.md updated in the same change).

Written by hand: migrations/env.py refuses to autogenerate foreign keys.

FOREIGN KEYS — dictated by the account purge job
------------------------------------------------
* product_id -> tbl_products ON DELETE CASCADE. A NEW FK to tbl_products, so
  ACCOUNT_PURGE_JOB_HANDOFF.md section 25 assertion 9 aborts the purge run
  until Rivollo.AccountPurge.Job allow-lists it. Deploy that job change in the
  same window as this migration (Q8; fold it into model-variants-purge/supriya).
* accepted_variant_id -> tbl_product_model_variants ON DELETE SET NULL, the
  same reasoning as ADR-014's asset FKs.
* NO FK from created_by / updated_by to tbl_users (ADR-010).

Candidate blobs live under {user_id}/{product_id}/model-variants/{generation id}/,
inside the purge's user prefix, so they need no new purge rule.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "b7d3f1a2c9e4"
down_revision: Union[str, Sequence[str], None] = "a5c1e9d4b7f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create tbl_model_variant_generations."""

    op.create_table(
        "tbl_model_variant_generations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("product_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source_image_url", sa.Text(), nullable=False),
        sa.Column("model_key", sa.Text(), nullable=False),
        sa.Column("credit_cost", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'queued'"), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("candidate_glb_url", sa.Text()),
        sa.Column("candidate_glb_blob_url", sa.Text()),
        sa.Column("candidate_size_bytes", sa.BigInteger()),
        sa.Column("accepted_variant_id", postgresql.UUID(as_uuid=True)),
        sa.Column("auto_accept", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("client_ref", sa.Text()),
        sa.Column("created_by", postgresql.UUID(as_uuid=True)),
        sa.Column(
            "created_date",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True)),
        sa.Column("updated_date", sa.TIMESTAMP(timezone=True)),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["tbl_products.id"],
            name="fk_generations_product",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["accepted_variant_id"],
            ["tbl_product_model_variants.id"],
            name="fk_generations_accepted_variant",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'generating', 'ready', 'failed', 'accepted', 'discarded')",
            name="ck_generations_status",
        ),
        sa.CheckConstraint("credit_cost >= 0", name="ck_generations_credit_cost"),
    )
    # Full, not partial: also serves the product -> generation CASCADE.
    op.create_index(
        "ix_generations_product_created",
        "tbl_model_variant_generations",
        ["product_id", "created_date"],
    )
    op.create_index(
        "ix_generations_product_client_ref",
        "tbl_model_variant_generations",
        ["product_id", "client_ref"],
    )
    # Serves the variant -> generation SET NULL lookup.
    op.create_index(
        "ix_generations_accepted_variant",
        "tbl_model_variant_generations",
        ["accepted_variant_id"],
    )
    # The stale-generation sweep: in-flight rows by age.
    op.create_index(
        "ix_generations_in_flight",
        "tbl_model_variant_generations",
        ["started_at"],
        postgresql_where=sa.text("status IN ('queued', 'generating')"),
    )


def downgrade() -> None:
    """Drop tbl_model_variant_generations.

    Accepted variants are untouched (they are ordinary model variants). Candidate
    blobs are NOT deleted; they sit under the seller's prefix and are removed by
    the account purge.
    """
    op.drop_index("ix_generations_in_flight", table_name="tbl_model_variant_generations")
    op.drop_index("ix_generations_accepted_variant", table_name="tbl_model_variant_generations")
    op.drop_index("ix_generations_product_client_ref", table_name="tbl_model_variant_generations")
    op.drop_index("ix_generations_product_created", table_name="tbl_model_variant_generations")
    op.drop_table("tbl_model_variant_generations")
