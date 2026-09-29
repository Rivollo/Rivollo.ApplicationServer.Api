"""add tbl_api_keys for third-party integration API keys

Revision ID: a5c1e9d4b7f2
Revises: e3b9c6a1d27f
Create Date: 2026-09-29 00:00:00.000000

Long-lived, revocable API keys a seller creates in the portal and pastes into
an integration (the Shopify app first). See docs/api_keys.md.

Purely additive: one new table and its indexes. No existing table is altered
and no data is written.

Written by hand, like c7a4e0d51b83 and e3b9c6a1d27f: migrations/env.py refuses
to autogenerate foreign keys, and the schema has drifted from the chain.

FOREIGN KEYS
------------
* user_id -> tbl_users ON DELETE CASCADE: a key dies with its owner. The purge
  deletes the user row last, so this cascade is how it removes keys.
  🔴 DEPLOYMENT DEPENDENCY: Rivollo.AccountPurge.Job must add
  ("tbl_api_keys", "user_id") to USER_CASCADE_FKS first, or its contract check
  (A14) aborts every run. See docs/account-purge-job-changes.md.
* created_by / updated_by are plain UUIDs, as AuditMixin declares them.

Only the SHA-256 of a key is stored (key_hash, unique). The raw key is never
written anywhere.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "a5c1e9d4b7f2"
down_revision: Union[str, Sequence[str], None] = "e3b9c6a1d27f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create tbl_api_keys."""

    op.create_table(
        "tbl_api_keys",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("key_prefix", sa.Text(), nullable=False),
        sa.Column(
            "scopes",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("ARRAY['read','write','convert']::text[]"),
        ),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("last_used_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("isactive", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("revoked_at", sa.TIMESTAMP(timezone=True)),
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
            ["user_id"], ["tbl_users.id"], name="fk_api_keys_user", ondelete="CASCADE"
        ),
    )
    # Authentication is a single lookup on this index; uniqueness also makes a
    # hash collision fail loudly instead of authenticating as the wrong user.
    op.create_index("ux_api_keys_key_hash", "tbl_api_keys", ["key_hash"], unique=True)
    # The portal lists a user's keys newest first; also serves the user -> key cascade.
    op.create_index("ix_api_keys_user_created", "tbl_api_keys", ["user_id", "created_date"])


def downgrade() -> None:
    """Drop tbl_api_keys. Every issued key stops working."""
    op.drop_index("ix_api_keys_user_created", table_name="tbl_api_keys")
    op.drop_index("ux_api_keys_key_hash", table_name="tbl_api_keys")
    op.drop_table("tbl_api_keys")
