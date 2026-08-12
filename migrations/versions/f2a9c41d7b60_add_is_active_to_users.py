"""add is_active flag to tbl_users

Revision ID: f2a9c41d7b60
Revises: d8b6b3c4d9f1
Create Date: 2026-08-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f2a9c41d7b60"
down_revision: Union[str, Sequence[str], None] = "d8b6b3c4d9f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE tbl_users "
        "ADD COLUMN IF NOT EXISTS is_active boolean NOT NULL DEFAULT true"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE tbl_users DROP COLUMN IF EXISTS is_active")
