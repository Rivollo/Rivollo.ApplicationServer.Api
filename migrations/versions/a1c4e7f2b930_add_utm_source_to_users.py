"""add utm_source to tbl_users

Revision ID: a1c4e7f2b930
Revises: f2a9c41d7b60
Create Date: 2026-08-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1c4e7f2b930"
down_revision: Union[str, Sequence[str], None] = "f2a9c41d7b60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE tbl_users "
        "ADD COLUMN IF NOT EXISTS utm_source VARCHAR(100)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE tbl_users DROP COLUMN IF EXISTS utm_source")
