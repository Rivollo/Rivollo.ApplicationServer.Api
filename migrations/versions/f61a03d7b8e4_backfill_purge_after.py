"""backfill purge_after for accounts soft-deleted before the column existed

Revision ID: f61a03d7b8e4
Revises: d94b62e8c1f5
Create Date: 2026-08-20 00:00:00.000000

c5e81a7f3d94 added tbl_users.purge_after but did not backfill it, so accounts
soft-deleted before that migration have deleted_at set and purge_after NULL.
They are stranded in both directions: the purge job selects on
``purge_after IS NOT NULL AND purge_after <= now()`` so they can never be erased,
and AccountService.restore_account treats ``purge_after IS NULL`` as
non-restorable. 3 such rows in DEV; the production count is unknown.

The window is ``now() + 30 days``, NOT ``deleted_at + 30 days``.

deleted_at + 30 days would compute a date already in the past for every one of
these rows — they were deleted well over a month ago — so all of them would
become immediately purgeable and be erased on the purge job's first execute run,
with no operator ever seeing them in a dry-run first. now() + 30 days is safe at
any row count and buys a full 30 days in which a wrong backfill shows up in the
nightly dry-run output while it is still reversible. The cost is that these
accounts wait an extra 30 days to be erased, which is not a real cost given they
have been waiting since before the column existed.

Idempotent: the WHERE clause matches only rows still missing purge_after, so a
second run updates nothing. It cannot touch accounts deleted through the normal
flow either — AccountService writes deleted_at and purge_after in the same
UPDATE, so a live soft-delete never presents this NULL combination.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f61a03d7b8e4"
down_revision: Union[str, Sequence[str], None] = "d94b62e8c1f5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE tbl_users
           SET purge_after = now() + interval '30 days'
         WHERE deleted_at  IS NOT NULL
           AND purge_after IS NULL
        """
    )


def downgrade() -> None:
    # Intentionally a no-op. The rows this touched are indistinguishable from
    # accounts deleted normally after it ran, so clearing purge_after by the
    # same predicate would strip the value from live pending deletions and
    # strand them exactly as this migration was written to fix.
    pass
