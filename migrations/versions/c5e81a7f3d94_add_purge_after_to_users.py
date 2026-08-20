"""add purge_after to tbl_users for 30-day account soft-delete

Revision ID: c5e81a7f3d94
Revises: a1c4e7f2b930
Create Date: 2026-08-17 00:00:00.000000

Foundation for the 30-day recoverable account deletion. No behaviour changes
here — this only adds the column and the index the future purge job reads.

The lifecycle reuses the two columns that already exist on tbl_users:

    is_active   = false  -> access kill-switch (login + token validation refuse)
    deleted_at  != NULL  -> deletion requested at this instant; account is
                            recoverable until purge_after
    purge_after != NULL  -> the instant permanent deletion becomes due

purge_after is stored rather than derived from ``deleted_at + 30 days`` so the
retention window is data, not a constant compiled into a query. That lets a
single account be put on legal hold or given an extension, and lets the window
change without a backfill.

Written as raw idempotent DDL rather than op.add_column/op.create_index to match
the house style (see f2a9c41d7b60) and because this database has drifted from
the migration chain: columns exist that no revision created, so every statement
here must tolerate already having been applied by hand.

Deliberately NOT touched:
  - tbl_users_email_key stays a plain UNIQUE(email). Email reuse during the
    recovery window must stay blocked, otherwise a re-registration would strand
    the original account with no way to restore it.
  - tbl_auth_identities is left alone. It is kept until permanent purge so a
    Google account can still be restored.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c5e81a7f3d94"
# Originally authored against f2a9c41d7b60, but a1c4e7f2b930 (utm_source) landed
# on main from the same parent while this branch was open, which left the project
# with two heads. Repointed here rather than merged: both revisions only ADD a
# nullable column to tbl_users, so ordering between them carries no meaning, and
# a linear chain is cheaper to reason about than a merge revision. DEV was
# already advanced a1c4e7f2b930 -> c5e81a7f3d94, so this matches deployed state.
down_revision: Union[str, Sequence[str], None] = "a1c4e7f2b930"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # timestamptz to match every other timestamp on this table (created_at,
    # created_date, updated_date, deleted_at). Nullable with no default: NULL
    # means "no deletion pending", which is the correct state for all existing
    # rows, so this is a metadata-only change and does not rewrite the table.
    op.execute(
        "ALTER TABLE tbl_users "
        "ADD COLUMN IF NOT EXISTS purge_after timestamp with time zone"
    )

    # Drives the purge job's only query:
    #     WHERE purge_after IS NOT NULL AND purge_after <= now()
    #
    # Partial on IS NOT NULL because accounts pending deletion are a tiny
    # fraction of the table (3 of 118 today), so the index stays proportional to
    # the work queue rather than to the user count.
    #
    # Not CONCURRENTLY: that cannot run inside Alembic's transaction, and it
    # would leave an INVALID index behind on failure. tbl_users is small enough
    # that the brief SHARE lock is a non-issue, and no other migration in this
    # project uses CONCURRENTLY.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_tbl_users_purge_after "
        "ON tbl_users (purge_after) "
        "WHERE purge_after IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_tbl_users_purge_after")
    op.execute("ALTER TABLE tbl_users DROP COLUMN IF EXISTS purge_after")
