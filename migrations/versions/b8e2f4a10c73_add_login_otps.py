"""add tbl_login_otps for the email OTP login flow

Revision ID: b8e2f4a10c73
Revises: f61a03d7b8e4
Create Date: 2026-08-31 00:00:00.000000

Creates the table behind POST /auth/otp/request and POST /auth/otp/verify.
Purely additive: no existing table, column, constraint or index is touched, so
this revision cannot affect signup, password login, Google login or password
reset.

Written as raw idempotent DDL rather than op.create_table to match the house
style (see c5e81a7f3d94 and f2a9c41d7b60) and because this database has drifted
from the migration chain — every statement here must tolerate having already
been applied by hand from sql/add_login_otps_table.sql, which is kept in sync
with this file.

ONE ROW PER (email, purpose)
----------------------------
A row is a challenge SERIES, not a single code. The resend budget spans up to
three codes and the lockout outlives all of them, so neither fits on a per-code
row. Issuing a code overwrites otp_hash, which is what invalidates the previous
one. UNIQUE (email, purpose) is what makes the series addressable by upsert —
the same shape tbl_app_tokens uses for client_key — and it bounds the table to
one row per address that has ever attempted OTP login.

NO FOREIGN KEY TO tbl_users
---------------------------
ACCOUNT_PURGE_JOB_HANDOFF.md section 25 assertion 9 aborts the entire account
purge run if an unexpected new foreign key references tbl_users. Beyond that,
rows are written for addresses that may have no account at all, which is what
keeps the lockout response free of an enumeration oracle. The purge job needs
an email-keyed DELETE for this table, alongside the one it already has for
tbl_signup_otps.

Deliberately NOT done here:
  - No UNIQUE index on otp_hash. tbl_password_resets.token carries exactly that
    over a 6-digit space, which makes two concurrent resets collide into a 500.
    Constrain the subject, never the secret.
  - No column added to tbl_users. The lockout lives entirely in locked_until on
    this table and has no effect on account state, Google login or password
    login.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b8e2f4a10c73"
down_revision: Union[str, Sequence[str], None] = "f61a03d7b8e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # email is TEXT rather than CITEXT. CITEXT would match tbl_users.email and
    # give case-insensitivity at the column level, but Azure Database for
    # PostgreSQL rejects CREATE EXTENSION for anything outside its
    # `azure.extensions` allow-list, and citext is not on it for this server.
    #
    # Safe here because the application lowercases every address before it
    # reaches this table — LoginOtpService normalises on both the request and
    # the verify path — which is also what tbl_signup_otps does today.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS tbl_login_otps (
            id                        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

            -- challenge identity: the address, not the account
            email                     TEXT        NOT NULL,
            purpose                   VARCHAR(32) NOT NULL DEFAULT 'login',

            -- current code only; NULL means no live code
            otp_hash                  TEXT,
            expires_at                TIMESTAMPTZ,

            -- resend budget (series-scoped)
            resend_count              SMALLINT    NOT NULL DEFAULT 0,
            max_resends               SMALLINT    NOT NULL DEFAULT 2,

            -- verification attempts (code-scoped; reset on every send)
            verification_attempts     SMALLINT    NOT NULL DEFAULT 0,
            max_verification_attempts SMALLINT    NOT NULL DEFAULT 5,

            last_sent_at              TIMESTAMPTZ,

            -- the entire OTP lockout state
            locked_until              TIMESTAMPTZ,
            lock_reason               VARCHAR(32),

            consumed_at               TIMESTAMPTZ,
            invalidated_reason        VARCHAR(32),

            request_ip                TEXT,

            created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at                TIMESTAMPTZ DEFAULT NOW(),

            CONSTRAINT uq_login_otps_email_purpose UNIQUE (email, purpose)
        )
        """
    )

    # The addressing key. Targeted by name from
    # LoginOtpRepository.upsert_challenge via on_conflict_do_update, exactly as
    # AuthService.generate_app_token targets uq_app_tokens_client_key.
    #
    # It MUST be a CONSTRAINT, not merely a unique index. ON CONFLICT ON
    # CONSTRAINT resolves the name against pg_constraint, and a bare unique
    # index is invisible there -- Postgres answers
    #   UndefinedObjectError: constraint ... does not exist
    # even though \d shows an index by that name. The declaration is inline
    # above; this block only repairs databases created before it was.
    op.execute(
        """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_login_otps_email_purpose'
    ) THEN
        -- Drop a bare unique INDEX of the same name if an earlier version of
        -- this script created one. ON CONFLICT ON CONSTRAINT requires a real
        -- constraint in pg_constraint; a unique index alone does not satisfy it.
        DROP INDEX IF EXISTS uq_login_otps_email_purpose;
        ALTER TABLE tbl_login_otps
            ADD CONSTRAINT uq_login_otps_email_purpose UNIQUE (email, purpose);
    END IF;
END $$;
        """
    )

    # Supports the cleanup sweep in scripts/cleanup_login_otps.py, which
    # selects on last_sent_at and skips rows that are still locked.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_login_otps_last_sent_at "
        "ON tbl_login_otps (last_sent_at)"
    )

    # Partial, because locked rows are a tiny fraction of the table: keeps the
    # index proportional to the number of locked addresses rather than to the
    # number of addresses. Supports "how many addresses are locked right now",
    # which is the operational question during a rollout.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_login_otps_locked_until "
        "ON tbl_login_otps (locked_until) "
        "WHERE locked_until IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_login_otps_locked_until")
    op.execute("DROP INDEX IF EXISTS ix_login_otps_last_sent_at")
    op.execute(
        "ALTER TABLE IF EXISTS tbl_login_otps "
        "DROP CONSTRAINT IF EXISTS uq_login_otps_email_purpose"
    )
    op.execute("DROP TABLE IF EXISTS tbl_login_otps")
