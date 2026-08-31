-- Create the email OTP login challenge table.
-- One row per (email, purpose) — a row is a challenge SERIES, not a single
-- code. Issuing a code overwrites otp_hash, which invalidates the previous one.
--
-- Kept in sync with migrations/versions/b8e2f4a10c73_add_login_otps.py. Run
-- this script manually against the database in environments where Alembic is
-- not executed (neither deploy workflow runs it).
--
-- NO foreign key to tbl_users, deliberately: the account purge job's schema
-- contract (ACCOUNT_PURGE_JOB_HANDOFF.md section 25, assertion 9) aborts its
-- entire run on an unexpected new FK to tbl_users, and rows here are written
-- for addresses that may have no account at all. The purge job needs an
-- email-keyed DELETE for this table.
--
-- NO unique index on otp_hash. tbl_password_resets.token has exactly that over
-- a 6-digit space, which makes two concurrent resets collide into a 500.
--
-- email is TEXT, not CITEXT: Azure Database for PostgreSQL does not allow-list
-- the citext extension. The application lowercases every address before it
-- reaches this table, on both the request and the verify path, so the two
-- behave identically here.

CREATE TABLE IF NOT EXISTS tbl_login_otps (
    id                        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- challenge identity: the address (always lowercased by the app), not the account
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

    -- the entire OTP lockout state; never touches tbl_users
    locked_until              TIMESTAMPTZ,
    lock_reason               VARCHAR(32),

    consumed_at               TIMESTAMPTZ,
    invalidated_reason        VARCHAR(32),

    request_ip                TEXT,

    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                TIMESTAMPTZ DEFAULT NOW(),

    -- The addressing key. MUST be a CONSTRAINT, not just a unique index:
    -- the application upserts with ON CONFLICT ON CONSTRAINT, which resolves
    -- names against pg_constraint. Same shape as tbl_app_tokens.
    CONSTRAINT uq_login_otps_email_purpose UNIQUE (email, purpose)
);

-- For databases where the table was created before the line above existed.
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

CREATE INDEX IF NOT EXISTS ix_login_otps_last_sent_at
    ON tbl_login_otps (last_sent_at);

CREATE INDEX IF NOT EXISTS ix_login_otps_locked_until
    ON tbl_login_otps (locked_until)
    WHERE locked_until IS NOT NULL;
