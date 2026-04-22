-- Create app tokens table
-- One row per client_key — token is updated in place on every new issuance.
-- Run this script manually against the database.

CREATE TABLE IF NOT EXISTS tbl_app_tokens (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    client_key    VARCHAR     NOT NULL,
    token         TEXT        NOT NULL,
    expires_at    TIMESTAMPTZ NOT NULL,
    isactive      BOOLEAN     NOT NULL DEFAULT TRUE,

    -- audit columns
    created_by    UUID,
    created_date  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by    UUID,
    updated_date  TIMESTAMPTZ,

    CONSTRAINT uq_app_tokens_client_key UNIQUE (client_key)
);
