"""add tbl_product_events (viewer analytics, replaces ClickHouse)

Revision ID: a4c8e2f61b05
Revises: c9e5a3b1d8f6
Create Date: 2026-11-20 00:00:00.000000

Viewer analytics events: product views, hotspot clicks, link clicks. Replaces
the ClickHouse table analytics.product_events, whose VM has been shut down.
Columns are one-to-one with that table so the Viewer API's analytics endpoints
return the same shapes. No historical data is migrated: the ClickHouse VM is
gone, so counts restart from zero.

Written by:  Rivollo.Viewer.Api  (Repositories/AnalyticsRepository.cs)
Read by:     Rivollo.Viewer.Api  (Repositories/ProductAnalyticsRepository.cs)

Append-only. Purely additive: no existing table is altered, no data is written.

This table is unrelated to tbl_analytics_events / tbl_analytics_daily_product,
which this API's /analytics routes use.

NO FOREIGN KEYS, deliberately: an analytics write must never fail because of
the state of another table, and the account purge job aborts on unrecognised
FKs to tbl_users / tbl_products. Deleting a product leaves its events behind;
every read filters by product_id, so orphans are never returned.

IF NOT EXISTS: dev may already hold this table from the hand-run script
Rivollo.Viewer.Api/sql/create_tbl_product_events.sql (now removed). The DDL
below is identical to it, so applying this migration there is a no-op.

Written by hand: migrations/env.py refuses to autogenerate foreign keys.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "a4c8e2f61b05"
down_revision: Union[str, Sequence[str], None] = "c9e5a3b1d8f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS tbl_product_events (
            id             BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

            event_time     TIMESTAMPTZ  NOT NULL DEFAULT now(),
            -- 'product_view' | 'hotspot_click' | 'link_click'
            event_type     VARCHAR(32)  NOT NULL,

            product_id     UUID         NOT NULL,
            public_id      TEXT         NOT NULL,

            -- reserved, currently always NULL
            client_id      UUID,
            user_id        UUID,
            feature_id     UUID,

            session_id     TEXT         NOT NULL,
            -- 'apple' | 'other' | 'unknown'
            device_type    VARCHAR(16)  NOT NULL,
            user_agent     TEXT         NOT NULL,

            -- hotspot_click only
            hotspot_id     UUID,
            hotspot_title  TEXT,

            -- link_click only
            link_id        BIGINT,
            link_url       TEXT,
            link_title     TEXT
        )
        """
    )
    # Every read filters on product_id + event_type.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_product_events_product_type "
        "ON tbl_product_events (product_id, event_type)"
    )
    # Time-range reporting and any future retention clean-up.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_product_events_event_time "
        "ON tbl_product_events (event_time)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tbl_product_events")
