"""add tbl_agent_usage and tbl_llm_model_prices (3D Agent billing)

Revision ID: f3c7a9e2b1d4
Revises: a4c8e2f61b05, d2f8b4c6a1e3
Create Date: 2026-11-26 00:00:00.000000

AI Assistant chat messages are charged in AI credits by the LLM tokens they
actually use. tbl_agent_usage holds one row per message (its credit hold while
it runs, then its final bill); tbl_llm_model_prices is the per-deployment price
list the cost is computed from. The DDL, formula and seed prices are the same
as sql/create_agent_billing.sql, which is what gets applied by hand.

MERGES TWO HEADS. a4c8e2f61b05 (tbl_product_events) and d2f8b4c6a1e3
(configuration dimensions) both branched from c9e5a3b1d8f6, which left the
history with two heads and `alembic upgrade head` refusing to run. This
revision descends from both, so the history has a single head again.

Purely additive: two new tables, their indexes, and the price seed. No existing
table is altered — tbl_license_assignments stays as it is; open holds are summed
from tbl_agent_usage.

Written by hand: migrations/env.py refuses to autogenerate foreign keys.

NO FOREIGN KEYS, on purpose: a billing record outlives the product and user it
describes (as payments do, d94b62e8c1f5), and no new FK to tbl_users or
tbl_products means Rivollo.AccountPurge.Job needs no change. created_by /
updated_by carry no FK either (ADR-010).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "f3c7a9e2b1d4"
down_revision: Union[str, Sequence[str], None] = ("a4c8e2f61b05", "d2f8b4c6a1e3")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _audit_columns() -> list[sa.Column]:
    return [
        sa.Column("created_by", postgresql.UUID(as_uuid=True)),
        sa.Column(
            "created_date",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True)),
        sa.Column("updated_date", sa.TIMESTAMP(timezone=True)),
    ]


def upgrade() -> None:
    """Create tbl_agent_usage and tbl_llm_model_prices; seed the prices."""

    op.create_table(
        "tbl_agent_usage",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("hold_key", sa.Text(), nullable=False, unique=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", postgresql.UUID(as_uuid=True)),
        sa.Column("session_id", sa.Text()),
        sa.Column("agent_model_id", sa.Text()),
        sa.Column("mode", sa.Text()),
        sa.Column("status", sa.Text(), server_default=sa.text("'holding'"), nullable=False),
        sa.Column("credits_held", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("credits_charged", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_credits", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), server_default=sa.text("0"), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("cached_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("llm_calls", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("breakdown", postgresql.JSONB()),
        sa.Column("multiplier", sa.Numeric(6, 3)),
        sa.Column("credit_usd_value", sa.Numeric(10, 6)),
        sa.Column("error", sa.Text()),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("settled_at", sa.TIMESTAMP(timezone=True)),
        *_audit_columns(),
        sa.CheckConstraint(
            "status IN ('holding', 'settled', 'released', 'expired')",
            name="ck_agent_usage_status",
        ),
        sa.CheckConstraint("credits_held >= 0", name="ck_agent_usage_credits_held"),
        sa.CheckConstraint("credits_charged >= 0", name="ck_agent_usage_credits_charged"),
        sa.CheckConstraint("max_credits > 0", name="ck_agent_usage_max_credits"),
        sa.CheckConstraint("cost_usd >= 0", name="ck_agent_usage_cost"),
        sa.CheckConstraint(
            "input_tokens >= 0 AND cached_tokens >= 0 AND output_tokens >= 0 "
            "AND llm_calls >= 0",
            name="ck_agent_usage_tokens",
        ),
    )
    op.create_index(
        "ix_agent_usage_user_created",
        "tbl_agent_usage",
        ["user_id", sa.text("created_date DESC")],
    )
    op.create_index(
        "ix_agent_usage_user_holding",
        "tbl_agent_usage",
        ["user_id"],
        postgresql_where=sa.text("status = 'holding'"),
    )
    op.create_index(
        "ix_agent_usage_holding_expires",
        "tbl_agent_usage",
        ["expires_at"],
        postgresql_where=sa.text("status = 'holding'"),
    )

    prices = op.create_table(
        "tbl_llm_model_prices",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("model_key", sa.Text(), nullable=False),
        sa.Column("input_per_m", sa.Numeric(12, 6), nullable=False),
        sa.Column("cached_input_per_m", sa.Numeric(12, 6), nullable=False),
        sa.Column("output_per_m", sa.Numeric(12, 6), nullable=False),
        sa.Column("currency", sa.Text(), server_default=sa.text("'USD'"), nullable=False),
        sa.Column(
            "effective_from",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("isactive", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notes", sa.Text()),
        *_audit_columns(),
        sa.UniqueConstraint(
            "model_key", "effective_from", name="uq_llm_model_prices_model_effective"
        ),
        sa.CheckConstraint(
            "input_per_m >= 0 AND cached_input_per_m >= 0 AND output_per_m >= 0",
            name="ck_llm_model_prices_non_negative",
        ),
    )

    # Azure Retail Prices API, Global Standard, short context, westus3, checked
    # 2026-11-26 — same rows as sql/create_agent_billing.sql.
    effective = sa.text("'2026-11-01T00:00:00Z'::timestamptz")
    op.execute(
        sa.insert(prices).values(
            [
                {"model_key": "Kimi-K2.7-Code", "input_per_m": 0.95, "cached_input_per_m": 0.19,
                 "output_per_m": 4.00, "effective_from": effective,
                 "notes": "Azure meters: K2.7 Code Inp/cached/Outp glbl Tokens (per 1K, converted)"},
                {"model_key": "gpt-6-astra", "input_per_m": 10.00, "cached_input_per_m": 1.00,
                 "output_per_m": 50.00, "effective_from": effective,
                 "notes": "Azure meters: 6-astra ShortCo Inp/Cd Inp/Opt Std Gl 1M Tokens"},
                {"model_key": "gpt-6.1-sol", "input_per_m": 2.00, "cached_input_per_m": 0.10,
                 "output_per_m": 10.00, "effective_from": effective,
                 "notes": "Azure meters: 6.1-sol ShortCo Inp/Cd Inp/Opt Std Gl 1M Tokens"},
                {"model_key": "gpt-5.6-sol", "input_per_m": 4.00, "cached_input_per_m": 0.40,
                 "output_per_m": 20.00, "effective_from": effective,
                 "notes": "Azure meters: 5.6 sol ShortCo Inp/Cd Inp Std Gl 1M Tokens; output from Std DZ / 1.1"},
            ]
        )
    )


def downgrade() -> None:
    """Drop both tables (their indexes and seed rows go with them)."""
    op.drop_table("tbl_llm_model_prices")
    op.drop_index("ix_agent_usage_holding_expires", table_name="tbl_agent_usage")
    op.drop_index("ix_agent_usage_user_holding", table_name="tbl_agent_usage")
    op.drop_index("ix_agent_usage_user_created", table_name="tbl_agent_usage")
    op.drop_table("tbl_agent_usage")
