"""retain tbl_payments when a user is erased: user_id nullable + ON DELETE SET NULL

Revision ID: d94b62e8c1f5
Revises: e7a15c93f0b2
Create Date: 2026-08-20 00:00:00.000000

Unblocks assertion A13 of the Rivollo.AccountPurge.Job schema contract.

tbl_payments.user_id is NOT NULL with ON DELETE CASCADE, so erasing a user
destroys their payment rows. Payments are RETAINED: books-of-account retention
sits with us as merchant of record, not with Razorpay as processor, so the
gateway holding its own copy does not discharge the obligation. After this
migration a purge nulls user_id and leaves an anonymised financial record —
amount, currency, status, timestamps, razorpay_order_id and razorpay_payment_id
all survive, with nothing tying the row to a person.

Safe for the application: Payment is imported in exactly three places
(app/models/models.py registers it with the metadata; the INR and USD webhook
services use it). Across both services the only reads are idempotency lookups
keyed on razorpay_order_id, and the only writes are constructors that always
supply a live user_id. Nothing filters, joins, orders or serialises on user_id,
so widening it to nullable changes no existing code path.

The FK name is looked up from pg_constraint rather than assumed, for the same
reason as A07 and A09 — DEV happens to carry the conventional
``tbl_payments_user_id_fkey``, but this migration must not depend on that.

The index is new. Before this change user_id was NOT NULL and only ever read
through the (indexed) razorpay_order_id path; afterwards the purge job needs
``WHERE user_id = ...`` per account, and Postgres would otherwise sequential-scan
the payments table once per purged user.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d94b62e8c1f5"
down_revision: Union[str, Sequence[str], None] = "e7a15c93f0b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _swap_user_fk(rule_sql: str, already_code: str) -> str:
    """Rewrite the tbl_payments.user_id -> tbl_users FK's delete rule.

    confdeltype: 'c' = CASCADE, 'n' = SET NULL.
    """
    return f"""
DO $$
DECLARE
    fk record;
BEGIN
    SELECT c.conname, c.confdeltype INTO fk
      FROM pg_constraint c
      JOIN pg_class t      ON t.oid  = c.conrelid
      JOIN pg_namespace tn ON tn.oid = t.relnamespace
      JOIN pg_class rt     ON rt.oid = c.confrelid
      JOIN pg_attribute a  ON a.attrelid = t.oid AND a.attnum = c.conkey[1]
     WHERE c.contype = 'f'
       AND tn.nspname = current_schema()
       AND t.relname  = 'tbl_payments'
       AND rt.relname = 'tbl_users'
       AND a.attname  = 'user_id'
       AND array_length(c.conkey, 1) = 1;

    IF fk IS NULL THEN
        RAISE EXCEPTION
            'No FK found on tbl_payments.user_id -> tbl_users; schema has drifted.';
    END IF;

    IF fk.confdeltype = '{already_code}' THEN
        RETURN;
    END IF;

    EXECUTE format('ALTER TABLE tbl_payments DROP CONSTRAINT %I', fk.conname);
    EXECUTE format(
        'ALTER TABLE tbl_payments ADD CONSTRAINT %I '
        'FOREIGN KEY (user_id) REFERENCES tbl_users(id) {rule_sql}',
        fk.conname
    );
END $$;
"""


def upgrade() -> None:
    # Must precede the FK swap: SET NULL against a NOT NULL column is accepted
    # as DDL and only fails later, at the moment a user is deleted.
    op.execute("ALTER TABLE tbl_payments ALTER COLUMN user_id DROP NOT NULL")

    op.execute(_swap_user_fk("ON DELETE SET NULL", "n"))

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction, and Alembic
    # wraps every migration in one. autocommit_block() suspends that for the
    # duration. CONCURRENTLY (rather than the plain CREATE INDEX used by
    # c5e81a7f3d94) because tbl_payments is a live billing table and a plain
    # build takes an ACCESS EXCLUSIVE lock that would block webhook writes.
    #
    # A cancelled CONCURRENTLY build leaves an INVALID index behind that
    # IF NOT EXISTS would then happily skip over forever, so clear that first.
    with op.get_context().autocommit_block():
        op.execute(
            """
            DO $$
            DECLARE
                bad_index text;
            BEGIN
                SELECT c.relname INTO bad_index
                  FROM pg_class c
                  JOIN pg_index i ON i.indexrelid = c.oid
                 WHERE c.relname = 'ix_tbl_payments_user_id'
                   AND NOT i.indisvalid;

                IF bad_index IS NOT NULL THEN
                    EXECUTE format('DROP INDEX %I', bad_index);
                END IF;
            END $$;
            """
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_tbl_payments_user_id "
            "ON tbl_payments (user_id)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_tbl_payments_user_id")

    op.execute(_swap_user_fk("ON DELETE CASCADE", "c"))

    # Deliberately NOT restoring NOT NULL. Once a purge has run there are rows
    # with user_id IS NULL, and re-adding the constraint would fail against
    # exactly the anonymised records this migration exists to preserve.
    # Reinstating it is a manual decision that has to deal with those rows.
