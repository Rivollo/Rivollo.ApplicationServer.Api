"""created_by / updated_by FKs -> tbl_users become ON DELETE SET NULL

Revision ID: e7a15c93f0b2
Revises: b3f8d21c4a76
Create Date: 2026-08-20 00:00:00.000000

Unblocks assertion A09 of the Rivollo.AccountPurge.Job schema contract.

43 foreign keys point at tbl_users(id) from created_by / updated_by audit
columns, all of them ON DELETE NO ACTION, so a bare ``DELETE FROM tbl_users``
fails outright. SET NULL is the intended behaviour: the audit row survives and
only the link to the erased person is dropped, which is exactly what anonymised
retention means. Confirmed against DEV 2026-08-20: 43 NO ACTION, 1 already
SET NULL.

The statements are generated from pg_constraint at run time rather than written
out. Two reasons, both load-bearing:

  * The names are not derivable. tbl_product_assets carries
    ``tbl_assets_created_by_fkey`` / ``tbl_assets_updated_by_fkey`` — the table
    was renamed and the constraints were not. Assuming {table}_{column}_fkey
    would miss them.
  * 43 hand-written pairs is 86 chances to typo a name into a silent no-op.

NOT touched, deliberately: tbl_products.created_by is ALREADY SET NULL and is
excluded by the ``confdeltype = 'a'`` filter. The account purge deletes products
before their owner precisely because that rule exists; changing it would silently
orphan every product of a deleted user.

These FKs are invisible to the ORM — AuditMixin declares created_by/updated_by
as plain UUID columns with no ForeignKey — so ``alembic revision --autogenerate``
would happily emit DROP statements for all 43. migrations/env.py installs an
include_object hook that refuses to autogenerate foreign key constraints at all,
which is what actually protects this migration. See the note on AuditMixin.

Idempotent: the loop selects only NO ACTION constraints, so a second run finds
nothing and changes nothing.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "e7a15c93f0b2"
down_revision: Union[str, Sequence[str], None] = "b3f8d21c4a76"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _convert(from_code: str, to_rule: str, extra_predicate: str = "") -> str:
    """Rewrite every audit FK currently at `from_code` to use `to_rule`.

    confdeltype: 'a' = NO ACTION, 'n' = SET NULL.
    """
    return f"""
DO $$
DECLARE
    r record;
    n_changed int := 0;
BEGIN
    FOR r IN
        SELECT c.conname,
               t.relname     AS table_name,
               a.attname     AS column_name,
               a.attnotnull  AS col_not_null
          FROM pg_constraint c
          JOIN pg_class t      ON t.oid  = c.conrelid
          JOIN pg_namespace tn ON tn.oid = t.relnamespace
          JOIN pg_class rt     ON rt.oid = c.confrelid
          JOIN pg_attribute a  ON a.attrelid = t.oid AND a.attnum = c.conkey[1]
         WHERE c.contype = 'f'
           AND tn.nspname = current_schema()
           AND rt.relname = 'tbl_users'
           AND a.attname IN ('created_by', 'updated_by')
           AND array_length(c.conkey, 1) = 1
           AND c.confdeltype = '{from_code}'
           {extra_predicate}
    LOOP
        -- SET NULL on a NOT NULL column is accepted as DDL and then fails at
        -- DELETE time, which would move the purge job's breakage from today
        -- (visible, blocked) to the first execute run (irreversible). Refuse now.
        IF r.col_not_null THEN
            RAISE EXCEPTION
                'Cannot apply ON DELETE SET NULL to %.% - column is NOT NULL.',
                r.table_name, r.column_name;
        END IF;

        EXECUTE format('ALTER TABLE %I DROP CONSTRAINT %I', r.table_name, r.conname);
        EXECUTE format(
            'ALTER TABLE %I ADD CONSTRAINT %I FOREIGN KEY (%I) '
            'REFERENCES tbl_users(id) {to_rule}',
            r.table_name, r.conname, r.column_name
        );
        n_changed := n_changed + 1;
    END LOOP;

    RAISE NOTICE 'audit FKs rewritten to {to_rule}: %', n_changed;
END $$;
"""


def upgrade() -> None:
    op.execute(_convert("a", "ON DELETE SET NULL"))


def downgrade() -> None:
    # tbl_products.created_by was SET NULL before this migration ran and must
    # stay that way — reverting it would break the purge job's delete ordering
    # in a way nothing else here would catch.
    op.execute(
        _convert(
            "n",
            "ON DELETE NO ACTION",
            extra_predicate="AND NOT (t.relname = 'tbl_products' AND a.attname = 'created_by')",
        )
    )
