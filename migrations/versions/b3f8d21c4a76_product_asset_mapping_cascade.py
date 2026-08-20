"""tbl_product_asset_mapping.productid -> ON DELETE CASCADE

Revision ID: b3f8d21c4a76
Revises: c5e81a7f3d94
Create Date: 2026-08-20 00:00:00.000000

Unblocks assertion A07 of the Rivollo.AccountPurge.Job schema contract.

The ORM has always declared this FK as ``ondelete="CASCADE"``
(app/models/models.py, ProductAssetMapping.productid) but the database was
created with the default NO ACTION, so model and schema disagree. Any
``DELETE FROM tbl_products`` for a product that has asset mappings — 4818
mappings across 1191 products in DEV — raises a foreign key violation. This
brings the database up to what the model already says.

The constraint is dropped and recreated under its EXISTING name rather than a
generated one. That name is not derivable: DEV carries
``tbl_product_asset_mapping_tbl_products_fk``, not the
``{table}_{column}_fkey`` Postgres would have chosen, and other environments may
differ again. It is therefore looked up from pg_constraint at run time — the
same reason A09 is catalog-driven.

Idempotent: if the FK is already CASCADE the block returns without touching it.
Raises rather than silently creating one if the FK is missing entirely, because
that would mean drift this migration was not written to reason about.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b3f8d21c4a76"
down_revision: Union[str, Sequence[str], None] = "c5e81a7f3d94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# confdeltype codes in pg_constraint: 'a' = NO ACTION, 'c' = CASCADE, 'n' = SET NULL.
_FIND_FK = """
    SELECT c.conname, c.confdeltype
      FROM pg_constraint c
      JOIN pg_class t       ON t.oid  = c.conrelid
      JOIN pg_namespace tn  ON tn.oid = t.relnamespace
      JOIN pg_class rt      ON rt.oid = c.confrelid
      JOIN pg_attribute a   ON a.attrelid = t.oid AND a.attnum = c.conkey[1]
     WHERE c.contype = 'f'
       AND tn.nspname = current_schema()
       AND t.relname  = 'tbl_product_asset_mapping'
       AND rt.relname = 'tbl_products'
       AND a.attname  = 'productid'
       AND array_length(c.conkey, 1) = 1
"""


def _swap_delete_rule(rule_sql: str, already_code: str) -> str:
    return f"""
DO $$
DECLARE
    fk record;
BEGIN
    {_FIND_FK} INTO fk;

    IF fk IS NULL THEN
        RAISE EXCEPTION
            'No FK found on tbl_product_asset_mapping.productid -> tbl_products. '
            'Expected exactly one; the schema has drifted further than this migration handles.';
    END IF;

    IF fk.confdeltype = '{already_code}' THEN
        RETURN;
    END IF;

    EXECUTE format('ALTER TABLE tbl_product_asset_mapping DROP CONSTRAINT %I', fk.conname);
    EXECUTE format(
        'ALTER TABLE tbl_product_asset_mapping ADD CONSTRAINT %I '
        'FOREIGN KEY (productid) REFERENCES tbl_products(id) {rule_sql}',
        fk.conname
    );
END $$;
"""


def upgrade() -> None:
    op.execute(_swap_delete_rule("ON DELETE CASCADE", "c"))


def downgrade() -> None:
    op.execute(_swap_delete_rule("ON DELETE NO ACTION", "a"))
