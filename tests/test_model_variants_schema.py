"""Schema of tbl_product_model_variants — ORM metadata and migration e3b9c6a1d27f.

Like test_configurator_models.py, this asserts declared shape, not database
behaviour: there is no Postgres in the test environment. The migration is held
to being purely ADDITIVE (ADR-014): one new table, one nullable column, no data
written, nothing existing dropped. Most FK rules exist because of
Rivollo.AccountPurge.Job, so the migration's DDL is checked as well as the ORM.
"""

import ast
import importlib.util
import io
import re
import tokenize
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint

from app.models.configurator import (
    PartOption,
    PartOptionTexture,
    ProductModelVariant,
    ProductPart,
)

MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "migrations"
    / "versions"
    / "e3b9c6a1d27f_add_product_model_variants.py"
)
AUDIT_COLUMNS = ("created_by", "created_date", "updated_by", "updated_date")


def _code_only(source: str) -> str:
    """Source with comments and docstrings removed; string literals kept.

    The migration's prose explains what it deliberately does NOT do, so
    assertions about emitted DDL must not match the explanation. Tokens are
    re-joined without whitespace.
    """
    tree = ast.parse(source)
    docstrings = [
        doc
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and (doc := ast.get_docstring(node))
    ]
    code = "".join(
        token.string if token.type != tokenize.COMMENT else ""
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
    )
    for doc in docstrings:
        code = code.replace(doc, "", 1)
    return code


def _table():
    return ProductModelVariant.__table__


def _fk(table, column):
    (fk,) = table.c[column].foreign_keys
    return fk


@pytest.fixture(scope="module")
def migration():
    spec = importlib.util.spec_from_file_location("model_variants_migration", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def code() -> str:
    return re.sub(r"\s+", "", _code_only(MIGRATION_PATH.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# ORM
# --------------------------------------------------------------------------- #
def test_table_name():
    assert _table().name == "tbl_product_model_variants"


def test_columns():
    assert set(_table().c.keys()) == {
        "id",
        "product_id",
        "name",
        "glb_asset_id",
        "usdz_asset_id",
        "thumbnail_url",
        "thumbnail_blob_url",
        "original_glb_url",
        "original_glb_blob_url",
        "original_size_bytes",
        "compressed_size_bytes",
        "compression_status",
        "compression_error",
        "width_m",
        "depth_m",
        "height_m",
        "order_index",
        "isactive",
        *AUDIT_COLUMNS,
    }


def test_no_default_flag_the_original_model_is_always_the_default():
    """The product's original model has no row and is the permanent default."""
    assert "is_default" not in _table().c
    assert not [i for i in _table().indexes if i.unique]


def test_nullability():
    nullable = {c.name for c in _table().c if c.nullable}
    assert nullable == {
        # Nullable ONLY because of ON DELETE SET NULL — see test below.
        "glb_asset_id",
        "usdz_asset_id",
        "thumbnail_url",
        "thumbnail_blob_url",
        "original_glb_url",
        "original_glb_blob_url",
        "original_size_bytes",
        "compressed_size_bytes",
        "compression_error",
        "width_m",
        "depth_m",
        "height_m",
        "created_by",
        "updated_by",
        "updated_date",
    }


def test_product_fk_cascades():
    fk = _fk(_table(), "product_id")
    assert fk.column.table.name == "tbl_products"
    assert fk.ondelete == "CASCADE"


@pytest.mark.parametrize("column", ["glb_asset_id", "usdz_asset_id"])
def test_asset_fks_set_null(column):
    """The purge deletes tbl_product_assets BEFORE tbl_products.

    RESTRICT / NO ACTION would abort that step for every seller with a variant;
    CASCADE would let a stray asset delete wipe a variant's configuration.
    """
    fk = _fk(_table(), column)
    assert fk.column.table.name == "tbl_product_assets"
    assert fk.ondelete == "SET NULL"


def test_part_variant_is_optional_and_cascades():
    """NULL = the product's original model, so no existing part is rewritten."""
    column = ProductPart.__table__.c.variant_id
    assert column.nullable is True
    fk = _fk(ProductPart.__table__, "variant_id")
    assert fk.column.table.name == "tbl_product_model_variants"
    assert fk.ondelete == "CASCADE"


def test_part_slug_uniqueness_is_unchanged():
    names = {c.name for c in ProductPart.__table__.constraints}
    assert "uq_parts_product_slug" in names
    assert "uq_parts_variant_slug" not in names


def test_no_foreign_key_to_users():
    for column in AUDIT_COLUMNS:
        assert not _table().c[column].foreign_keys
    assert "tbl_users" not in {fk.column.table.name for fk in _table().foreign_keys}


def test_configurator_foreign_keys_to_products_are_exactly_the_known_two():
    """What Rivollo.AccountPurge.Job assertion A14 must allow-list — no more."""
    targeting = sorted(
        (model.__table__.name, fk.parent.name)
        for model in (ProductModelVariant, ProductPart, PartOption, PartOptionTexture)
        for fk in model.__table__.foreign_keys
        if fk.column.table.name == "tbl_products"
    )
    assert targeting == [
        ("tbl_product_model_variants", "product_id"),
        ("tbl_product_parts", "product_id"),
    ]


@pytest.mark.parametrize(
    "name, columns",
    [
        ("ix_model_variants_product_order", ["product_id", "order_index"]),
        ("ix_model_variants_glb_asset", ["glb_asset_id"]),
        ("ix_model_variants_usdz_asset", ["usdz_asset_id"]),
    ],
)
def test_purge_hot_path_indexes_are_full_indexes(name, columns):
    """The purge's CASCADE and SET NULL lookups need non-partial indexes."""
    index = next(i for i in _table().indexes if i.name == name)
    assert [c.name for c in index.columns] == columns
    assert index.dialect_options["postgresql"]["where"] is None


def test_compression_status_values():
    check = next(
        c for c in _table().constraints
        if isinstance(c, CheckConstraint) and c.name == "ck_model_variants_compression_status"
    )
    text = str(check.sqltext)
    assert "'compressed'" in text and "'fallback_original'" in text
    assert "'legacy'" not in text, "nothing is backfilled, so nothing is legacy"


@pytest.mark.parametrize("column, expected", [("order_index", "1"), ("isactive", "true")])
def test_server_defaults(column, expected):
    assert str(_table().c[column].server_default.arg) == expected


# --------------------------------------------------------------------------- #
# Migration — purely additive
# --------------------------------------------------------------------------- #
def test_revision_chain(migration):
    assert migration.revision == "e3b9c6a1d27f"
    assert migration.down_revision == "c7a4e0d51b83"
    assert callable(migration.upgrade) and callable(migration.downgrade)


def _upgrade(code: str) -> str:
    return code.split("defdowngrade")[0]


def test_migration_writes_no_data(code):
    """Every data write in Alembic goes through op.execute or op.bulk_insert."""
    upgrade = _upgrade(code)
    assert "op.execute" not in upgrade, "the migration must not run SQL against data"
    assert "bulk_insert" not in upgrade


def test_migration_touches_no_existing_core_table(code):
    upgrade = _upgrade(code)
    for table in ("tbl_products", "tbl_product_assets", "tbl_product_asset_mapping"):
        assert f'add_column("{table}"' not in upgrade
        assert f'alter_column("{table}"' not in upgrade
        assert f'drop_constraint("{table}"' not in upgrade


def test_migration_drops_nothing_on_upgrade(code):
    upgrade = _upgrade(code)
    assert "drop_constraint" not in upgrade
    assert "drop_column" not in upgrade
    assert "drop_index" not in upgrade


def test_variant_id_is_added_nullable(code):
    assert 'add_column("tbl_product_parts",sa.Column("variant_id",postgresql.UUID(as_uuid=True),nullable=True)' in code


def test_migration_declares_no_fk_to_users(code):
    assert "tbl_users" not in code


def test_migration_fk_delete_rules(code):
    assert '"fk_model_variants_product"' in code
    assert code.count('ondelete="SETNULL"') == 2  # whitespace is squashed
    assert '"fk_parts_model_variant"' in code
