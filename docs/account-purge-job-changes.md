# Rivollo.AccountPurge.Job — changes required for API keys, layout generations and Shopify

**Audience:** whoever changes `Rivollo.AccountPurge.Job`.
**Why:** three new Application Server schema pieces add foreign keys to `tbl_users` and
`tbl_products`. The job's contract check **A14** ("no unexpected FKs reference users or
products") rejects any FK it does not know, so until the job is updated **every purge run aborts
at the contract check** — safely, before deleting anything, but no account is erased.
**Base branch:** `model-variants-purge/supriya` (it already carries D10 for the configurator and
must ship first or together).

---

## 1. What is new upstream

| Application Server revision | Table | FK | Rule |
|---|---|---|---|
| `a5c1e9d4b7f2` | `tbl_api_keys` | `user_id → tbl_users` | **CASCADE** |
| `b7d3f1a2c9e4` | `tbl_model_variant_generations` | `product_id → tbl_products` | **CASCADE** |
| `b7d3f1a2c9e4` | `tbl_model_variant_generations` | `accepted_variant_id → tbl_product_model_variants` | SET NULL |
| `c9e5a3b1d8f6` | `tbl_shopify_connections` | `user_id → tbl_users` | **CASCADE** |
| `c9e5a3b1d8f6` | `tbl_shopify_connections` | `api_key_id → tbl_api_keys` | CASCADE |
| `c9e5a3b1d8f6` | `tbl_shopify_products` | `user_id → tbl_users` | **CASCADE** |
| `c9e5a3b1d8f6` | `tbl_shopify_products` | `rivollo_product_id → tbl_products` | **CASCADE** |
| `c9e5a3b1d8f6` | `tbl_shopify_product_variants` | `shopify_product_ref → tbl_shopify_products` | CASCADE |
| `c9e5a3b1d8f6` | `tbl_shopify_layouts` | `shopify_product_ref → tbl_shopify_products` | CASCADE |

**Bold** rows reference `tbl_users` / `tbl_products`, so A14 sees them: **3 new user FKs, 2 new
product FKs.** None of the new tables declares an audit FK: `created_by` / `updated_by` are plain
UUIDs, so the 43 audit FKs (A09) are unchanged.

### How each table is erased — no new DELETE or UPDATE target

| Table | Reached by | At step |
|---|---|---|
| `tbl_model_variant_generations` | cascade from `tbl_products` | 6 |
| `tbl_shopify_products` → variants, layouts | cascade from `tbl_products` (and from `tbl_users`) | 6 (9) |
| `tbl_api_keys` → `tbl_shopify_connections` | cascade from `tbl_users` | 9 |
| `tbl_shopify_connections` | cascade from `tbl_users` and from `tbl_api_keys` | 9 |

PostgreSQL runs referential actions as the table owner, so **the purge role needs no new grant** —
the same reason `tbl_variant_assets` and the configurator tables work (D10). Step order is
unchanged.

`accepted_variant_id` is SET NULL, not RESTRICT: when step 6 cascades a product, its model
variants and generations are both removed, and a RESTRICT here could abort that cascade.

### Blobs — no new prefix needed

| Blob | Path | Swept by |
|---|---|---|
| Generation candidate GLB | media `{user_id}/{product_id}/model-variants/{generation_id}/candidate.glb` | prefix 1 `{user_id}/` |
| Generation source photo | uploads `users/{user_id}/uploads/…` (seller's own upload) | prefix 2 `users/{user_id}/` |
| Shopify images copied into Rivollo | uploads `users/{user_id}/uploads/{id}/shopify-<hash>.<ext>` | prefix 2 |
| Accepted variant GLB / thumbnail | media `{user_id}/{product_id}/model-variants/{variant_id}/…` | prefix 1 (unchanged, D10) |

API keys and Shopify rows hold no blobs.

---

## 2. Code changes

### 2.1 `src/rivollo_purge/db/expectations.py`

```python
USER_CASCADE_FKS: Final[tuple[tuple[str, str], ...]] = (
    ...existing 11...,
    # Integrations — Application Server a5c1e9d4b7f2, c9e5a3b1d8f6 (D11).
    ("tbl_api_keys", "user_id"),
    ("tbl_shopify_connections", "user_id"),
    ("tbl_shopify_products", "user_id"),
)

PRODUCT_CASCADE_FKS: Final[tuple[tuple[str, str], ...]] = (
    ...existing 11...,
    # Layout from photo (ADR-015) — Application Server b7d3f1a2c9e4 (D11).
    ("tbl_model_variant_generations", "product_id"),
    # Shopify integration (ADR-016) — Application Server c9e5a3b1d8f6 (D11).
    ("tbl_shopify_products", "rivollo_product_id"),
)

# Second-level FKs below the new tables. None references users or products, so
# A14 cannot see them; A22 asserts them so a RESTRICT/NO ACTION upstream cannot
# make a cascade from step 6 or step 9 fail mid-transaction.
# Format: (source table, source column, target table, required rule).
INTEGRATION_CHILD_FKS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("tbl_shopify_connections", "api_key_id", "tbl_api_keys", "CASCADE"),
    ("tbl_shopify_product_variants", "shopify_product_ref", "tbl_shopify_products", "CASCADE"),
    ("tbl_shopify_layouts", "shopify_product_ref", "tbl_shopify_products", "CASCADE"),
    ("tbl_model_variant_generations", "accepted_variant_id", "tbl_product_model_variants", "SET NULL"),
)
```

Add the four second-level tables to `CASCADE_TABLES` so A01 requires them to exist:

```python
CASCADE_TABLES: Final[tuple[str, ...]] = tuple(
    sorted(
        {t for t, _ in USER_CASCADE_FKS}
        | {t for t, _ in PRODUCT_CASCADE_FKS}
        | {"tbl_variant_assets"}
        | {t for t, _, _ in CONFIGURATOR_CASCADE_FKS}
        | {t for t, _, _, _ in INTEGRATION_CHILD_FKS}
    )
)
```

Add to `REQUIRED_COLUMNS` (bound by A22 and useful to A03):

```python
    # Integrations (D11).
    ("tbl_api_keys", "user_id", "uuid", False),
    ("tbl_shopify_connections", "user_id", "uuid", False),
    ("tbl_shopify_connections", "api_key_id", "uuid", False),
    ("tbl_shopify_products", "user_id", "uuid", False),
    ("tbl_shopify_products", "rivollo_product_id", "uuid", False),
    ("tbl_model_variant_generations", "product_id", "uuid", False),
    ("tbl_model_variant_generations", "accepted_variant_id", "uuid", True),   # SET NULL => nullable
```

Add to `HOT_PATH_INDEX_COLUMNS` (A19, advisory). All of these are indexed upstream:

```python
    ("tbl_api_keys", "user_id"),                        # ix_api_keys_user_created (leading)
    ("tbl_shopify_connections", "user_id"),             # ix_shopify_connections_user
    ("tbl_shopify_connections", "api_key_id"),          # ix_shopify_connections_api_key
    ("tbl_shopify_products", "user_id"),                # ix_shopify_products_user
    ("tbl_shopify_products", "rivollo_product_id"),     # ix_shopify_products_rivollo_product
    ("tbl_model_variant_generations", "product_id"),    # ix_generations_product_created (leading)
    ("tbl_model_variant_generations", "accepted_variant_id"),  # ix_generations_accepted_variant
```

Update the two docstrings that state counts: `expected_user_fks` "All **59** FKs referencing
tbl_users" (with `PaymentsPolicy.DELETE`), `expected_product_fks` "All **15** FKs referencing
tbl_products", and the `# §6.2 — the 13 FKs referencing tbl_products` comment → 15.

### 2.2 `src/rivollo_purge/db/contract.py` — new assertion A22

Same shape as A20:

```python
def a22_integration_child_fks(snap: SchemaSnapshot) -> AssertionResult:
    """FKs below the integration tables have the rules the deletion depends on (D11).

    They are reached only by cascade from users (step 9) or products (step 6);
    a RESTRICT or NO ACTION here aborts the account's transaction.
    """
    findings: list[str] = []
    for table, column, target, expected in X.INTEGRATION_CHILD_FKS:
        rule = _rule_of(snap, table, column, target)
        if rule is None:
            findings.append(f"no FK from {table}.{column} to {target}")
        elif rule != expected:
            findings.append(f"{table}.{column} -> {target} is {rule}, expected {expected}")
    total = len(X.INTEGRATION_CHILD_FKS)
    return _result(
        "A22",
        "integration tables cascade below users and products",
        Severity.REQUIRED,
        not findings,
        f"{total}/{total} correct",
        f"{len(findings)}/{total} wrong - integration rows could block or survive the purge",
        findings,
    )
```

Register it in `run_all(...)` next to A20/A21 and update the module docstring
("assertions A01..A15, A20..A22"). A10 and A14 need no code change: they read
`PRODUCT_CASCADE_FKS` / `expected_user_fks`.

### 2.3 Optional — `src/rivollo_purge/sql/select/inventory_counts.sql`

Only if the dry-run report should show what the cascades will remove. Cascade tables are not
counted today (the configurator tables aren't either), so this is **not required**:

```sql
    (SELECT count(*) FROM tbl_api_keys WHERE user_id = $1)                         AS tbl_api_keys,
    (SELECT count(*) FROM tbl_shopify_connections WHERE user_id = $1)              AS tbl_shopify_connections,
    (SELECT count(*) FROM tbl_shopify_products WHERE user_id = $1)                 AS tbl_shopify_products,
    (SELECT count(*) FROM tbl_model_variant_generations
      WHERE product_id = ANY($2::uuid[]))                                          AS tbl_model_variant_generations,
```

---

## 3. Tests

### 3.1 `tests/unit/test_expectations.py`

```python
    assert len(X.expected_user_fks(PaymentsPolicy.DELETE)) == 59   # was 56
    assert len(X.USER_CASCADE_FKS) == 14                           # was 11
    assert len(X.expected_product_fks()) == 15                     # was 13
    assert len(X.PRODUCT_CASCADE_FKS) == 13                        # was 11
```

Update the docstring "56 FKs reference tbl_users — 11 CASCADE, 2 SET NULL, 43 audit" to
"59 … 14 CASCADE …". Add: `INTEGRATION_CHILD_FKS` has 4 entries and none targets `tbl_users` or
`tbl_products` (so A14 and A22 never overlap).

### 3.2 `tests/fixtures/schema_compliant.sql` — append (after `tbl_product_model_variants`)

```sql
-- Integrations (Application Server a5c1e9d4b7f2, b7d3f1a2c9e4, c9e5a3b1d8f6; D11).
CREATE TABLE tbl_api_keys (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES tbl_users(id) ON DELETE CASCADE,
    key_hash text NOT NULL UNIQUE,
    created_by uuid, updated_by uuid
);
CREATE INDEX ix_api_keys_user_created ON tbl_api_keys (user_id);

CREATE TABLE tbl_model_variant_generations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id uuid NOT NULL REFERENCES tbl_products(id) ON DELETE CASCADE,
    accepted_variant_id uuid REFERENCES tbl_product_model_variants(id) ON DELETE SET NULL,
    candidate_glb_url text,
    created_by uuid, updated_by uuid
);
CREATE INDEX ix_generations_product_created ON tbl_model_variant_generations (product_id);
CREATE INDEX ix_generations_accepted_variant ON tbl_model_variant_generations (accepted_variant_id);

CREATE TABLE tbl_shopify_connections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES tbl_users(id) ON DELETE CASCADE,
    api_key_id uuid NOT NULL REFERENCES tbl_api_keys(id) ON DELETE CASCADE,
    shop_domain text NOT NULL,
    created_by uuid, updated_by uuid
);
CREATE INDEX ix_shopify_connections_user ON tbl_shopify_connections (user_id);
CREATE INDEX ix_shopify_connections_api_key ON tbl_shopify_connections (api_key_id);

CREATE TABLE tbl_shopify_products (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES tbl_users(id) ON DELETE CASCADE,
    rivollo_product_id uuid NOT NULL REFERENCES tbl_products(id) ON DELETE CASCADE,
    created_by uuid, updated_by uuid
);
CREATE INDEX ix_shopify_products_user ON tbl_shopify_products (user_id);
CREATE INDEX ix_shopify_products_rivollo_product ON tbl_shopify_products (rivollo_product_id);

CREATE TABLE tbl_shopify_product_variants (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    shopify_product_ref uuid NOT NULL REFERENCES tbl_shopify_products(id) ON DELETE CASCADE
);
CREATE TABLE tbl_shopify_layouts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    shopify_product_ref uuid NOT NULL REFERENCES tbl_shopify_products(id) ON DELETE CASCADE
);
```

(Only the columns the job binds are needed, matching how the fixture trims other tables.)

### 3.3 `tests/fixtures/schema_drifted.sql`

Add one drift A22 must catch, e.g.:

```sql
ALTER TABLE tbl_shopify_product_variants DROP CONSTRAINT tbl_shopify_product_variants_shopify_product_ref_fkey;
ALTER TABLE tbl_shopify_product_variants ADD FOREIGN KEY (shopify_product_ref)
    REFERENCES tbl_shopify_products(id) ON DELETE RESTRICT;
```

### 3.4 `tests/unit/test_contract_assertions.py`

- A22 passes on the compliant snapshot; fails on a RESTRICT, a NO ACTION and a missing FK;
  fails when `accepted_variant_id` is CASCADE instead of SET NULL.
- A14 passes with the three new user FKs and two new product FKs; still fails on an unrelated
  unknown FK (regression guard).
- A01 fails when any of the six new tables is missing.

### 3.5 `tests/fixtures/seed_purge.sql` + `tests/integration/test_purge_end_to_end.py`

Seed, for the purge target: one API key, one Shopify connection bound to it, one Shopify product
linked to one of the user's products with one variant and one layout, and one generation on that
product whose `accepted_variant_id` points at a model variant. Seed the same for a **second,
untouched user**. After `execute`, assert every seeded row of the target is gone, and every row
of the second user is intact.

---

## 4. `docs/DECISIONS.md` — add D11

```markdown
| D11 | **Integration tables are part of the contract** (A01, A10, A14, A22) | 2026-09-29 | Allow-lists
three FKs to `tbl_users` (`tbl_api_keys`, `tbl_shopify_connections`, `tbl_shopify_products`) and two to
`tbl_products` (`tbl_model_variant_generations`, `tbl_shopify_products`), all CASCADE; A22 asserts the
four second-level FKs below them. No new DELETE/UPDATE target, grant or blob prefix. Deploy together with
Application Server revisions a5c1e9d4b7f2, b7d3f1a2c9e4, c9e5a3b1d8f6. |
```

And a `## D11` section in the D10 format covering: what arrived upstream (section 1 above), why each
change is load-bearing (A14 would abort; A22 guards the cascades), what did not change (no new
target, grant or prefix — every row goes by cascade), and the deploy order below.

---

## 5. Deploy order

Same rule as D10 — **ship both sides between two nightly runs (00:00 UTC)**:

1. Merge and deploy the job change (on top of `model-variants-purge/supriya`, or after it).
2. Apply the Application Server tables (migrations `a5c1e9d4b7f2` → `b7d3f1a2c9e4` →
   `c9e5a3b1d8f6`, or the equivalent SQL script).
3. Run the job in `validate` mode; every assertion must pass.

Either side alone makes the next run abort at the contract check **before anything is deleted**
(the job without the tables fails A01; the tables without the job fail A14). The cost of a gap is
one day's erasure delay, not data.

---

## 6. Verify on the database after applying the tables

```sql
-- The five FKs A14 must now recognise.
SELECT conrelid::regclass AS src, a.attname AS col, confrelid::regclass AS target,
       CASE confdeltype WHEN 'c' THEN 'CASCADE' WHEN 'n' THEN 'SET NULL'
                        WHEN 'r' THEN 'RESTRICT' WHEN 'a' THEN 'NO ACTION' END AS on_delete
FROM pg_constraint c
JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey)
WHERE c.contype = 'f'
  AND confrelid IN ('tbl_users'::regclass, 'tbl_products'::regclass)
  AND conrelid::regclass::text IN ('tbl_api_keys', 'tbl_shopify_connections',
        'tbl_shopify_products', 'tbl_model_variant_generations')
ORDER BY 1, 2;
-- Expect 5 rows, all CASCADE.
```

---

## 7. Rollback

- Job side: revert the job to the previous image. It will then fail A14 while the tables exist,
  which aborts safely — so roll back the tables too, or keep the new job.
- Table side: drop in reverse order (`tbl_shopify_layouts`, `tbl_shopify_product_variants`,
  `tbl_shopify_products`, `tbl_shopify_connections`, `tbl_model_variant_generations`,
  `tbl_api_keys`), or downgrade the three migrations.
