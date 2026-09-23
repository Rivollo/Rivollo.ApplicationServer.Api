# Known bugs — found in passing, not fixed

Recorded while building Configurator model variants (ADR-014). None of these are in scope
for that work and none were changed. Each needs its own change and owner.

## Publish endpoint — `POST /products/{product_id}/publish` (`app/api/routes/products.py`)

1. **Unpublishing always fails with a 500.** The unpublish branch sets
   `product.status = ProductStatus.UNPUBLISHED`, but `ProductStatus` has no `UNPUBLISHED`
   member (`draft, queue, processing, ready, published, archived`), so it raises
   `AttributeError`. Even with that fixed, the response builder reads `now`, which is only
   assigned on the publish branch, so unpublish would then raise `UnboundLocalError`.
2. **No ownership check.** The product is looked up by id and `deleted_at IS NULL` only —
   any authenticated user can publish or unpublish any other seller's product. The
   Configurator resolves products through `created_by == current_user.id` and returns 404
   (ADR-008); this endpoint predates that.
3. **Two publish implementations.** `Rivollo.Viewer.Api` `PublishController` also writes
   `tbl_products.status`. Its `ConfiguratorRepository` compares status case-insensitively
   because the two writers do not agree on casing.

## Test suite

As of 2026-09-21 on `main` (`6090265`), 34 tests fail before any model-variant change:

- `tests/test_account_deletion.py` (14) and `tests/test_account_deletion_subscription_guard.py`
  (5) — `TypeError: AccountService.delete_account() got an unexpected keyword argument
  'password'`. The tests were not updated when password confirmation was removed from account
  deletion (commit `957fdd3`).
- `tests/test_pricing_endpoint.py` (15) — `ValueError: Invalid format string`, observed on
  Windows. Not investigated further.

## Environment drift — rivollo-dev-db

`alembic_version` reads `f61a03d7b8e4`, two revisions behind the chain, although the
objects created by `b8e2f4a10c73` (`tbl_login_otps`) and `c7a4e0d51b83` (the three
Configurator tables, with identical constraint and index names) already exist — they were
applied from the hand-run SQL files. `alembic upgrade head` fails at `c7a4e0d51b83` with
"relation already exists" until the database is stamped:

```bash
alembic stamp c7a4e0d51b83
```

Check production for the same drift before its next migration.

## Other repositories

- **`Rivollo.Viewer.Api/Rivollo.Viewer.Api/appsettings.json` contains committed credentials** —
  the PostgreSQL and ClickHouse connection strings, with passwords, and a JWT signing key.
  Rotate them and move them to environment variables / Key Vault. (Found while adding the
  `Configurator:EnableModelVariants` flag to that file; the values are not repeated here.)
- **`Rivollo.Converter.Service` sets the product to `READY` after every product-level USDZ
  conversion** (`update_product_status(db, product_id, "READY")` in `job.py`). A conversion
  that finishes after the product was published silently un-publishes it. Model-variant
  conversions (`--model-variant-id`) skip this call; the product-level path is unchanged.
