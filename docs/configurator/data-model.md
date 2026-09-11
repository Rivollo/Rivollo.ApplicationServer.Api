# Product Configurator — Data Model

> Status: **FINALIZED design, not implemented.** No table, column, ORM model, or migration
> described here exists yet. Facts about the *existing* schema are tagged **CONFIRMED**;
> design choices are **PROPOSED**; anything that cannot be settled from this repository is
> **NEEDS VERIFICATION**.

See [architecture.md](architecture.md) for context and [decisions.md](decisions.md) for the
ADRs and open questions this document depends on.

**Phase-1 shape: exactly three tables.** `tbl_product_parts`, `tbl_part_options`,
`tbl_part_option_textures`. No normalized material table, no separate texture-option table,
no `option_type` / `source_type` discriminator column.

---

## 1. Existing schema, as observed

### 1.1 Conventions (CONFIRMED)

| Convention | Value | Evidence |
|---|---|---|
| Table prefix | `tbl_` | 30 of 33 `__tablename__` values in `app/models/models.py` |
| Primary key | `UUID`, `server_default gen_random_uuid()` + Python `default=uuid.uuid4` | `UUIDMixin` `models.py:34-35`; `6317c2563d0b:52+` |
| `pgcrypto` extension | already created by the initial migration | `6317c2563d0b:37` |
| Audit columns | `created_by`, `created_date`, `updated_by`, `updated_date` | `AuditMixin`, `models.py:38-60` |
| `created_date` | `TIMESTAMP(timezone=True)`, `server_default=now()`, NOT NULL | `models.py:57-59` |
| `updated_date` | `TIMESTAMP(timezone=True)`, nullable | `models.py` |
| `created_by`/`updated_by` | `UUID`, **no declared ForeignKey** | `models.py:38-56` |
| Soft delete | `deleted_at TIMESTAMPTZ` via `SoftDeleteMixin` | `models.py:84-85` |
| Active flag | `isactive BOOLEAN` (one word, no underscore) | `tbl_product_asset_mapping`, `tbl_hotspot_type` |
| JSON columns | `JSONB` with `server_default=text("'[]'::jsonb")`, **no CHECK constraints** | `ProductColorVariant.overrides`, `models.py:860-862` |
| "One default per parent" | partial unique index | `ux_color_variants_one_default`, `sql/create_color_variants.sql:79-81` |

**CONFIRMED — the `created_by` FK subtlety.** Those columns *are* foreign keys to
`tbl_users(id)` in the database (43 of them, `ON DELETE SET NULL` since revision
`e7a15c93f0b2`) but are deliberately **not declared** in the ORM. `migrations/env.py`
installs an `include_object` hook that refuses to autogenerate foreign keys at all, because
autogenerate would otherwise emit DDL dropping every one of them. Read
`migrations/env.py:60-86` before touching anything in this area.

### 1.2 Tables the Configurator reads (CONFIRMED)

```
tbl_products
  id                UUID PK
  created_by        UUID          ← ownership check hangs off this
  deleted_at        TIMESTAMPTZ   ← soft delete
  name, status, ...

tbl_product_assets
  id                UUID PK       ← candidate glb_version value, see §7
  asset_id          INTEGER       ← FORMAT id: 9 = GLB, 11 = USDZ, 17 = Draco glTF zip
  image             TEXT          ← the URL, despite the column name
  size_bytes        BIGINT

tbl_product_asset_mapping
  productid         UUID          ← FK → tbl_products, ON DELETE CASCADE (b3f8d21c4a76)
  product_asset_id  UUID          ← no declared FK (commented out, models.py:938-940)
  isactive          BOOLEAN
  created_date      TIMESTAMPTZ   ← "newest active mesh wins" ordering key

tbl_asset            (note: singular; a lookup table)
  id                INTEGER PK    ← the FORMAT id above
  assetid           INTEGER       ← the KIND: 1 = image, 2 = 3D model
  name              TEXT
```

The mesh URL for a product is resolved by joining these three, filtering
`asset_id == 9 AND isactive`, ordering `created_date DESC`, taking the first row
(`app/database/color_variant_repo.py:43-68`).

**CONFIRMED caveat.** `tbl_products.model_asset_id -> tbl_assets` still exists in the ORM but
`tbl_assets` **does not exist in the live database** (`sql/create_color_variants.sql:152-153`,
`migrations/env.py:80-83`). Do not use that relation.

### 1.3 Existing colour-variant tables (CONFIRMED)

`tbl_product_color_variants` and `tbl_variant_assets` exist, created by the hand-run script
`sql/create_color_variants.sql`. They are **not** in the Alembic chain. The Configurator does
not read, write, or modify them. See
[architecture.md §9](architecture.md#9-relationship-to-the-existing-colour-variant-feature).

### 1.4 🔴 The account-purge schema contract (CONFIRMED — governs this design)

`ACCOUNT_PURGE_JOB_HANDOFF.md` §25 defines a **schema contract** between this repository and
`Rivollo.AccountPurge.Job`, a production job in a different repository. The contract is the
database schema itself; there is no runtime call in either direction. A contract check runs
**before every purge** and aborts the entire run on any mismatch. Assertion 9:

> 🔴 **No unexpected new FK references `tbl_users` or `tbl_products`**
>
> Assertion 9 is the most valuable: it catches a table added in the Application Server that
> this job does not know about.

There is direct precedent for designing around this. `app/models/login_otp.py:19-33`
documents omitting a `tbl_users` FK specifically because *"assertion 9 aborts the entire run
if an unexpected new foreign key references tbl_users. A user_id FK here would break a
production job in another repository."*

**Three rules follow, and they are load-bearing for every table below:**

1. **Configurator tables declare NO foreign key from `created_by` / `updated_by` to
   `tbl_users`** — neither in the ORM nor in the migration. They are plain `UUID` columns,
   exactly as `AuditMixin` already declares them. This keeps assertions 5 and 9 clean.
2. **`tbl_product_parts.product_id → tbl_products` IS a new FK to `tbl_products`.** It is
   kept, because it is required — the purge deletes products before their owner, and without
   `ON DELETE CASCADE` that `DELETE` fails outright. This is the same reason `b3f8d21c4a76`
   exists. **The answer to assertion 9 is to update the job, not to remove the FK.**
3. **The purge job must be updated before production deployment** — see §13.

§25 "Change protocol" states it directly: *any* migration touching `tbl_products` or its
dependents **must** be accompanied by a review of that job's contract check and deletion
order. Recorded as [ADR-010](decisions.md#adr-010).

---

## 2. Configurator entities

```
   tbl_products                     (existing, unchanged)
        │ 1
        │                           ON DELETE CASCADE
        │ N
   tbl_product_parts                "Seat", "Backrest", "Legs"
        │  · owns a set of glTF material indices (JSONB)
        │  · pinned to a specific GLB version
        │ 1
        │                           ON DELETE CASCADE
        │ N
   tbl_part_options                 "Black", "Red", "Walnut", "Herringbone.png"
        │  · holds a recipe (JSONB) — recolour OR uploaded image
        │  · carries is_default and the bake lifecycle
        │ 1
        │                           ON DELETE CASCADE
        │ N
   tbl_part_option_textures         one baked texture per affected material index
           · material_index + CDN url
```

Naming follows the existing `tbl_` prefix and plural-noun convention exactly. No deviation
from repository convention is required.

---

## 3. `tbl_product_parts` (PROPOSED)

A seller-defined, persisted, configurable region of a product.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | `gen_random_uuid()` | PK |
| `product_id` | `UUID` | no | — | FK → `tbl_products(id)` `ON DELETE CASCADE` — **the only new FK to `tbl_products`**, see §1.4 |
| `name` | `TEXT` | no | — | seller-facing, 1-100 chars (enforced in the schema layer) |
| `slug` | `TEXT` | no | — | URL-safe, unique per product, derived server-side |
| `material_indices` | `JSONB` | no | `'[]'::jsonb` | array of non-negative ints — the glTF material indices this part owns. **Validated in `PartService`, not by a CHECK.** See §9 |
| `material_type` | `TEXT` | yes | `NULL` | `'fabric' \| 'wood' \| 'metal' \| 'leather' \| 'plastic'` — informational, drives editor defaults |
| `order_index` | `INTEGER` | no | `0` | display order in editor and viewer |
| `shopper_selectable` | `BOOLEAN` | no | `true` | false = seller-only, hidden from the public payload |
| `glb_version` | `TEXT` | no | — | identity of the GLB this part was authored against — see §7 |
| `isactive` | `BOOLEAN` | no | `true` | matches repo convention |
| `created_by` | `UUID` | yes | — | audit — **NO FK to `tbl_users`** (§1.4) |
| `created_date` | `TIMESTAMPTZ` | no | `now()` | audit |
| `updated_by` | `UUID` | yes | — | audit — **NO FK to `tbl_users`** (§1.4) |
| `updated_date` | `TIMESTAMPTZ` | yes | — | audit |

**There is no `default_option_id` column.** The default lives on the option as `is_default`,
guaranteed by a partial unique index — see §4 and [ADR-011](decisions.md#adr-011).

### Constraints

```sql
CONSTRAINT uq_parts_product_slug  UNIQUE (product_id, slug)
CONSTRAINT ck_parts_material_type CHECK (material_type IS NULL OR material_type IN
                                    ('fabric','wood','metal','leather','plastic'))
```

**No JSONB CHECK constraints.** `jsonb_typeof(...) = 'array'` and
`jsonb_array_length(...) > 0` would duplicate Pydantic validation that must exist regardless,
and no existing JSONB column in this repository carries either —
`ProductColorVariant.overrides` has none. Shape validation belongs in the schema layer;
`ck_parts_material_type` is kept because it guards a value the service could plausibly get
wrong, matching the precedent of `ck_variant_swatch_hex`.

### Indexes

```sql
CREATE INDEX ix_parts_product_order  ON tbl_product_parts (product_id, order_index);
CREATE INDEX ix_parts_product_active ON tbl_product_parts (product_id)
    WHERE isactive AND shopper_selectable;
```

**No GIN index on `material_indices`.** GIN-on-JSONB does have precedent here
(`ix_analytics_events_payload_gin`, `6317c2563d0b:261-266`), but the overlap check in §9 loads
all of a product's parts anyway — a single-digit row count reached through
`ix_parts_product_order`. An index over that set earns nothing.

---

## 4. `tbl_part_options` (PROPOSED)

A shopper-selectable appearance for one part. **One option model covers both a generated
recolour and an uploaded texture** — see §5.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | `gen_random_uuid()` | PK |
| `part_id` | `UUID` | no | — | FK → `tbl_product_parts(id)` `ON DELETE CASCADE` |
| `name` | `TEXT` | no | — | "Charcoal", 1-100 chars |
| `slug` | `TEXT` | no | — | unique per part, derived server-side |
| `swatch_hex` | `TEXT` | no | — | `#RRGGBB`, the picker dot; no model download needed to render the UI. **Required for every method, including `image`** — see §5.3 |
| `recipe` | `JSONB` | no | `'{}'::jsonb` | how the appearance is produced — §5 |
| `recipe_hash` | `TEXT` | no | — | SHA-256 over canonical recipe + `glb_version` + baker version; the idempotency key |
| `order_index` | `INTEGER` | no | `0` | display order |
| `is_default` | `BOOLEAN` | no | `false` | the look loaded first for this part — at most one per part, enforced by a partial unique index |
| `isactive` | `BOOLEAN` | no | `true` | publish toggle |
| `bake_status` | `TEXT` | no | `'pending'` | `pending \| baking \| completed \| failed` — §8 |
| `bake_error` | `TEXT` | yes | `NULL` | truncated failure message, seller-visible |
| `bake_started_at` | `TIMESTAMPTZ` | yes | `NULL` | **required for stale detection** — §8.2 |
| `bake_completed_at` | `TIMESTAMPTZ` | yes | `NULL` | |
| `bake_attempts` | `INTEGER` | no | `0` | retry accounting |
| `created_by` | `UUID` | yes | — | audit — **NO FK to `tbl_users`** (§1.4) |
| `created_date` | `TIMESTAMPTZ` | no | `now()` | audit |
| `updated_by` | `UUID` | yes | — | audit — **NO FK to `tbl_users`** (§1.4) |
| `updated_date` | `TIMESTAMPTZ` | yes | — | audit |

### Constraints

```sql
CONSTRAINT uq_options_part_slug   UNIQUE (part_id, slug)
CONSTRAINT ck_options_swatch_hex  CHECK (swatch_hex ~* '^#[0-9A-F]{6}$')
CONSTRAINT ck_options_bake_status CHECK (bake_status IN
                                    ('pending','baking','completed','failed'))
```

`ck_options_bake_status` is what makes the state model non-arbitrary at the database level.
The existing colour-variant table has the equivalent constraint
(`sql/create_color_variants.sql:61-62`) but its ORM column is a plain `Text` and the response
schema widens it to `BakeStatus | str` (`app/schemas/color_variants.py:148`) — the `| str`
arm defeats the literal, so no client can rely on the union. **Do not repeat that widening.**
Type it as the `Literal` alone.

**No `ck_options_recipe_obj`.** Recipe shape is a Pydantic concern; see §3 for the reasoning.

### Indexes

```sql
CREATE INDEX ix_options_part_order ON tbl_part_options (part_id, order_index);

-- At most ONE default per part, enforced by the database rather than by code.
-- Mirrors ux_color_variants_one_default (sql/create_color_variants.sql:79-81).
CREATE UNIQUE INDEX ux_part_options_one_default ON tbl_part_options (part_id)
    WHERE is_default;

CREATE INDEX ix_options_stale ON tbl_part_options (bake_started_at)
    WHERE bake_status = 'baking';
```

`ix_options_stale` is a partial index sized to the number of in-flight bakes — normally a
handful of rows — which makes the reaper sweep in §8.2 effectively free.

### Default-option rules

- `is_default` may only be `true` on an option that is `isactive` **and**
  `bake_status = 'completed'`. There is no column recording a *deferred* intent to become the
  default, and none is wanted — see the two setters below.
- **Automatic, first-baked-wins.** When an option's bake reaches `completed`, if the owning
  part has no default, that option becomes the default. This is how a part gets a working
  default without a second API call, and it is why `set_as_default` is not accepted on
  create. **If the part already has a default, a completing bake never replaces it** —
  promotion fills a vacancy, it does not compete for an occupied slot.
- **Explicit,** via `PATCH /options/{id}` with `set_as_default: true`, once the option is
  active and completed.
- Setting a new default clears the previous one in the same transaction, mirroring
  `repo.clear_default` / `color_variant_service.py:197-204`.
- Deactivating the current default is rejected with `400` — set another default first,
  mirroring `color_variant_service.py:189-195`.
- Deleting the current default promotes the next suitable option — lowest `order_index` among
  options that are `isactive` and `completed` — mirroring
  `color_variant_service.delete_variant` (`color_variant_service.py:250-263`). If no option
  qualifies, the part is left with no default and the shopper payload omits it (§9 of
  [api-spec.md](api-spec.md#9-shopper-api)).
- **API responses continue to expose `default_option_id` on the part.** It is a *computed*
  field — `next((o.id for o in part.options if o.is_default), None)` — never a stored column.
  In the shopper payload it is computed over the SURVIVING options and returns `null` when the
  configured default was filtered out; it is never substituted with another option.

---

## 5. Recipe format (PROPOSED)

Stored in `tbl_part_options.recipe`. It describes *how* to derive the option's appearance. It
is the contract the frontend preview and the backend bake both implement.

### 5.1 Generated recolour

```json
{
  "version": 1,
  "method": "luminance",
  "color": "#C0182B",
  "brightness": 1.0,
  "overrides": [
    { "material_index": 3, "method": "factor", "color": "#8A8A8A" }
  ]
}
```

### 5.2 Uploaded texture

```json
{
  "version": 1,
  "method": "image",
  "image_url": "https://cdn.example.net/uploads/users/{user_id}/uploads/{uuid}/herringbone.png"
}
```

### 5.3 Field rules

| Field | Type | Rules |
|---|---|---|
| `version` | int | recipe schema version; currently `1`. Reject unknown versions. |
| `method` | enum | `factor \| luminance \| remap \| image`. **`auto` is never persisted** — it is resolved to a concrete method at save time. |
| `color` | string | `#RRGGBB`, normalised uppercase before storage. Required for `factor`/`luminance`/`remap`; **ignored for `image`**. |
| `brightness` | float | `0.1 <= b <= 2.0`, HSL lightness multiplier. **Ignored for `image`**. |
| `overrides` | array | optional per-material-index deviations, for parts whose materials should not all take the same treatment |
| `image_url` | string | **Required for `image`, rejected for every other method.** Must pass the ownership check below. |

**`swatch_hex` is required for `image` options.** It is `NOT NULL` on the table and, for the
recolour methods, defaults server-side to `recipe.color`. `image` options have no
`recipe.color` to default from, so the API requires it explicitly rather than deriving an
average — see [api-spec.md §7](api-spec.md#7-options).

### 5.4 Why `image` is a method and not an `option_type` column

`recipe.method` is **already** the behavioural discriminator: `factor`, `luminance` and
`remap` dispatch to different code paths today (`glb_recolor.py:436-464`). Adding a fourth
value to an enum inside an existing JSONB document is not a schema change, adds no column, no
table, and no join.

An `option_type` / `source_type` column would be a *second*, redundant discriminator that
could contradict `recipe.method`. It is deliberately **not** introduced. If a future
requirement needs to query options by kind without opening the JSONB, that is the moment to
reconsider — not before.

### 5.5 `image_url` ownership validation (🔴 NEW security requirement)

Before an `image` recipe is persisted, `OptionService` **must** verify that `image_url`:

1. is absolute and begins with `settings.CDN_BASE_URL`;
2. resolves inside the configured uploads container;
3. sits under **this seller's own** upload namespace —
   `users/{current_user.id}/uploads/…`, the path shape written by
   `storage_service.upload_file_content` (`app/services/storage.py:164-175`) and confirmed in
   `ACCOUNT_PURGE_JOB_HANDOFF.md` §14;
4. has an allowed image extension.

An unvalidated URL is an SSRF vector (the baker fetches it server-side) and a cross-tenant
hotlink vector. Reject with `400`. **Never** dereference the URL to decide whether it is
acceptable.

### 5.6 Uploaded images are copied, not referenced

The bake for an `image` option **copies** the uploaded file into the Configurator storage
namespace before recording it:

```
users/{user_id}/uploads/{uuid}/herringbone.png          ← seller's upload, left untouched
        │ fetch + validate + normalise
        ▼
configurator/{product_id}/{glb_version}/{option_id}/{material_index}-{recipe_hash}.png
        │
        ▼
tbl_part_option_textures row
```

Referencing the seller's upload directly would mean deleting an option purges a blob under
`users/{user_id}/uploads/…` that may be referenced by another option, another product, or
nothing at all. Copying makes the Configurator the unambiguous owner of everything it
purges, and keeps the lifecycle in §10 correct.

Uploaded-texture options otherwise use **the same option, bake and texture lifecycle** as
every other option: same table, same `recipe_hash`, same `bake_status`, same staleness rule.

The recipe is deliberately a JSONB document rather than columns: it is read whole, written
whole, never queried by field, and its shape will evolve as methods are added. `version`
makes that evolution explicit.

### 5.7 Resolving `auto`

The existing implementation resolves `"auto"` at save time so that "the preview the seller
approved and the file we bake are produced by the same rule"
(`app/services/color_variant_service.py:107-110`). The Configurator keeps that rule and closes
the hole in it: `resolve_methods` currently swallows inspection failure and returns the
overrides with `"auto"` still in them (`variant_bake_service.py:152-155`), so an unresolved
`auto` can reach the database despite the invariant. The Configurator **fails the write**
rather than persisting an unresolved method.

`auto` is meaningless for `image` and is rejected alongside it.

**Ambiguity is an error, not a guess.** A part may own several material indices whose albedo
suggests different methods — one near-white material suggesting `factor` beside a mid-tone
one suggesting `luminance`. There is no specified rule for choosing between them, and
inventing one (first wins, majority wins) would silently give the seller a treatment they did
not ask for on some of the part's materials. When `auto` does not resolve to exactly one
method across the part's materials, the write is **rejected with `400`** naming the candidates,
and the seller re-sends a concrete `method`. A single-material part, and every per-material
`override`, always resolve unambiguously.

---

## 6. `tbl_part_option_textures` (PROPOSED)

One baked texture, for one material index, belonging to one option. Structure unchanged from
the original proposal.

| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | `UUID` | no | `gen_random_uuid()` | PK |
| `option_id` | `UUID` | no | — | FK → `tbl_part_options(id)` `ON DELETE CASCADE` |
| `material_index` | `INTEGER` | no | — | the glTF material whose base colour this texture replaces |
| `url` | `TEXT` | no | — | CDN URL served to clients |
| `blob_url` | `TEXT` | yes | `NULL` | raw Azure blob URL, for reprocessing/cleanup |
| `content_type` | `TEXT` | no | — | `image/png` or `image/jpeg` |
| `width` / `height` | `INTEGER` | yes | `NULL` | for the viewer's memory budgeting |
| `size_bytes` | `BIGINT` | yes | `NULL` | |
| `recipe_hash` | `TEXT` | no | — | must equal the parent option's; if not, the texture is stale |
| `created_by` | `UUID` | yes | — | audit — **NO FK to `tbl_users`** (§1.4) |
| `created_date` | `TIMESTAMPTZ` | no | `now()` | audit |
| `updated_by` | `UUID` | yes | — | audit — **NO FK to `tbl_users`** (§1.4) |
| `updated_date` | `TIMESTAMPTZ` | yes | — | audit |

### Constraints

```sql
CONSTRAINT uq_option_texture_material UNIQUE (option_id, material_index)
CONSTRAINT ck_option_texture_index    CHECK (material_index >= 0)
CONSTRAINT ck_option_texture_mime     CHECK (content_type IN ('image/png','image/jpeg'))
```

### Index

```sql
CREATE INDEX ix_option_textures_option ON tbl_part_option_textures (option_id);
```

### The staleness rule

`recipe_hash` on the texture must equal `recipe_hash` on the parent option. When it does not,
the file was baked from a superseded recipe and **must not be served**. This mirrors the
existing, sound pattern at `app/services/color_variant_service.py:373-381`, where assets
whose hash no longer matches are filtered out of the response rather than served as the wrong
colour while a re-bake is in flight. Adopt it verbatim.

### How many texture rows an option has

One per material index the part owns **that has a base-colour texture**. A material with no
base-colour image can only be treated by `factor`, which sets `baseColorFactor` and produces
no file (`glb_recolor.py:440-443`). So `len(textures) <= len(part.material_indices)`, and an
option may legitimately have **zero** texture rows. That is correct, not a failed bake — see
[baking.md §3.2](baking.md#32-which-textures-a-bake-produces).

`image` options produce one row per material index in the part, since the uploaded image
replaces the base colour of each.

---

## 7. GLB identity and versioning

> **Status: [ADR-006](decisions.md#adr-006) is Proposed / Needs Verification.** The
> *column* below is settled and safe to build. The *value semantics* are not. Do not
> hardcode either strategy; the prefix scheme is the hedge.

### What is CONFIRMED

- No hash of GLB **bytes** exists anywhere in the repository.
- `variant_bake_service.compute_config_hash` hashes the source **URL string** plus the recipe
  (`variant_bake_service.py:72-86`). `model_cache._key` hashes the **URL string**
  (`model_cache.py:53`). Neither proves anything about file content.
- Every observed mesh write path inserts a **new** `ProductAsset` row with a randomised blob
  name (`_sanitize_filename` appends 5 hex chars, `storage.py:60-63`). No endpoint replaces a
  product's GLB in place.
- The codebase nonetheless mutates an asset row's URL in place for a replace operation
  elsewhere — `primary_asset.image = blob_url` for `asset_id == 1`
  (`product_service.py:2140`).
- Every upload passes `overwrite=True`.
- `get_product_model_url` returns the **newest active** mesh row, so the "current" GLB can
  change while every existing row stays untouched.

### What follows

URL-as-identity is *presently* accurate and *not* guaranteed. The randomised suffix is
collision avoidance, not policy.

### PROPOSED

`glb_version TEXT NOT NULL` on every part, set server-side at creation, validated on every
write and every bake. Two candidate value sources:

| Option | Value | Pros | Cons |
|---|---|---|---|
| **A** — mesh asset id | `asset:{ProductAsset.id}` of the resolved `asset_id == 9` row | free; already a stable per-file database identity; no download | proves the *row* is the same, not the *bytes*; defeated by an in-place `.image` mutation |
| **B** — content hash | `sha256:{hex}` of the GLB bytes | proves byte identity; survives any storage-layer change | requires downloading the GLB once per version; needs a cache for the result |

**Phase 1 uses A**, with the column **prefix-discriminated** (`asset:` / `sha256:`) so
upgrading to B is a data migration, not a schema change. **Under no circumstances store a
hash of the URL string** — the existing code's use of exactly that pattern is the mistake this
section exists to avoid.

### Version mismatch behaviour

When a part's `glb_version` no longer matches the product's current GLB:

- `GET` endpoints return the part with `glb_stale: true` (computed, not stored).
- `POST`/`PATCH` on that part → `409 Conflict`, telling the seller to re-map materials.
- Bake requests → `409 Conflict`.
- The shopper payload **omits** stale parts entirely rather than showing a shopper an option
  that paints the wrong mesh.

---

## 8. Bake state

### 8.1 The lifecycle

```
                  ┌─────────┐
   option created │ pending │
   or recipe      └────┬────┘
   changed             │  runner picks it up
                       ▼
                  ┌─────────┐
                  │ baking  │──── bake_started_at set here
                  └────┬────┘
                       │
            ┌──────────┴───────────┐
            ▼                      ▼
       ┌───────────┐          ┌────────┐
       │ completed │          │ failed │
       └───────────┘          └───┬────┘
                                  │ retry / POST .../bake
                                  └──▶ pending
```

Constrained by `ck_options_bake_status` at the database level and by a Pydantic `Literal` in
the schema layer. Not a free string.

**Note the naming difference from the existing feature.** `sql/create_color_variants.sql`
uses `ready`; the Configurator uses `completed`. The two features have separate tables, so
there is no conflict — but do not copy-paste the colour-variant `BakeStatus` literal.

`image` options traverse the identical lifecycle. Their bake is short (fetch, validate,
normalise, copy, record) but it is a real bake with a real status, not a special case.

### 8.2 Stale bake detection (mandatory)

**CONFIRMED problem.** The existing bake service commits `bake_status = 'baking'`
(`variant_bake_service.py:242-244`), then performs 30-90 seconds of download, recolour and
upload before the commit that sets the terminal status. There is no `bake_started_at`, no
reaper, and no startup sweep. A replica recycled inside that window leaves the row in
`baking` forever, which the UI renders as a permanent spinner.

**PROPOSED — three mechanisms, all cheap:**

1. **`bake_started_at`** is set in the same transaction that sets `baking`.
2. **A lifespan sweep** on application startup flips any row with
   `bake_status = 'baking' AND bake_started_at < now() - interval '15 minutes'` to `failed`
   with `bake_error = 'Bake interrupted; please retry.'`. This turns a permanent spinner into
   a retryable error. It hooks into the existing `lifespan` context manager in
   `app/main.py:87-118`, alongside the deactivation loop already there.
3. **A periodic sweep** on the same 5-minute cadence as `_deactivation_loop`
   (`app/main.py:62-84`), so a bake lost mid-session recovers without waiting for a deploy.

The timeout constant belongs in `app/core/config.py` next to the other feature flags.

**These five fields are sufficient.** The one race worth documenting: if a sweep flips a row
to `failed` and the original replica *later* finishes, it re-reads the row and compares
`recipe_hash` before writing (the guard at `variant_bake_service.py:265-273`). A late
completion therefore either writes a valid `completed` — the correct outcome — or discards
itself as superseded. No additional column is needed to make that safe.

A `bake_worker_id` is the field a queue-based Phase 2 would add. Omitting it is correct at
`Semaphore(1)`, in-process scale.

### 8.3 Idempotency

`recipe_hash` is the idempotency key. `POST /options/{id}/bake` is safe to call repeatedly:

- if `bake_status = 'completed'` and every texture's `recipe_hash` matches the option's →
  return the existing result, do no work;
- if `bake_status = 'baking'` and `bake_started_at` is recent → return the in-flight ticket,
  do not start a second bake;
- otherwise → set `pending` and enqueue.

`?force=true` bypasses the first check only, for the case where a blob was deleted out of
band.

Because `image_url` is part of the canonical recipe, re-uploading a different image produces a
different `recipe_hash` and correctly triggers a re-bake.

**What `recipe_hash` deliberately does NOT cover: the parent part's `material_indices`.** The
hash fingerprints the recipe, `glb_version` and the baker version — the inputs that decide what
a texture *looks like*. Material membership decides *how many* textures an option needs, which
is a different question. So changing a part's `material_indices` leaves every option's hash
untouched while genuinely invalidating their bakes, and the reset is therefore **explicit**:
`PartService` sets every option of the part back to `pending` and returns their ids as
`invalidated_option_ids`. Folding material membership into the hash would work too, but it
would make the hash mean two things at once and break the "same recipe, same bytes" reading
that the staleness rule in §6 depends on.

---

## 9. Material-index uniqueness: JSONB + a row lock

**The invariant.** Within a product, a given `material_index` belongs to **at most one**
active part. Two parts claiming material 3 means two options could paint the same mesh with
conflicting textures, and the viewer would show whichever loaded last. No business reason to
allow overlap has been identified, so overlap is forbidden.

**Decision: keep `material_indices` as JSONB. No trigger. No normalized table.**
See [ADR-012](decisions.md#adr-012).

| | JSONB + trigger | Normalized `tbl_product_part_materials` | **JSONB + row lock (chosen)** |
|---|---|---|---|
| Declarative DB guarantee | trigger (imperative) | plain `UNIQUE` ✅ | none |
| New FKs to `tbl_products` | 0 extra | **+1** — aggravates §1.4 | 0 extra |
| Alembic precedent | **zero triggers in the entire chain** | ordinary `create_table` | n/a |
| Visible to autogenerate | ❌ drift risk | ✅ | n/a |
| Table count | 3 | **4** | **3** |

The normalized table is more orthodox in the abstract, but here it costs a second foreign key
to `tbl_products` — directly worsening the assertion-9 problem in §1.4 — and breaks the
three-table shape. A trigger costs a pattern that has never existed in this migration chain
and that `migrations/env.py` cannot see, in a repository already burned by schema drift.

**Enforcement:**

1. **`PartService` takes a row lock on the product, then checks siblings, then writes — all in
   one transaction.** The lock is what a database constraint would otherwise have bought:
   protection against two concurrent writes both claiming index 3.

   ```python
   # PartService.create / update, inside the write transaction
   await db.execute(
       select(Product)
       .where(Product.id == product_id, Product.created_by == user_id,
              Product.deleted_at.is_(None))
       .with_for_update()
   )
   siblings = await configurator_repo.get_active_parts(db, product_id, exclude_id=part_id)
   # reject any intersection, naming the conflicting part
   ```

   The ownership check and the lock are the same query — no extra round trip.

2. **The service check also produces the error message**, which a constraint never could:
   `"Material index 3 already belongs to part 'Backrest'"`, not a constraint-violation string.

3. Every index is additionally validated as `0 <= i < material_count` for **that product's
   current GLB**, and the array is validated as non-empty, unique, and ≤ 64 entries — all in
   the Pydantic layer plus the service.

Parts per product are single-digit and the sibling load runs on `ix_parts_product_order`, so
the lock is held for microseconds.

**Revisit** the normalized table if bulk material→part querying appears, or if a second writer
(an import job, a bulk editor) is introduced.

---

## 10. Ownership, cascade, and blob lifecycle

```
tbl_users
   │ created_by  — plain UUID on Configurator tables, NO FK (§1.4)
   ▼
tbl_products ──── created_by is the ownership anchor
   │ ON DELETE CASCADE          ← the one new FK to tbl_products
   ▼
tbl_product_parts
   │ ON DELETE CASCADE
   ▼
tbl_part_options
   │ ON DELETE CASCADE
   ▼
tbl_part_option_textures
```

Deleting a product removes its entire configurator tree. Deleting a *user* leaves the tree
intact and the audit columns populated — the product's own deletion path governs, which is
exactly how the purge job already works (it deletes products before their owner).

**Soft delete.** `tbl_products` uses `deleted_at`. Configurator queries filter
`Product.deleted_at IS NULL` on every ownership check, matching
`app/api/routes/products.py:1347-1348`. Configurator tables themselves use `isactive` rather
than `deleted_at` — a hidden part is a publish decision, not a deletion.

**Blob cleanup.** Database `CASCADE` does not delete Azure blobs.
`PartService.delete` / `OptionService.delete` must purge texture blobs *before* removing rows,
following `variant_bake_service.purge_variant_assets` (`variant_bake_service.py:314-319`) via
`storage_service.delete_blob_by_cdn_url` — **CONFIRMED** to be the only delete method in
`storage.py`, and to have no enumeration counterpart at all
(`ACCOUNT_PURGE_JOB_HANDOFF.md` §14). A blob left behind is a cost leak; a row pointing at a
deleted blob is a broken viewer.

Because §5.6 copies uploaded images into the Configurator namespace, purging an option never
touches a seller's own upload.

---

## 11. ORM sketch (PROPOSED)

Place in `app/models/models.py` beside the existing colour-variant models, following their
style exactly.

```python
class ProductPart(UUIDMixin, AuditMixin, Base):
    __tablename__ = "tbl_product_parts"
    __table_args__ = (
        Index("ix_parts_product_order", "product_id", "order_index"),
        UniqueConstraint("product_id", "slug", name="uq_parts_product_slug"),
    )

    product_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("tbl_products.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    material_indices: Mapped[list[int]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    material_type: Mapped[Optional[str]] = mapped_column(Text)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    shopper_selectable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    glb_version: Mapped[str] = mapped_column(Text, nullable=False)
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))

    # NOTE: no default_option_id. The default lives on PartOption.is_default,
    # guaranteed by ux_part_options_one_default. See §4.

    product: Mapped["Product"] = relationship("Product")
    options: Mapped[list["PartOption"]] = relationship(
        "PartOption", back_populates="part",
        cascade="all, delete-orphan", lazy="selectin",
    )

    @property
    def default_option_id(self) -> Optional[uuid.UUID]:
        """Computed, never stored — the API exposes this; the database does not."""
        return next((o.id for o in self.options if o.is_default), None)
```

`PartOption` carries `is_default` plus the five bake columns and follows the same shape;
`PartOptionTexture` likewise. Use `lazy="selectin"` on both collection relationships, matching
`ProductColorVariant.assets` (`app/models/models.py:884-886`) — it is what keeps the list
endpoint to a bounded number of queries.

**`AuditMixin` supplies `created_by` / `updated_by` as plain UUID columns with no
`ForeignKey`. Do not add one.** See §1.4.

---

## 12. Migration requirements

**All Configurator DDL goes through Alembic.** Do not repeat
`sql/create_color_variants.sql` — a hand-run script whose tables no revision creates, so a
fresh environment brought up with `alembic upgrade head` does not have them.

**CONFIRMED constraints on how the migration must be written:**

- **Current head is `b8e2f4a10c73`** (`add_login_otps`). The chain is linear:
  `6317c2563d0b → c133643a9626 → 97c7878c3123 → d8b6b3c4d9f1 → f2a9c41d7b60 → a1c4e7f2b930
  → c5e81a7f3d94 → b3f8d21c4a76 → e7a15c93f0b2 → d94b62e8c1f5 → f61a03d7b8e4 → b8e2f4a10c73`.
  Re-check with `alembic heads` at implementation time — it will move.
- **Write it by hand.** `migrations/env.py:60-86` refuses to autogenerate foreign keys, and its
  own docstring calls autogenerate untrustworthy repo-wide because the schema has drifted from
  the chain. Explicit `op.create_foreign_key` / `op.create_table` are unaffected by the hook.
- **`pgcrypto` already exists** (`6317c2563d0b:37`). Do not re-create it;
  `server_default=sa.text("gen_random_uuid()")` is safe to use directly.
- **Declare no `created_by` / `updated_by` FK to `tbl_users`** (§1.4, assertion 9,
  `login_otp.py:19-33` precedent).
- Assert the `product_id → tbl_products` cascade rather than assuming it. `b3f8d21c4a76`
  exists precisely because the ORM and the database once disagreed about exactly this.
- Create `ux_part_options_one_default` as a **partial unique index**
  (`op.create_index(..., unique=True, postgresql_where=sa.text("is_default"))`), matching
  `ux_color_variants_one_default`.
- Follow the file style of `migrations/versions/97c7878c3123_add_hotspot_type_table.py`:
  module docstring with revision/date, typed `revision` / `down_revision` identifiers, both
  `upgrade()` and `downgrade()` implemented.
- Use `postgresql.UUID(as_uuid=True)` and `sa.TIMESTAMP(timezone=True)` with
  `server_default=sa.text("now()")`.

One revision creating all three tables is appropriate; they are a single cohesive unit and a
partial application helps nobody. There is **no circular FK to sequence** — removing
`default_option_id` (§4) eliminated it, so a single `upgrade()` body suffices.

`downgrade()` drops in reverse dependency order: textures, options, parts.

---

## 13. 🔴 Deployment dependency: the account-purge job

**This is a deployment blocker, not a design problem, and it must not be "solved" by removing
the `product_id` foreign key.**

Per §1.4 and `ACCOUNT_PURGE_JOB_HANDOFF.md` §25, `Rivollo.AccountPurge.Job` must be updated
**before** this migration reaches production. Required changes on that side:

1. **§4.2 table inventory** — add `tbl_product_parts`, `tbl_part_options`,
   `tbl_part_option_textures`.
2. **§7 cascade list ("From `DELETE FROM tbl_products`")** — add all three. They cascade
   transitively through `tbl_product_parts.product_id`.
3. **Schema contract check, assertion 9** — allow-list the new
   `tbl_product_parts.product_id → tbl_products` FK. Without this the check aborts every purge
   run.
4. **§15 blob deletion order** — add the Configurator prefix
   `{container}/configurator/{product_id}/…`. Note this is **product-scoped, not user-scoped**,
   the same shape §14 already flags 🔴 for `products/{product_id}/variants/…`, so it is deleted
   during *product* teardown rather than by user-prefix enumeration.
5. Confirm the job's own prefix-enumeration handles the extra path depth
   (`{glb_version}/{option_id}/`).

**Sequencing:** open this with the purge-job owner as soon as implementation starts. It is
cross-repository and is the longest-lead item in the plan.
