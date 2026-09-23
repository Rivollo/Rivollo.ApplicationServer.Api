# Product Configurator — Architecture Decision Records

Format: context → decision → status → consequences. A record is **Accepted** only when the
evidence supporting it exists in this repository. Anything depending on runtime behaviour,
Azure configuration, database contents, or the frontend codebase is **Proposed / Needs
Verification** until someone checks.

| ADR | Title | Status |
|---|---|---|
| [001](#adr-001) | Dedicated Configurator API/domain | **Accepted** |
| [002](#adr-002) | Original GLB as canonical model | **Accepted** |
| [003](#adr-003) | Texture baking instead of per-option GLB generation | **Accepted** |
| [004](#adr-004) | Seller-owned persisted Product Parts | **Accepted** |
| [005](#adr-005) | Preview / Bake separation | **Proposed / Needs Verification** |
| [006](#adr-006) | GLB identity and versioning strategy | **Proposed / Needs Verification** |
| [007](#adr-007) | Bake execution stays in-process behind a seam | **Accepted** |
| [008](#adr-008) | Product ownership enforced on every Configurator endpoint | **Accepted** |
| [009](#adr-009) | Alembic owns all Configurator DDL | **Accepted** |
| [010](#adr-010) | Account-purge schema contract is a deployment dependency | **Accepted** (with a **deployment blocker**) |
| [011](#adr-011) | `is_default` on the option, not `default_option_id` on the part | **Accepted** |
| [012](#adr-012) | Material-index uniqueness: JSONB + row lock, no trigger, no child table | **Accepted** |
| [013](#adr-013) | Uploaded textures are `recipe.method = "image"`, not an option type | **Accepted** |
| [014](#adr-014) | Model variants: extra shapes, added without changing the existing system | **Accepted** (with a **deployment blocker**, Q8) |

---

## ADR-001

### Dedicated Configurator API/domain

**Status: Accepted**

**Context.** Configurator logic must not be mixed into generic Product routes.
`app/api/routes/products.py` is 2,250 lines holding products, assets, backgrounds, currency
types, hotspot types, and the legacy `tbl_configurators` blob — a working demonstration of
what happens without a boundary.

The brief suggests `/api/v1/configurator`. **CONFIRMED**, the repository has no such
convention: `API_PREFIX` defaults to `""` (`app/core/config.py:10`), all 27 routers mount at
that bare prefix (`app/main.py:219-247`), paths are flat and resource-first, and the only
versioned mount in the codebase is `products_v2_router` at `/v2`. Adopting `/api/v1/...`
would make the Configurator the sole endpoint under a prefix nothing else uses.

**Decision.**

1. A dedicated domain: `app/api/routes/configurator.py` + `configurator_public.py`,
   `app/schemas/configurator.py`, `app/services/configurator/` (package),
   `app/database/configurator_repo.py`.
2. Its own routers with `tags=["configurator"]`, router-level auth dependencies.
3. **Flat, resource-first paths at the existing prefix** — `/products/{id}/parts`,
   `/parts/{id}/options` — consistent with hotspots, dimensions and colour variants.
4. **No new deployable.** A domain boundary is not a deployment boundary.
5. No duplication of auth, storage, CDN, product lookup, GLB inspection, or DB infrastructure.

**Alternative left open for review.** Mounting at `f"{_api_prefix}/configurator"` gives an
explicit namespace without inventing a version scheme. It costs one line in `main.py` and a
frontend base-URL change. Raise it at review if the team wants the stronger boundary; the
decision above is the lower-friction default, not a strong preference.

**Consequences.** Configurator changes touch Configurator files. Onboarding reads one
package. The trade is one more router in `main.py` and the discipline not to let product
routes grow Configurator logic. If the Configurator later needs its own scaling profile, the
package boundary is already where a split would cut.

---

## ADR-002

### Original GLB as canonical model

**Status: Accepted**

**Context.** The seller uploads one GLB. It is the geometry, UVs, normals, and
roughness/metallic maps — the expensive, quality-defining artifact. **CONFIRMED**, it is
resolved through `tbl_product_assets` / `tbl_product_asset_mapping` at `asset_id == 9`,
newest active row wins (`app/database/color_variant_repo.py:43-68`).

**CONFIRMED**, it is normally Draco-compressed: `ENABLE_DRACO_COMPRESSION` and
`ENABLE_GLTF_DRACO_PACKAGE` both default `True` (`app/core/config.py:292,306`), and
`_compress_mesh_for_storage` returns the compressed bytes as asset 9
(`app/services/product_service.py:63-81`).

**Decision.** The Configurator treats the original GLB as read-only and canonical. It never
rewrites geometry, never produces a modified copy for normal colour/material configuration,
and never becomes a second source of truth for the mesh. Configurator data attaches to it by
glTF **material index**, which is both the array index in the GLB and the index the browser
exposes as `model.materials[i]`.

**Consequences.** Geometry downloads once per shopper session. Mesh quality is preserved by
construction. Configurator data is coupled to the GLB's material ordering, which makes
[ADR-006](#adr-006) mandatory rather than optional. Any feature genuinely requiring different
geometry falls outside the Configurator.

**Verification still owed** (does not block the decision, blocks specific features):

- That `pygltflib` inspection works against a real Draco product GLB. Textures are not
  Draco-compressed, so it should — **NEEDS VERIFICATION**.
- That `accessor.min`/`max` on POSITION survive Draco, for `_material_centers`.
- That `indices.count` is readable for triangle counts. See Q4.

---

## ADR-003

### Texture baking instead of per-option GLB generation

**Status: Accepted**

**Context.** The existing colour-variant feature bakes a **complete GLB per colourway**
(`variant_bake_service._bake_bytes` → `glb_recolor.recolor` → `_rebuild_and_write`,
CONFIRMED). Every colourway is a full copy of the mesh.

At the sizes this application actually handles — `model_cache.py:5-6` documents typical Tripo
meshes at 40-80 MB — that is:

| | Whole-GLB per option | Texture per option |
|---|---|---|
| Stored, 60 MB model × 4 options | ~300 MB | ~68 MB |
| Transferred on a swatch click | ~60 MB | ~2 MB |
| Geometry re-parsed per click | yes | no |
| Combinatorics: 3 parts × 4 options | 64 GLBs if combinations are baked | 12 textures |

The last row is the decisive one. Parts multiply. Whole-model baking multiplies with them;
texture baking adds.

**Decision.** A Part Option bakes **one texture per affected material index**, stored as a
row in `tbl_part_option_textures` with a CDN URL. The viewer swaps
`material.map` per material index. No complete GLB is generated for any colour or material
combination.

**Consequences.**

- Storage and bandwidth scale with parts + options, not their product.
- Bakes are seconds, not minutes — no geometry decode/encode.
- The viewer must support runtime texture swapping by material index. This is a real frontend
  requirement, not a given; confirm it against the viewer's renderer.
- **Limits:** base colour and surface appearance only. Not normal maps, not
  roughness/metallic, not UVs, not geometry. Materials with no base-colour image use the
  `factor` path and produce no texture file at all — the option's `textures` array may be
  short or empty, and that is correct, not a bug.
- The `remap`/`luminance` maths is reused unchanged; only the container changes.

---

## ADR-004

### Seller-owned persisted Product Parts

**Status: Accepted**

**Context.** A glTF material index is a rendering detail. "Seat" is what a seller and a
shopper think in. AI mesh tools split one visual part across many materials, so the mapping
is genuinely many-to-one.

The existing engine already computes a `group_id`
(`app/services/color/glb_recolor.py:307-358`, CONFIRMED). It is tempting to treat it as the
Part. It must not be, for four reasons all visible in the code:

1. **Not persisted.** Recomputed on every `/materials` call.
2. **Not stable.** Depends on the GLB's material ordering and average colours; a re-upload
   can renumber every group.
3. **Heuristic, and transitively so.** Materials union if they share a base-colour image
   **or** their average colours are within `_GROUP_COLOR_DISTANCE = 42.0` RGB units. Union-find
   over a proximity relation is transitive: a model with a smooth grey ramp can collapse every
   material into one group.
4. **Unnamed.** A seller cannot call a group "Backrest".

**Decision.** `tbl_product_parts` is a first-class, seller-owned, persisted entity. It has a
name, a slug, an explicit `material_indices` array, an order, a visibility flag, and a
`glb_version`. Membership is validated server-side against the actual GLB.

`group_id` is surfaced only as `similarity_group_hint` on the materials response — renamed
precisely so no client mistakes it for identity — to pre-populate the editor's suggestions.
It is never stored on a part.

**Invariant:** within one product, a material index belongs to **at most one** part.
*Amended by [ADR-014](#adr-014):* within one **model** — the original (`variant_id IS NULL`)
or one extra model variant. Material indices are only meaningful against one GLB. No
business reason to allow overlap has been identified; overlap would let two options paint the
same mesh with conflicting textures, and the viewer would show whichever loaded last.
Enforced in the service (for the error message) *and* in the database (as the backstop) — see
[data-model.md §9](data-model.md#9-uniqueness-one-material-index-one-part) and Q5.

**Consequences.** Sellers do real authoring work once per product. The editor should make
that cheap by seeding from `similarity_group_hint`. Parts survive re-inspection, which the
computed groups do not.

---

## ADR-005

### Preview / Bake separation

**Status: Proposed / Needs Verification**

Two independent prerequisites are unverified. Both are outside this repository.

**Context.** Interactive colour selection needs sub-frame latency. A backend bake takes
30-90 seconds. These cannot be the same operation, but they must produce the same picture.

**Decision.**

1. **Preview** runs entirely in the browser. No backend request per interaction. For a baked
   option the frontend uses the baked texture URL directly — the fast, authoritative path.
   Client-side recolouring is needed only in the seller editor, for an option not yet baked.
2. **Bake** is asynchronous, produces persistent CDN-hosted textures, and is polled.
3. Both implement **one specification**, recorded in
   [baking.md §5](baking.md#5-frontendbackend-consistency).
4. `auto` is resolved to a concrete method **server-side at save time**, so preview and bake
   can never diverge because two heuristics disagreed. Unlike the existing implementation,
   resolution failure **fails the write** rather than persisting an unresolved `auto`
   (`variant_bake_service.py:152-155` swallows it today).
5. A shared golden-fixture test suite runs in **both** repositories against the same inputs
   and the same expected outputs.

**Blocking verification — Q1, CDN CORS.** Client-side recolouring reads texture pixels.
Cross-origin pixel readback taints the canvas and throws without
`Access-Control-Allow-Origin` on the CDN response plus `crossorigin="anonymous"` on the
loader. **CONFIRMED**: nothing in this repository configures blob or Front Door CORS. There is
no IaC — `.github/` holds only workflow YAML. `app/main.py:202-208` sets
`allow_origins=["*"]` on the **API**, which does not apply to CDN responses.

**Blocking verification — the `remap` divergence.** Reported: the frontend uses the absolute
darkest and brightest pixels; the backend uses the 2nd and 98th percentiles
(`glb_recolor.py:383`, CONFIRMED for the backend half). If true, preview and bake differ
visibly on exactly the assets `remap` exists for — a near-black texture where a handful of
specular pixels pin `max` but not the 98th percentile.

Recommendation: **standardise on the percentile form**, because outlier robustness is the
method's whole purpose, and approximate it on the GPU (coarse histogram, or a CPU pass on a
small mip). **Requires confirming the actual frontend implementation before either side
changes.** Do not silently pick one.

**Promote to Accepted when:** CORS is verified or configured, *and* the frontend `remap`
implementation is read and a single specification is agreed and written into
[baking.md §5.3](baking.md#53-the-remap-divergence--decision-required).

---

## ADR-006

### GLB identity and versioning strategy

**Status: Proposed / Needs Verification**

**Context.** Configurator data is a set of integer material indices. If the GLB changes, those
integers may point at different materials — the seller's "Seat" silently becomes the legs.
So every part must be pinned to the GLB it was authored against.

**CONFIRMED evidence, gathered rather than assumed:**

| Question | Answer | Evidence |
|---|---|---|
| Are GLB URLs immutable? | **In practice yes, by policy no.** | `_sanitize_filename` appends 5 random hex chars, `storage.py:60-63` — every upload lands at a new name |
| Does replacing a GLB create a new URL? | **Yes, in every observed path.** | every mesh write constructs a new `ProductAsset` row; there is no "replace GLB" endpoint |
| Can one URL point at different bytes over time? | **Nothing prevents it.** | every upload passes `overwrite=True`; and the codebase already mutates an asset URL in place for a replace — `primary_asset.image = blob_url` for `asset_id == 1`, `product_service.py:2140` |
| Is there an existing version or content hash? | **No.** | no hash of GLB bytes anywhere. `compute_config_hash` hashes the URL *string* (`variant_bake_service.py:85`); `model_cache._key` hashes the URL *string* (`model_cache.py:53`); `tbl_assets.checksum_sha256` exists as a column but that table is not in the live database and nothing writes it |
| Can a product's current GLB change without any row changing? | **Yes.** | `get_product_model_url` takes the newest active `asset_id == 9` row — adding a mesh silently changes the answer |

So URL-as-identity is *currently* accurate and *not* guaranteed. The randomised suffix is
collision avoidance, not an immutability policy, and an in-place-mutation precedent already
exists one asset kind over.

**Decision.**

1. Every part carries a `glb_version TEXT NOT NULL`, set server-side at creation.
2. It is validated on every write and every bake. Mismatch → `409 Conflict` on writes and
   bakes; `glb_stale: true` on seller reads; **omitted entirely** from the shopper payload.
3. The value is **prefix-discriminated** — `asset:<uuid>` or `sha256:<hex>` — so the strategy
   can be upgraded by data migration, not schema change.
4. **Phase 1 uses `asset:<ProductAsset.id>`** of the resolved mesh row: free, requires no
   download, and is a stable per-file database identity that the observed write paths never
   reuse.
5. **Never** use a hash of the URL string. It proves nothing about file contents, and the
   existing code's use of exactly that pattern is the mistake this ADR exists to avoid.

**Needs verification before promoting to Accepted:**

- Whether any process outside this repository — the Container Apps USDZ job, the Viewer API,
  manual ops, a storage lifecycle policy — overwrites a mesh blob at a stable path.
- Whether any planned "replace model" feature will mutate `ProductAsset.image` in place, the
  way `PUT /products/{id}/original-image` already does for images. If so, `asset:<uuid>` is
  insufficient and the strategy must move to `sha256:<bytes>`.

**Amended by [ADR-014](#adr-014).** A part of the original model (`variant_id IS NULL`)
resolves its GLB exactly as before; a part of an extra model variant resolves it from the
variant's `glb_asset_id`. Either way `glb_version` stays `asset:<ProductAsset.id>`. The value
semantics are as unsettled as before; this ADR stays **Proposed / Needs Verification**.

**Consequences.** If a seller re-uploads a model, their parts go stale and must be re-mapped.
That is honest: the material indices genuinely may not mean the same thing. A future
improvement is best-effort re-mapping by material name and mesh name, offered as a suggestion
the seller confirms — never applied silently.

---

## ADR-007

### Bake execution stays in-process, behind a swappable seam

**Status: Accepted**

**Context.** The brief says not to introduce a queue without justification. The existing
implementation uses FastAPI `BackgroundTasks` (CONFIRMED). Two Azure-native alternatives
already exist in the repository: an ACA Job trigger (`usdz_trigger_service.py`) and a Service
Bus publisher (`service_bus_publisher.py`).

**Decision.**

1. Phase 1 runs bakes on `BackgroundTasks`, one at a time per worker via
   `asyncio.Semaphore(1)`, matching `variant_bake_service._BAKE_SEMAPHORE`.
2. All of it hides behind `bake_runner.enqueue()`. `BakeService`, routes, schemas and the
   public API never reference `BackgroundTasks`.
3. **`bake_started_at`, a startup sweep, and a periodic sweep ship in phase 1**, not later.
   Without them a recycled replica leaves rows in `baking` forever — which is the state the
   existing colour-variant implementation is in today.
4. Move to an ACA Job or Service Bus when a stated trigger fires: stuck/lost bakes after the
   sweep exists, API p99 degradation, sustained volume above ~30 bakes/hour, or memory
   pressure.

**Consequences.** Phase 1 is small and matches its neighbours. The durability gap is real but
bounded — the worst case is a retryable `failed`, not a permanent spinner. Moving later is a
one-module change.

**Note for whoever does the move:** a memory note from 2026-09-01 records the existing
GLB→USDZ Container Apps Job OOMing at 1 GiB. Size a bake job's memory deliberately; do not
inherit that limit unmeasured. **NEEDS VERIFICATION** against current Azure config.

---

## ADR-008

### Product ownership enforced on every Configurator endpoint

**Status: Accepted**

**Context.** **CONFIRMED**: the two closest neighbours check that a product exists but never
that the caller owns it. `hotspot_service._ensure_product_exists` and
`color_variant_service._ensure_product_exists` both call a bare `db.get(Product, product_id)`
with no `created_by` filter (`app/database/color_variant_repo.py:35-39`). Any authenticated
user can read and modify any product's hotspots and colourways. Meanwhile the product
*listing* endpoints do scope correctly (`Product.created_by == current_user.id`,
`app/api/routes/products.py:2149`).

Following the neighbours here would propagate a security gap into a new feature.

**Decision.**

1. Every seller endpoint resolves the product through
   `created_by == current_user.id AND deleted_at IS NULL`.
2. Ownership failure returns **404, not 403** — a 403 confirms that a product with that id
   exists and belongs to someone else, which is an enumeration oracle over every seller's
   catalogue.
3. Endpoints keyed on `part_id` / `option_id` resolve upward to the product and run the same
   check. A client-supplied `product_id` alongside a `part_id` is never trusted.
4. Enforced in **services**, not routes. Routes stay thin.
5. Covered by API tests that assert a second user gets 404 on every endpoint.

**Consequences.** One extra predicate per query — negligible, and `tbl_products` is indexed on
its primary key. The Configurator diverges from its neighbours, deliberately.

**Out of scope, but worth raising separately:** the same gap in hotspots and colour variants
is a live issue in shipped code. It is not this project's to fix, and it should not be
silently inherited either.

---

## ADR-009

### Alembic owns all Configurator DDL

**Status: Accepted**

**Context.** **CONFIRMED**: the colour-variant tables were created by
`sql/create_color_variants.sql`, a hand-run script. No revision in `migrations/versions/`
creates them. A fresh environment brought up with `alembic upgrade head` therefore does not
have `tbl_product_color_variants` or `tbl_variant_assets` at all.

`migrations/env.py:78-86` states the problem plainly: the schema has drifted from the
migration chain, "columns exist that no revision created, and three modelled tables do not
exist at all", and autogenerate output "always needs reading line by line".

**Decision.**

1. Every Configurator schema change is an Alembic revision. No hand-run SQL, ever.
2. Migrations are **written by hand**, not autogenerated. `env.py`'s `include_object` hook
   already refuses to autogenerate foreign keys — it would otherwise emit DDL dropping the 43
   undeclared `created_by`/`updated_by` FKs.
3. Follow the file style of `migrations/versions/97c7878c3123_add_hotspot_type_table.py`:
   docstring with revision/date, typed identifiers, both `upgrade()` and `downgrade()`
   implemented.
4. One revision creates all three tables; `downgrade()` drops them in reverse dependency
   order.
5. `down_revision` chains from the actual head at implementation time — check `alembic heads`,
   do not copy a revision id from this document.

**Consequences.** A fresh environment gets a working Configurator from `upgrade head`. The
cost is writing DDL by hand, which is the correct cost given the state of autogenerate here.

---

## ADR-010

### Account-purge schema contract is a deployment dependency

**Status: Accepted** — the decision is settled; the **coordination is a deployment blocker**.

**Context.** **CONFIRMED**: `ACCOUNT_PURGE_JOB_HANDOFF.md` §25 defines a schema contract
between this repository and `Rivollo.AccountPurge.Job`, a production job in a different
repository. The contract is the database schema itself — no runtime call in either direction.
A contract check runs **before every purge** and aborts the run on mismatch. Assertion 9:

> 🔴 **No unexpected new FK references `tbl_users` or `tbl_products`**
>
> Assertion 9 is the most valuable: it catches a table added in the Application Server that
> this job does not know about.

`tbl_product_parts.product_id → tbl_products` is exactly such a FK. There is direct precedent
for designing around this constraint: `app/models/login_otp.py:19-33` documents omitting a
`tbl_users` FK because "a user_id FK here would break a production job in another repository."

**Decision.**

1. **Configurator tables declare no FK from `created_by` / `updated_by` to `tbl_users`** —
   not in the ORM, not in the migration. Plain `UUID` columns, exactly as `AuditMixin`
   already declares them. This keeps assertions 5 and 9 clean and costs nothing: the ORM has
   never declared those FKs anyway.
2. **The `product_id → tbl_products ON DELETE CASCADE` FK is kept.** It is required — the
   purge deletes products before their owner, and without the cascade that `DELETE` fails
   outright. This is the same reason revision `b3f8d21c4a76` exists.
3. **The purge job is updated before this migration reaches production.** Required changes
   there: add the three tables to §4.2 inventory and §7 cascade list; allow-list the new FK in
   assertion 9; add the product-scoped blob prefix `{container}/configurator/{product_id}/…`
   to §15 deletion order.

**Explicitly rejected:** removing the `product_id` FK to avoid the contract check. That would
trade a coordination task for permanent orphaned rows and a broken cascade. §25's "Change
protocol" already anticipates this: any migration touching `tbl_products` or its dependents
**must** be accompanied by a review of the job's contract check and deletion order.

**Consequences.** One cross-repository task, and it is the longest-lead item in the plan —
open it the day implementation starts. Until it lands, the migration must not be deployed to
production. Full detail in [data-model.md §13](data-model.md#13--deployment-dependency-the-account-purge-job).

---

## ADR-011

### `is_default` on the option, not `default_option_id` on the part

**Status: Accepted**

**Context.** The initial design put `default_option_id` on `tbl_product_parts` pointing at
`tbl_part_options`, which points back at the part — a circular foreign key requiring either a
deferrable constraint or a two-step migration, plus a nullable pointer the service must keep
consistent by hand.

**CONFIRMED**: the repository already solves this exact problem the other way round:

```sql
-- sql/create_color_variants.sql:79-81
CREATE UNIQUE INDEX ux_color_variants_one_default
    ON tbl_product_color_variants (product_id) WHERE is_default;
```

**Decision.** Drop `default_option_id`. Add `is_default BOOLEAN NOT NULL DEFAULT false` to
`tbl_part_options` with a partial unique index `ux_part_options_one_default ON (part_id)
WHERE is_default`.

API responses continue to expose `default_option_id` on the part as a **computed** field.

**One setter — explicit.** *(Revised 2026-09-11; previously also first-baked-wins.)*
A part with **no default shows the model's Original appearance**, and that is the intended
starting state. `is_default` is set only by `PATCH /options/{id}` with
`set_as_default: true`, once the option is active and completed. `set_as_default` is **not**
accepted on create — a new option is always `pending`, and no column records a deferred intent.

Returning to Original is explicit too: `set_as_default: false` on the current default clears
it, and hiding or deleting the default clears it without promoting a replacement.

**Why the revision.** The original rule auto-promoted the first option whose bake completed,
and on deleting the default promoted the next suitable option. Both made a choice on the
seller's behalf — which option a shopper sees first — and left the seller's untouched model
unreachable as a starting state once any option existed. With Original as the fallback,
`default_option_id` means exactly one thing: a starting colour the seller deliberately chose.
`set_as_default: false`, previously ignored because no operation removed a default without
naming a replacement, now has one: Original.

**Consequences.** No circular FK, no deferrable constraint, no two-step migration, and
"at most one default per part" becomes a database guarantee instead of a service convention.
The service still owns the *rules* — only an `isactive`, `completed` option may become the
default; hiding or deleting the default returns the part to Original rather than choosing a
replacement. This deliberately diverges from `color_variant_service.py:189-204, 250-263`,
which rejects hiding the default and auto-promotes on delete. Also removes one FK from the
surface ADR-010 has to coordinate.

---

## ADR-012

### Material-index uniqueness: JSONB + row lock

**Status: Accepted**

**Context.** Within a product a material index must belong to at most one active part, or two
options could paint the same mesh with conflicting textures. `material_indices` is a JSONB
array, which a plain `UNIQUE` cannot cover. Two enforcement mechanisms were considered.

| | JSONB + trigger | Normalized `tbl_product_part_materials` | **JSONB + row lock** |
|---|---|---|---|
| Declarative DB guarantee | trigger (imperative) | plain `UNIQUE` ✅ | none |
| New FKs to `tbl_products` | 0 extra | **+1** — aggravates ADR-010 | 0 extra |
| Alembic precedent | **zero triggers in the chain** | ordinary `create_table` | n/a |
| Visible to autogenerate | ❌ drift risk | ✅ | n/a |
| Table count | 3 | **4** | **3** |

**CONFIRMED**: no Alembic revision in this repository has ever created a trigger. Triggers
appear only in the hand-run `normalize_license_assignments.sql` and the `rivolloschema.sql`
dump. `migrations/env.py` already documents that this schema has drifted from its own
migrations; a trigger would be invisible to autogenerate and add to that drift.

**Decision.** Keep `material_indices` as JSONB. **No trigger. No normalized child table.**
Enforce in `PartService` with a `SELECT ... FOR UPDATE` on the product row taken in the same
query as the ownership check, before loading sibling parts and writing.

**Consequences.** The row lock buys precisely what the constraint would have: protection
against two concurrent writes claiming the same index. The service check keeps producing the
useful error message a constraint never could — *"Material index 3 already belongs to part
'Backrest'"*. Parts per product are single-digit and the sibling load runs on
`ix_parts_product_order`, so the lock is held for microseconds. Also allows dropping the GIN
index and both JSONB CHECK constraints.

*Amended by [ADR-014](#adr-014):* the overlap rule is scoped to one **model** (siblings =
active parts with the same `variant_id`, NULL meaning the original). The lock is still the
**product** row: one lock per product serialises every model's writes, which costs nothing at
single-digit parts and keeps a single lock order.

**Accepted cost:** there is no database-level backstop. If a second writer appears — an import
job, a bulk editor — or bulk material→part querying is needed, revisit the normalized table.

---

## ADR-013

### Uploaded textures are `recipe.method = "image"`, not an option type

**Status: Accepted**

**Context.** A Part Option must be able to represent both a generated recolour and a
seller-uploaded texture. The obvious move is an `option_type` / `source_type` discriminator
column, or a second table for uploaded textures.

**CONFIRMED**: `recipe.method` is *already* the behavioural discriminator — `factor`,
`luminance` and `remap` dispatch to different code paths today
(`app/services/color/glb_recolor.py:436-464`).

**Decision.** Add `image` as a fourth `recipe.method` value. No new column, no new table, no
`option_type`, no `source_type`.

```json
{ "version": 1, "method": "image", "image_url": "https://cdn/.../users/{uid}/uploads/{id}/x.png" }
```

Three supporting rules:

1. **`image_url` is validated server-side** as belonging to the current seller's own upload
   namespace (`users/{current_user.id}/uploads/…`). Unvalidated, it is an SSRF vector — the
   baker dereferences it — and a cross-tenant hotlink vector.
2. **The uploaded file is copied** into `configurator/{product_id}/{glb_version}/{option_id}/…`
   before the Configurator records or purges it, so deleting an option never touches a blob
   under the seller's own uploads prefix.
3. **`swatch_hex` is required** for `image` options — there is no `recipe.color` to default
   from.

**Consequences.** One option model, one bake lifecycle, one staleness rule, one purge path.
Extending an enum inside an existing JSONB document is not a schema change. A separate
`option_type` column would be a *second*, redundant discriminator that could contradict
`recipe.method`.

A useful side effect: `image` options are unaffected by the unresolved `remap` algorithm
dispute in [ADR-005](#adr-005), so they can ship even while that stays open.

**Revisit** only if a concrete requirement emerges to query options by kind without opening
the JSONB. None exists today.

---

## ADR-014

### Model variants: extra shapes, added without changing the existing system

**Status: Accepted** — the decision is settled; like ADR-010 it carries a **deployment
blocker** (Q8).

**Context.** A product can come in several shapes — "2 Seater", "3 Seater", "6 Seater
Corner" — each a separate GLB. Configurator parts are bound to glTF material indices, and
those differ between GLBs (Tripo names and orders materials per generation), so a part
must belong to exactly one GLB.

Every reader of "the product's model" — the product list and its thumbnails
(`ProductRepository.get_primary_assets_for_products`), `GET /products/{id}` and
`/assets`, `ConfiguratorRepository.get_product_mesh_asset`, colour variants,
`Rivollo.Viewer.Api` `/assets` and `/configurator`, the editor's `getGlbUrl`, the USDZ
job — resolves it through `tbl_product_asset_mapping`: the newest **active mapped** row of
the wanted format (CONFIRMED, every one). That fact is what lets this feature be additive.

**Requirement (product decision, 2026-09-22): the existing system must not change.** The
product's original model stays its model and its **permanent default**.

**Decision.**

1. **The original model has no row.** The GLB created with the product
   (`createProductFromGlb` or AI generation) is the product's model, as today, and is always
   the default. There is no `is_default` column and no "set as default" action.
2. **`tbl_product_model_variants` holds the EXTRA shapes only**: `name`,
   `glb_asset_id → tbl_product_assets`, `usdz_asset_id → tbl_product_assets`, thumbnail,
   original-upload columns, compression sizes and status, bounding-box dimensions in metres,
   `order_index` (the original is implicitly 0), `isactive`, audit columns.
3. **A variant's GLB is a `tbl_product_assets` row with NO mapping row.** Unmapped, it is
   invisible to every reader above, so product lists, thumbnails, `/assets`, AR and colour
   variants are untouched. Referencing an asset row (rather than storing a URL) keeps the
   Configurator's GLB identity `asset:<id>` (ADR-006) for variants too.
4. **`tbl_product_parts.variant_id` is NULLABLE.** `NULL` means the original model, so every
   existing part keeps its meaning and no row is rewritten. Part slugs stay unique per
   product (`uq_parts_product_slug` unchanged); the service suffixes a clashing slug.
5. **Options and textures are unchanged** — they reach the variant through the part.
6. **The migration is purely additive** (`e3b9c6a1d27f`): one table, one nullable column,
   indexes. No data written, nothing dropped, no change to `tbl_products`,
   `tbl_product_assets` or `tbl_product_asset_mapping`.
7. **`ENABLE_MODEL_VARIANTS`, on by default** (product decision, 2026-09-23; it shipped off
   and was flipped before the first deploy). Set to `false`, every variant route answers 404
   and the shopper payload omits `variants`, exactly as before this feature. On — the default
   — an environment MUST have migration `e3b9c6a1d27f` applied: without the table the variant
   routes and the public shopper payload both fail.
8. **Every variant GLB is Draco-compressed on upload**, reusing
   `glb_compression_service.compress()` (gltf-transform, in-process Node) off the event loop.
   The compressed file is re-inspected and must have identical material and mesh names in
   the same order — parts attach to material indices. On any failure or mismatch the
   original is served and `compression_status = 'fallback_original'`. The untouched upload is
   kept in `original_glb_*` columns (never an asset row) when it differs from the served file.
9. **Legacy colour variants are untouched.** `tbl_product_color_variants` and its routes stay
   product-level (Q6 is still open for them).
10. **AR per variant, behind its own flag.** `ENABLE_VARIANT_USDZ` is **off by default**
    (product decision, 2026-09-23): a variant has no USDZ, `usdz_url` stays `null` and the
    viewer hides AR for that shape. With it on, an upload starts the existing converter job
    with `--model-variant-id`; the job writes the USDZ to the variant's folder, inserts an
    asset row with **no** mapping, sets `usdz_asset_id`, and leaves the product status alone.
    Without the argument the job behaves exactly as before. Deploy the converter image that
    accepts it **before** switching the flag on — an older image rejects the unknown argument.
    Turning it on later converts variants uploaded from then on; earlier ones need a re-upload
    or a manual job run.
11. **Shopper payload.** `variants[]` appears only when a product has live extra variants;
    top-level fields stay the original model's. `Rivollo.Viewer.Api` mirrors it behind
    `Configurator:EnableModelVariants`, and never reads the variants table while that is off.

**Foreign keys — chosen for `Rivollo.AccountPurge.Job`:**

| FK | Rule | Why |
|---|---|---|
| `product_id → tbl_products` | CASCADE | a **new** FK to `tbl_products`; assertion A14 must allow-list it (Q8) |
| `glb_asset_id`, `usdz_asset_id → tbl_product_assets` | **SET NULL** (hence nullable) | the purge deletes `tbl_product_assets` (step 4) **before** `tbl_products` (step 6); RESTRICT would abort that step, CASCADE would let a stray asset delete wipe a variant's configuration |
| `tbl_product_parts.variant_id → variants` | CASCADE | only a hard delete reaches it; the app soft-deletes variants |
| `created_by`, `updated_by` | **no FK** | as ADR-010 |

Full indexes on `product_id`, `glb_asset_id` and `usdz_asset_id` serve the purge's CASCADE
and SET NULL lookups. Variant blobs live under `{user_id}/{product_id}/model-variants/…`,
which the purge's existing user prefix already sweeps.

**Consequences.**

- `CLAUDE.md`'s "exactly three tables" rule becomes four. The spirit is unchanged: no
  normalised material table, no texture-option table, no type discriminator column.
- Code that lists a product's parts must now say *which* model: `variant_id IS NULL` for the
  original, `= :id` for an extra variant. Until the variant-scoped part routes land (step d),
  only original-model parts exist.
- Legacy consumers always show the original model. A seller cannot make an extra shape the
  product's main model; that would need a mapping re-point, which this design rules out.
- Unmapped variant asset rows are found by the purge through `created_by` and through the
  variant's own FK (job decision D10) — so the service always sets `created_by`.
- A variant with `glb_asset_id IS NULL` can exist after an out-of-band asset delete; services
  treat it as unusable rather than failing.

---

## Open questions

Resolve before or during implementation. Each names who can answer it and what unblocks.

### Q1 — Is CORS configured on the CDN? · **blocks preview** · Infrastructure

**CONFIRMED**: nothing in this repository configures it, and there is no IaC to inspect.
Frontend pixel readback fails without it.

```bash
curl -I -H "Origin: https://portal.example.com" "$CDN_BASE_URL/$CONTAINER/<known-texture>"
# expect: access-control-allow-origin
```

If absent: configure blob-service CORS and the Front Door rule set. Also confirm the CDN does
not strip the header. Blocks [ADR-005](#adr-005).

### Q2 — Should Configurator errors use the `api_error` envelope? · Backend team

**CONFIRMED**: business errors raise `HTTPException` → `{"detail": ...}`, while unhandled 500s
return `api_error(...)`. Two shapes, repo-wide. The spec follows the neighbours
(`HTTPException`) rather than inventing a third. Unifying is a separate repo-wide change.
Decide whether the Configurator is the place to start.

### Q3 — Extract source textures server-side, or read them in the browser? · Frontend + backend

Option A (extract and upload, recommended) makes "what the bake started from" an explicit
addressable artifact and keeps the frontend's job to display. Option B (frontend reads the
GLB) needs no backend work but requires CORS regardless and duplicates extraction. Affects
whether `base_color_texture_url` exists on the materials response.

### Q4 — Are triangle counts obtainable from a Draco-compressed product GLB? · Backend

**CONFIRMED**: nothing in this repo reads triangle counts today, and the canonical GLB is
normally Draco-compressed. Under `KHR_draco_mesh_compression` the `indices` accessor is
specified to remain present with a valid `count` and no `bufferView`, which would make
`indices.count / 3` readable without a decoder — but that is the extension's specification,
**not something observed here**. Verify against a real product asset before promising the
field:

```python
gltf = GLTF2().load("real-product.glb")
for m in gltf.meshes:
    for p in m.primitives:
        print(p.material, gltf.accessors[p.indices].count // 3 if p.indices is not None else None)
```

The same check should confirm `accessor.min`/`max` on POSITION, which `_material_centers`
depends on.

### Q5 — Trigger or normalised table for the material-index uniqueness constraint? · ✅ **RESOLVED**

Superseded by **[ADR-012](#adr-012)**: neither. `material_indices` stays JSONB, enforcement is
a `SELECT ... FOR UPDATE` on the product row inside `PartService`, and the table count stays
at three. A trigger has no precedent in this migration chain and is invisible to autogenerate;
a normalised child table would add a second FK to `tbl_products`, aggravating
[ADR-010](#adr-010). Revisit only if a second writer or bulk material→part querying appears.

### Q6 — What happens to the existing colour-variant feature? · Product + backend

**CONFIRMED**: it exists, is fully implemented, has zero tests, its tables are outside
Alembic, and there is no evidence any row has ever held a genuinely baked `model_url` — the
backfill creates only `is_original` rows, which short-circuit to `ready` without baking.

The Configurator supersedes it functionally. This spec assumes **coexistence**: additive work
only, no modification to colour-variant code or tables.

One concrete conflict must be resolved regardless:
`GET /products/{product_id}/materials` is **already registered** by
`app/api/routes/color_variants.py:41`. Two routers cannot own one path. Options: the
Configurator uses `/products/{id}/configurator/materials`; the colour-variant route is
retired; or one shared handler serves both. Deprecation timing is a product decision.

### Q7 — Can the viewer swap textures by glTF material index at runtime? · ✅ **ANSWERED: yes** (product decision, 2026-09-21) · Frontend

Answered yes. Supporting evidence: `Rivollo.Web.Portal` `components/shared/preview/ThreeViewer.tsx`
already replaces base-colour textures at runtime on a loaded `<model-viewer>` with
`mv.createTexture()` + `baseColorTexture.setTexture()` across `model.materials`. The public viewer
applies option textures per material index the same way. The original question is kept below.

**[ADR-003](#adr-003) rests entirely on this** and it has never been verified. The whole
texture-baking design assumes the 3D viewer can replace `material.map` for a specific glTF
material index at runtime, on a model it has already loaded, without reloading the GLB.

Not verifiable from this repository — the viewer is a separate codebase. If the answer is no,
texture-only baking cannot render, and the architecture reverts to per-option GLBs (with all
the storage and bandwidth costs ADR-003 documents) or the viewer must be extended first.

Should be a short conversation, but it invalidates the core approach if the answer is no, so
**get it before any Configurator implementation begins** — earlier than every other question
here except Q8.

### Q8 — Has the account-purge job been updated? · 🔴 **deployment blocker** · Cross-repo

Tracked by [ADR-010](#adr-010). The decision is settled; the coordination is not done. Until
`Rivollo.AccountPurge.Job` allow-lists the new `tbl_product_parts.product_id → tbl_products`
FK in assertion 9 and adds the three tables plus the `configurator/{product_id}/…` blob prefix
to its inventory and deletion order, **and** (ADR-014) allow-lists
`tbl_product_model_variants.product_id → tbl_products` and inventories that table's
`thumbnail_blob_url` / `original_glb_blob_url` blobs, **the migration must not be deployed to production** —
the contract check aborts every purge run on an unrecognised FK.

This does not block writing the migration, the ORM, or any service code. It blocks the deploy.

**Status 2026-09-22 — change written, not merged or deployed.** `Rivollo.AccountPurge.Job` branch
`model-variants-purge/supriya` (its decision D10) allow-lists both product FKs, adds assertions
A20 (configurator cascade tree) and A21 (every FK into `tbl_product_assets` is SET NULL), the
`configurator/{product_id}/` blob prefix, and a UNION branch for unmapped model-variant assets.

**CONFIRMED on dev:** `tbl_product_parts.product_id → tbl_products` already exists there, so the
deployed job's contract check rejects it — dev purge runs are aborting today, until that branch
ships. **Deploy it together with `e3b9c6a1d27f`, between two nightly (00:00 UTC) runs:** either
side alone makes the contract check abort the run — safely, before anything is deleted.

---

## Status discipline

Two ADRs remain deliberately unpromoted. Neither may be described as Accepted anywhere in
these documents, in `CLAUDE.md`, or in a pull request, until the stated evidence exists:

- **[ADR-005](#adr-005) Preview / Bake separation — Proposed / Needs Verification.**
  Blocked on Q1 (CDN CORS is not configured anywhere in this repository and there is no IaC to
  inspect) **and** on reading the frontend's actual `remap` implementation to settle the
  percentile-vs-absolute-min/max divergence. Do not silently standardise on either.
- **[ADR-006](#adr-006) GLB identity and versioning — Proposed / Needs Verification.**
  The `glb_version TEXT NOT NULL` **column** is settled and safe to build. The **value
  semantics** are not: `asset:` is the Phase-1 choice, `sha256:` the upgrade path, and the
  prefix discriminator is the hedge that keeps the upgrade a data migration. Do not hardcode
  either strategy, and do not promote this ADR until it is confirmed that no out-of-band
  process overwrites a mesh blob at a stable path and that no planned "replace model" feature
  will mutate `ProductAsset.image` in place the way `PUT /products/{id}/original-image`
  already does for `asset_id == 1`.

[ADR-003](#adr-003) is Accepted as a *design decision* but carries the Q7 runtime dependency
above. If Q7 comes back negative, ADR-003 must be reopened, not worked around.
