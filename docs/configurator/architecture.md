# Product Configurator — Architecture

> Status: **specification only**. No Configurator code, schema, or migration exists yet.
> Every claim below is tagged **CONFIRMED** (observed in this repository at `main`),
> **PROPOSED** (a design recommendation), or **NEEDS VERIFICATION** (depends on runtime,
> Azure configuration, database contents, or a real asset — not establishable from code).

Companion documents:

| Document | Purpose |
|---|---|
| [data-model.md](data-model.md) | Entities, tables, columns, constraints |
| [api-spec.md](api-spec.md) | Endpoints, schemas, auth, errors |
| [baking.md](baking.md) | Preview vs bake, recolouring, status lifecycle |
| [decisions.md](decisions.md) | ADRs, including everything still unresolved |

---

## 1. Goals

Build a seller-facing Product Configurator backend and a shopper-facing read API that
together support:

1. Inspecting the materials of a product's 3D model.
2. Letting a seller define named, configurable **Parts** ("Seat", "Legs") over those materials.
3. Letting a seller define shopper-selectable **Options** per part ("Black", "Red").
4. Producing **baked textures** per option, persisted to blob storage and served by CDN.
5. Serving a compact configuration payload to the public 3D viewer.
6. Keeping the seller's original GLB as the single canonical geometry.

## 2. Scope

**In scope**

- A dedicated Configurator domain: routes, schemas, services, repositories.
- Materials inspection endpoint tailored to the Part Editor.
- Parts / Options / Option Textures persistence and validation.
- A `BakeService` with a stable public contract and a swappable execution backend.
- A shopper read endpoint.
- Alembic migrations for all Configurator tables.
- 🔴 Coordinating the `Rivollo.AccountPurge.Job` schema contract before production deployment
  — see [ADR-010](decisions.md#adr-010). Cross-repository, longest lead time in the plan.
- Tests at unit, service, repository, and API level.

**Out of scope (non-goals)**

- Replacing or deleting the existing colour-variant feature in this phase. See §9.
- Generating a complete GLB per colour/material combination. See [ADR-003](decisions.md#adr-003).
- A separate deployable service, message bus, or event-sourced design. See §11 and [ADR-001](decisions.md#adr-001).
- Frontend implementation. This document specifies the backend contract the frontend must honour.
- Geometry editing, UV editing, normal/roughness map editing.
- Pricing, inventory, or cart integration for configured products.

---

## 3. Existing backend, as observed

**CONFIRMED — stack**

| Concern | What the repo uses | Evidence |
|---|---|---|
| Framework | FastAPI 0.128 | `requirements.txt:27`, `app/main.py` |
| ORM | SQLAlchemy 2.0.45, async, `asyncpg` | `requirements.txt:58`, `app/core/db.py` |
| Migrations | Alembic 1.17.2 | `requirements.txt:7`, `migrations/` |
| Schemas | Pydantic v2 | `app/schemas/*.py` |
| Storage | Azure Blob (`azure-storage-blob`) + CDN/Front Door | `app/services/storage.py` |
| Hosting | Azure Container Apps | `.github/workflows/deploy-prod.yml` |
| Telemetry | OpenTelemetry → Azure Monitor | `app/main.py:174-193` |
| Tests | pytest + `pytest-asyncio` (`asyncio_mode = "auto"`) | `pyproject.toml:78-79` |
| 3D | `pygltflib`, `Pillow`, `numpy`; `gltf-transform` via Node subprocess | `requirements.txt:72-74`, `app/services/glb_compression_service.py` |

**CONFIRMED — layering.** The repository already uses the layering this project wants.
The newest domains (hotspots, dimensions, colour variants) all follow:

```
app/api/routes/<domain>.py     thin route: parse ids, call service, wrap in api_success
        |
app/schemas/<domain>.py        Pydantic request/response contracts
        |
app/services/<domain>_service.py   business rules, HTTPException on rule violations
        |
app/database/<domain>_repo.py      SQLAlchemy statements only
        |
app/models/models.py               ORM tables
```

Note two competing repository locations exist: `app/database/*_repo.py` (11 modules, used by
every recent domain) and `app/repositories/support_repository.py` (1 module). **PROPOSED:**
the Configurator uses `app/database/`, matching the dominant and most recent convention.

**CONFIRMED — response envelope.** `app/utils/envelopes.py` provides
`api_success(data) -> {"success": True, "data": ...}` and
`api_error(code, message, details)`. Routes declare `response_model=dict` and return
`api_success(...)`.

**CONFIRMED — error handling is not uniform.** Business errors are raised as
`HTTPException`, which FastAPI renders as `{"detail": "..."}` — *not* the `api_error`
envelope. Only unhandled 500s go through `app/main.py:307` and produce `api_error`. So a
client sees two different error shapes. The Configurator should follow the existing
`HTTPException` pattern for consistency with its neighbours; unifying the envelope is a
separate, repo-wide change and is recorded as an open question in
[decisions.md](decisions.md#open-questions).

**CONFIRMED — authentication tiers.**

| Tier | Dependency | Used by |
|---|---|---|
| Seller (JWT bearer) | `get_current_user` → `CurrentUser` | `products`, `hotspots`, `color_variants`, most routers |
| Public (HTTP Basic) | `verify_public_basic_auth` on `public_router` | `/public/products/{id}/assets` |
| Fully open | `public_noauth_router` | `/remove-background` only |
| App token | `verify_app_token` | mobile/app integrations |

`PUBLIC_API_USERNAME` / `PUBLIC_API_PASSWORD` gate the `/public/...` surface
(`app/api/routes/products.py:88-104`). This is the shopper/viewer precedent.

**CONFIRMED — authorization gap in the closest neighbours.** `hotspot_service` and
`color_variant_service` check that a product *exists* but never that the caller *owns* it:

```python
# app/services/hotspot_service.py:184-191  (color_variant_service.py:269-274 is identical)
async def _ensure_product_exists(db, product_id) -> None:
    if not await hotspot_repository.get_product_by_id(db, product_id):
        raise HTTPException(status_code=404, detail="Product not found")
```

`get_product_by_id` is a bare `db.get(Product, product_id)` with no `created_by` filter
(`app/database/color_variant_repo.py:35-39`). By contrast, the product *listing* endpoints
do scope by owner (`Product.created_by == current_user.id`,
`app/api/routes/products.py:2149`). **The Configurator must not inherit this gap** — see
§10 and [api-spec.md](api-spec.md#authorization).

**CONFIRMED — no `/api/v1` convention exists.** `_api_prefix = settings.API_PREFIX.rstrip("/")`
and `API_PREFIX` defaults to `""` (`app/core/config.py:10`, `app/main.py:216`). Every router
is mounted at that bare prefix; the single exception is `products_v2_router` at
`f"{_api_prefix}/v2"`. Paths are flat and resource-first (`/products/{id}/hotspots`).
Introducing `/api/v1/configurator` would create a *new* convention, not follow one — see
[ADR-001](decisions.md#adr-001) for the recommendation and the alternative.

---

## 4. What the Configurator reuses

The Configurator must not duplicate any of these. **CONFIRMED** to exist and be reusable:

| Capability | Module | Reuse as |
|---|---|---|
| JWT auth, current user | `app/api/deps.py` (`CurrentUser`, `DB`) | direct dependency |
| DB session | `app/core/db.py` (`get_db`, `new_session`) | direct dependency |
| Response envelope | `app/utils/envelopes.py` | direct call |
| Product lookup / mesh resolution | `app/database/color_variant_repo.py:43-68` | extract to a shared helper, see §9 |
| Azure Blob upload + CDN URL | `app/services/storage.py` | direct call, plus one new method |
| Blob→CDN URL rewriting | `app/middleware/cdn.py` | automatic, no action |
| Generic image upload endpoint | `POST /uploads/content` | reuse as-is, see §8 |
| GLB parsing / material inspection | `app/services/color/glb_recolor.py:177` `inspect()` | extend, see §7 |
| Recolour mathematics | `app/services/color/glb_recolor.py`, `colors.py` | extract texture-level core, see [baking.md](baking.md) |
| Source-model disk cache | `app/services/model_cache.py` | direct call |
| Background job trigger (ACA Job) | `app/services/usdz_trigger_service.py` | pattern for future worker |
| Queue publisher (Service Bus) | `app/integrations/service_bus_publisher.py` | pattern for future worker |
| Structured logging + tracing | `app/main.py` logging config, OTel | automatic |

---

## 5. System architecture

```
                          ┌──────────────────────────────┐
   Seller Portal ────────▶│  Configurator seller API     │
   (Part Editor)          │  bearer JWT + ownership      │
                          │                              │
                          │  /products/{id}/materials    │
                          │  /products/{id}/parts        │
                          │  /parts/{id}/options         │
                          │  /options/{id}/bake          │
                          └───────────┬──────────────────┘
                                      │
                          ┌───────────▼──────────────────┐
                          │  Configurator services       │
                          │  PartService                 │
                          │  OptionService               │
                          │  MaterialInspectionService   │
                          │  BakeService  ◀── swappable  │
                          └───────────┬──────────────────┘
                                      │
                   ┌──────────────────┼───────────────────┐
                   │                  │                   │
        ┌──────────▼────────┐ ┌───────▼────────┐ ┌────────▼─────────┐
        │ configurator_repo │ │ storage_service│ │ colour engine    │
        │ (SQLAlchemy)      │ │ (Azure Blob)   │ │ app/services/    │
        └──────────┬────────┘ └───────┬────────┘ │ color/*          │
                   │                  │          └──────────────────┘
        ┌──────────▼────────┐ ┌───────▼────────┐
        │ PostgreSQL        │ │ Blob + CDN     │
        │ tbl_product_parts │ │ baked textures │
        │ tbl_part_options  │ │                │
        │ tbl_part_option_  │ └───────┬────────┘
        │   textures        │         │
        └───────────────────┘         │
                                      │
                          ┌───────────▼──────────────────┐
   Shopper Viewer ───────▶│  Configurator shopper API    │
   (3D viewer)            │  HTTP Basic (public tier)    │
                          │  GET /public/products/{id}/  │
                          │      configurator            │
                          └──────────────────────────────┘
```

### 5.1 Module layout (PROPOSED)

```
app/
  api/routes/
    configurator.py             seller routes (materials, parts, options, bake)
    configurator_public.py      shopper route
  schemas/
    configurator.py             all Configurator Pydantic models
  services/
    configurator/
      __init__.py
      part_service.py           part CRUD + material-membership invariants
      option_service.py         option CRUD + recipe validation
      material_service.py       GLB inspection → editor-facing material list
      bake_service.py           public bake contract + status lifecycle
      bake_runner.py            execution backend (BackgroundTasks today)
      texture_baker.py          pure texture in → texture out (no DB, no Azure)
  database/
    configurator_repo.py        all Configurator persistence
migrations/versions/
  <rev>_add_configurator_tables.py
tests/
  test_configurator_parts.py
  test_configurator_options.py
  test_configurator_bake.py
  test_configurator_public.py
  test_texture_baker.py
```

Rationale for a `services/configurator/` package rather than flat modules: the existing
`app/services/color/` package sets the precedent for grouping a cohesive, pure sub-domain,
and the Configurator has five distinct services that would otherwise clutter a directory
already holding 45 modules.

---

## 6. Original GLB strategy

**Principle.** The seller's uploaded GLB is the canonical model. The Configurator never
rewrites its geometry and never produces a per-option copy of it.

```
Product (tbl_products)
  │
  └── Original GLB          asset_id = 9 in tbl_product_assets   [CONFIRMED]
        │                   resolved via tbl_product_asset_mapping
        │
        ├── glTF materials  material_index 0..N-1                [CONFIRMED join key]
        │     │
        │     └── Product Part          seller-named, persisted  [PROPOSED table]
        │           │                   owns 1..N material indices (JSONB)
        │           │                   pinned to a glb_version
        │           │
        │           └── Part Option     shopper-selectable       [PROPOSED table]
        │                 │             holds a recipe: a recolour
        │                 │             (factor|luminance|remap) OR an
        │                 │             uploaded image (method: "image")
        │                 │             carries is_default + bake lifecycle
        │                 │
        │                 └── Option Texture                     [PROPOSED table]
        │                       one baked texture per affected
        │                       material index → CDN URL
        │
        └── Draco glTF package  asset_id = 17 (.zip)             [CONFIRMED]
        └── USDZ                asset_id = 11 (AR)               [CONFIRMED]
```

**CONFIRMED — how the mesh is resolved today.** `color_variant_repo.get_product_model_url`
selects `ProductAsset.image` where `asset_id == 9` and the mapping `isactive`, ordered by
`created_date DESC LIMIT 1`. A product may therefore hold several mesh rows; the newest
active one wins. `MESH_ASSET_ID = 9` is defined at `app/database/color_variant_repo.py:27`.

**CONFIRMED — the canonical GLB is normally Draco-compressed.**
`ENABLE_DRACO_COMPRESSION` and `ENABLE_GLTF_DRACO_PACKAGE` both default to `True`
(`app/core/config.py:292,306`), and `_compress_mesh_for_storage` returns `package.glb` —
the Draco-compressed bytes — which is what gets uploaded and mapped as asset 9
(`app/services/product_service.py:63-81`). `pygltflib` does not decode Draco.

Consequences the implementation must respect:

- Base-colour **textures are unaffected** — `KHR_draco_mesh_compression` compresses geometry
  attributes only, so image bufferViews decode normally. Material inspection therefore works.
  **NEEDS VERIFICATION** against a real product GLB.
- **Triangle counts**: under Draco the primitive's `indices` accessor is still present with a
  `count`, with the `bufferView` omitted, so `indices.count / 3` is obtainable from JSON
  metadata without a Draco decoder. This is the glTF extension's specified behaviour, **not**
  something observed in this repository — nothing here reads triangle counts today.
  **NEEDS VERIFICATION** against a real product GLB before it is relied upon. See
  [decisions.md](decisions.md#open-questions) Q4.
- `glb_recolor._material_centers` reads `accessor.min` / `accessor.max` on POSITION, which
  glTF requires even under Draco. **NEEDS VERIFICATION**.

---

## 7. Materials inspection

**CONFIRMED — what exists.** `GET /products/{id}/materials`
(`app/api/routes/color_variants.py:41`) downloads the GLB and returns, per material:
`material_index`, `name`, `mesh_names`, `has_base_color_texture`, `average_color`,
`suggested_method`, `group_id`, `center`. `average_color` is the mean of a 16×16 downsample
of the base-colour image (`glb_recolor.py:196-199`).

**CONFIRMED — what it does not return.** There is no base-colour texture URL and no triangle
count. The texture lives inside the GLB's binary buffer and is never extracted or uploaded.

**PROPOSED — what the Part Editor additionally needs.**

| Field | Why | Status |
|---|---|---|
| `base_color_texture_url` | frontend preview must read source pixels to recolour them live | requires a new *texture extraction* step, §7.1 |
| `triangle_count` | lets the editor rank materials and hide slivers | requires the Draco check above |
| `eligible_for_part` | server's opinion on whether a material can be configured | derived |
| `assigned_part_id` | which Part already claims this index | join |
| `glb_version` | ties the response to the exact model inspected | §8.2 |

`group_id` **must be surfaced as a hint only**. It is recomputed on every call from a
union-find over "shares a base-colour image" OR "average colours within 42.0 RGB units"
(`glb_recolor.py:307-358`), it is never persisted, it is not stable across a model
re-upload, and colour-proximity unions are transitive — a model with a smooth grey ramp can
collapse every material into one group. See [ADR-004](decisions.md#adr-004).

### 7.1 Source texture extraction (PROPOSED)

For the frontend to preview a recolour it needs the source texture pixels in the browser.
Two candidate strategies:

- **A — Extract on demand, cache in blob storage.** On first `GET .../materials`, extract
  each distinct base-colour image from the GLB, upload to
  `configurator/{product_id}/{glb_version}/source/{image_index}.{ext}`, and return CDN URLs.
  Idempotent (path is content-scoped), cacheable forever, one cost per model.
- **B — Frontend reads the GLB itself.** The viewer already loads the GLB; three.js exposes
  decoded textures. No backend work, no extra storage — but pixel readback from a
  cross-origin texture requires CDN CORS (§10.3) and forces the frontend to reimplement
  the extraction.

**Recommendation: A**, because it makes the source-of-truth for "what the bake started from"
an explicit, addressable artifact, and because it keeps the frontend's job to *display*.
Recorded as an open decision — see [decisions.md](decisions.md#open-questions) Q3.

---

## 8. Preview vs bake

This split is the core of the design. Full detail in [baking.md](baking.md).

```
 PREVIEW  (interactive, no backend round-trip per click)
 ─────────────────────────────────────────────────────────
   shopper clicks "Red"
        │
        ├─ frontend updates configuration state
        ├─ frontend applies the recipe to the source texture in-browser
        └─ viewer swaps the material's map  →  visible in < 1 frame budget

 BAKE  (persistent, asynchronous, once per option)
 ─────────────────────────────────────────────────────────
   seller saves an option
        │
        ├─ POST /options/{id}/bake      → 202, bake_status = "pending"
        │
        ├─ BakeService
        │     ├─ fetch source GLB (model_cache)
        │     ├─ extract base-colour texture for each affected material index
        │     ├─ apply recipe  (factor | luminance | remap | image)
        │     ├─ upload each result → CDN URL
        │     └─ write tbl_part_option_textures, set completed
        │
        └─ client polls GET /options/{id}/bake-status until completed | failed
```

The two paths **must implement the same recolouring specification** or the shopper sees one
colour while previewing and a different one after load. There is a known suspected
divergence in the `remap` method — see [baking.md](baking.md#5-frontendbackend-consistency)
and [ADR-005](decisions.md#adr-005).

**Uploaded-texture options traverse the same pipeline.** A `recipe.method = "image"` option
bakes by fetching the seller's validated upload, normalising it, copying it into the
Configurator namespace and recording texture rows — same statuses, same `recipe_hash`, same
staleness rule ([ADR-013](decisions.md#adr-013)). Their preview needs no client-side
recolouring at all, which means they are unaffected by the unresolved `remap` divergence and
can ship while ADR-005 stays open.

### 8.1 Storage

**CONFIRMED — existing helpers.**

| Helper | Path shape | Notes |
|---|---|---|
| `upload_file_content` | `{container}/users/{user_id}/uploads/{uuid4}/{name}` | behind `POST /uploads/content` |
| `upload_product_image` | `{container}/{user_id}/{product_id}/{name}` | product media |
| `upload_variant_model` | `{container}/products/{product_id}/variants/{config_hash}.glb` | sets `Cache-Control: public, max-age=31536000, immutable` |

`_sanitize_filename` appends a random 5-hex suffix to every name, and every upload passes
`overwrite=True` (`app/services/storage.py:25-63`).

**PROPOSED — one new method,** `upload_configurator_texture`, mirroring
`upload_variant_model`'s content-addressed, immutable-cache shape:

```
{container}/configurator/{product_id}/{glb_version}/{option_id}/{material_index}-{recipe_hash}.{ext}
```

Content-addressing by `recipe_hash` makes re-baking an unchanged recipe a no-op overwrite of
identical bytes, and makes the URL safe to cache forever.

**Lifecycle.** Deleting an option deletes its textures' blobs before the rows, following
`variant_bake_service.purge_variant_assets`. Superseded blobs are deleted only *after* the
replacement row is committed, following `variant_bake_service` lines 290-293, so a viewer
mid-request never hits a 404.

A seller-uploaded image is **copied** out of `users/{user_id}/uploads/…` into the Configurator
namespace before it is recorded, so purging an option never touches a blob the seller owns
elsewhere ([data-model.md §5.6](data-model.md#56-uploaded-images-are-copied-not-referenced)).

**🔴 The Configurator blob prefix is product-scoped**, the same shape
`ACCOUNT_PURGE_JOB_HANDOFF.md` §14 already flags for `products/{product_id}/variants/…`.
It must be added to that job's blob deletion order before production deployment — see
[ADR-010](decisions.md#adr-010).

### 8.2 GLB identity

**CONFIRMED**

- No content hash of GLB bytes exists anywhere in the repository. `compute_config_hash`
  hashes the source URL *string* plus the recipe (`variant_bake_service.py:72-86`), and
  `model_cache._key` hashes the URL string (`model_cache.py:53`).
- Every observed write path that stores a mesh **creates a new `ProductAsset` row** with a
  freshly randomised blob name. There is no "replace the GLB" endpoint.
- However, the codebase does mutate an asset row's URL in place for a replace operation
  elsewhere: `PUT /products/{id}/original-image` sets `primary_asset.image = blob_url` on the
  existing `asset_id == 1` row (`app/services/product_service.py:2140`). The precedent for
  in-place URL mutation exists; it simply has not been applied to meshes.
- All uploads use `overwrite=True`. Nothing at the storage layer *forbids* rewriting a path.

**Therefore: URL-as-identity is currently accurate but not guaranteed.** The randomised
filename suffix is a collision-avoidance measure, not an immutability policy.

**PROPOSED** — the Configurator stores a `glb_version` on every Part, and validates it on
every write and bake. Which value fills it is [ADR-006](decisions.md#adr-006), still open:
the mesh `ProductAsset.id` (a stable database identity for "this exact uploaded file", and
already immutable per-row), or a SHA-256 of the GLB bytes. **Do not use a hash of the URL
string** — it proves nothing about the file's contents.

---

## 9. Relationship to the existing colour-variant feature

**CONFIRMED — what exists.** A working, shipped-but-unexercised colour configurator:
`tbl_product_color_variants` + `tbl_variant_assets`, seven routes, a full recolour engine,
and a bake service. Details in this repo's audit; the load-bearing facts:

- It bakes a **complete GLB per colourway** (`variant_bake_service._bake_bytes` →
  `glb_recolor.recolor` → `_rebuild_and_write`), which is exactly the pattern
  [ADR-003](decisions.md#adr-003) rejects for the Configurator.
- Its tables were created by a **hand-run SQL script**, `sql/create_color_variants.sql`, not
  by Alembic. No revision in `migrations/versions/` creates them.
- Bakes run on FastAPI `BackgroundTasks` with no `bake_started_at`, no reaper, and no retry.
- It has **no test coverage**.
- It enforces authentication but not product ownership.

**PROPOSED — coexistence, not replacement, in this phase.**

1. The Configurator is additive. Do not modify, delete, or migrate the colour-variant tables
   or routes as part of Configurator work.
2. **Extract, do not fork.** The pure colour maths in `app/services/color/` is the shared
   asset. The Configurator's `texture_baker` should call into it. Refactoring
   `glb_recolor._recolor_pixels` into a texture-in/texture-out function that both features
   call is the one sanctioned change to existing code — and it belongs in its own reviewed
   PR, not in the Configurator's first phase.
3. Product/mesh resolution (`get_product_model_url`, `MESH_ASSET_ID`) is duplicated logic
   waiting to happen. Extract it to a shared helper both repositories call.
4. Decide the colour-variant feature's fate separately once the Configurator ships — it is a
   product decision, not an architectural one. Recorded in
   [decisions.md](decisions.md#open-questions) Q6.

---

## 10. Seller vs shopper boundary

### 10.1 Seller surface

Requires bearer JWT **and** product ownership. Exposes the full editing model: parts,
material index membership, recipes, bake status, bake errors, timestamps.

### 10.2 Shopper surface

Served from the existing `/public/...` tier (HTTP Basic, `verify_public_basic_auth`).
Returns only what the viewer renders:

```json
{
  "product_id": "...",
  "model_url": "https://cdn/.../model.glb",
  "ar_model_url": "https://cdn/.../model.usdz",
  "parts": [
    {
      "id": "...",
      "name": "Seat",
      "material_indices": [0, 3],
      "default_option_id": "...",
      "options": [
        {
          "id": "...",
          "name": "Red",
          "swatch_hex": "#C0182B",
          "textures": [{"material_index": 0, "url": "https://cdn/..."}]
        }
      ]
    }
  ]
}
```

Excluded by rule: `bake_status`, `bake_error`, `bake_started_at`, `recipe`, `glb_version`,
`blob_url`, `created_by`, `updated_by`, any option whose bake is not `completed`, and any
part or option where `shopper_selectable` is false.

### 10.3 Security requirements

Enforced server-side, in services, never in routes and never in the frontend:

1. Authenticated user (existing `CurrentUser`).
2. **Product ownership** — `Product.created_by == current_user.id`, plus
   `deleted_at IS NULL`. This is new; the neighbouring domains do not do it (§3).
3. Part belongs to the product named in the path.
4. Option belongs to the part named in the path.
5. Every `material_index` in a part is `0 <= i < len(gltf.materials)` for *that product's*
   current GLB.
6. A material index belongs to at most one active part per product. Enforced in
   `PartService` under a `SELECT ... FOR UPDATE` on the product row — see
   [ADR-012](decisions.md#adr-012) and
   [data-model.md §9](data-model.md#9-material-index-uniqueness-jsonb--a-row-lock).
7. Every recipe colour is validated and normalised before storage.
8. For an `image` recipe, `image_url` must resolve inside **this seller's own** upload
   namespace (`users/{current_user.id}/uploads/…`). Unvalidated it is an SSRF vector — the
   baker dereferences it server-side — and a cross-tenant hotlink vector. See
   [ADR-013](decisions.md#adr-013).

Failing 1 → 401. Failing 2 → 404 (not 403 — do not confirm the existence of another
seller's product). Failing 3-7 → 400.

---

## 11. Deployment and future worker evolution

**The Configurator is not a microservice.** It ships inside this FastAPI application, in the
same container, behind the same auth. [ADR-001](decisions.md#adr-001) explains why a domain
boundary is not a deployment boundary.

**Bake execution is the one part designed to move.** `BakeService` exposes a stable contract:

```python
async def request_bake(db, option_id, *, force: bool = False) -> BakeTicket
async def get_bake_status(db, option_id) -> BakeStatusView
```

Behind it, `bake_runner` holds the execution strategy. Phase 1 uses FastAPI
`BackgroundTasks`, matching `variant_bake_service`. Nothing outside `bake_runner` may know
that. When scale demands it, the runner is swapped without touching routes, schemas, or the
public API contract.

**CONFIRMED — two migration targets already exist in this repo:**

- `app/services/usdz_trigger_service.py` fires an Azure Container Apps Job over the ARM API
  with `DefaultAzureCredential`, retries with backoff, fire-and-forget.
- `app/integrations/service_bus_publisher.py` publishes to Azure Service Bus.

Either is a viable phase-2 backend. Neither should be introduced in phase 1 —
[ADR-005](decisions.md#adr-005) and §21 of the brief both say build the seam, not the queue.

**CONFIRMED risk that forces the seam to be real.** Prod runs on Azure Container Apps with
`minReplicas=1` and autoscaling (`.github/workflows/deploy-prod.yml:445-449`). A bake that
commits `baking` and then spends 30-90s on I/O will be lost if the replica is recycled
mid-flight, leaving the row stuck in `baking` forever. The exact scale ceiling and
termination grace period live in Azure configuration, not in this repo —
**NEEDS VERIFICATION**. The mitigation (`bake_started_at` + a startup sweep) is mandatory
from day one regardless, and is specified in [baking.md](baking.md#4-failure-handling).

---

## 12. CORS

**CONFIRMED — the API sends permissive CORS.** `app/main.py:202-208`:
`allow_origins=["*"]`, all methods, all headers.

**NOT CONFIRMED — the CDN.** Nothing in this repository configures CORS on the Azure Blob
service or the Front Door endpoint. There is no IaC (`.github/` holds only workflow YAML;
no Bicep, no Terraform). `storage_service` sets `ContentSettings(content_type=...)` and, for
variant models, `cache_control` — no CORS rule. `BlobToCdnMiddleware` rewrites URL strings in
JSON bodies and adds no headers.

**This is a hard prerequisite, not a nicety.** The preview path reads texture pixels in the
browser. Without `Access-Control-Allow-Origin` on the CDN response — and a
`crossorigin="anonymous"` attribute on the loader — the canvas is tainted and `getImageData`
throws. Preview cannot work.

**Action: verify before implementation begins.**

```bash
curl -I -H "Origin: https://portal.example.com" "$CDN_BASE_URL/$CONTAINER/<a-known-texture>"
# expect: Access-Control-Allow-Origin
```

Tracked as [ADR-005](decisions.md#adr-005) prerequisite and
[decisions.md](decisions.md#open-questions) Q1.

---

## 13. Observability

Log via the stdlib logger already configured in `app/main.py` (file + console + OTel trace
correlation). Emit at minimum:

| Event | Level | Fields |
|---|---|---|
| bake requested | INFO | `option_id`, `part_id`, `product_id`, `recipe_hash`, `glb_version` |
| bake started | INFO | + `source_texture_count`, `source_bytes` |
| bake completed | INFO | + `duration_ms`, `texture_count`, `output_bytes` |
| bake failed | ERROR | + `duration_ms`, exception (`logger.exception`) |
| bake skipped (already current) | DEBUG | `option_id`, `recipe_hash` |
| stale bake recovered | WARNING | `option_id`, `stuck_for_seconds` |
| invalid material index rejected | WARNING | `product_id`, `material_index`, `material_count` |
| GLB version mismatch rejected | WARNING | `product_id`, `expected`, `actual` |
| storage upload failure | ERROR | `option_id`, `blob_path`, exception |

Never log: bearer tokens, `Authorization` headers, connection strings, SAS tokens, user email
addresses, or raw texture bytes. Note `app/main.py:294-302` already logs `net.peer.ip` and
`http.user_agent` per request — do not add more PII on top.

**PROPOSED metrics** (via OTel, if the Azure Monitor exporter is enabled): bake duration
histogram, bake outcome counter by status, stale-recovery counter, queue depth once a real
runner exists.

---

## 14. Frontend / backend responsibility split

| Concern | Frontend | Backend |
|---|---|---|
| Show the model | ✅ loads GLB | serves URL |
| Show parts and options | ✅ | serves configuration payload |
| Interactive preview | ✅ applies recipe to texture in-browser | — |
| Recipe *specification* | must match the spec | **owns the spec** |
| Validate material membership | may hint | ✅ **authoritative** |
| Validate ownership | ❌ never trusted | ✅ **authoritative** |
| Persist parts/options | — | ✅ |
| Bake and store textures | — | ✅ |
| Bake status display | ✅ polls | ✅ reports |
| Decide which options a shopper sees | — | ✅ filters before sending |

The rule: the frontend may *duplicate* a backend rule for responsiveness; it may never be
the *only* place a rule is enforced.
