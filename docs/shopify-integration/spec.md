# Shopify Integration — Design Specification

> **Status: IMPLEMENTED on branch `shopify-integration-apis/supriya` (uncommitted), 2026-09-29.**
> The section below, "As built", is authoritative where it differs from the design sections
> that follow them.

## As built (read this first)

| Piece | Where | Migration |
|---|---|---|
| API keys (`riv_live_…`, SHA-256, scopes) | `docs/api_keys.md`, `app/*/api_key*` | `a5c1e9d4b7f2` |
| Layout from photo (ADR-015) | `app/services/configurator/model_variant_generation_service.py`, `generation_runner.py` | `b7d3f1a2c9e4` |
| Shopify module (ADR-016) | `app/api/routes/shopify.py`, `app/services/shopify/`, `app/models/shopify.py` | `c9e5a3b1d8f6` |

**Differences from the design sections below:**

1. **Auth:** no per-shop key table. The plugin uses a general **API key** (docs/api_keys.md),
   then binds it to the shop once with `POST /integrations/shopify/connect {shop_domain}`
   (`tbl_shopify_connections`). The shop for every other call comes from that binding.
   Scopes: `read` (connect, state), `write` (sync, options, accept, discard, unlink, redact),
   `convert` (the two `glb` calls, which spend AI credits).
2. **Publishing is not in this module.** It stays in `Rivollo.Viewer.Api`
   (`POST /api/products/{id}/publish`). `GET …/products/{id}` returns `public_id` and
   `viewer_url` (`VIEWER_BASE_URL` + `/` + `public_id`) once the product is published.
3. **Layouts' models are derived, not stored.** `tbl_shopify_layouts` has no
   `model_variant_id`: a layout's model is the live variant of its newest accepted generation
   (`client_ref = "shopify-layout:<id>"`), so auto-accept needs no bookkeeping.
4. **Path ids are numeric** (`/products/10575538127127`); a URL-encoded GID cannot route.
   Bodies accept either form.
5. **Sync ignores unknown fields**, so newer plugin builds do not break it; a variant's old
   single `image_url` is folded into `image_urls`.
6. `tbl_shopify_products.main_glb_requested_at` was added to detect a stalled main GLB.
7. `/createProductFal` is untouched; the new paths use `app/services/generation_gate.py`
   (same rules and messages).
8. **Standard foreign keys, not the "no FK" isolation rule** in the design notes below
   (decision 2026-09-29, because the purge job is being updated alongside). Ownership and
   links are real FKs, all `ON DELETE CASCADE`: `user_id → tbl_users` on `tbl_api_keys`,
   `tbl_shopify_connections`, `tbl_shopify_products`; `rivollo_product_id → tbl_products`;
   `api_key_id → tbl_api_keys`. Audit columns stay plain UUIDs (`AuditMixin`). The purge job's
   changes are in [../account-purge-job-changes.md](../account-purge-job-changes.md).

### Endpoints (all `Authorization: Bearer riv_live_…`, behind `ENABLE_SHOPIFY_INTEGRATION`)

| Method | Path | Scope | Result |
|---|---|---|---|
| POST | `/integrations/shopify/connect` | read | `{shop_domain}` → connection (idempotent; 409 if the shop is on another account) |
| GET | `/integrations/shopify/connection` | read | the binding |
| POST | `/integrations/shopify/uninstall` | any | unbind (key stays valid) |
| POST | `/integrations/shopify/shop/redact` | write | delete the shop's Shopify rows, unbind; Rivollo products kept |
| POST | `/integrations/shopify/products/sync` | write | 201 first time (draft Rivollo product created), 200 after |
| GET | `/integrations/shopify/models` | read | models, credit cost, ETA (the plugin cannot call `/ai/3d-models`) |
| GET | `/integrations/shopify/products` | read | synced products |
| GET | `/integrations/shopify/products/{id}` | read | full state — **poll this** |
| PUT | `/integrations/shopify/products/{id}/options` | write | `{roles, original_layout_value}` → state |
| POST | `/integrations/shopify/products/{id}/glb` | convert | 202, main GLB (`{image_url, model?, retry?}`) |
| POST | `/integrations/shopify/products/{id}/layouts/{layout_id}/glb` | convert | 202, candidate (`{image_url, model?, auto_accept?}`) |
| POST | `/integrations/shopify/products/{id}/generations/{gid}/accept` | write | 201 (200 if already accepted) |
| DELETE | `/integrations/shopify/products/{id}/generations/{gid}` | write | discard |
| DELETE | `/integrations/shopify/products/{id}` | write | unlink |
| GET | `/public/products/{product_id}/shopify` | none | shopper payload (published products only) |

State values: `rivollo_product.main_glb_state` = `none | generating | stalled | ready`;
`layouts[].state` = `none | generating | ready_for_review | ready | failed`.

### Deployment checklist

1. `alembic upgrade head` (three new revisions; each only creates tables).
2. 🔴 Deploy the `Rivollo.AccountPurge.Job` change in
   [../account-purge-job-changes.md](../account-purge-job-changes.md) **before or with** these
   tables in any environment where the purge runs: it allow-lists 3 new FKs to `tbl_users` and
   2 new FKs to `tbl_products`. Without it the purge's contract check (A14) aborts every run —
   safely, before deleting anything.
3. Set `VIEWER_BASE_URL` (e.g. `https://view.rivollo.com`). `ENABLE_SHOPIFY_INTEGRATION` is **on by default** (2026-09-29): every environment running this build needs the tables; set it `false`
   where wanted.
4. Portal: Settings → API Keys screen (docs/api_keys.md). Plugin: §12, with the auth change above.
5. Viewer: read `/public/products/{id}/shopify` (directly or mirrored in Viewer.Api — D5).

---

> Original design notes follow.
> Tags: **CONFIRMED** = observed in this repository · **PROPOSED** = this document's
> recommendation · **NEEDS VERIFICATION** = runtime, Azure, database, Shopify, or another
> repository.

Date: 2026-09-28 · Branch at time of writing: `save-variants-glb/supriya`

---

## 0. Summary

The Rivollo 3D Shopify app (plugin, separate codebase `rivollo3d/`) lets a merchant sync a
Shopify product into Rivollo, turn its photos into 3D, and push the GLB back to Shopify. This
document specifies the backend the plugin needs.

It is built in three phases, each shippable on its own:

| Phase | What | Where |
|---|---|---|
| **1. Layout from photo** | Generate a model variant (a "Layout" tile) from one image: *generate → preview candidate → accept*. Generic, usable by the Rivollo portal too. | Configurator domain (ADR-015) |
| **2. Shopify module** | Per-shop API keys; sync creates (or updates) a **draft Rivollo product** linked to the Shopify id; "create GLB" calls for the **main model** and for each **layout**; option roles; uninstall / shop-redact. | New, isolated `shopify` domain (ADR-016) |

### End-to-end flow (v1)

```
1. Sync            POST /integrations/shopify/products/sync
                   ─► Rivollo product created, status DRAFT, thumbnail = main Shopify image (copied)
                   ─► Shopify ids stored in tbl_shopify_products, linked to that product
                   (re-sync updates the Shopify mirror; the same product is reused)

2. Option roles    PUT …/products/{id}/options     "Layout" = layout, "Color" = info, …

3. Main GLB        POST …/products/{id}/glb {image_url}
                   ─► existing fal pipeline on the SAME product: DRAFT → QUEUE → PROCESSING → READY
                      (WebSocket, notification, Draco, USDZ, all as today)

4. Layout GLBs     POST …/products/{id}/layouts/{layout_id}/glb {image_url}   (per layout, any image)
                   ─► candidate ─► preview ─► accept ─► model variant (Layout tile)
                   (can start as soon as sync is done; does not wait for step 3)

5. Back to Shopify GET …/products/{id} ─► GLB URLs ─► plugin uploads the main GLB to Shopify media

6. Publish         plugin calls Viewer.Api publish when ready ─► publicId + full viewer link
                   ─► viewer shows layouts + Shopify price / add to cart
```
| **3. Viewer read** | Public payload mapping each Shopify variant to a layout + price + availability + add-to-cart link. | Shopify domain + `Rivollo.Viewer.Api` + viewer frontend |

**Deferred (not in v1):** colour. The configurator's colour options (flat recolour and
uploaded image textures, ADR-013) will later be auto-matched to Shopify colour values and fed
from Shopify swatch images. v1 treats colour as an informational option (§7.4). The schema is
laid out so that colour arrives without a migration to the tables defined here.

### Principles (from the review discussion)

1. **The existing system does not change.** Existing endpoints, tables and responses keep their
   behaviour. The complete list of edits to existing files is in §10.
2. **Reuse, don't duplicate.** Generation reuses `fal_queue_client.generate_3d`; model
   variants reuse `ModelVariantService.create_variant`; the main GLB reuses
   `ProductService._run_fal_3d_generation_background` on the product created at sync.
3. **Shopify is the source of commerce truth.** Rivollo stores a mirror of what the viewer
   displays (price, availability, options), refreshed by the plugin on every change.
4. **Generation is seller-side only.** Shoppers never trigger generation. It costs AI credits
   and takes minutes.

---

## 1. What exists today

| Fact | Evidence | Tag |
|---|---|---|
| `POST /createProductFal` creates a product from `{userId, name, imageURL, model?}`, gates on plan, charges `ai_credits`, and generates in a FastAPI `BackgroundTask`. **It has no auth dependency and trusts `userId` from the body.** | `app/api/routes/products.py:474-621` | CONFIRMED |
| It stores `imageURL` **verbatim** as the product's asset-1 (thumbnail) row. A Shopify CDN URL passed there becomes the product thumbnail on Shopify's CDN. | `app/services/product_service.py:757-760` | CONFIRMED |
| Credits are charged **before** generation runs and are **not refunded** on failure. | `products.py:537-572` | CONFIRMED |
| `GET /products/{id}/assets` requires a JWT but **does not check ownership**. | `products.py:1109-1125` | CONFIRMED |
| No `/products/sync`, no API-key mechanism, no Shopify identifiers anywhere. | grep | CONFIRMED |
| JWT access tokens last 60 minutes by default. A static token in the plugin's env expires. | `app/core/config.py:35` | CONFIRMED |
| `hash_token()` (SHA-256) exists and `tbl_app_tokens` already stores token hashes, not tokens. | `app/core/security.py:59`, `docs/app_token_authentication.md` | CONFIRMED |
| A model variant can **only** be created by uploading a GLB. | `app/api/routes/configurator.py:180-239` | CONFIRMED |
| `ModelVariantService.create_variant(db, product_id, user_id, *, name, glb: UploadedFile, thumbnail)` takes raw bytes and owns ownership check, Draco + re-check, blob path, unmapped asset row, `created_by`, USDZ request. | `app/services/configurator/model_variant_service.py:343-462` | CONFIRMED |
| `fal_queue_client.generate_3d(spec=, product_id=, image_url=)` returns `glb_bytes` and is independent of product creation. | `product_service.py:1107-1111` | CONFIRMED |
| One image per generation: `FalModelSpec.build_body(image_url)` substitutes a single field. | `app/integrations/fal/registry.py:83-90` | CONFIRMED |
| `validate_image_url_ownership(image_url, user_id)` checks a URL is in the caller's `users/{user_id}/uploads/` namespace. | `app/services/configurator/recipe.py:85` | CONFIRMED |
| Durable background work pattern: `bake_runner.enqueue()` (`asyncio.create_task`, semaphore, own session) + a startup and periodic sweep wired in `lifespan`. | `app/services/configurator/bake_runner.py`, `app/main.py:94-177` | CONFIRMED |
| `Rivollo.Viewer.Api` mirrors the configurator shopper payload (`/configurator`). | ADR-014 §11 | CONFIRMED (documented); its code is in another repo |
| Migration head is `e3b9c6a1d27f`. | `migrations/versions/` | CONFIRMED |
| Configurator is fixed at "exactly four tables". | `CLAUDE.md`, `data-model.md` | CONFIRMED |

---

## 2. Architecture

```
Shopify store ──OAuth──► Shopify app (rivollo3d)                         Rivollo portal (seller)
                              │  X-Rivollo-Api-Key                                │ JWT
                              ▼                                                   ▼
               ┌──────── /integrations/shopify/* ────────┐     /products/{id}/configurator/model-variants/generate
               │  app/api/routes/shopify.py               │                        │
               │  app/services/shopify/*                  │                        │
               │  app/database/shopify_repo.py            │                        │
               │  app/models/shopify.py  (tbl_shopify_*)  │                        │
               └───────┬──────────────────────┬───────────┘                        │
                       │ calls (unchanged)    │ calls                               │
                       ▼                      ▼                                     ▼
     ProductService: draft product     ModelVariantGenerationService  ◄────────────┘   (Phase 1, configurator)
       (sync) + fal background               │ generation_runner.enqueue()
       generation (main GLB)                 │
                                             ├─► fal_queue_client.generate_3d   (unchanged)
                                             └─► accept ─► ModelVariantService.create_variant (unchanged)

Viewer ──► Viewer.Api /configurator (layouts; unchanged)   +   /public/products/{id}/shopify (Phase 3, new)
```

- **Phase 1** belongs to the configurator, because model variants do, and the portal
  benefits too (ADR-015).
- **Phase 2** is an isolated module with its own namespace, tables, auth and feature flag.
  It **calls** existing services and never modifies them (ADR-016).
- Removing Phase 2 = delete one `include_router` line + downgrade one migration.

---

## 3. Phase 1 — Layout from photo (configurator)

### 3.1 Flow

```
POST …/model-variants/generate {name, image_url}
   ownership · flag · image in caller's uploads · model resolved · plan gate · credits charged
   ─► row status=queued ─► generation_runner.enqueue(id, spec)
                               status=generating, started_at
                               fal_queue_client.generate_3d(image_url)
                               upload raw GLB as a *candidate* blob
                           ─► status=ready  (candidate_glb_url set)     ─or─►  status=failed (error)
seller previews candidate (portal / plugin; NOT the shopper viewer)
POST …/generations/{id}/accept   ─► ModelVariantService.create_variant(glb=candidate bytes,
                                                                        thumbnail=source image)
                                 ─► status=accepted, accepted_variant_id; candidate blob deleted
DELETE …/generations/{id}        ─► status=discarded; candidate blob deleted
```

A candidate is **never** a model variant row. `tbl_product_model_variants` keeps its invariant
that every variant has a GLB, so no existing reader (`list_models`, shopper payload,
Viewer.Api) has to learn a "generating" state.

### 3.2 Endpoints (JWT, `CurrentUser`; ownership → 404 per ADR-008; behind `ENABLE_MODEL_VARIANTS`)

| Method | Path | Result |
|---|---|---|
| `POST` | `/products/{product_id}/configurator/model-variants/generate` | `202`, generation |
| `GET` | `/products/{product_id}/configurator/model-variants/generations` | `200`, list (`?status=` filter; newest first) |
| `GET` | `/configurator/model-variant-generations/{generation_id}` | `200`, generation |
| `POST` | `/configurator/model-variant-generations/{generation_id}/accept` | `201`, the created model variant (existing `ModelVariantCreateResponse`) |
| `DELETE` | `/configurator/model-variant-generations/{generation_id}` | `200`, generation (`discarded`) |

**Request: `POST …/generate`** (snake_case, no aliases)

```json
{
  "name": "4 Seater-corner",
  "image_url": "https://<cdn>/<container>/users/<user_id>/uploads/<upload_id>/corner.jpg",
  "model": null,
  "client_ref": null
}
```

- `name`: same validation as `create_variant` (`_validate_name`).
- `image_url`: **must** pass `validate_image_url_ownership`. Sellers upload through the
  existing `POST /uploads/content`. The server never fetches an arbitrary URL, and fal is given
  only our own CDN URL.
- `model`: registry key, `null` = registry default. Same semantics as `/createProductFal`.
- `client_ref`: optional opaque tag (≤ 200 chars) the caller uses to find its generations. The
  Shopify module sets `shopify-layout:<uuid>`. The configurator never interprets it.

**Response: generation**

```json
{
  "id": "uuid",
  "product_id": "uuid",
  "name": "4 Seater-corner",
  "source_image_url": "https://…/corner.jpg",
  "model": "tripo-h3.1",
  "credit_cost": 10,
  "status": "queued | generating | ready | failed | accepted | discarded",
  "error": null,
  "candidate_glb_url": null,
  "accepted_variant_id": null,
  "client_ref": null,
  "estimate": { "...": "same shape as /createProductFal's `gpu` field" },
  "started_at": null,
  "completed_at": null,
  "created_at": "2026-09-28T11:00:00Z"
}
```

`candidate_glb_url` is set only in `ready`. It is the raw fal output (not Draco-compressed),
fine for a seller preview. Compression happens at accept, inside `create_variant`.

**Errors** (`HTTPException` → `{"detail": …}`, matching neighbours; Q2 unchanged)

| Case | Status |
|---|---|
| Feature off | 404 |
| Product not found / not owned; generation not found / not owned | 404 |
| `image_url` not in caller's uploads | 400 |
| Unknown / deactivated `model` | 400 |
| Plan does not allow the model | 403 (same message as `/createProductFal`) |
| Not enough credits | 400 (same message as `/createProductFal`) |
| `accept` when status ≠ `ready` | 409 (`accepted` returns the existing variant, idempotent: `200`) |
| `accept` when the candidate GLB fails `inspect_glb` | 400, generation → `failed` |
| `DELETE` when `accepted` | 409 (delete the model variant instead) |

### 3.3 Table `tbl_model_variant_generations` (5th configurator table)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | `UUIDMixin` |
| `product_id` | UUID NOT NULL | FK → `tbl_products` **ON DELETE CASCADE** (new product FK, see §9) |
| `name` | TEXT NOT NULL | |
| `source_image_url` | TEXT NOT NULL | caller's upload URL |
| `model_key` | TEXT NOT NULL | resolved at request time |
| `credit_cost` | INTEGER NOT NULL | what was charged |
| `status` | TEXT NOT NULL default `'queued'` | no CHECK constraint (repo convention) |
| `error` | TEXT | seller-safe message only |
| `started_at`, `completed_at` | TIMESTAMPTZ | `started_at` drives the sweep |
| `candidate_glb_url`, `candidate_glb_blob_url` | TEXT | |
| `candidate_size_bytes` | BIGINT | |
| `accepted_variant_id` | UUID | FK → `tbl_product_model_variants` **ON DELETE SET NULL** |
| `client_ref` | TEXT | |
| `created_by`, `created_date`, `updated_by`, `updated_date` | | `AuditMixin`; **no FK** on `created_by` (ADR-010) |

Indexes: `(product_id, created_date DESC)`; `(product_id, client_ref)`; `accepted_variant_id`;
partial `ix_generations_in_flight ON (started_at) WHERE status IN ('queued','generating')`
for the sweep.

### 3.4 Runner and durability (ADR-007 pattern)

- `app/services/configurator/generation_runner.py`: `enqueue(generation_id, spec)` →
  `asyncio.create_task`, strong-reference set, `asyncio.Semaphore(settings.GENERATION_CONCURRENCY)`
  (PROPOSED default **2**: fal work is remote, but the downloaded GLB is held in memory), own
  session via `new_session()`. It never raises. Every path ends in `ready` or `failed`.
- The resolved `FalModelSpec` is passed in memory, as the product path does, so a registry
  edit cannot change a paid-for run. After a restart the in-memory spec is gone, and the
  sweep fails the row (no silent retry of a charged job).
- **Sweep:** `recover_stale_generations()` runs at startup and every
  `GENERATION_SWEEP_INTERVAL_SECONDS`, next to the bake sweep in `lifespan`.
  - `generating` with `started_at` older than `GENERATION_STALE_AFTER_SECONDS` (PROPOSED: the
    spec's `max_wait_seconds` + 10 min margin, default 30 min) → `failed`, "Generation was
    interrupted. Please try again."
  - `queued` older than the same threshold → `failed`, same message.
  - `ready` older than `GENERATION_CANDIDATE_TTL_DAYS` (PROPOSED 30) → `discarded`, blob deleted.
- Generation duration feeds `generation_estimate_service.record(...)` exactly as the product
  path does, so ETAs stay honest.

### 3.5 Storage

Candidates: `{user_id}/{product_id}/model-variants/candidates/{generation_id}/model.glb`, inside
the purge's user prefix (ADR-014 rule). Deleted on accept (after `create_variant` has stored
its own copy), discard and TTL. **NEEDS VERIFICATION:** whether an existing `storage_service`
method can write this path, or a small new helper is needed.

### 3.6 Plan and credit gate

The checks in `/createProductFal` (`products.py:496-572`: resolve spec, plan gate with the
free-model message, `check_quota`, `increment_usage`) must apply identically.

- **PROPOSED:** extract them, unchanged, into one helper (e.g.
  `LicensingService.authorize_generation(db, user_id, model_key) -> (spec, cost)` plus
  `charge_generation(...)`), and switch `/createProductFal` to call it. This is behaviour-
  preserving and gets a regression test, but **it edits `products.py`**. See decision **D2**.
- Alternative: duplicate the ~40 lines. Rejected by CLAUDE.md "reuse, don't duplicate" unless
  D2 says otherwise.

### 3.7 ADR-015 (Proposed) — Model variants can be generated from a photo, via reviewed candidates

**Context.** ADR-014 variants are upload-only. Sellers and the Shopify app need to create a
layout from a product photo. Generation is paid and slow, and output quality varies by photo.

**Decision.** A generation row tracks each attempt. The output is a private *candidate* until
the seller accepts it. Acceptance reuses `ModelVariantService.create_variant` unchanged. One
image per generation (the registry is single-image). The configurator gains a fifth table,
`tbl_model_variant_generations`.

**Consequences.** `CLAUDE.md` "exactly four tables" → five, with the spirit unchanged (no
normalised material table, no discriminator column). One more FK to `tbl_products` → the
account-purge job must allow-list it (Q8 grows). ADR-014 is amended to note that variants may
originate from an accepted generation. Credits are charged per attempt (refund policy: **D1**).

---

## 4. Phase 2 — Shopify module

### 4.1 Authentication: per-shop connections

A merchant connects a shop to their Rivollo account once. The plugin then sends
`X-Rivollo-Api-Key: rvl_shp_<random>` on every call.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/integrations/shopify/connections` | JWT (portal) | `{shop_domain}` → creates a connection, returns the key **once** |
| `GET` | `/integrations/shopify/connections` | JWT | list (shop, key prefix, created, last used, revoked) |
| `DELETE` | `/integrations/shopify/connections/{id}` | JWT | revoke |
| `POST` | `/integrations/shopify/uninstall` | API key | plugin's `app/uninstalled` webhook → revoke the calling key |

- Key = `rvl_shp_` + `secrets.token_urlsafe(32)`. Stored as `hash_token(key)` plus an
  8-character display prefix. The key is never logged.
- `get_shopify_connection` dependency (in `app/services/shopify/auth.py`): hash the header →
  active row → `(connection_id, user_id, shop_domain)`. Unknown, revoked or missing key → 401,
  with one generic message. Updates `last_used_at` at most once per minute.
- **The shop comes from the key, never from the body.** This removes `shop_domain` from the
  sync payload entirely and makes cross-shop writes impossible by construction.
- `shop_domain` is normalised (lower-case, must match `^[a-z0-9][a-z0-9-]*\.myshopify\.com$`).
  One active connection per shop (partial unique index).
- The user must still exist and be active. A deleted Rivollo user → 401.

**Portal work:** a small "Connect Shopify" settings screen that calls the three JWT endpoints
and shows the key once. (Frontend, other repo.)

### 4.2 Tables (`app/models/shopify.py`)

**Isolation rule:** no FK from any `tbl_shopify_*` table to a core table (`tbl_users`,
`tbl_products`, configurator tables). References are plain UUIDs, and services treat a
missing or deleted target as "not there" (§4.8). FKs *between* `tbl_shopify_*` tables are
normal CASCADE FKs.

**`tbl_shopify_connections`**

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `user_id` | UUID NOT NULL | plain UUID |
| `shop_domain` | TEXT NOT NULL | |
| `key_hash` | TEXT NOT NULL UNIQUE | |
| `key_prefix` | TEXT NOT NULL | display only |
| `last_used_at`, `revoked_at` | TIMESTAMPTZ | |
| audit | | `AuditMixin`, no FK |

`ux_shopify_connections_active_shop ON (shop_domain) WHERE revoked_at IS NULL`.

**`tbl_shopify_products`**

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `user_id` | UUID NOT NULL | owner (from connection) |
| `shop_domain` | TEXT NOT NULL | |
| `shopify_product_id` | BIGINT NOT NULL | numeric part of the GID |
| `title`, `handle` | TEXT NOT NULL | |
| `description_html`, `vendor`, `product_type` | TEXT | |
| `tags` | JSONB `'[]'` | |
| `shopify_status` | TEXT NOT NULL | `ACTIVE` / `DRAFT` / `ARCHIVED`. **Never** written to `tbl_products.status` |
| `currency` | TEXT NOT NULL | ISO 4217, e.g. `INR` |
| `images` | JSONB `'[]'` | `[{id, url, alt_text}]` |
| `options` | JSONB `'[]'` | `[{name, values[]}]`, derived from variants |
| `option_roles` | JSONB `'{}'` | `{"Layout": "layout", "Color": "info"}` (§4.5) |
| `rivollo_product_id` | UUID NOT NULL | plain UUID (no FK, isolation rule), the draft product created at first sync |
| `synced_at` | TIMESTAMPTZ NOT NULL | |
| audit | | |

`ux_shopify_products_shop_product ON (shop_domain, shopify_product_id)`; index on `user_id`;
index on `rivollo_product_id` (Phase 3 lookup).

**`tbl_shopify_product_variants`**

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `shopify_product_ref` | UUID NOT NULL | FK → `tbl_shopify_products` CASCADE |
| `shopify_variant_id` | BIGINT NOT NULL | |
| `title` | TEXT NOT NULL | |
| `sku` | TEXT | |
| `price` | NUMERIC(12,2) NOT NULL | exact decimal, never float |
| `compare_at_price` | NUMERIC(12,2) | |
| `inventory_quantity` | INTEGER | **never exposed publicly** |
| `available` | BOOLEAN NOT NULL | |
| `image_urls` | JSONB `'[]'` | all media for this variant |
| `options` | JSONB `'[]'` | `[{name, value}]` |
| `position` | INTEGER NOT NULL | Shopify order |

`ux_shopify_variants_product_variant ON (shopify_product_ref, shopify_variant_id)`.

**`tbl_shopify_layouts`**: one row per value of the product's `layout` option

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `shopify_product_ref` | UUID NOT NULL | FK → `tbl_shopify_products` CASCADE |
| `option_value` | TEXT NOT NULL | e.g. `4 Seater-corner` |
| `is_original` | BOOLEAN NOT NULL default false | exactly one per product (partial unique index) |
| `model_variant_id` | UUID | plain UUID; NULL = Original, or not yet accepted |
| `position` | INTEGER NOT NULL | |

`ux_shopify_layouts_value ON (shopify_product_ref, option_value)`;
`ux_shopify_layouts_one_original ON (shopify_product_ref) WHERE is_original`.

Generations for a layout are found via `client_ref = 'shopify-layout:<layout id>'` (§3.2). No
Shopify column is added to the configurator table.

Deferred colour will add `tbl_shopify_colour_mappings` (option value → part option). **No
change to the four tables above.**

### 4.3 Endpoints (all `X-Rivollo-Api-Key`, prefix `/integrations/shopify`)

`{shopify_product_id}` is the **numeric** Shopify id (`10575538127127`). The plugin strips
`gid://shopify/Product/`. A product belonging to another shop → 404.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/products/sync` | Upsert: first sync **creates the draft Rivollo product**; later syncs update the mirror (§4.4). `201` created / `200` updated |
| `GET` | `/products/{shopify_product_id}` | Full integration state: roles, layouts, candidate images, generations, Rivollo product status + GLB URL |
| `PUT` | `/products/{shopify_product_id}/options` | Set option roles + which layout value is the Original (§4.5) |
| `POST` | `/products/{shopify_product_id}/glb` | Create the **main GLB** on the synced product (§4.6) |
| `POST` | `/products/{shopify_product_id}/layouts/{layout_id}/glb` | Create a **layout GLB** candidate from one image (§4.7) |
| `POST` | `/products/{shopify_product_id}/generations/{generation_id}/accept` | Accept → model variant; sets `layout.model_variant_id` |
| `DELETE` | `/products/{shopify_product_id}/generations/{generation_id}` | Discard a candidate |
| `DELETE` | `/products/{shopify_product_id}` | Unlink (Shopify rows only; the Rivollo product remains) |
| `POST` | `/uninstall` | Revoke the calling key (§4.1) |
| `POST` | `/shop/redact` | Shopify `shop/redact`: delete all `tbl_shopify_*` rows for the shop and revoke (§4.9, **D3**) |

`GET /products/{id}` replaces the plugin's polling of `GET /products/{id}/assets`. The
Shopify module reads the product's GLB itself (ownership-scoped through the connection), so
it does not inherit that endpoint's missing ownership check.

### 4.4 Sync payload: changes from the plugin's `rivollo-sync-api.md`

| Plugin spec | This spec | Why |
|---|---|---|
| `Authorization: Bearer <RIVOLLO_API_TOKEN>` | `X-Rivollo-Api-Key` | JWT expires in 60 min; per-shop identity |
| `POST /products/sync` | `POST /integrations/shopify/products/sync` | isolation; no collision with `/products/*` |
| 409 + unspecified `PUT /products/sync/:id` | **upsert**: 201 or 200 | idempotent; one endpoint; webhooks reuse it |
| `shopify_product_id` GID string | same (GID accepted, numeric id parsed and stored) | |
| — | **`currency`** (required, ISO 4217) | prices are meaningless without it |
| variant `image_url` | **`image_urls: []`** | a variant can have several media (**NEEDS VERIFICATION:** GraphQL field for variant media in API 2026-07) |
| root `status` → Rivollo status | stored as `shopify_status` only | `tbl_products.status` is the 3D pipeline state |
| `{id, status:"synced", …}` | `{"success": true, "data": {...}}` | repo envelope |
| `400 {"error":"validation_error", fields}` | `422 {"detail": [...]}` | FastAPI default; no custom handler exists |
| `401/409/500 {"error":…}` | `{"detail": "..."}`; unhandled 500 → `api_error` | repo convention (Q2) |

**Request**

```json
{
  "shopify_product_id": "gid://shopify/Product/10575538127127",
  "title": "Sofa with variant",
  "handle": "sofa-with-variant",
  "description": "<p>…</p>",
  "vendor": "Rivollo",
  "product_type": "Furniture",
  "tags": ["sofa"],
  "status": "ACTIVE",
  "currency": "INR",
  "images": [{ "id": "gid://shopify/ProductImage/1001", "url": "https://cdn.shopify.com/…", "alt_text": "…" }],
  "variants": [{
    "shopify_variant_id": "gid://shopify/ProductVariant/44001111",
    "title": "4 Seater-corner / Red",
    "sku": "SOF-C-R",
    "price": "499.00",
    "compare_at_price": "599.00",
    "inventory": 10,
    "available": true,
    "image_urls": ["https://cdn.shopify.com/…/corner-red.jpg"],
    "options": [{ "name": "Layout", "value": "4 Seater-corner" }, { "name": "Color", "value": "Red" }]
  }]
}
```

**Validation** (Pydantic, 422): `variants` 1–100; prices `^\d+(\.\d{1,2})?$`;
`currency` `^[A-Z]{3}$`; every URL in `images` / `image_urls` must be `https://cdn.shopify.com/…`
(stored, not fetched, at sync time); GIDs must match their resource type; string length caps
on every field. Simple products send one variant titled `Default Title` with `options: []`.

**First sync creates the Rivollo product**, in one transaction:

1. Copy the first product image (§6.2) into the user's uploads namespace. If the product has
   no image, the product is created without a thumbnail.
2. Create `tbl_products`: `name = title`, `description` = the Shopify description **converted
   to plain text** (the HTML stays in the mirror, since the portal and viewer must not render
   merchant HTML), `status = DRAFT`, `created_by = user_id`, unique slug via the existing
   `_generate_unique_slug`. When a thumbnail exists, add an asset-1 `tbl_product_assets` row
   plus a mapping row, exactly as `create_product_with_fal_image_urls` does
   (`product_service.py:743-782`). This is the same state `/createProductFal` leaves a product
   in before generation, so the portal lists it as an ordinary draft (CONFIRMED:
   `products_repo.py:61-101` reads the thumbnail from asset 1).
3. Insert the `tbl_shopify_*` rows with `rivollo_product_id` = the new product.

No AI credits and no plan requirement, the same as `createProductFromGlb`. The product-count
quota does not apply (CONFIRMED: generation is limited by AI credits, not product count,
`products.py:534`).

**Later syncs** update only the Shopify mirror (variants, prices, images, options). They
**do not** overwrite the Rivollo product's name, description or thumbnail, because the seller
may have edited them in the portal (**D10**). If the linked product was deleted in the portal,
the next sync creates a new draft product and re-links it.

The Shopify id lives in `tbl_shopify_products` (the link row), **not** as a new column on
`tbl_products`. That keeps the migration free of `ALTER` on a core table.

**Semantics:** upsert on `(shop_domain, shopify_product_id)`. The submitted variant list
**replaces** the stored one (variants no longer selected are removed). Layout rows are
reconciled with the `layout` option's current values: new values → new rows; removed
values → rows deleted, **their model variants are left untouched** in Rivollo (the seller
can delete them in the portal).

**Response**

```json
{ "success": true, "data": {
  "id": "uuid", "shopify_product_id": "10575538127127",
  "rivollo_product_id": "uuid", "rivollo_status": "draft",
  "synced_at": "2026-09-28T11:00:00Z", "variants_synced": 2, "created": true
}}
```

The plugin should also call sync from Shopify `products/update` webhooks (same endpoint), and
call `DELETE /products/{id}` from `products/delete`.

### 4.5 Option roles

`PUT /products/{id}/options`

```json
{ "roles": { "Layout": "layout", "Color": "info", "Size": "info" },
  "original_layout_value": "4 Seater-lounge" }
```

- Roles in v1: `layout` (each value is a 3D model) and `info` (shown as a selector, affects
  price / cart only). `colour` is reserved for the deferred phase and rejected with 400 in v1.
- At most **one** `layout` option. No `layout` option = single-model product.
- `original_layout_value` is required when a `layout` option exists. That value's layout is
  `is_original`.
- Unlisted options default to `info`.

### 4.6 Create the main GLB

`POST /products/{id}/glb`, body `{ "image_url": "<one of the synced images>", "model": null }`

The synced product already exists (§4.4). This call generates **its** model. It is the
product's Original: the main GLB that Shopify media, `/assets`, AR and the viewer use.

1. `image_url` must be one of this product's synced `images[].url` or any variant's
   `image_urls` (**allow-list from our own rows**, never a free URL). Else 400.
2. The linked product must be live and owned. Its status must be `draft` and it must have no
   active mapped GLB. `queue` / `processing` → 409 "already generating"; `ready` / `published`
   or an existing GLB → 409 "already has a model" (regenerating is a later feature, **D4**).
3. **Copy** the image into the user's uploads namespace (§6.2). fal is given our CDN URL,
   never Shopify's.
4. Authorize and charge through the shared gate (§3.6).
5. Schedule `ProductService._run_fal_3d_generation_background(user_id, product_id,
   mesh_asset_id=9, name, blob_url=<copied url>, spec)` **unchanged** (CONFIRMED: it works on
   an existing product, `product_service.py:1371-1426`). It moves the product
   `draft → queue → processing → ready`, broadcasts on the product WebSocket, writes the GLB
   asset + mapping, the Draco glTF package, the ETA sample and the "Product Ready"
   notification, and requests the USDZ, exactly as `/createProductFal` does. On failure the
   product returns to `draft` and the call can be retried.
   *The method is underscore-private today. PROPOSED: add a thin public alias,
   `ProductService.start_fal_generation(...)`, that calls it. That is additive, and no
   existing caller changes.*
6. Response: `{ rivollo_product_id, status: "queue", estimate }` (estimate shape as
   `/createProductFal`'s `gpu` field).

The plugin polls `GET /products/{id}` (or subscribes to the existing product WebSocket) until
`ready`, then reads the GLB URL and uploads it to Shopify media.

**Known limitation (existing behaviour, not changed here):** this generation runs as an
in-process background task with no sweep. A recycled replica leaves the product in `queue` /
`processing`. PROPOSED for v1: `GET /products/{id}` reports a main GLB stuck longer than
`GENERATION_STALE_AFTER_SECONDS` as `stalled`, and `POST …/glb` accepts `"retry": true` to
reset it to `draft` and try again (charged again, subject to **D1**).

### 4.7 Create a layout GLB

`POST /products/{id}/layouts/{layout_id}/glb`, body `{ "image_url": "…", "model": null }`

1. The layout must not be `is_original` (the Original's model is the main GLB, §4.6). Else 409.
2. `image_url` allow-listed from this product's synced images (§4.6.1). The picker in the
   plugin should offer that layout value's variant images first, then the product gallery.
3. Copy into uploads (§6.2) → call `ModelVariantGenerationService.request(...)` with
   `name = option_value`, `client_ref = "shopify-layout:<layout_id>"`. That service enforces
   ownership, plan, credits and durability (Phase 1).
4. **No need to wait for the main GLB.** A model variant only needs the product row, which
   exists from sync, so layouts can be generated in parallel with the main model. The viewer
   is only meaningful once the main GLB is `ready`.
5. Any number of attempts per layout. `GET /products/{id}` lists them per layout with their
   candidate GLB URLs for preview in the plugin.
6. Accept (`POST …/generations/{generation_id}/accept`) → Phase 1 accept → store
   `model_variant_id` on the layout. Accepting another candidate for a layout that already
   has a model variant is **409 in v1** ("replace layout model" is later).
7. Optional `"auto_accept": true` on the request: the runner accepts the candidate as soon as
   it is `ready`, for a one-click flow without preview (**D11**).

### 4.8 Dangling references

`rivollo_product_id` / `model_variant_id` can point to rows the seller deleted in the portal.
On read the module checks liveness. A deleted product → reported `null`, and the **next sync
creates a fresh draft product** and re-links it. A deleted model variant → the layout is
reported "not generated". Nothing is auto-cleaned in v1.

### 4.9 Uninstall and privacy webhooks

Shopify requires public apps to handle `customers/data_request`, `customers/redact` and
`shop/redact`. The plugin receives them.

- `customers/*`: **Rivollo stores no Shopify customer data**, so the plugin can acknowledge
  these without calling us. **NEEDS VERIFICATION** once the colour/analytics phases land.
- `shop/redact` → `POST /integrations/shopify/shop/redact`: delete every `tbl_shopify_*` row
  for the calling key's shop, then revoke the key. Whether Rivollo **products** generated
  for that shop are also deleted is **D3** (they belong to the Rivollo account, not to
  Shopify).
- `app/uninstalled` → `POST /integrations/shopify/uninstall` (revoke only; data kept until
  `shop/redact`, which Shopify sends 48 h later).

### 4.11 Publish from the plugin

**The GLB URL does not need publishing.** When the main GLB is `ready`, `GET /products/{id}`
returns its CDN URL (`{CDN_BASE_URL}/{container}/…`, built by `storage_service._cdn_url`;
CONFIRMED `storage.py:82-99`). The plugin downloads it and uploads it to Shopify, which then
hosts its own copy. **Publishing** creates the public **viewer** link (`PublishLink.public_id`),
which Phase 3 and a storefront embed need.

**Publishing is owned by `Rivollo.Viewer.Api`, not by this repository or the Shopify module**
(product decision, 2026-09-29). The portal already publishes through
`POST /api/products/{productId}/publish` there (`PublishController.cs`,
`PublishLinkService.cs`). The plugin calls the same endpoint. Nothing in this module writes
`tbl_publish_links` or the product status for publishing.

- Publishing requires status ready, and the main GLB is asynchronous. So the plugin calls
  Viewer.Api publish when its status poll first sees `ready`, the same moment it uploads the
  GLB to Shopify. Exposed as a plugin option ("Publish when ready", default on).
- Layouts accepted **after** publishing appear in the viewer automatically (the shopper payload
  reads live model variants). No republish is needed.
- **Full viewer link.** The viewer (`Rivollo.Viewer.Portal`) serves one page per product at
  `/[publicId]` (CONFIRMED `app/[publicId]/page.tsx`), so the link is
  `https://<viewer-host>/<publicId>`. Viewer.Api returns only `publicId` today. **Change in
  Viewer.Api:** add a `ViewerBaseUrl` setting per environment and return
  `viewerUrl = ViewerBaseUrl + "/" + publicId` in the publish response.
- `GET /integrations/shopify/products/{id}` returns `public_id` and `viewer_url` whenever the
  product is published (read from `tbl_publish_links`, using the same `VIEWER_BASE_URL` value
  configured here), so the plugin can re-read the link at any time.

**Viewer.Api changes needed for the plugin (cross-repo):**
1. Accept the plugin's API key (`riv_live_…`, SHA-256 lookup in the shared `tbl_api_keys`)
   on the publish endpoint.
2. Return `viewerUrl`.
3. Recommended while touching it: check that the caller owns the product. The method is
   `[AllowAnonymous]` today (`PublishController.cs:49`), so no credentials are needed to
   publish or unpublish a product by UUID.

This repository's own `POST /products/{id}/publish` (`products.py:1653`) duplicates the
Viewer.Api logic and appears unused by the portal. It is left untouched.

### 4.10 ADR-016 (Proposed) — Shopify integration is an isolated module

**Decision.** Own namespace (`/integrations/shopify`), own tables (`tbl_shopify_*`, no FKs into
core tables), own auth (hashed per-shop API keys), own flag (`ENABLE_SHOPIFY_INTEGRATION`,
off by default). It calls `ProductService`, `ModelVariantGenerationService` and
`LicensingService` and never modifies them. Commerce data is a mirror refreshed by the plugin.
**Sync creates an ordinary draft Rivollo product** (product row + asset-1 thumbnail +
mapping, the same rows `/createProductFal` writes before generating) and links it by id. No
column is added to any core table.

**Consequences.** Zero behaviour change to existing endpoints. Synced products appear in the
seller's portal as drafts, which is intended: the seller can finish them there too. No
account-purge FK contract change for these tables, but the purge job will not delete Shopify
rows for a purged user until it is taught to (follow-up, not a blocker). Products created by
sync are ordinary products and are purged normally. Dangling references are handled on read.

---

## 5. Phase 3 — Viewer read

### 5.1 `GET /public/products/{product_id}/shopify`

Unauthenticated, mirroring `/public/products/{id}/configurator` (whose router has no auth
dependency; CONFIRMED `configurator.py:87`). Returns 404 unless the product is **published**
(status `published` and an enabled `PublishLink`, the same test as `products.py:1270-1278`)
**and** linked to a Shopify product with a live connection. The viewer treats 404 as "no
commerce panel".

Separate shopper schema, never the internal one:

```json
{ "success": true, "data": {
  "title": "Sofa with variant",
  "currency": "INR",
  "product_url": "https://store.myshopify.com/products/sofa-with-variant",
  "layout_option": "Layout",
  "options": [
    { "name": "Layout", "role": "layout", "values": ["4 Seater-lounge", "4 Seater-corner"] },
    { "name": "Color",  "role": "info",   "values": ["Red", "Blue"] }
  ],
  "layouts": [
    { "value": "4 Seater-lounge", "model": "original" },
    { "value": "4 Seater-corner", "model": "<model_variant_id>" }
  ],
  "variants": [
    { "id": "44001111", "title": "4 Seater-corner / Red",
      "options": { "Layout": "4 Seater-corner", "Color": "Red" },
      "price": "499.00", "compare_at_price": "599.00", "available": true,
      "image_url": "https://cdn.shopify.com/…",
      "add_to_cart_url": "https://store.myshopify.com/cart/44001111:1" }
  ]
}}
```

- `layouts[].model` uses the same identifiers as the configurator shopper payload's model
  list (`"original"` or the variant id). **NEEDS VERIFICATION** against the exact field names
  in `shopper_service.py` / Viewer.Api before building.
- Layout values whose model is not accepted yet are **omitted** from `layouts`. Their Shopify
  variants still appear, and the viewer shows the Original for them.
- Never exposed: `inventory_quantity`, `sku`, GIDs, `shopify_status`, connection or user ids,
  audit columns.
- Prices are strings (exact decimals).
- `product_url` / `add_to_cart_url` use the `*.myshopify.com` domain. A custom storefront
  domain is a later addition.

### 5.2 Viewer behaviour

Picking a layout tile switches the model (existing) **and** narrows the Shopify variant. The
`info` selectors (colour, size) narrow it further. The viewer shows price / compare-at /
availability and an **Add to cart** button for the single matching variant.

### 5.3 Viewer.Api

ADR-014 §11 says Viewer.Api mirrors the configurator payload. **D5:** either Viewer.Api
mirrors this endpoint too (a query over `tbl_shopify_*`), or the viewer calls this API directly
for it. Cross-repo either way.

---

## 6. Security

### 6.1 Enforcement order (services, never routes)

Plugin: API key → connection active → user active → Shopify product belongs to the
connection's shop → layout / generation belongs to that product → image URL allow-listed
from our own synced rows.
Portal (Phase 1): JWT → product ownership (`created_by`, `deleted_at IS NULL`, 404) →
generation belongs to product → image URL in caller's uploads.

### 6.2 Copying Shopify images (the only outbound fetch)

`ShopifyImageImporter.copy_to_uploads(user_id, url)`:
- URL must be byte-for-byte one of the stored allow-listed URLs, scheme `https`, host exactly
  `cdn.shopify.com`.
- No redirects followed to any other host. Timeout 20 s. Size cap
  (`SHOPIFY_IMAGE_MAX_BYTES`, PROPOSED 20 MB) enforced while streaming. Content type
  `image/jpeg|png|webp`, verified by decoding with Pillow.
- Stored through the existing `storage_service` uploads path
  (`users/{user_id}/uploads/{upload_id}/…`, `storage.py:125`), so the result passes
  `validate_image_url_ownership` exactly like a seller upload.

### 6.3 Other

- Keys hashed at rest. Never logged. Revocation immediate.
- All generation paths charge credits through one gate. No credit-free path through the
  Shopify module.
- Rate limiting: none exists in the repo today. A per-connection cap on the two `glb` calls
  (and on product creation by sync) is recommended before a public App Store launch (**D6**).

---

## 7. Behaviour notes

1. **Shopify variant ≠ Rivollo variant.** Code says `shopify_variant`, `model variant`,
   `colour variant`. Never plain "variant".
2. **Credits:** sync is free; main GLB = 1 charge; every layout attempt = 1 charge. The
   plugin shows the cost on each button (from `GET /ai/3d-models`).
3. **Product status** is only ever moved by the existing pipeline:
   `draft` (sync) → `queue` → `processing` → `ready` (main GLB) → `published` (existing
   publish). Layout generations never change the product status.
4. **Layout tile thumbnail** = the source photo (passed as `thumbnail` to `create_variant`).
5. **Colour in v1** is an `info` option: a plain selector for price and cart that does not
   change the 3D model. Setting it to `colour` later additionally applies the configurator
   option. The shopper-facing interaction stays the same.

---

## 8. Configuration

| Setting | Default | Phase |
|---|---|---|
| `ENABLE_MODEL_VARIANTS` | existing (on) | 1 (generation rides on it) |
| `GENERATION_CONCURRENCY` | 2 | 1 |
| `GENERATION_STALE_AFTER_SECONDS` | 1800 | 1 |
| `GENERATION_SWEEP_INTERVAL_SECONDS` | 300 | 1 |
| `GENERATION_CANDIDATE_TTL_DAYS` | 30 | 1 |
| `ENABLE_SHOPIFY_INTEGRATION` | **true** (changed 2026-09-29) | 2, 3 |
| `SHOPIFY_IMAGE_MAX_BYTES` | 20 MB | 2 |
| `VIEWER_BASE_URL` | none; required when the Shopify flag is on (same value as Viewer.Api's `ViewerBaseUrl`) | 2 |

---

## 9. Migrations and the account-purge job

Written by hand (ADR-009). Purely additive: create tables and indexes, alter nothing.

| Revision | Down revision | Creates | New FK to `tbl_products`/`tbl_users`? |
|---|---|---|---|
| Phase 1 `…_add_model_variant_generations` | `e3b9c6a1d27f` | `tbl_model_variant_generations` | **Yes**: `product_id → tbl_products` CASCADE |
| Phase 2 `…_add_shopify_integration` | Phase 1 revision | 4 `tbl_shopify_*` tables | **No** |

- **Phase 1 is a deployment dependency on `Rivollo.AccountPurge.Job`** (same mechanism as Q8):
  allow-list the new FK, add `tbl_model_variant_generations` to its inventory, and sweep
  `candidate_glb_blob_url` blobs (already under the user prefix). Deploy together, between
  two nightly runs. Ideally fold it into the unmerged `model-variants-purge/supriya` branch.
- `accepted_variant_id → tbl_product_model_variants` is SET NULL, consistent with ADR-014's
  purge ordering.
- Phase 2 does not trip assertion 9. Follow-up: teach the purge job to delete
  `tbl_shopify_*` rows by `user_id`.

---

## 10. Edits to existing files (complete list)

| File | Phase | Change |
|---|---|---|
| `app/api/routes/configurator.py` | 1 | 5 new route functions (appended) |
| `app/schemas/configurator.py` | 1 | new request/response schemas |
| `app/models/configurator.py` | 1 | `ModelVariantGeneration` model |
| `app/database/configurator_repo.py` | 1 | new repository methods |
| `app/main.py` | 1 | generation sweep beside the bake sweep in `lifespan` |
| `app/main.py` | 2 | one `include_router(shopify_router)` + public router |
| `app/models/__init__.py` | 2 | register `app.models.shopify` |
| `app/core/config.py` | 1, 2 | settings in §8 |
| `app/api/routes/products.py` + `LicensingService` | 1 | **only if D2 = extract**: `/createProductFal` calls the shared gate; behaviour identical |
| `app/services/product_service.py` | 2 | **additive only**: public `start_fal_generation(...)` alias for `_run_fal_3d_generation_background` (§4.6), and `create_draft_product(...)` for sync (**D9**). No existing method changes |
| `CLAUDE.md`, `docs/configurator/decisions.md` (ADR-014 amendment, ADR-015), `data-model.md`, `api-spec.md` | 1 | same change as the code |

New files: `app/services/configurator/generation_service.py`, `generation_runner.py`;
`app/api/routes/shopify.py`; `app/schemas/shopify.py`; `app/services/shopify/`
(`auth.py`, `sync_service.py`, `main_glb_service.py`, `layout_service.py`,
`image_importer.py`, `public_service.py`); `app/database/shopify_repo.py`;
`app/models/shopify.py`; two migrations; `tests/configurator/test_generation_*.py`;
`tests/shopify/`.

---

## 11. Test plan

Pytest, `asyncio_mode = "auto"`, `get_db` overridden, no real database. fal and storage are
mocked.

**Phase 1**
- Unit: state transitions (all legal and illegal), sweep thresholds, TTL discard, name and
  `client_ref` validation, image-ownership rejection.
- Service: request → charged once; plan-gate and credit failures charge nothing; runner
  success → `ready` with a candidate; fal failure / empty bytes / storage failure → `failed`
  with a seller-safe error and no orphan blob; accept → `create_variant` called with the
  candidate bytes + source thumbnail, candidate blob deleted, idempotent second accept;
  discard deletes the blob; discard while generating → the result is dropped on completion.
- Repository: FK cascade from product; SET NULL from variant; in-flight partial index query.
- API: contracts, 404 for a second user on every endpoint, feature-off 404.
- Regression (if D2 = extract): `/createProductFal` plan / credit / error messages unchanged.

**Phase 2**
- Auth: missing / unknown / revoked key → 401; key for shop A cannot read or write shop B
  (404); deleted user → 401; key never appears in logs.
- Sync: first sync creates exactly one draft product (+ asset-1 thumbnail + mapping, owned by
  the connection's user) and returns 201; re-sync returns 200, reuses the product and does
  **not** change its name / description / thumbnail; description stored as plain text; a
  product without images gets no thumbnail; a product deleted in the portal is recreated on
  the next sync; variants replaced; layout rows reconciled; removed layouts do not touch model
  variants; all validation rules (422); non-Shopify image hosts rejected.
- Options: one layout max, `colour` rejected, Original required.
- Main GLB: allow-listed images only; charges once; schedules the existing generation with the
  **copied** URL; 409 when queued / processing / ready / already has a GLB; `retry` only when
  stalled; a failed generation leaves the product `draft` and retryable.
- Layout GLB: allowed before the main GLB is ready; Original layout rejected; `auto_accept`
  accepts on `ready`; second accept on a layout with a model variant → 409.
- Image copy: host allow-list; SSRF cases (other hosts, redirects, oversize, non-image)
  rejected; dangling references treated as absent.
- Uninstall and `shop/redact` delete exactly the calling shop's rows.

**Phase 3**
- Not published → 404; unlinked → 404; revoked connection → 404; shopper schema contains no
  inventory, SKU, GID or audit fields; unaccepted layouts omitted; cart URL format.

---

## 12. Plugin changes required (other repository)

1. Replace the `RIVOLLO_API_TOKEN` / `RIVOLLO_USER_ID` env pair with a per-shop key entered in
   (or obtained by) the app's settings, sent as `X-Rivollo-Api-Key`.
2. Call `/integrations/shopify/...` instead of `/createProductFal`, `/products/:id/assets` and
   `/products/sync`.
3. Sync payload: add `currency` (`shop.currencyCode`); send `image_urls[]` per variant (query
   variant media); read `{success, data}` and `detail`.
4. New UI: option-role mapping; "Create 3D" for the main model (pick an image); per-layout
   image picker → Create 3D → preview → Accept; credit cost on every button.
5. Webhooks: `products/update` → sync; `products/delete` → unlink; `app/uninstalled` →
   uninstall; `shop/redact` → redact; `customers/*` → acknowledge.
6. Metafields: `rivollo_product_id` = the id returned by **sync** (the draft product);
   `rivollo_status` follows the product status from `GET /products/{id}`.
7. Upload to Shopify media only after the main GLB is `ready` (unchanged staged-upload code).

---

## 13. Decisions required

| # | Question | Recommendation |
|---|---|---|
| **D1** | Refund credits when a generation fails? Today's product path does not. | Keep consistent (no refund) for v1. Revisit repo-wide |
| **D2** | Extract the plan/credit gate from `products.py` (touches existing code, behaviour-preserving) or duplicate it? | Extract, with a regression test |
| **D3** | On `shop/redact`, delete only Shopify integration data, or also the Rivollo products created for that shop? | Integration data only. Products belong to the Rivollo account |
| **D4** | Regenerate / replace the **main GLB** once it is `ready` (and preview it before it becomes the product's model)? | Not in v1. Needs a mapping re-point, which ADR-014 avoided |
| **D5** | Viewer reads Shopify data via Viewer.Api mirror, or directly from this API? | Ask the viewer owner |
| **D6** | Per-connection rate limit on the `glb` calls and sync before App Store launch? | Yes, before public launch; not needed for a pilot |
| **D7** | Who builds the portal "Connect Shopify" screen? | Frontend team |
| **D8** | Approve the 5th configurator table and the `CLAUDE.md` rule change | ✅ approved in review (2026-09-28) |
| **D9** | Draft-product creation for sync: new additive `ProductService.create_draft_product` (repeats ~30 lines of `create_product_with_fal_image_urls`), or extract those lines and have that method call it (touches an existing method)? | Additive method in v1; fold the duplication later with a regression test |
| **D10** | Should re-sync overwrite the Rivollo product's name / description / thumbnail? | No. Set once at creation; the seller owns them in the portal |
| **D11** | Offer `auto_accept` on layout GLBs (one click, no preview)? | Yes, as an option; default `false` |
| ~~Sync creates the Rivollo product~~ | ✅ decided in review (2026-09-28): sync creates a draft product linked to the Shopify id | — |

## 14. Needs verification before building

1. Shopify Admin API 2026-07 field for **multiple media per variant**.
2. Exact model identifiers in the configurator shopper payload (`shopper_service.py`) and
   Viewer.Api, for `layouts[].model`.
3. A `storage_service` method for the candidate path (§3.5).
4. For the deferred colour phase: material count of a real fal-generated GLB via
   `GET …/configurator/materials` (AI output is expected to be one material).
