# Product Configurator — API Specification

> Status: **specification only**. None of these endpoints exist. Do not implement from this
> document until [decisions.md](decisions.md) open questions Q1-Q6 are resolved.

---

## 1. Base path and versioning

### What the repository actually does (CONFIRMED)

- `_api_prefix = settings.API_PREFIX.rstrip("/")` and `API_PREFIX` defaults to `""`
  (`app/core/config.py:10`, `app/main.py:216`). In every deployed configuration observed,
  routes are served at the bare root.
- All 27 routers are mounted at that prefix. The **only** versioned mount is
  `products_v2_router` at `f"{_api_prefix}/v2"` (`app/main.py:226`).
- Paths are flat and resource-first: `/products/{id}/hotspots`, `/color-variants/{id}`,
  `/products/{id}/dimensions`.
- There is **no `/api/v1` anywhere in this repository.**

### Decision

The brief suggests `/api/v1/configurator`. Adopting it would introduce a new convention
rather than follow one, and would leave the Configurator as the only domain reachable under a
different prefix — which complicates the reverse proxy, the frontend base URL, and every
future router.

**PROPOSED: keep flat, resource-first paths at the existing prefix**, and get the domain
boundary from module structure, a router `tags=["configurator"]`, and a dedicated service
package — not from a URL segment. This is [ADR-001](decisions.md#adr-001).

**DECIDED (Phase 3, implemented).** The namespaced variant was chosen: Configurator paths
carry a `/configurator` segment — `/products/{id}/configurator/parts`,
`/configurator/parts/{id}` — at the existing `_api_prefix`. No version segment is invented.

That choice also resolves the §5 path collision outright: the colour-variant feature keeps
`/products/{id}/materials` and `/products/{id}/color-variants` untouched, and no
colour-variant route was modified. See [decisions.md](decisions.md#open-questions) Q6.

All paths below are relative to `_api_prefix`.

---

## 2. Authentication

| Router | Dependency | Applies to |
|---|---|---|
| `configurator_router` | `Depends(get_current_user)` at router level | all seller endpoints |
| `configurator_public_router` | `Depends(verify_public_basic_auth)` | the shopper endpoint |

Router-level dependencies match `color_variants.py:26-29` and `hotspots.py:17-20`. Handlers
additionally take `current_user: CurrentUser` where they need the id.

**Bearer JWT** for seller endpoints. `get_current_user` already rejects deleted (403) and
deactivated (403) accounts and invalid tokens (401) — see `app/api/deps.py:34-93`. No
Configurator-specific auth code is needed.

**HTTP Basic** for the shopper endpoint, reusing `verify_public_basic_auth`
(`app/api/routes/products.py:88-101`), which is how `/public/products/{id}/assets` is already
protected.

---

## 3. Authorization

**This is the one place the Configurator must diverge from its neighbours.**

**CONFIRMED gap.** `hotspot_service._ensure_product_exists` and
`color_variant_service._ensure_product_exists` check existence via a bare
`db.get(Product, product_id)` with no owner filter. Any authenticated user can currently read
and modify any product's hotspots and colourways.

**PROPOSED — every seller endpoint runs this check first:**

```python
async def _require_owned_product(db, product_id, user_id) -> Product:
    """Load a product the caller owns, or 404.

    404 rather than 403 on an ownership failure: answering 403 confirms that a
    product with this id exists and belongs to someone else, which is an
    enumeration oracle over every seller's catalogue.
    """
    product = await configurator_repo.get_owned_product(db, product_id, user_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
```

backed by:

```python
select(Product).where(
    Product.id == product_id,
    Product.created_by == user_id,
    Product.deleted_at.is_(None),
)
```

For endpoints keyed on `part_id` or `option_id`, resolve upward to the product and run the
same check. Never trust a client-supplied `product_id` alongside a `part_id`.

**The full authorization chain**, enforced in services:

| # | Rule | Failure |
|---|---|---|
| 1 | Valid bearer token | 401 |
| 2 | Product exists, not soft-deleted, `created_by == caller` | 404 |
| 3 | Part belongs to that product | 404 |
| 4 | Option belongs to that part | 404 |
| 5 | Every `material_index` valid for the product's current GLB | 400 |
| 6 | No material index claimed by another part | 400 |
| 7 | Part's `glb_version` matches the product's current GLB | 409 |

---

## 4. Conventions

**Response envelope.** Every success returns `api_success(data)`:

```json
{ "success": true, "data": { } }
```

Routes declare `response_model=dict` and return `api_success(...)`, matching every existing
router.

**Field naming.** `snake_case`, no aliases. **CONFIRMED**: the four newest schema modules —
`hotspots.py`, `dimensions.py`, `color_variants.py`, `products.py` — contain **zero**
`alias=` declarations. The camelCase-alias style survives only in older modules
(`uploads.py`, `analytics.py`, `branding.py`, `dashboard.py`, `galleries.py`). The
Configurator follows its neighbours: snake_case.

**Errors.** Raise `HTTPException(status_code, detail)`, matching every existing service.
FastAPI renders `{"detail": "..."}`.

> **CONFIRMED inconsistency, flagged not fixed.** This means business errors do *not* use the
> `api_error` envelope; only unhandled 500s do (`app/main.py:307-321`). Clients see two error
> shapes. The Configurator matches its neighbours rather than unilaterally introducing a
> third convention. Unifying this is a repo-wide change — [decisions.md](decisions.md#open-questions) Q2.

**Ids in paths** are UUID strings, parsed with a helper that returns `400` on a malformed
value, matching `color_variants._parse_uuid` (`app/api/routes/color_variants.py:32-36`).

---

## 5. Materials

### `GET /products/{product_id}/configurator/materials`

Auth: bearer. Authorization: product ownership.

Returns the colourable materials of the product's current GLB, annotated for the Part Editor.

> **Collision RESOLVED (Phase 3).** `color_variants.py:41` already owns
> `/products/{id}/materials`, so the Configurator uses
> **`GET /products/{product_id}/configurator/materials`** instead — option (a) of the three
> this note used to list. The colour-variant route is untouched and keeps working.

**Response**

```json
{
  "success": true,
  "data": {
    "glb_version": "asset:6f1c...",
    "model_url": "https://cdn/.../model.glb",
    "material_count": 7,
    "materials": [
      {
        "material_index": 0,
        "name": "Fabric_Seat",
        "mesh_names": ["seat_cushion"],
        "has_base_color_texture": true,
        "base_color_texture_url": "https://cdn/.../source/0.png",
        "average_color": "#7A6B5D",
        "triangle_count": 12480,
        "suggested_method": "luminance",
        "similarity_group_hint": 0,
        "center": [0.0, 0.42, 0.0],
        "assigned_part_id": "9c2e...",
        "eligible_for_part": true
      }
    ]
  }
}
```

| Field | Status | Note |
|---|---|---|
| `material_index`, `name`, `mesh_names`, `has_base_color_texture`, `average_color`, `center` | CONFIRMED available | `glb_recolor.inspect()` already produces these |
| `suggested_method` | CONFIRMED available | from `_suggest_method` |
| `similarity_group_hint` | CONFIRMED available, **renamed** | this is today's `group_id`. Renamed so no client mistakes it for a Part identity — it is recomputed per call, unstable across re-upload, and transitively unions colour-proximate materials. See [ADR-004](decisions.md#adr-004). |
| `base_color_texture_url` | **NOT SHIPPED** — see §5.1 | extraction not implemented, and real assets include external-URI images the loader rejects |
| `triangle_count` | **NOT SHIPPED** — see §5.1 | computable, but not reliably attributable to a material on real assets |
| `glb_version` | **PROPOSED** | [ADR-006](decisions.md#adr-006) |
| `assigned_part_id`, `eligible_for_part` | **PROPOSED** | derived from existing parts |

### 5.1 Why `triangle_count` and `base_color_texture_url` are not in the response

Investigated against real dev assets (Phase 3 follow-up), not reasoned about. Findings:

**`triangle_count` — computable, but not meaningfully per-material.**
Every primitive in all four sampled GLBs carried an `indices` accessor with a valid `count`
and a present `bufferView`, so `count // 3` is available without decoding geometry. The
blocker is attribution: in the sampled assets most primitives have **no material assigned**
(`primitive.material` is null). One 3.8 MB mesh reported 51,730 triangles under no material
and **2** under material 0. A per-material `triangle_count` would therefore be `0` or
near-zero for most materials on real products — worse than absent, because a seller would
read it as "this part is tiny". A whole-model triangle count is trivially available if that
turns out to be what the editor actually wants.

**`base_color_texture_url` — blocked on extraction, and on assets that defeat it.**
The engine reads images from the GLB's binary buffer
(`glb_recolor._load_pil_from_image`), which raises for any image referenced by an external
`uri` rather than a `bufferView`. One sampled asset had exactly that: one material, one image,
`external_uri`. `inspect()` swallows that failure and reports
`has_base_color_texture: false`, so such a material already looks untextured today. Serving a
URL would additionally require the extract-and-upload step of
[architecture.md §7.1](architecture.md#71-source-texture-extraction-proposed), which is not
built.

**Both stay out of the contract until the work behind them exists.** Adding a field that is
usually zero, or usually absent, is worse than omitting it.

**Correction to a documented assumption:** ADR-002 and architecture.md §6 state the canonical
GLB is *normally Draco-compressed*, on the strength of `ENABLE_DRACO_COMPRESSION` defaulting
to `True`. **None of the four sampled dev assets is Draco-compressed** — `extensionsUsed` was
empty on every one. The setting's default is not evidence of what is stored. See
[decisions.md](decisions.md#open-questions) Q4.

**Performance.** This endpoint downloads and parses the GLB. `variant_bake_service.inspect_model`
already runs it in `asyncio.to_thread` behind `model_cache`; reuse both. Typical meshes are
40-80 MB (`model_cache.py:5-6`). Called once when the editor opens, not per interaction.

**Errors:** `400` malformed id · `404` product not owned/missing · `400` product has no GLB ·
`502` GLB could not be fetched or parsed.

---

## 6. Parts

### `GET /products/{product_id}/parts`

Auth: bearer + ownership. Returns all parts, active and inactive, ordered by `order_index`,
each with its options.

```json
{
  "success": true,
  "data": [
    {
      "id": "9c2e...",
      "product_id": "1a4f...",
      "name": "Seat",
      "slug": "seat",
      "material_indices": [0, 3],
      "material_type": "fabric",
      "order_index": 0,
      "shopper_selectable": true,
      "default_option_id": "77bd...",
      "glb_version": "asset:6f1c...",
      "glb_stale": false,
      "isactive": true,
      "options": [ /* see §7 */ ],
      "created_at": "2026-09-08T10:00:00Z",
      "updated_at": null
    }
  ]
}
```

Two fields on this response are **computed, not stored**:

- `glb_stale` — `part.glb_version != current_glb_version`.
- `default_option_id` — read from the option carrying `is_default = true`
  (`next((o.id for o in part.options if o.is_default), None)`), or `null` when no option
  qualifies. **There is no `default_option_id` column.** See
  [data-model.md §4](data-model.md#4-tbl_part_options-proposed).

### `POST /products/{product_id}/parts` → `201`

```json
{
  "name": "Seat",
  "material_indices": [0, 3],
  "material_type": "fabric",
  "shopper_selectable": true,
  "order_index": 0
}
```

**Validation**

| Rule | Failure |
|---|---|
| `name` 1-100 chars after strip | 422 |
| `material_indices` non-empty, ≤ 64 entries, unique, all ≥ 0 | 422 |
| every index `< material_count` of the current GLB | 400 `"Material index 12 does not exist on this model (7 materials)"` |
| no index already claimed by another active part | 400 `"Material index 3 already belongs to part 'Backrest'"` |
| `material_type` in the allowed set or null | 422 |
| product has a GLB | 400 |

`slug` is derived server-side from `name` and de-duplicated per product, following
`color_variant_service._unique_slug` (`color_variant_service.py:349-366`). Clients do not
send it.

`glb_version` is set server-side from the product's current GLB. Clients do not send it.

### `GET /parts/{part_id}`

Auth: bearer + ownership resolved through the part. Same shape as one list element.

### `PATCH /parts/{part_id}`

All fields optional. Only what is sent changes.

```json
{ "name": "Seat Cushion", "material_indices": [0, 3, 5], "shopper_selectable": false }
```

Additional rules beyond create:

- Changing `material_indices` re-runs the full membership and overlap validation.
- Changing `material_indices` **invalidates every option's bake**. Every option of the part is
  reset to `bake_status: "pending"` and the response names them:
  `"invalidated_option_ids": [...]`.

  Note this is an *explicit* reset, not a hash comparison. `recipe_hash` covers the recipe,
  `glb_version` and the baker version — **not** the part's material membership — so adding or
  removing a material index does not change any option's hash. The bake is nonetheless stale:
  the set of textures an option must produce is derived from the part's materials, so an
  option baked against the old set is incomplete against the new one.
- `409` if the part's `glb_version` is stale — re-map against the current model first.
- Setting `shopper_selectable: false` on a part is allowed even if it is the only part.

### `DELETE /parts/{part_id}` → `200`

Cascades to options and textures. **Purges texture blobs before deleting rows**, following
`variant_bake_service.purge_variant_assets`. Returns
`{"success": true, "data": {"message": "Part deleted successfully"}}`, matching the delete
convention at `color_variants.py:172`.

---

## 7. Options

### `GET /parts/{part_id}/options`

```json
{
  "success": true,
  "data": [
    {
      "id": "77bd...",
      "part_id": "9c2e...",
      "name": "Charcoal",
      "slug": "charcoal",
      "swatch_hex": "#3A3A3A",
      "recipe": {
        "version": 1, "method": "luminance", "color": "#3A3A3A", "brightness": 1.0,
        "overrides": []
      },
      "recipe_hash": "b41d9f...",
      "order_index": 0,
      "is_default": true,
      "isactive": true,
      "bake_status": "completed",
      "bake_error": null,
      "bake_started_at": "2026-09-08T10:02:11Z",
      "bake_completed_at": "2026-09-08T10:02:47Z",
      "textures": [
        { "material_index": 0, "url": "https://cdn/...", "content_type": "image/png",
          "width": 2048, "height": 2048, "size_bytes": 1841203 }
      ],
      "created_at": "2026-09-08T10:02:10Z"
    }
  ]
}
```

`textures` is filtered to entries whose `recipe_hash` equals the option's — stale files are
hidden rather than served as the wrong colour, mirroring
`color_variant_service._to_response` (`color_variant_service.py:373-381`).

### `POST /parts/{part_id}/options` → `201`

**One endpoint creates both kinds of option.** A generated recolour and an uploaded texture
differ only in `recipe.method`. There is **no separate Configurator texture endpoint**, no
`option_type` field, and no upload-specific route — see §7.3.

Generated recolour:

```json
{
  "name": "Charcoal",
  "swatch_hex": "#3A3A3A",
  "recipe": { "version": 1, "method": "auto", "color": "#3A3A3A", "brightness": 1.0 },
  "order_index": 0
}
```

Uploaded texture:

```json
{
  "name": "Herringbone",
  "swatch_hex": "#6B5B4A",
  "recipe": {
    "version": 1,
    "method": "image",
    "image_url": "https://cdn.example.net/uploads/users/{user_id}/uploads/{uuid}/herringbone.png"
  },
  "order_index": 1
}
```

**Validation**

| Rule | Failure |
|---|---|
| `name` 1-100 chars | 422 |
| `recipe.version == 1` | 422 |
| `recipe.method` in `auto \| factor \| luminance \| remap \| image` | 422 |
| `recipe.color` matches `#RGB`/`#RRGGBB` — required unless `method == "image"` | 422 |
| `0.1 <= recipe.brightness <= 2.0` — ignored when `method == "image"` | 422 |
| `recipe.image_url` **required** when `method == "image"`, **rejected** otherwise | 422 |
| `recipe.image_url` passes the ownership check in §7.2 | 400 |
| `recipe.method: "auto"` resolves to exactly one method across the part's materials | 400 |
| `swatch_hex` **required** when `method == "image"` | 422 |
| every `overrides[].material_index` is in the parent part's `material_indices` | 400 |
| part's `glb_version` current | 409 |
| ≤ 32 options per part | 400 |

**`set_as_default` is not accepted on create.** `is_default` is only valid on an option that
is `isactive` and `completed` ([data-model.md §4](data-model.md#4-tbl_part_options-proposed)),
and a newly created option is always `pending` — so there is nothing a create-time flag could
legitimately set. There is deliberately no column recording a deferred intent.

A part does **not** acquire a default automatically. Until the seller chooses one with
`PATCH /options/{id}` (§7.4), the part starts on the model's Original appearance.

**Method-specific field handling**

| `method` | `color` | `brightness` | `image_url` | `swatch_hex` |
|---|---|---|---|---|
| `factor` / `luminance` / `remap` | required | optional, default `1.0` | rejected | optional — defaults to `recipe.color` |
| `image` | **ignored** | **ignored** | **required** | **required** |

Colours are normalised to uppercase `#RRGGBB` before storage, reusing `_normalize_hex`
(`app/schemas/color_variants.py:24-32`). `swatch_hex` cannot be defaulted for an `image`
option because there is no `recipe.color` to default from, so the API requires it rather than
deriving an average from the uploaded file.

**`auto` is resolved server-side at save time** into a concrete method per material index, so
the preview the seller approved and the file the backend bakes come from the same rule. If
resolution fails — the GLB cannot be fetched or parsed — the request **fails with `502`**. It
does not persist an unresolved `auto`. (The existing implementation swallows this failure and
stores `auto`, `variant_bake_service.py:152-155`; do not repeat that.) `auto` is meaningless
for `image` and is rejected alongside it.

Creating an option sets `bake_status: "pending"` and enqueues a bake — **for `image` options
too**. Their bake is short (fetch, validate, normalise, copy, record) but it is a real bake
with a real status, not a special case. Response is `201` with the option; the client polls §8.

### 7.1 One lifecycle for both kinds of option

An uploaded texture uses the same table, the same `recipe_hash`, the same `bake_status`
transitions, the same staleness rule and the same purge behaviour as a generated recolour.
Nothing downstream of `recipe.method` branches on "is this an upload".

Because `image_url` is part of the canonical recipe, re-uploading a different image changes
`recipe_hash` and correctly triggers a re-bake.

### 7.2 🔴 `image_url` ownership validation (server-side, mandatory)

Before an `image` recipe is persisted, `OptionService` **must** verify that `image_url`:

1. is absolute and begins with `settings.CDN_BASE_URL`;
2. resolves inside the configured uploads container;
3. sits under **this seller's own** upload namespace — `users/{current_user.id}/uploads/…`,
   the path shape written by `storage_service.upload_file_content`
   (`app/services/storage.py:164-175`);
4. carries an allowed image extension.

Reject with `400` otherwise. An unvalidated URL is an SSRF vector — the baker dereferences it
server-side — and a cross-tenant hotlink vector. **Never** fetch the URL in order to decide
whether it is acceptable.

The bake then **copies** the file into the Configurator namespace rather than referencing it,
so that deleting an option never purges a blob under the seller's own uploads prefix. See
[data-model.md §5.6](data-model.md#56-uploaded-images-are-copied-not-referenced).

### 7.3 Uploading the image itself

Use the **existing** `POST /uploads/content` — do not build a Configurator upload endpoint.
**CONFIRMED**: it accepts `.png/.jpg/.jpeg/.webp`, stores under
`users/{user_id}/uploads/{uuid4}/`, returns a CDN URL, writes only the `uploads` table, and
never touches `tbl_product_asset_mapping`, so it cannot pollute `GET /products/{id}/assets`.

The seller flow is two calls, both already specified:

```
POST /uploads/content            → { "url": "https://cdn/.../herringbone.png", ... }
POST /parts/{part_id}/options    → recipe.method = "image", recipe.image_url = that url
```

### 7.4 How a part acquires its default option

**A part with no default shows the model's Original appearance.** That is the intended
starting state, not a gap, and it is what a part has until the seller chooses otherwise.

`is_default` is set by exactly one thing: **`PATCH /options/{id}` with
`set_as_default: true`**, once the option is `isactive` and `bake_status = "completed"`.
Nothing sets it automatically — a completing bake does not, and deleting the current default
does not promote a replacement.

A part returns to Original when its default is cleared: `set_as_default: false` on the default
option, hiding it (`isactive: false`), or deleting it.

> **Revised 2026-09-11.** This section previously specified *first-baked-wins*: the first
> option to finish baking became the default automatically, and deleting the default promoted
> the next suitable option. Both were removed so that the untouched model is the starting state
> and `default_option_id` only ever reflects a choice the seller made. See
> [ADR-011](decisions.md#adr-011).

### `GET /options/{option_id}` · `PATCH /options/{option_id}` · `DELETE /options/{option_id}`

`PATCH` accepts `name`, `swatch_hex`, `recipe`, `order_index`, `isactive`, `set_as_default`.

- Changing `recipe` recomputes `recipe_hash`. **If and only if the hash changed**, set
  `bake_status: "pending"` and enqueue. A rename or reorder must not trigger a bake — the
  existing `needs_rebake` distinction at `color_variants.py:118-120` is correct and is kept.
- `set_as_default: true` may only be applied to an option that is `isactive` **and**
  `bake_status == "completed"`; otherwise `400`. Setting it clears the previous default in
  the same transaction (`repo.clear_default`, `color_variant_service.py:197-204`), and the
  partial unique index `ux_part_options_one_default` is the backstop.
- `set_as_default: false` on the current default **clears it**, returning the part to
  Original. On an option that is not the default it is a no-op.
- Deactivating the current default is allowed and **clears the default**, returning the part
  to Original — a hidden option cannot be the default.
- `DELETE` purges blobs before rows. Deleting the current default does **not** promote another
  option: the part returns to Original and `default_option_id` becomes `null`.

---

## 8. Bake

### `POST /options/{option_id}/bake` → `202 Accepted`

Auth: bearer + ownership. Idempotent.

Query: `force` (bool, default `false`) — re-bake even when the current result is valid, for
recovering from a blob deleted out of band.

```json
{
  "success": true,
  "data": {
    "option_id": "77bd...",
    "bake_status": "pending",
    "recipe_hash": "b41d9f...",
    "already_current": false,
    "poll_url": "/options/77bd.../bake-status"
  }
}
```

**Idempotency rules** (keyed on `recipe_hash`):

| Current state | Behaviour |
|---|---|
| `completed`, all textures' hash matches, `force=false` | no work; `already_current: true`, status `completed` |
| `baking`, `bake_started_at` within the stale threshold | no second bake; returns the in-flight ticket |
| `baking`, `bake_started_at` older than the threshold | ticket returned unchanged; the **stale-bake sweep** reclaims it, not this endpoint |
| `pending` | no duplicate enqueue |
| `failed` | re-enqueued |

**`bake_attempts` increments when a real attempt BEGINS**, in
`BakeService.mark_baking` — not when a bake is enqueued. An earlier draft of this
table said "`failed` → re-enqueued, `bake_attempts += 1`"; that conflicted with
[baking.md §4.5](baking.md#45-retry-and-idempotency) ("per real attempt") and has
been corrected here. An enqueue that never runs must not burn a retry, because
the count is what the automatic cap is measured against.

**Reclaiming a stale `baking` row is the sweep's job, not this endpoint's.** This
endpoint deliberately never starts a second bake for a row that is already
`baking`, however old — two runners on one option is worse than waiting. See
[baking.md §4.3](baking.md#43-required-mitigations).

`409` if the part's `glb_version` is stale. `400` if the product has no GLB.

### `GET /options/{option_id}/bake-status`

```json
{
  "success": true,
  "data": {
    "option_id": "77bd...",
    "bake_status": "baking",
    "bake_error": null,
    "bake_started_at": "2026-09-08T10:02:11Z",
    "bake_completed_at": null,
    "bake_attempts": 1,
    "recipe_hash": "b41d9f...",
    "progress": { "textures_total": 2, "textures_done": 1 },
    "textures": []
  }
}
```

**`progress` is derived, not stored.** No column tracks it:

- `textures_total` — the number of material indices in the parent part **expected to yield a
  texture**. For `image` options that is every index in `part.material_indices`; for the
  recolour methods it excludes indices whose material has no base-colour image, since those
  are treated by `factor` and produce no file
  ([data-model.md §6](data-model.md#how-many-texture-rows-an-option-has)). Determining that
  exactly requires GLB inspection, so the API MAY report `len(part.material_indices)` as an
  upper bound.
- `textures_done` — `COUNT(*)` of `tbl_part_option_textures` rows for this option whose
  `recipe_hash` equals the option's current `recipe_hash`.

`textures_done` is exact. `textures_total` is **best-effort** and may over-count. The whole
`progress` object MAY be omitted. Clients must treat it as a progress hint only and must
drive completion off `bake_status`, never off `textures_done == textures_total`.

**Polling guidance for clients:** 2s interval, backing off to 10s after 30s, giving up at
5 minutes and showing a retry affordance. Do not poll faster than 1s.

---

## 9. Shopper API

### `GET /public/products/{product_id}/configurator`

Auth: HTTP Basic (`verify_public_basic_auth`), matching `/public/products/{id}/assets`.

Read-only. Returns only what the viewer renders.

```json
{
  "success": true,
  "data": {
    "product_id": "1a4f...",
    "product_name": "Aria Lounge Chair",
    "model_url": "https://cdn/.../model.glb",
    "ar_model_url": "https://cdn/.../model.usdz",
    "parts": [
      {
        "id": "9c2e...",
        "name": "Seat",
        "slug": "seat",
        "material_indices": [0, 3],
        "order_index": 0,
        "default_option_id": "77bd...",
        "options": [
          {
            "id": "77bd...",
            "name": "Charcoal",
            "slug": "charcoal",
            "swatch_hex": "#3A3A3A",
            "order_index": 0,
            "textures": [
              { "material_index": 0, "url": "https://cdn/...", "content_type": "image/png" }
            ]
          }
        ]
      }
    ]
  }
}
```

**Filtering rules — applied server-side, before serialisation:**

| Excluded | Why |
|---|---|
| parts where `isactive = false` or `shopper_selectable = false` | seller-only configuration |
| parts where `glb_version` is stale | would paint the wrong mesh |
| options where `isactive = false` | unpublished |
| options where `bake_status != "completed"` | no texture to show |
| textures whose `recipe_hash` ≠ the option's | stale file, wrong colour |
| a part left with zero options after filtering | nothing to choose |

**Fields never present in this response:** `recipe` (including `image_url`), `recipe_hash`,
`bake_status`, `bake_error`, `bake_started_at`, `bake_completed_at`, `bake_attempts`,
`glb_version`, `blob_url`, `size_bytes`, `is_default`, `isactive`, `created_by`, `updated_by`,
`created_date`, `updated_date`, `material_type`.

`default_option_id` **is** present, computed from the option carrying `is_default = true`
exactly as in §6. The raw `is_default` flag is not exposed — the shopper needs to know which
option to load first, not the internal representation of that fact.

**`null` means "start on Original"** — show the model as uploaded, with no option applied.
That is the case when the seller has not chosen a default, and also when their chosen default
did not survive filtering (it is re-baking after a recipe change). It is NEVER substituted with
another option: `default_option_id` means "the option this seller deliberately chose", and
returning the first surviving option would report a choice the seller never made.

A part whose only options are unbaked is dropped entirely (a part with zero surviving options
is filtered out), so a part that *is* present with `null` has visible options — the shopper
simply starts on Original and can pick one.

Note that `recipe.image_url` being excluded is deliberate: it points into the seller's own
upload namespace. The shopper receives only the **copied** texture URLs under
`configurator/{product_id}/…` from `textures[]`.

**On exposing `id` and `material_indices`.** Both are deliberate parts of the public
contract: the viewer needs a stable key per option and needs to know which glTF material to
swap. Neither is sensitive — the material index is derivable from the GLB the shopper already
downloaded. All other database identifiers stay internal.

Use a **separate response schema class** (`PublicConfiguratorResponse`), not the seller schema
with fields omitted. An omission-based approach leaks the next field somebody adds.

**Errors:** `404` product missing, soft-deleted, unpublished, or has no configurator ·
`401` bad Basic credentials.

**Caching.** `Cache-Control: public, max-age=60`. Texture and model URLs are content-addressed
and separately immutable-cached, so a short TTL on the manifest is enough.

---

## 9a. Model variants ([ADR-014](decisions.md#adr-014))

An extra model variant is another **shape** of a product with its own GLB. The product's
original model stays its model and permanent default; it has no variant row. Every route here
answers **404** while `ENABLE_MODEL_VARIANTS` is `false`; it is **on by default**, so an
environment on this build needs migration `e3b9c6a1d27f` applied.

### `POST /products/{product_id}/configurator/model-variants` → `201`

`multipart/form-data`, the same style as `POST /createProductFromGlb`:

| Field | Type | Required | Rules |
|---|---|---|---|
| `name` | text | yes | trimmed, 1–100 chars |
| `glb` | file | yes | `.glb`, binary glTF 2.0 with at least one mesh, at most `MAX_VARIANT_GLB_BYTES` (default 150 MB) |
| `thumbnail` | file | no | PNG / JPEG / WebP, at most `MAX_VARIANT_THUMBNAIL_BYTES` (default 5 MB). Usually sent later, after the editor captures `toBlob()` |

What the server does, in order: ownership (404) → file checks → read material/mesh names and
the bounding box → Draco-compress with `glb_compression_service` (skipped if the upload is
already Draco) → re-read the compressed file and require identical material and mesh names in
the same order → upload to `{user_id}/{product_id}/model-variants/{variant_id}/` → insert one
`tbl_product_assets` row (asset 9, **no mapping row**) and one variant row. If compression
fails or changes a name, the original file is served, `compression_status` is
`fallback_original`, a warning is logged and returned.

```json
{
  "success": true,
  "data": {
    "id": "7f3c…",
    "product_id": "…",
    "name": "Corner",
    "glb_url": "https://cdn…/dev/{user}/{product}/model-variants/7f3c…/model3f2a1.glb",
    "thumbnail_url": null,
    "order_index": 1,
    "is_original": false,
    "isactive": true,
    "compression_status": "compressed",
    "compression_error": null,
    "original_size_bytes": 48211000,
    "compressed_size_bytes": 9120000,
    "width_m": 2.9, "depth_m": 2.1, "height_m": 0.8,
    "created_at": "2026-09-22T10:15:00Z",
    "warnings": []
  }
}
```

`width_m` / `height_m` / `depth_m` are the model's bounding box in metres (glTF is Y-up:
width = X, height = Y, depth = Z). `warnings` is advisory — the variant was created. It
flags a largest side under 0.1 m or over 10 m (usually a units problem), an unmeasurable
model, or a compression fallback.

| Status | When |
|---|---|
| 400 | bad `productId`, empty or over-long `name`, not `.glb`, empty file, not a glTF 2.0 GLB, no meshes, bad thumbnail type |
| 404 | feature off; product missing, deleted, or not the caller's |
| 413 | GLB or thumbnail over its limit |
| 422 | `name` or `glb` missing from the form |
| 502 | blob storage failed (anything already uploaded is deleted) |

`usdz_url` is `null`: per-variant iOS AR is **off by default**. With
`ENABLE_VARIANT_USDZ` set, the upload requests a conversion after the commit (fire-and-forget,
never fails the upload): the converter job is started with `--model-variant-id`, writes the
USDZ to the variant's folder and sets `usdz_asset_id`, with **no** product mapping, and
`usdz_url` fills in once it finishes. While the flag is off nothing is requested, and the
viewer hides AR for that shape.

### Managing variants

Every route resolves ownership (404, never 403) and answers 404 while the feature is off.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/products/{id}/configurator/model-variants` | — | list; the original model first (`"id": "original"`, `is_original: true`), then live variants by `order_index` |
| PATCH | `/configurator/model-variants/{variant_id}` | `{"name": "L-Shape"}` | the variant |
| POST | `/products/{id}/configurator/model-variants/reorder` | `{"variant_ids": [...]}` — every live variant exactly once, else `400` | the new list |
| PUT | `/configurator/model-variants/{variant_id}/thumbnail` | multipart `thumbnail` (PNG/JPEG/WebP) | `{id, thumbnail_url}`; the previous file is deleted |
| DELETE | `/configurator/model-variants/{variant_id}` | — | soft delete (`isactive = false`); its parts become unreachable. The original model cannot be deleted |

The original model is the permanent default: there is no "set default". Its entry reads the
product's mapped GLB, USDZ and thumbnail — the same rows the product list and `/assets` use —
and carries no compression or dimension fields.

### Parts and materials of one model

`{model}` is `original` or a variant id (a variant of **this** product, else 404).

| Method | Path |
|---|---|
| GET | `/products/{id}/configurator/model-variants/{model}/materials` |
| GET | `/products/{id}/configurator/model-variants/{model}/parts` |
| POST | `/products/{id}/configurator/model-variants/{model}/parts` |

Same shapes as §5 and §6; parts gain `variant_id` (`null` for the original model). The
material-index overlap rule applies within one model only. Part slugs stay unique per product,
so a second "Seat" in another model is `seat-2`. The product-level routes (§5, §6) are
unchanged and mean the original model. Routes that take a part or option id work for parts of
any live model; a soft-deleted variant's parts answer 404.

### Shopper payload

`GET /public/products/{id}/configurator` (§9) gains a `variants` array **only** when the
feature is on and the product has at least one live variant with a GLB. The key is omitted
otherwise, so a single-model product's payload is unchanged. Top-level `model_url`,
`ar_model_url` and `parts` always stay the original model's.

```json
"variants": [
  { "id": "original", "name": "Default", "glb_url": "…", "usdz_url": "…",
    "thumbnail_url": "…", "is_default": true, "order_index": 0,
    "width_m": null, "depth_m": null, "height_m": null, "parts": [ /* as top-level */ ] },
  { "id": "7f3c…", "name": "Corner", "glb_url": "…", "usdz_url": null,
    "thumbnail_url": "…", "is_default": false, "order_index": 1,
    "width_m": 2.9, "depth_m": 2.1, "height_m": 0.8, "parts": [ /* Corner's own */ ] }
]
```

Each variant's parts pass the §9 filters against **that variant's** GLB. `Rivollo.Viewer.Api`
mirrors this behind `Configurator:EnableModelVariants`.

---

## 10. Error reference

| Status | When | Body |
|---|---|---|
| `400` | invalid uuid; material index out of range; index claimed by another part; product has no GLB; default-option rule violated; limit exceeded | `{"detail": "..."}` |
| `401` | missing/invalid bearer token; bad Basic credentials | `{"detail": "..."}` |
| `403` | account deleted or deactivated (from `get_current_user`) | `{"detail": "..."}` |
| `404` | product/part/option missing **or not owned** | `{"detail": "..."}` |
| `409` | `glb_version` stale — model changed since authoring | `{"detail": "...", }` |
| `422` | Pydantic validation | FastAPI's standard validation body |
| `500` | unhandled | `api_error("INTERNAL_SERVER_ERROR", ...)` via `app/main.py:307` |
| `502` | GLB fetch/parse failure; storage upload failure during a synchronous step | `{"detail": "..."}` |

Detail strings are seller-facing: name the offending value and the fix.
`"Material index 3 already belongs to part 'Backrest'"`, not `"validation failed"`.

---

## 11. Endpoint summary

As implemented in `app/api/routes/configurator.py` (Phase 3).

| Method | Path | Auth | Ownership | Success | Notes |
|---|---|---|---|---|---|
| GET | `/products/{id}/configurator/materials` | bearer | ✅ | 200 | slow: parses the GLB |
| GET | `/products/{id}/configurator/parts` | bearer | ✅ | 200 | |
| POST | `/products/{id}/configurator/parts` | bearer | ✅ | **201** | |
| GET | `/configurator/parts/{id}` | bearer | ✅ via part | 200 | |
| PATCH | `/configurator/parts/{id}` | bearer | ✅ via part | 200 | returns `invalidated_option_ids` |
| DELETE | `/configurator/parts/{id}` | bearer | ✅ via part | 200 | purges blobs |
| GET | `/configurator/parts/{id}/options` | bearer | ✅ via part | 200 | |
| POST | `/configurator/parts/{id}/options` | bearer | ✅ via part | **201** | starts `pending` |
| GET | `/configurator/options/{id}` | bearer | ✅ via option | 200 | |
| PATCH | `/configurator/options/{id}` | bearer | ✅ via option | 200 | re-bakes only on recipe change |
| DELETE | `/configurator/options/{id}` | bearer | ✅ via option | 200 | purges blobs |
| POST | `/configurator/options/{id}/bake` | bearer | ✅ via option | **501** | see below |
| GET | `/configurator/options/{id}/bake-status` | bearer | ✅ via option | 200 | poll target |
| GET | `/public/products/{id}/configurator` | Basic | — | 200 | filtered payload |
| POST | `/products/{id}/configurator/model-variants` | bearer | ✅ | **201** | multipart; Draco-compressed; behind `ENABLE_MODEL_VARIANTS` (§9a) |
| GET | `/products/{id}/configurator/model-variants` | bearer | ✅ | 200 | original first (§9a) |
| PATCH | `/configurator/model-variants/{id}` | bearer | ✅ via variant | 200 | rename |
| POST | `/products/{id}/configurator/model-variants/reorder` | bearer | ✅ | 200 | |
| PUT | `/configurator/model-variants/{id}/thumbnail` | bearer | ✅ via variant | 200 | multipart |
| DELETE | `/configurator/model-variants/{id}` | bearer | ✅ via variant | 200 | soft delete |
| GET | `/products/{id}/configurator/model-variants/{model}/materials` | bearer | ✅ | 200 | `original` or variant id |
| GET | `/products/{id}/configurator/model-variants/{model}/parts` | bearer | ✅ | 200 | |
| POST | `/products/{id}/configurator/model-variants/{model}/parts` | bearer | ✅ | **201** | |

**`POST .../bake` returns `501 Not Implemented` until Phase 4.** The `202 Accepted` contract in
§8 stands as the target, but no runner exists yet: answering `202` would leave the option at
`pending` and the client polling a bake that never runs. Ownership is still resolved first, so
the endpoint answers `404` for another seller's option like every other route.

**Reused, not rebuilt:** `POST /uploads/content` for every seller-supplied image, including
uploaded-texture options. **CONFIRMED** it accepts `.png/.jpg/.jpeg/.webp`, stores under
`users/{user_id}/uploads/{uuid4}/`, returns a CDN URL, writes only the `uploads` table, and
never touches `tbl_product_asset_mapping` — so it cannot pollute
`GET /products/{id}/assets`.

**Textures have no Configurator upload endpoint and need none.** (Model-variant GLBs do —
§9a — because they are compressed and inspected server-side.) An uploaded texture reaches
the Configurator as `recipe.method = "image"` on the ordinary option-creation endpoint —
see §7.3. Do not build a second upload mechanism.
