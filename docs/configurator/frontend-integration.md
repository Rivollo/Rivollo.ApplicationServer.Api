# Product Configurator — Frontend Integration

Guide for building against the Configurator API. Describes the **currently
implemented** backend, verified against `app/api/routes/configurator.py`,
`app/schemas/configurator.py` and `app/services/configurator/`. Where the spec
docs disagree with the code, the code wins and the gap is marked **Backend
discrepancy**.

---

## 1. Overview

A seller splits a product's 3D model into **Parts** (seat, frame, legs) and gives
each Part **Options** (Charcoal, Oak, Brass). A shopper picks one Option per Part
and the viewer updates in place.

| | Seller Portal (`Rivollo.Web.Portal`) | Shopper Viewer (`Rivollo.Viewer.Portal`) |
|---|---|---|
| Does | Defines Parts/Options, triggers bakes | Renders the model, applies an Option |
| Auth | Bearer JWT | none |
| Endpoints | 13 | 1 |

Two facts shape everything below. **The seller's uploaded GLB is the one canonical
model** — never rewritten, always the file the viewer loads whatever the shopper
selects. And **Options ship textures, not models**: a bake produces one image per
affected glTF material, swapped onto the already-loaded GLB. No Option produces a
whole GLB — that was the colour-variant feature this replaces.

The join key between backend and viewer is the **glTF material index**.

---

## 2. API Quick Reference

Paths sit at `settings.API_PREFIX`, default `""` — locally, the bare root
(`http://127.0.0.1:8000/…`). There is no `/api/v1` in this repo.

### Seller APIs — Bearer token required

| Method | Path | OK | Purpose |
|---|---|---|---|
| GET | `/products/{product_id}/configurator/materials` | 200 | Material list for the editor (slow — parses the GLB) |
| GET | `/products/{product_id}/configurator/parts` | 200 | All Parts, Options nested |
| POST | `/products/{product_id}/configurator/parts` | **201** | Create a Part |
| GET | `/configurator/parts/{part_id}` | 200 | One Part |
| PATCH | `/configurator/parts/{part_id}` | 200 | Update a Part |
| DELETE | `/configurator/parts/{part_id}` | 200 | Delete a Part (cascades) |
| GET | `/configurator/parts/{part_id}/options` | 200 | Options of a Part |
| POST | `/configurator/parts/{part_id}/options` | **201** | Create an Option (auto-bakes) |
| GET | `/configurator/options/{option_id}` | 200 | One Option |
| PATCH | `/configurator/options/{option_id}` | 200 | Update an Option (may auto re-bake) |
| DELETE | `/configurator/options/{option_id}` | 200 | Delete an Option |
| POST | `/configurator/options/{option_id}/bake` | **202** | Request a bake |
| GET | `/configurator/options/{option_id}/bake-status` | 200 | Poll a bake |

### Shopper API — no authentication

| Method | Path | OK |
|---|---|---|
| GET | `/public/products/{product_id}/configurator` | 200 |

> **Backend discrepancy.** `api-spec.md` §11 says `POST …/bake` returns `501` until
> Phase 4. It is implemented and returns **202**.

### Do not use — these are a different, older feature

| Path | What it actually is |
|---|---|
| `GET /products/{id}/materials` | The colour-variant material list. Ours is `/products/{id}/configurator/materials`. |
| `PATCH /products/{id}/configurator` | Legacy free-form JSON settings blob. One segment **shorter** than the new paths; writes nothing the Configurator reads. |
| `/products/{id}/color-variants` | The whole-GLB-per-colour feature this replaces. |

**The rule:** every Configurator path contains a `/configurator` segment or starts
with `/configurator/`.

---

## 3. Authentication

**Seller endpoints:** `Authorization: Bearer <access_token>`.
`401` for a missing/expired token; `403` if the account is deactivated or pending
deletion.

**Ownership:** a seller reaches only their own products. Another seller's product,
Part or Option returns **`404`** — never `403`. Don't write a 403 branch for it.
Treat an unexpected 404 on something you just listed as "deleted or not yours"
and refresh.

**Shopper endpoint:** send **no credentials**. Do not attach an `Authorization`
header.

> **Backend discrepancy.** `api-spec.md` §2 and §9 describe the shopper endpoint as
> HTTP Basic. The implementation has no auth dependency on it at all.

---

## 4. Response and Error Handling

Every success, deletes included: `{ "success": true, "data": <object|array> }`.
Unwrap once — `res.data.data`.

| Status | Shape |
|---|---|
| `400` `401` `403` `404` `409` `502` | `{ "detail": "Material index 3 already belongs to part 'Frame'." }` |
| `422` | `{ "detail": [ { "loc": [...], "msg": "...", "type": "..." } ] }` |
| `500` | `{ "success": false, "data": null, "error": { "code": "…", "message": "…" } }` |

```ts
function errorMessage(body: any, fallback = 'Something went wrong'): string {
  if (typeof body?.detail === 'string') return body.detail;
  if (Array.isArray(body?.detail))
    return body.detail.map((e: any) => `${e.loc?.slice(1).join('.')}: ${e.msg}`).join('; ');
  return body?.error?.message ?? fallback;
}
```

`detail` strings are written for sellers — display them verbatim.

What each status means: `400` a bad value, named in the message (inline error);
`404` missing or not yours (refresh); `409` the Part was authored against an older
GLB (send the seller to re-map materials — retrying won't help); `422` field
validation; `502` the GLB is unreachable or unparseable (retry, and see §8.4).

Configurator fields are **snake_case, no aliases**. (`/uploads/content` in §5.5 is
camelCase — it predates this feature.) A malformed UUID in a path gives
`400 {"detail": "Invalid partId format"}`.

---

## 5. Seller Portal Integration

### 5.1 Load Materials

`GET /products/{product_id}/configurator/materials`

```json
{ "success": true, "data": {
  "glb_version": "asset:3f2b91c4-…", "model_url": "https://cdn/…/chair.glb",
  "material_count": 3,
  "materials": [ {
    "material_index": 0, "name": "Seat_Fabric", "mesh_names": ["Seat"],
    "has_base_color_texture": true, "average_color": "#6B6B6B",
    "suggested_method": "luminance", "similarity_group_hint": 0,
    "center": [0.0, 0.42, 0.0],
    "assigned_part_id": "9c1e7a52-…", "eligible_for_part": false } ] } }
```

⚠️ **Downloads and parses the GLB — seconds, not milliseconds.** Call once when the
editor opens and hold it in state. Never in a render path.

| Field | Use |
|---|---|
| `material_index` | The material's identity. Everything keys off this. |
| `name`, `mesh_names` | Picker labels |
| `average_color` | Swatch before anything is configured |
| `suggested_method` | Preselect the method: `factor` / `luminance` / `remap` |
| `has_base_color_texture` | `false` → only a factor recolour is possible |
| `similarity_group_hint` | Powers "select all similar". **Unstable** — recomputed per call, changes on re-upload. Never persist; never treat as a Part id. |
| `center` | `[x,y,z]` or `null`, for a 3D label |
| `assigned_part_id`, `eligible_for_part` | Which Part claims this index. Disable claimed checkboxes from these — an index belongs to at most one active Part. |

**`base_color_texture_url` and `triangle_count` are not returned**, though
`api-spec.md` §5 lists them (its own §5.1 explains why they were dropped). Don't
build UI needing them.

### 5.2 Create a Part

`POST /products/{product_id}/configurator/parts` → `201`

```json
{ "name": "Seat Fabric", "material_indices": [0, 2], "material_type": "fabric",
  "order_index": 0, "shopper_selectable": true }
```

| Field | Req | Rules |
|---|---|---|
| `name` | yes | 1–100 chars, trimmed; blank → `422` |
| `material_indices` | yes | 1–64 entries, all `>= 0`, no duplicates; each must exist on the GLB and be unclaimed |
| `material_type` | no | one of `fabric` `wood` `metal` `leather` `plastic` |
| `order_index` | no | `>= 0`; server appends if omitted |
| `shopper_selectable` | no | default `true` |

**Server-generated — do not send:** `slug`, `glb_version`, `isactive`,
`default_option_id`, `glb_stale`, timestamps. No limit on Parts per product.

The Part object is `Part` in §7. Two of its fields are **computed, never stored**: `default_option_id` (from whichever Option has `is_default`) and `glb_stale`.

`GET …/parts` returns Parts **with Options nested**, ordered by `order_index` —
one request renders the editor, don't N+1 it. Inactive Parts and Options are
included on the seller side; filter for display yourself.

**`glb_stale: true` is a hard gate, not a warning.** While true the server refuses
material changes, recipe changes and bakes (all `409`), and the shopper payload
drops the Part. Render a blocking "Re-map materials" state.

### 5.3 Edit a Part

`PATCH /configurator/parts/{part_id}` — all fields optional, plus `isactive`.

```json
{ "name": "Seat Cushion", "material_indices": [0, 2, 5] }
```

Response is the Part plus `"invalidated_option_ids": ["5d6e7f80-…"]`.

Changing `material_indices` discards every bake on the Part: each Option goes back
to `pending` and is listed. Renaming or reordering does not.

> 🔴 **Backend discrepancy — confirmed by test.** Those Options are set to
> `pending` but **no bake is enqueued**, and `POST …/bake` cannot start one:
> `pending` counts as in-flight, so the call returns the existing ticket with
> `enqueued: false` and nothing runs. The recovery sweep only reclaims rows stuck
> in `baking`, never `pending`. **After re-mapping a Part's materials, its Options
> stay `pending` indefinitely.**
>
> This is a backend fix (`PartService.update_part` needs the
> `OptionService.schedule_bake` call that option-create already makes). No frontend
> workaround exists — `recipe_hash` excludes `material_indices`, so re-submitting
> the same recipe is also a no-op. Until it lands, surface the returned ids as
> "re-bake required (blocked)" rather than implying the UI can fix it.

### 5.4 Options

```
GET    /configurator/parts/{part_id}/options
POST   /configurator/parts/{part_id}/options     → 201
GET    /configurator/options/{option_id}
PATCH  /configurator/options/{option_id}
DELETE /configurator/options/{option_id}
```

Create a colour Option:

```json
{ "name": "Charcoal", "swatch_hex": "#3A3A3A", "order_index": 0,
  "recipe": { "version": 1, "method": "luminance", "color": "#3A3A3A", "brightness": 1.0 } }
```

The response is `Option` in §7, with `bake_status: "pending"`, `bake_attempts: 0` and `textures: []`.

**The recipe**

| `method` | Needs | Produces |
|---|---|---|
| `factor` | `color` | **no texture** — a `baseColorFactor` change |
| `luminance` | `color` | one texture per textured material, shading kept |
| `remap` | `color` | as above, for very dark materials |
| `image` | `image_url` + `swatch_hex` | the uploaded texture (§5.5) |
| `auto` | `color` | resolved server-side, never stored |

- `brightness` `0.1`–`2.0`, default `1.0`; ignored for `image`.
- `color` required unless `method` is `image`; `image_url` rejected unless it is.
  Hex is normalised (`"3a3"` → `"#33AA33"`).
- `overrides` (optional, invalid for `image`) deviate per material:
  `[{ "material_index": 2, "method": "factor", "color": "#FFFFFF" }]`. Each index
  must belong to the Part.
- **`auto` is a `400` only when the Part's materials suggest *differing* methods**
  — not whenever a Part has several materials. A 5-material Part whose materials all
  suggest `luminance` resolves fine. Compute the distinct set of `suggested_method`
  client-side and offer `auto` when it has one member. Mixed suggestions give
  `"'auto' is ambiguous for this part: its materials suggest luminance, remap.
  Specify an explicit method instead."` Safe policy — `auto` for a single-material
  Part, an explicit `suggested_method` otherwise.
- Max **32 Options per Part**.

**Creating an Option enqueues its bake automatically** — it returns `pending` with
the runner already holding it. Go straight to polling (§5.7); do not call
`POST …/bake`.

**PATCH** accepts `name`, `swatch_hex`, `recipe`, `order_index`, `isactive`,
`set_as_default`. Only a `recipe` change that alters `recipe_hash` re-bakes, and
that re-bake is enqueued for you too. A rename or reorder never triggers one.

**`textures` holds only textures from the Option's *current* recipe.** You do not
need to check them for staleness — `completed` guarantees they are current. A bake
whose recipe moved on while it ran is **discarded**, not stored: `complete_bake`
returns early on a `recipe_hash` mismatch without setting `completed`, and
`_unlink_stale_textures` removes leftovers from earlier recipes first.

So `completed` + `textures: []` is **unambiguous** — it is always the legitimate
empty case of §6.6 (a `factor` treatment, or materials with no base-colour image),
never "stale, needs a re-bake". Drive the editor off `bake_status` alone.

**`is_default`, practically**

| Situation | Behaviour |
|---|---|
| On create | not accepted |
| First Option to reach `completed` | becomes the default; a later bake never displaces it |
| Moving the default | `PATCH set_as_default: true` on the **new** Option — only once it is `isactive` and `completed`, else `400` |
| `set_as_default: false` | **ignored**, not an error |
| Hiding the default | `400` — promote a replacement first |
| Deleting the default | next eligible Option is promoted, else `default_option_id: null` |

### 5.5 Image Options

Not a separate type or endpoint — just `recipe.method = "image"`.

```
POST /uploads/content  (multipart, Bearer)  →  take data.imageURL
POST /configurator/parts/{part_id}/options  →  recipe.method = "image"
```

Upload response (**camelCase**, unlike the Configurator):

```json
{ "success": true, "data": {
  "uploadId": "…", "contentType": "image/png", "sizeBytes": 284410,
  "url":       "https://cdn/uploads/users/{user_id}/uploads/{id}/weave.png",
  "imageURL":  "https://cdn/uploads/users/{user_id}/uploads/{id}/weave.png",
  "publicURL": "https://<account>.blob.core.windows.net/…" } }
```

⚠️ **Use `imageURL` (or `url`). Never `publicURL`** — that is the raw Azure Blob
URL and the backend rejects it; validation anchors on the CDN prefix.

```json
{ "name": "Herringbone", "swatch_hex": "#8B7355",
  "recipe": { "version": 1, "method": "image",
              "image_url": "https://cdn/uploads/users/{user_id}/uploads/{id}/weave.png" } }
```

`image_url` must sit under the **calling seller's own** upload namespace
(`…/users/{your_user_id}/uploads/…`), be `http(s)`, contain no `..`, and end in
`.png` `.jpg` `.jpeg` `.webp`. Every failure returns the same message:
`"recipe.image_url must be a file you uploaded via POST /uploads/content."`

So **never let a seller paste a URL** — only submit a value your client just
received from the upload call. `swatch_hex` is **required** here (there is no
`recipe.color` to derive it from), so show a colour picker beside the file input.
The file is copied into the Configurator namespace, so deleting the original upload
later won't break the Option.

### 5.6 Bake

`POST /configurator/options/{option_id}/bake` → `202`

```json
{ "success": true, "data": {
  "option_id": "5d6e7f80-…", "bake_status": "pending",
  "recipe_hash": "a3f91c2e…", "already_current": false,
  "poll_url": "/configurator/options/5d6e7f80-…/bake-status" } }
```

**You rarely need this call** — create and recipe-change already enqueue. Its two
real uses: **retrying a `failed` Option**, and **`force=true`** to rebuild a
`completed` Option whose file went missing.

| Stored state | What happens |
|---|---|
| `completed` and current, no `force` | nothing re-baked; `already_current: true` |
| `pending` or `baking` | existing ticket returned; no second bake, even with `force=true` |
| `failed`, or `completed` + `force=true` | a bake is enqueued |

`force=true` is implemented (`?force=true`). It does **not** bypass the in-flight
guard or the stale-GLB check. `poll_url` omits `API_PREFIX` — if you deploy with a
non-empty prefix, build it from `option_id` yourself.

### 5.7 Poll Bake Status

`GET /configurator/options/{option_id}/bake-status`

```json
{ "success": true, "data": {
  "option_id": "5d6e7f80-…", "bake_status": "baking", "bake_error": null,
  "bake_started_at": "2026-09-10T08:15:02Z", "bake_completed_at": null,
  "bake_attempts": 1, "recipe_hash": "a3f91c2e…",
  "progress": { "textures_total": 2, "textures_done": 1 }, "textures": [] } }
```

| `bake_status` | UI |
|---|---|
| `pending` | queued — spinner, keep polling |
| `baking` | in progress — `progress` drives a bar |
| `completed` | done — read `textures` |
| `failed` | show `bake_error` (seller-safe, ≤500 chars) + Retry |

**Drive completion off `bake_status`, never off `textures_done === textures_total`**
— `textures_total` is best-effort and may over-count. Poll at ~**2s**, back off
after ~30s, stop at **5 minutes** with a retry affordance, never faster than 1s.
Poll per Option and stop when the editor closes.

```ts
const TERMINAL = new Set(['completed', 'failed']);

async function waitForBake(optionId: string, signal?: AbortSignal) {
  let delay = 2000;
  const deadline = Date.now() + 5 * 60_000;
  while (Date.now() < deadline) {
    if (signal?.aborted) return null;
    const { data } = await api.get(
      `/configurator/options/${optionId}/bake-status`, { signal });
    if (TERMINAL.has(data.data.bake_status)) return data.data;
    await sleep(delay);
    delay = Math.min(delay * 1.5, 10_000);
  }
  return null;   // show retry
}
```

`bake_attempts` counts real attempts; the backend auto-retries up to 3 for bakes it
recovers itself. A `failed` Option needs an explicit `POST …/bake`.

---

## 6. Shopper Viewer Integration

### 6.1 Fetch Configuration

`GET /public/products/{product_id}/configurator` — no auth header.

```json
{ "success": true, "data": {
  "product_id": "1a2b3c4d-…", "product_name": "Lounge Chair",
  "model_url": "https://cdn/…/chair.glb", "ar_model_url": "https://cdn/…/chair.usdz",
  "parts": [ {
    "id": "9c1e7a52-…", "name": "Seat Fabric", "slug": "seat-fabric",
    "material_indices": [0, 2], "order_index": 0,
    "default_option_id": "5d6e7f80-…",
    "options": [ {
      "id": "5d6e7f80-…", "name": "Charcoal", "slug": "charcoal",
      "swatch_hex": "#3A3A3A", "order_index": 0,
      "textures": [ { "material_index": 0,
                      "url": "https://cdn/uploads/configurator/…/0-a3f91c2e….png",
                      "content_type": "image/png" } ] } ] } ] } }
```

**The viewer needs no filtering of its own.** The server already removed:

- unpublished, soft-deleted or missing products → **`404`** (unpublished is
  deliberately indistinguishable from missing)
- Parts with `isactive: false` or `shopper_selectable: false`
- Parts whose `glb_version` ≠ the product's current GLB
- Options not `completed`, or `isactive: false`
- textures whose `recipe_hash` ≠ the Option's current recipe
- Parts left with zero visible Options

`parts: []` is valid — *nothing configurable right now*. Render the plain model,
not an error. Parts arrive ordered by `order_index`, Options by
`(order_index, name)`.

This is a separate schema from the seller one: it never carries `recipe`,
`recipe_hash`, `bake_*`, `glb_version`, `blob_url`, `is_default`, `isactive`,
`material_type` or audit fields.

### 6.2 Load the GLB

Load `model_url` — the seller's original GLB. `ar_model_url` (USDZ) is for iOS AR;
either may be `null`.

**There is no Configurator-generated GLB.** Never re-fetch the model when the
shopper changes an Option — you only swap textures on the loaded scene.

### 6.3 Apply an Option

```
part → option → option.textures[] → texture.material_index
                                          ↓
                               model.materials[material_index]
```

**Always address materials by `material_index`.** `ModelViewerMaterial` exposes a
`readonly index`, and the project's own type declarations state why: glTF material
*names* are optional and exporters routinely emit duplicates. `getMaterialByName`
exists — do not make it the primary mechanism.

### 6.4 Apply a Texture

Confirmed against `@google/model-viewer` **^4.1.0** (both portals) and
`Rivollo.Web.Portal/types/model-viewer.d.ts`. The scene graph exists only **after
the element's `load` event fires**.

```ts
async function applyOption(
  mv: ModelViewerElement, part: PublicProductPart, option: PublicPartOption,
) {
  const model = mv.model;
  if (!model) return;                          // not loaded yet

  if (option.textures.length > 0) {
    for (const tex of option.textures) {
      const pbr = model.materials[tex.material_index]?.pbrMetallicRoughness;
      if (!pbr) continue;
      if (!pbr.baseColorTexture) {             // no base-colour slot on this material
        pbr.setBaseColorFactor(option.swatch_hex);
        continue;
      }
      const texture = await mv.createTexture(tex.url);
      // Neutral factor so the baked texture shows its true colours.
      pbr.setBaseColorFactor([1, 1, 1, pbr.baseColorFactor[3] ?? 1]);
      pbr.baseColorTexture.setTexture(texture);
    }
    return;
  }

  for (const index of part.material_indices) {  // no textures — see §6.6
    model.materials[index]?.pbrMetallicRoughness?.setBaseColorFactor(option.swatch_hex);
  }
}
```

- `setBaseColorFactor` accepts **either** a hex/CSS string **or** an `[r,g,b,a]`
  array of 0–1 floats. No manual conversion needed.
- `pbr.baseColorTexture` is **optional** — absent when the material has no
  base-colour slot, and you cannot `setTexture` on it. Fall back to the factor.

Texture URLs are content-addressed and immutable (`max-age=31536000, immutable`).
Cache hard; never add a cache-busting query string — a changed recipe yields a new
URL.

### 6.5 Multiple Material Indices

A Part owns a **list** of indices and one Option can carry several textures, one per
affected material. Always iterate — over `option.textures` when applying textures,
over `part.material_indices` for the fallback. Never assume one Part means one
material, and never apply only `textures[0]`.

### 6.6 Options With No Textures

An empty `textures` array is **legitimate, not a bake failure**. The backend emits
no file when the resolved treatment is `factor`, or when the material has no
base-colour image at all — both are coloured through `baseColorFactor` instead.

Use the §6.4 fallback: apply `option.swatch_hex` via `setBaseColorFactor` to every
index in `part.material_indices`. Never show an error; never skip the Option.

### 6.7 Default Option

```ts
const initial = part.default_option_id
  ? part.options.find(o => o.id === part.default_option_id) ?? part.options[0]
  : part.options[0];   // our deliberate fallback, not the seller's choice
```

`default_option_id` is computed over the **surviving** Options and deliberately
**not** substituted when the seller's configured default was filtered out (still
baking, or hidden). `null` means *no visible seller-configured default* — not *use
the first one*. Don't assume `options[0]` is what the seller chose.

---

## 7. TypeScript Types

```ts
export type UUID = string;
export type ISODateTime = string;
export type HexColor = string;                 // always "#RRGGBB" on output
export interface ApiSuccess<T> { success: true; data: T; }

export type BakeStatus = 'pending' | 'baking' | 'completed' | 'failed';
export type MaterialType = 'fabric' | 'wood' | 'metal' | 'leather' | 'plastic';
export type RecipeMethod = 'auto' | 'factor' | 'luminance' | 'remap' | 'image';
export type SuggestedMethod = 'factor' | 'luminance' | 'remap';

export const MAX_MATERIAL_INDICES_PER_PART = 64;   // 1–64 entries per Part
export const MAX_OPTIONS_PER_PART = 32;            // at most 32 options

export interface Recipe {
  version: 1;
  method: RecipeMethod;
  color?: HexColor | null;       // required unless method === 'image'
  brightness?: number;           // 0.1–2.0, default 1.0
  image_url?: string | null;     // required iff method === 'image'
  overrides?: Array<{ material_index: number;
                      method: Exclude<RecipeMethod, 'image'>;
                      color: HexColor }>;        // invalid for 'image'
}

export interface Material {
  material_index: number; name: string; mesh_names: string[];
  has_base_color_texture: boolean; average_color: HexColor;
  suggested_method: SuggestedMethod;
  similarity_group_hint: number;                 // unstable — never persist
  center: [number, number, number] | null;
  assigned_part_id: UUID | null; eligible_for_part: boolean;
}
export interface MaterialsResponse {
  glb_version: string; model_url: string;
  material_count: number; materials: Material[];
}

export interface Texture {
  material_index: number; url: string; content_type: string;
  width: number | null; height: number | null; size_bytes: number | null;
}

export interface Option {
  id: UUID; part_id: UUID; name: string; slug: string; swatch_hex: HexColor;
  recipe: Recipe;                                // `auto` always resolved on read
  recipe_hash: string; order_index: number;
  is_default: boolean; isactive: boolean;
  bake_status: BakeStatus; bake_error: string | null;
  bake_started_at: ISODateTime | null; bake_completed_at: ISODateTime | null;
  bake_attempts: number;
  textures: Texture[];                           // current recipe only; may be empty
  created_at: ISODateTime | null;
}

export interface Part {
  id: UUID; product_id: UUID; name: string; slug: string;
  material_indices: number[]; material_type: MaterialType | null;
  order_index: number; shopper_selectable: boolean; isactive: boolean;
  glb_version: string;
  default_option_id: UUID | null;                // computed
  glb_stale: boolean;                            // computed; blocks edits and bakes
  options: Option[];
  created_at: ISODateTime | null; updated_at: ISODateTime | null;
}
export interface PartUpdateResponse extends Part {
  invalidated_option_ids: UUID[];                // §5.3 — these do NOT bake on their own
}

// ---- requests ----
export interface PartCreate {
  name: string; material_indices: number[];
  material_type?: MaterialType; order_index?: number; shopper_selectable?: boolean;
}
export type PartUpdate = Partial<PartCreate & { isactive: boolean }>;

export interface OptionCreate {
  name: string;
  swatch_hex?: HexColor;                         // required when method === 'image'
  recipe: Recipe; order_index?: number;
}
export interface OptionUpdate {
  name?: string; swatch_hex?: HexColor; recipe?: Recipe;
  order_index?: number; isactive?: boolean;
  set_as_default?: true;                         // `false` is ignored server-side
}

// ---- bake ----
export interface BakeTicket {
  option_id: UUID; bake_status: BakeStatus; recipe_hash: string;
  already_current: boolean;
  poll_url: string;                              // lacks API_PREFIX
}
export interface BakeStatusResponse {
  option_id: UUID; bake_status: BakeStatus; bake_error: string | null;
  bake_started_at: ISODateTime | null; bake_completed_at: ISODateTime | null;
  bake_attempts: number; recipe_hash: string;
  /** Best-effort; may over-count. Never use for completion. */
  progress: { textures_total: number; textures_done: number } | null;
  textures: Texture[];
}

// ---- shopper ----
export interface PublicOptionTexture {
  material_index: number; url: string; content_type: string;
}
export interface PublicPartOption {
  id: UUID; name: string; slug: string;
  swatch_hex: HexColor; order_index: number;
  textures: PublicOptionTexture[];               // may be empty — §6.6
}
export interface PublicProductPart {
  id: UUID; name: string; slug: string;
  material_indices: number[]; order_index: number;
  default_option_id: UUID | null;                // null != "use options[0]"
  options: PublicPartOption[];
}
export interface PublicConfiguratorResponse {
  product_id: UUID; product_name: string;
  model_url: string | null; ar_model_url: string | null;
  parts: PublicProductPart[];                    // [] is valid
}
```

---

## 8. Frontend Prerequisites / Known Issues

### 8.1 🔴 Shopper viewer cannot swap textures yet

`Rivollo.Viewer.Portal/components/shared/preview/ThreeViewer.tsx` has **no**
`createTexture`, `setTexture` or `setBaseColorFactor` calls, and
`Rivollo.Viewer.Portal/types/model-viewer.d.ts` does not declare the scene-graph
API at all (no `model`, `materials`, `createTexture`). The Seller Portal has both,
so the technique is proven — the viewer just needs the work.

**To do:** port the scene-graph declarations from
`Rivollo.Web.Portal/types/model-viewer.d.ts`, then implement §6.4 keyed on
`material_index`. Nothing else in §6 can be exercised until then.

### 8.2 🔴 CDN CORS blocks baked textures entirely

Not a preview-only concern: **without it, no baked Option can be displayed at all**,
in either portal.

`mv.createTexture(url)` calls `textureUtils.loadImage`, which uses a three.js
`TextureLoader`. three.js's `Loader` base class sets `this.crossOrigin =
'anonymous'` (its documented default) and model-viewer never overrides it — it only
calls `setWithCredentials`. Every texture is therefore fetched as a CORS request,
and a response without `Access-Control-Allow-Origin` **fails to load**. It does not
merely taint a canvas.

The CDN serving `/uploads/configurator/…` must return
`Access-Control-Allow-Origin` for both portal origins. Nothing in this repository
configures that. Verify it before building §6.4, or you will debug an
infrastructure problem as a frontend one.

### 8.3 🔴 `remap` preview and bake do not match

Both sides compute `detail = 0.55 + 0.9 × normalized`, but normalise against
different dark/bright points:

| | Dark/bright points |
|---|---|
| Backend (`app/services/color/texture.py:137`) | **2nd / 98th percentile** of luminance |
| Frontend (`Rivollo.Web.Portal/lib/utils/textureRecolor.ts:186-195`) | **absolute min / max** |

On a texture with outlier pixels these visibly differ, so a seller can approve a
preview the bake won't reproduce. (The frontend comment claims "percentile-stretch"
but the code uses min/max — the comment is wrong.)

`luminance` is **identical** on both sides (`clip(lum / mean, 0.35, 1.8)`) and the
method-selection thresholds match. Until this is decided (ADR-005), prefer
`luminance` in seller-facing defaults and treat `remap` previews as approximate.

One more mismatch to fix when wiring the new editor: the existing preview keys its
overrides by `material.name`, while the Configurator addresses materials by
**index**. Use the index.

### 8.4 Some products' GLBs are unreachable by the backend

Many existing products have model URLs on storage accounts the API has no
credentials for. For those, `…/configurator/materials` and bakes fail with
`502 "The product's 3D model could not be read."` This needs an infrastructure
decision and is not a frontend bug — **test against a product whose GLB the backend
can actually read**, or you'll debug configuration as if it were code.

### 8.5 Re-mapped Parts don't re-bake

See §5.3. Affects editor flow design today.

---

## 9. Test Product

Verified against the dev database on 2026-09-11. This product exercises the whole
pipeline end to end and its GLB **is reachable by the backend**, so it is the one
to develop against (see §8.4).

```
product_id: 11efbbf1-cf13-4164-8710-07614a1114af    "shoe", PUBLISHED
```

| | |
|---|---|
| Mesh asset | on the CDN origin → **reachable**, `…/materials` works |
| `glb_version` | `asset:802fd78c-b1b9-4d5f-a031-47e632ad358f` |
| Parts | 1 — **Upper**, `material_indices: [1]`, active, shopper-selectable, `glb_stale: false` |
| Options | **Black** (default) and **Red** — both `completed`, `method: "luminance"`, 1 current texture each, 1 bake attempt |

So the public payload returns one Part with two Options, and
`default_option_id` points at Black. A good end-to-end smoke test:

```bash
curl -s http://127.0.0.1:8000/public/products/11efbbf1-cf13-4164-8710-07614a1114af/configurator
```

Note Black became the default automatically — it was simply the first bake to
reach `completed` (§5.4). Nobody set it.

If this ever stops working: `404` → unpublished or soft-deleted; `parts: []` → the
Part failed one of §6.1's filters, most often because a re-upload changed
`glb_version` and left it stale.

---

## 10. Related Documentation

| Doc | For |
|---|---|
| [api-spec.md](api-spec.md) | Endpoint contract of record — full validation tables and the complete error catalogue. Predates implementation in places; this guide marks where. |
| [baking.md](baking.md) | The recolour spec: what each method does to pixels, and the bake lifecycle. Read §5 before touching client-side preview — preview and bake must implement one spec. |
| [decisions.md](decisions.md) | ADRs and open questions. Relevant here: ADR-003 (textures not GLBs), ADR-005 (the `remap` divergence), ADR-011 (defaults), Q1 (CORS), Q7 (viewer texture swap). |
