# Product Configurator — Preview and Baking

> Status: **specification only**. The `BakeService` described here does not exist. The
> *existing* colour-variant bake service is described in §6 and is CONFIRMED.

---

## 1. The two operations

They share a specification and share nothing else.

|  | Preview | Bake |
|---|---|---|
| Trigger | every click | once per saved option |
| Runs on | the browser | the server |
| Latency budget | one frame | 30-90 seconds |
| Output | a texture in GPU memory | a file on the CDN + a database row |
| Lifetime | until the tab closes | permanent |
| Backend call | **none** | one, asynchronous |

The rule that makes the split safe: **both implement the same recipe specification.** If they
diverge, a shopper picks a colour, sees one thing, reloads, and sees another.

---

## 2. Preview

```
 shopper clicks "Charcoal"
      │
      ├─ frontend reads the option's recipe from the configurator payload
      ├─ frontend takes the part's source base-colour texture (already in memory
      │  from the GLB, or fetched from base_color_texture_url)
      ├─ frontend applies the recipe on a canvas / in a shader
      └─ viewer swaps material.map for the affected material indices
```

**No backend request per interaction.** The configurator payload (`GET /public/products/{id}/configurator`)
is fetched once and contains every option's baked texture URL. For options already baked, the
frontend can simply swap to the baked URL and skip client-side maths entirely — that is the
fast path and the visually authoritative one.

Client-side recolouring is needed for exactly two cases:

1. The **seller editor**, where an option is being authored and has not been baked yet.
2. A shopper interaction on an option whose bake has not completed — which the shopper API
   filters out, so in practice: case 1 only.

**CORS is a hard prerequisite for case 1.** Reading texture pixels
(`canvas.getImageData`, or `THREE.WebGLRenderer.readRenderTargetPixels`) from a cross-origin
image taints the canvas and throws unless the CDN sends `Access-Control-Allow-Origin` *and*
the image is loaded with `crossorigin="anonymous"`. See [architecture.md §12](architecture.md#12-cors)
— **NEEDS VERIFICATION**, currently unconfigured as far as this repository shows.

---

## 3. Bake

### 3.1 Texture-only, not whole-GLB

```
 source texture
   · recolour methods: the base-colour texture extracted from the original GLB
   · method "image":   the seller's validated upload (ADR-013)
      │
      ├─ apply recipe: factor | luminance | remap (then brightness), or image passthrough
      │
      ├─ encode PNG (alpha present) or JPEG q92 (no alpha)
      │
      ├─ upload → {container}/configurator/{product_id}/{glb_version}/
      │            {option_id}/{material_index}-{recipe_hash}.{ext}
      │
      └─ INSERT tbl_part_option_textures, set option completed
```

The geometry is never decoded, modified, or re-encoded. The original GLB is untouched.

**Why this matters concretely.** The existing colour-variant bake produces a full GLB per
colourway. A 60 MB mesh with four colourways is 300 MB stored and 60 MB downloaded per swatch
change. The same four colourways as textures are ~2 MB each: 68 MB stored, and a swatch
change transfers 2 MB. Geometry is downloaded exactly once, ever, and stays in GPU memory
across every option the shopper tries. That is the whole argument for
[ADR-003](decisions.md#adr-003).

**The known limit.** Texture-only baking changes base colour. It cannot change geometry,
normal maps, roughness/metallic maps, or UV layout. A "material" change that means a
different *finish* (matte → gloss) needs a PBR-factor change too; a change that means
different *geometry* (a different leg shape) is out of scope entirely and would need a
separate model. Document this to sellers as "colour and surface appearance", not "material".

### 3.2 Which textures a bake produces

One per material index the part owns that has a base-colour texture:

```
Part "Seat", material_indices = [0, 3]
   material 0  has base-colour texture  → bake a texture
   material 3  no texture, factor only  → no texture row; the recipe's colour is
                                          applied by the viewer as a baseColorFactor
```

For a material with no base-colour image, the `factor` method is the only correct treatment
(the existing engine already forces this fallback, `glb_recolor.py:440-443`). No file is
produced; the viewer sets the material's colour directly from the recipe. The option's
`textures` array is therefore allowed to be shorter than `material_indices`, and an option
may legitimately have zero textures.

### 3.2.1 `image` options ([ADR-013](decisions.md#adr-013))

An option whose `recipe.method` is `"image"` bakes from the seller's upload instead of from
the GLB's own texture:

```
1. validate  recipe.image_url is under users/{current_user.id}/uploads/…   (§7.2 of api-spec)
2. fetch     the uploaded file
3. normalise format / dimensions; encode PNG (alpha) or JPEG q92 (no alpha)
4. copy      → configurator/{product_id}/{glb_version}/{option_id}/{material_index}-{hash}.{ext}
5. record    one tbl_part_option_textures row per material index in the part
6. complete
```

Three properties worth stating plainly:

- **It is a real bake**, not a special case. Same `bake_status` transitions, same
  `bake_started_at`, same sweeps, same `recipe_hash` idempotency, same staleness rule.
- **The file is copied, never referenced.** Recording the seller's own
  `users/{user_id}/uploads/…` URL would mean deleting an option purges a blob that may be
  referenced elsewhere. Copying makes the Configurator the unambiguous owner of what it
  purges.
- **One row per material index in the part** — unlike the recolour methods, an uploaded image
  replaces the base colour of every material the part owns, including those that had no
  texture of their own.

Because there is no client-side recolouring involved, `image` options are unaffected by the
unresolved `remap` divergence in §5.3 and can ship while [ADR-005](decisions.md#adr-005)
stays open.

### 3.3 Reusing the existing engine

**CONFIRMED — the recolour maths already exists and is sound.**
`app/services/color/glb_recolor.py:364-408` `_recolor_pixels(pil, hex_color, remap)` is
already a pure function from a PIL image to encoded bytes. It has no database, storage, or
HTTP dependency. `app/services/color/colors.py` holds correct sRGB↔linear conversion and HSL
brightness adjustment.

**PROPOSED refactor — the only sanctioned change to existing code, and it belongs in its own
reviewed PR:**

```
app/services/color/
  colors.py        unchanged
  texture.py       NEW — recolor_texture(image_bytes, method, color, brightness)
                          -> (bytes, mime).  Lifts the body of _recolor_pixels.
  glb_recolor.py   inspect() + recolor() unchanged in behaviour;
                   _recolor_pixels becomes a thin wrapper over texture.py
```

Both features then call one implementation. **Do not copy the maths into a Configurator
module** — two copies of a colour algorithm diverge, and this one already has a suspected
frontend divergence (§5) without adding a third implementation.

Bump `BAKER_VERSION` (`app/services/color/__init__.py:24`, currently `"2"`) only if the
refactor changes output bytes. A pure lift should not.

---

## 4. Status lifecycle and failure handling

### 4.1 States

```
   pending ──▶ baking ──┬──▶ completed
      ▲                 │
      │                 └──▶ failed
      └──────────────────────────┘
            retry / recipe change
```

Four states, constrained by a database `CHECK` and typed as a Pydantic `Literal`. See
[data-model.md §8](data-model.md#8-bake-state).

**Do not widen the type.** The existing colour-variant response types this field as
`BakeStatus | str` (`app/schemas/color_variants.py:148`), and the `| str` arm defeats the
literal — the API is free to return a status outside the four, so no client can rely on the
union. Type the Configurator's as the literal alone.

### 4.2 The stale-bake problem

**CONFIRMED, in the existing implementation:**

```python
# app/services/variant_bake_service.py:242-263
variant.bake_status = "baking"
variant.bake_error = None
await db.commit()                         # ← committed here
...
cdn_url, blob_url, size = await asyncio.to_thread(_work)   # ← 30-90s of I/O
```

Between those two points there is no `bake_started_at`, no timeout, no reaper, and no startup
sweep. If the replica is recycled in that window the row stays `baking` forever and the UI
shows a permanent spinner. Prod runs on Azure Container Apps with autoscaling
(`.github/workflows/deploy-prod.yml:441-449`), so replica recycling is routine, not
exceptional. The exact scale ceiling and termination grace period live in Azure config and
are **NEEDS VERIFICATION**.

### 4.3 Required mitigations

All three ship in phase 1. None is expensive.

**1. `bake_started_at`**, written in the same transaction that sets `baking`.

**2. Startup sweep**, awaited in the existing `lifespan` context manager before traffic is
served, so a seller polling straight after a deploy is not told "baking" about work that died
with the old replica.

**3. Periodic sweep**, on the same cadence and in the same style as `_deactivation_loop`, so a
bake lost mid-session recovers without waiting for a deploy. Both call one function,
`BakeService.recover_stale_bakes` — there is no second scheduler.

**IMPLEMENTED behaviour per stale row** (this supersedes the single `UPDATE … SET
bake_status='failed'` sketch this section used to carry, which could only ever surface an
error and never actually retry):

| `bake_attempts` | Outcome |
|---|---|
| under `CONFIGURATOR_MAX_AUTOMATIC_BAKE_ATTEMPTS` | back to `pending` and **re-enqueued** |
| at or over it | marked `failed` with "Bake interrupted; please retry." |

The claim is a single guarded `UPDATE` whose `WHERE` repeats every condition — status,
`bake_started_at`, and `recipe_hash` — so two replicas sweeping at once cannot both re-enqueue
one option, and a row whose recipe changed meanwhile is left to the newer bake.

Thresholds live in `app/core/config.py`, not as magic numbers in the sweep:
`CONFIGURATOR_BAKE_STALE_AFTER_SECONDS` (900), `CONFIGURATOR_BAKE_SWEEP_INTERVAL_SECONDS`
(300), `CONFIGURATOR_BAKE_SWEEP_BATCH` (20),
`CONFIGURATOR_MAX_AUTOMATIC_BAKE_ATTEMPTS` (3).

### 4.4 Failure taxonomy

| Failure | `bake_error` shown to seller | Retryable |
|---|---|---|
| Product has no GLB | "This product has no 3D model yet." | after upload |
| GLB fetch failed | "Could not download the product model." | ✅ |
| GLB parse failed | "The product model could not be read." | ❌ needs re-upload |
| Material index missing from GLB | "Material 3 is no longer on this model." | ❌ needs re-map |
| Texture decode failed | "A texture on this model could not be read." | ❌ |
| Storage upload failed | "Could not save the generated texture." | ✅ |
| Out of memory | "The model is too large to process." | ❌ needs a smaller model |
| Interrupted (sweep) | "Bake interrupted; please retry." | ✅ |

`bake_error` is truncated to 500 chars before storage, matching
`variant_bake_service._mark_failed`. It is seller-facing: never put a stack trace, a blob
path, or a connection string in it. The full exception goes to `logger.exception`.

### 4.5 Retry and idempotency

`recipe_hash` = `sha256(canonical(recipe) + glb_version + BAKER_VERSION)` is the idempotency
key. It changes exactly when the output bytes would change.

- Re-baking an unchanged recipe is a no-op that returns the existing textures.
- Because the blob path embeds `recipe_hash`, a genuine re-bake overwrites the same path with
  identical bytes — safe, and safe to cache forever.
- **`bake_attempts` increments per real attempt** — in `BakeService.mark_baking`, the moment
  a worker claims the row, not when one is enqueued. An enqueue that never runs must not burn
  a retry. This is the authoritative reading; api-spec.md §8 previously said it incremented on
  re-enqueue and has been corrected to match.
- **DECIDED (implemented): automatic recovery stops after
  `CONFIGURATOR_MAX_AUTOMATIC_BAKE_ATTEMPTS` attempts (default 3).** At the cap the
  stale-bake sweep marks the row `failed` with "Bake interrupted; please retry." instead of
  re-enqueuing it, because retrying a deterministic failure (a corrupt GLB, an unreadable
  storage account) forever burns CPU and never succeeds.
- **The cap governs AUTOMATIC retries only.** An explicit `POST /options/{id}/bake` from the
  seller is never blocked by it — the seller may have fixed the underlying problem, and
  refusing them on a count they cannot see would be inexplicable.
- Concurrency: one bake per option at a time, guarded by the `baking` state plus
  `bake_started_at` freshness. A superseded bake — the recipe changed while it was running —
  discards its result rather than overwriting the newer one. The existing service already
  gets this right (`variant_bake_service.py:265-273`); copy the pattern.

---

## 5. Frontend/backend consistency

The frontend preview and the backend bake **must** implement the same specification. This
section is the specification.

### 5.1 The three methods — CONFIRMED backend behaviour

Read from `app/services/color/glb_recolor.py`.

**`factor`** — for materials with no base-colour image, and for near-white neutral ones.
No pixels are touched; the material's `baseColorFactor` is set to the target colour converted
sRGB→linear (`glb_recolor.py:453-455`, `colors.hex_to_linear_factor`). `white × colour = colour`,
and existing shading survives.

**`luminance`** — for already-coloured materials.

```python
# glb_recolor.py:372, 388-394
lum    = rgb @ [0.2126, 0.7152, 0.0722]          # Rec.709
mean   = max(lum.mean(), 1e-3)
detail = clip(lum / mean, 0.35, 1.8)             # centred on 1.0
out    = clip(detail[..., None] * target, 0, 1)
```

The key property: `detail` is centred on **1.0**, so the *average* pixel equals the target
colour exactly while highlights and shadows still read as highlights and shadows. Multiplying
the target by *raw* luminance instead makes dark textures come out muddy — the code comments
call this out explicitly as the bug this design avoids.

**`remap`** — for very dark materials.

```python
# glb_recolor.py:380-387
lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
if hi - lo < 1e-4: hi = lo + 1e-4
norm   = clip((lum - lo) / (hi - lo), 0.0, 1.0)
detail = 0.55 + 0.9 * norm                        # range ≈ 0.55 .. 1.45
```

**Brightness** is applied to the target colour *before* either path, as an HSL lightness
shift (`colors.adjust_brightness_hex`, `glb_recolor.py:451`), so factor and pixel paths stay
in sync.

**Encoding** (`glb_recolor.py:398-408`): PNG `compress_level=1` when the source had alpha,
otherwise JPEG `quality=92, subsampling=0`.

### 5.2 Automatic method selection — CONFIRMED thresholds

```python
# glb_recolor.py:43-45
_NEAR_WHITE_BRIGHTNESS = 0.82
_NEAR_WHITE_SATURATION = 0.18
_NEAR_BLACK_BRIGHTNESS = 0.16

# glb_recolor.py:163-171
def _suggest_method(avg_rgb, has_texture):
    if not has_texture:                     return "factor"
    brightness, saturation = _brightness_saturation(avg_rgb)
    if brightness >= 0.82 and saturation <= 0.18:  return "factor"
    if brightness <= 0.16:                          return "remap"
    return "luminance"
```

`brightness` is Rec.709 luma of the average colour. `saturation` is **HSV-style**
`(max-min)/max` (`glb_recolor.py:155-160`) — note this is a *different* saturation definition
from the HSL one used by `colors.adjust_brightness_hex`. Any frontend implementing `auto` must
use the HSV form here.

In the Configurator this matters less than it does today, because **`auto` is resolved
server-side at save time** and never persisted — but the editor will still want to *show* the
suggestion, so it needs the same rule.

### 5.3 The `remap` divergence — DECISION REQUIRED

**Reported (not verified from this repository — the frontend is not in this repo):**

| | Dark point | Bright point |
|---|---|---|
| Frontend | absolute darkest pixel (`min`) | absolute brightest pixel (`max`) |
| Backend | 2nd percentile | 98th percentile |

If accurate, preview and bake produce **visibly different results on the same recipe**, and
the difference is worst on exactly the assets `remap` exists for: a near-black texture with a
few specular highlights. `max` is pinned by those few pixels; the 98th percentile ignores
them. The stretch, and therefore the final colour, differ.

**Do not silently pick one.** Both are defensible:

- **Percentile (current backend)** is robust to outliers — a single blown-out pixel or a
  compression artifact cannot distort the whole image.
- **Absolute min/max (reported frontend)** is trivially cheap in a shader and needs no sort.

**Recommendation: standardise on the percentile form** and change the frontend, because
outlier robustness is the reason the method exists, and because a GPU-side approximation
(a coarse histogram, or a downsampled CPU pass on the small mip) is achievable. But this
**requires confirmation of the actual frontend implementation before either side changes.**

Tracked as [ADR-005](decisions.md#adr-005) and [decisions.md](decisions.md#open-questions) Q1.

### 5.4 Consistency test requirement

Once the algorithm is agreed, lock it down:

1. Commit a small fixture texture and a fixed set of recipes to `tests/fixtures/`.
2. Backend test asserts the bake output matches a committed golden PNG per recipe, byte-exact
   or within a tight per-pixel tolerance.
3. The frontend repository runs the **same fixtures** through its preview path and asserts
   against the **same goldens**.
4. Any change to the maths must update both, in the same change, and bump `BAKER_VERSION`.

Without step 3 the specification is a comment, not a contract.

---

## 6. What exists today

**CONFIRMED** — `app/services/variant_bake_service.py` (322 lines), a real implementation,
not a stub. Assessment against the four questions the brief asks:

### Works, keep and copy the pattern

- **Config-hash idempotency.** `compute_config_hash` folds source, recipe and baker version
  into one key; a re-bake at the same hash short-circuits (lines 227-232). Sound.
- **Stale-asset filtering.** Assets whose hash ≠ the parent's are hidden rather than served
  (`color_variant_service.py:373-381`). Exactly right.
- **Superseded-bake discard.** Re-reads the row after the long work and drops the result if
  the recipe changed underneath (lines 265-273). Correct.
- **Delete-after-commit blob purge.** Old blob removed only once the replacement is committed
  (lines 290-293), so no viewer hits a 404.
- **Blocking work off the event loop.** `asyncio.to_thread` around all of pygltflib/Pillow/Azure.
- **The disk cache.** `model_cache` turns the second bake of a product from a minute into
  seconds. Reuse as-is.
- **Never-raise background guard** with failure recorded on the row (lines 202-208).

### Reusable with a refactor

- `_recolor_pixels` → lift to `color/texture.py` (§3.3).
- `inspect()` → extend with texture URLs and triangle counts, do not rewrite.
- `get_product_model_url` / `MESH_ASSET_ID` → extract to a shared helper.

### Unsafe for production, do not carry forward

- **No `bake_started_at`, no reaper, no sweep.** Rows stick in `baking` forever. §4.2.
- **Whole-GLB bake.** The thing [ADR-003](decisions.md#adr-003) replaces.
- **Swallowed `auto` resolution failure** (lines 152-155) persists an unresolved method
  despite the schema comment claiming it cannot happen.
- **`bake_status: BakeStatus | str`** in the response schema defeats its own literal.
- **Optimistic rebake response** — `POST /color-variants/{id}/rebake` returns
  `"bake_status": "pending"` unconditionally (`color_variants.py:155`) before the task has
  run. The Configurator returns the *actual* state.
- **No ownership check.** Any authenticated user can bake any product's variants.
- **Tables outside Alembic.** `sql/create_color_variants.sql` is hand-run.
- **Zero test coverage.**

### Leave alone

The colour-variant tables, routes, and service stay as they are during Configurator work.
Coexistence, not replacement — [architecture.md §9](architecture.md#9-relationship-to-the-existing-colour-variant-feature).

### Has it ever run in production?

**No evidence that it has.** No seed or migration inserts into `tbl_variant_assets`; the
backfill in `sql/create_color_variants.sql:130-165` creates only `is_original` rows, which
short-circuit to `ready` without baking (`variant_bake_service.py:218-223`). `server.log`
contains no bake lines. There are no tests. Treat the bake path as **unexercised** — that is
an argument for building the Configurator's version with tests from the first commit, not for
assuming the existing one is broken.

---

## 7. Execution backend and its evolution

### Phase 1 — FastAPI `BackgroundTasks`

Matches the existing implementation and the brief's instruction not to introduce a queue
without justification. In-process, after the response is flushed.

```python
# app/api/routes/configurator.py — the route stays this thin
background_tasks.add_task(bake_service.run_bake, option_uuid)
```

Concurrency limited by a module-level `asyncio.Semaphore(1)`, as
`variant_bake_service._BAKE_SEMAPHORE` does — a large texture decodes to hundreds of MB and
parallel bakes are the fastest way to OOM the pod.

Known limits, all mitigated by §4.3 rather than by a queue: lost on replica recycle; no
cross-replica coordination; no durable retry; capacity tied to API pods.

### The seam

Everything above `bake_runner` is unaware of how a bake executes:

```python
# app/services/configurator/bake_service.py — the stable public contract
async def request_bake(db, option_id, *, force: bool = False) -> BakeTicket: ...
async def get_bake_status(db, option_id) -> BakeStatusView: ...

# app/services/configurator/bake_runner.py — the swappable part
async def enqueue(option_id: uuid.UUID) -> None: ...
```

Routes, schemas, services and the public API contract must never reference `BackgroundTasks`.
Swapping the runner is then a one-module change.

### Phase 2 — when, and to what

**Move when** any of: sellers report stuck or lost bakes after the sweep is in place; bake
latency degrades API p99; bakes exceed ~30/hour; or bakes need more memory than an API pod
should hold.

**CONFIRMED — two targets already exist in this repository:**

- `app/services/usdz_trigger_service.py` — fires an Azure Container Apps Job via the ARM API
  with `DefaultAzureCredential`, retry with backoff, fire-and-forget. Closest precedent; the
  GLB→USDZ conversion already works this way.
- `app/integrations/service_bus_publisher.py` — Azure Service Bus publisher.

**A memory note from 2026-09-01 records that the USDZ job OOMs at 1 GiB.** Size a bake job's
memory deliberately; do not copy the USDZ job's limits without measuring. **NEEDS VERIFICATION**
against current Azure configuration.

Do not build either in phase 1. Build the seam.

---

## 8. Storage layout

| Artifact | Path | Cache |
|---|---|---|
| Original GLB | existing product-asset paths, unchanged | existing |
| Extracted source texture (PROPOSED) | `configurator/{product_id}/{glb_version}/source/{image_index}.{ext}` | `immutable` |
| Baked option texture (PROPOSED) | `configurator/{product_id}/{glb_version}/{option_id}/{material_index}-{recipe_hash}.{ext}` | `immutable` |
| Seller-uploaded image | reuse `POST /uploads/content` → `users/{user_id}/uploads/{uuid4}/` | existing |

Container: `settings.STORAGE_CONTAINER_UPLOADS`. CDN URL via `storage_service._cdn_url`,
which raises if `CDN_BASE_URL` is unset — misconfiguration fails fast rather than returning
broken URLs.

`Cache-Control: public, max-age=31536000, immutable` on both configurator artifacts, matching
`upload_variant_model` (`storage.py:333-337`). Safe because both paths are content-addressed:
different bytes always mean a different path.

**Cleanup.** Deleting an option or part purges its texture blobs before the rows. Re-baking
overwrites the same path, so no orphan accumulates. A `glb_version` change orphans the whole
prefix for the old version — **PROPOSED:** a periodic reaper that deletes
`configurator/{product_id}/{old_glb_version}/**` once no part references it. Low priority;
note it now so it is not discovered as a cost surprise later.
