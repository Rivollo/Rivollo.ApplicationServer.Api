# Multi-Part 3D Generation

Generate a product model split into **separate, individually colourable parts** —
a chair whose seat, frame and legs can each be recoloured on their own.

`POST /createProductWithParts`

> Every number in this document was measured against a live run on a real
> product photo, not taken from vendor documentation.

---

## 1. What this gives you

The normal creation paths return **one solid model**. A seller can recolour it,
but only as a whole.

This path returns a model made of **many named parts**, each with its own
material. The colour configurator picks them up automatically, so a seller
colours the seat separately from the frame.

Measured on a chair photo:

```
19 parts   ·   19 materials   ·   57 texture images   ·   12.8 MB
```

The configurator reads all 19 as colourable parts with **no changes** to it.

---

## 2. Why it takes two steps

This is the single most important thing to understand about this feature.

Tripo **cannot** produce a segmented model and texture it in one call. Their
docs are explicit:

> `generate_parts` — *"Not compatible with `texture=true`, `pbr=true`, or
> `quad=true`. To use this, set all three to false."*

Asking for both at once fails. So the model is built in two passes:

```
STEP 1  Build the shape, split into parts     → bare grey geometry, no colour
STEP 2  Paint the parts using the same photo  → each part gets its own texture
```

Think of it like a model kit: first the pieces are moulded, then they're painted.
Tripo will not mould and paint in one action.

**One Rivollo endpoint hides both steps.** Callers make one request and get one
result — they never see the split.

---

## 3. The flow, end to end

```
Seller uploads a product photo
        │
        ▼
POST /createProductWithParts          ← returns 201 immediately, status "draft"
        │
        │  checks first, fails fast:
        │    · Tripo key configured?      no → 503
        │    · Pro/Enterprise plan?       no → 403
        │    · enough AI credits?         no → 400
        ▼
Background work begins                 status → queue → processing
        │
        ├─ STEP 1  POST /generation/image-to-model
        │            generate_parts: true, texture: false
        │            ~4 min · 40 Tripo credits
        │            → 19 parts, no colour
        │            → task id saved to tbl_jobs   ← lets a retry skip this step
        │
        ├─ STEP 2  POST /models/texture
        │            input: <step 1 task id>, photo re-supplied
        │            ~3 min · 20 Tripo credits
        │            → 19 parts, each with its own PBR texture
        │
        ├─ Draco compression        (optional, falls back to original on failure)
        ├─ Upload GLB to Azure      stored as asset_id 9
        ├─ Create asset + mapping rows
        ├─ USDZ conversion          (Pro/Enterprise only)
        └─ Notify the seller
        │
        ▼
status → ready                        Model opens in the configurator
```

Progress is streamed over the product-status WebSocket the whole time, using
Tripo's **real** progress percentage — not an estimate.

---

## 4. What we send Tripo right now

### Step 1 — geometry

```json
{
  "input":          "<product photo URL>",
  "model":          "v3.1-20260211",
  "generate_parts": true,
  "texture":        false,
  "pbr":            false,
  "quad":           false,
  "face_limit":     150000,
  "export_uv":      false
}
```

| Setting | Why |
|---|---|
| `generate_parts: true` | The whole point — splits the mesh into parts |
| `texture/pbr/quad: false` | **Mandatory.** Tripo refuses parts if any is true |
| `model: v3.1` | Newest geometry. `generate_parts` needs ≥ v3.0 |
| `face_limit: 150000` | Generous on purpose — decimating *before* segmentation risks merging parts we asked to keep apart |
| `export_uv: false` | UV unwrapping happens during texturing anyway; skipping it here is faster and produces a smaller intermediate |

### Step 2 — texture

```json
{
  "input":              "<step 1 task id>",
  "model":              "v3.0-20250812",
  "texture_prompt":     { "image": "<same product photo>" },
  "pbr":                true,
  "texture_quality":    "detailed",
  "texture_alignment":  "original_image",
  "bake":               true
}
```

| Setting | Why |
|---|---|
| `input: <task id>` | Chains directly to step 1 — the mesh never passes through our storage |
| `texture_prompt.image` | Tripo **strongly recommends** re-supplying the photo. Without it, texturing drifts toward the model's own priors instead of the real product |
| `model: v3.0` | Tripo pairs the v3.0 texture model with **both** v3.0 and v3.1 geometry. There is no v3.1 texture model |
| `pbr: true` | Metallic/roughness/normal maps. The colour configurator needs them, or recoloured parts look like flat paint |
| `bake: true` | Bakes advanced effects into the base textures. `<model-viewer>` cannot reproduce the unbaked versions |

---

## 5. Options we are not using

Everything below is available and currently left at a default. Change these in
[`app/integrations/tripo/tasks.py`](../app/integrations/tripo/tasks.py) — never
by accepting them from the API, or a client could produce a mesh the
configurator cannot read.

### Geometry

| Option | Values | Current | Notes |
|---|---|---|---|
| `model` | v3.1 / v3.0 / v2.5 | **v3.1** | v2.5 does not support parts |
| `face_limit` | up to 1.5M (v3.1) | **150,000** | Tripo suggests 10k–50k for web. Lower is lighter but risks merging parts |
| `geometry_quality` | standard / detailed | *(default)* | "detailed" = Ultra mode, slower. **Cost impact unmeasured** |
| `smart_low_poly` | true / false | false | Clean low-poly topology, 500–20k faces. May fail on complex inputs |
| `auto_size` | true / false | false | Scales to real-world metres. Useful for AR |
| `enable_image_autofix` | true / false | false | Enhances low-quality input photos |
| `model_seed` | integer | *(random)* | Same seed + same photo = identical mesh |
| `compress` | "geometry" | *(off)* | Tripo-side meshopt. We use Draco instead |
| `quad` | true / false | **false** | ⚠️ Never enable — forces FBX output and breaks everything downstream |

### Texture

| Option | Values | Current | Notes |
|---|---|---|---|
| `texture_quality` | standard / detailed / **extreme** | **detailed** | `extreme` = 8K textures and costs more. Amount unmeasured |
| `texture_alignment` | original_image / geometry | **original_image** | Match the photo's colours vs the generated shape |
| `texture_prompt.text` | free text | *(unused)* | e.g. "worn leather with scratches" |
| `texture_prompt.images` | 4 images | *(unused)* | front / left / back / right for multi-angle guidance |
| `part_names` | string[] | *(unused)* | **Texture only specific parts.** See below |
| `texture_seed` | integer | *(random)* | Same seed = identical texture |
| `bake` | true / false | **true** | Keep true for web viewers |

### Selective re-texturing (`part_names`)

Not used yet, but confirmed possible. Part names come from the **step 1 mesh** —
its glTF nodes are named `tripo_part_0` … `tripo_part_18`. Read them there and
pass a subset to re-texture only those parts:

```json
{ "part_names": ["tripo_part_3", "tripo_part_7"] }
```

Omitting it textures everything, which is what we do today.

---

## 6. Cost and time — measured

| Step | Tripo credits | Time |
|---|---|---|
| Geometry | **40** | ~4 min |
| Texture | **20** | ~3 min |
| **Total per product** | **60** | **~7 min** |

**1,000 Tripo credits ≈ 16 products.**

Compared with the other creation paths:

| Endpoint | Provider | Your AI credits | Parts? |
|---|---|---|---|
| `/createProduct` | fal SAM-3 | 20 | ✗ |
| `/createProductFal` | fal (SAM 3D 20; Tripo/Hunyuan/Trellis 100; Meshy 200) | 20–200 | ✗ |
| **`/createProductWithParts`** | **Tripo direct** | **200** | **✓** |
| `/createProductFromGlb` | none | 0 | n/a |

We charge **200 AI credits**, no longer 1:1 with what Tripo itself bills us
(60 — see the table above). The seller-facing price is a Rivollo pricing
decision independent of provider cost; keep `PARTS_PRODUCT_CREATION_AI_CREDIT_COST`
in `app/api/routes/products.py` and `AI_CREDIT_COST.multiPart` in the portal's
`lib/product/creditCosts.ts` in sync if it changes again.

> ⚠️ **Plan headroom.** `max_ai_credits_month` was **100 on Pro** (10 free, 5
> weekly) as of the last measurement in this doc — confirm the current value
> before relying on it, since 200 credits now exceeds that figure and would
> make multi-part unusable within a month's allowance if it hasn't been raised
> to match. There is also no `enterprise` row in `tbl_mstr_plans` — only
> `free`, `pro`, `weekly` — even though the route gates on `("pro", "enterprise")`.

**Untested cost variations:** `geometry_quality: detailed`, `texture_quality:
extreme`, and different `face_limit` values may all change the price. Tripo
publishes no per-task table — the **dashboard transaction history** is the
authoritative source.

---

## 7. How the code is organised

```
app/integrations/tripo/
    client.py     PROTOCOL  — Bearer auth, task polling, progress, download
    tasks.py      PAYLOADS  — one builder per Tripo task
    pipeline.py   SEQUENCE  — geometry → texture as one operation

app/services/product_service.py
    create_product_with_parts_image_urls    creates the record, schedules work
    generate_3d_with_parts_and_finalize     runs the pipeline, stores the result
    _run_parts_generation_background        background session wrapper

app/api/routes/products.py
    POST /createProductWithParts            gates, then delegates
```

Each layer knows one thing: `client.py` doesn't know what a "part" is,
`tasks.py` doesn't know how to poll, `pipeline.py` doesn't know about HTTP.
Adding another Tripo task means adding a builder and nothing else.

**This is deliberately separate from `app/integrations/fal/`.** Different auth,
different protocol, different billing, and two chained tasks instead of one.
Sharing code would couple things that fail differently.

---

## 8. When things go wrong

### Failure is resumable

Geometry is the expensive half (40 of the 60 credits). If texturing fails,
geometry has already succeeded and been paid for — so its task id is stored:

```sql
SELECT stage, status, external_task_id, provider_credits
FROM tbl_jobs
WHERE engine = 'tripo-parts'
ORDER BY created_date DESC;
```

- `stage='geometry'`, `status='failed'`, `external_task_id` set → **resumable**,
  pass it to `generate_3d_with_parts_and_finalize(resume_geometry_task_id=…)`
- `external_task_id` null → geometry never completed; must restart

### Non-fatal steps

These log a warning and continue rather than failing the product:

- **Draco compression** — falls back to uploading the original GLB
- **USDZ conversion** — the product is already `ready` with its GLB
- **Notifications** and **progress broadcasts**

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `503` on the request | `TRIPO_API_KEY` empty | Set it, **restart** — settings load at import |
| `403` on the request | Not Pro/Enterprise | Upgrade the account |
| `400` on the request | Under 60 AI credits | Top up (your credits, not Tripo's) |
| `201` then stuck at `draft` | Generation failed | **Read the console** — the Tripo `code` and message are logged |
| Tripo `code 2010` | Tripo account out of credit | Top up at Tripo |
| Tripo `code 2` "Invalid API key" | Wrong credential | See below |

### Getting the API key right

Tripo's console shows a **Client ID** (`tcli_…`) on the API Keys page. That is
**not** the secret — sending it gives `401`. The secret (`tsk_…`) is shown once
at creation. Use **rotate** to issue a new one.

Verify before spending anything — this call is free:

```bash
curl "https://openapi.tripo3d.ai/v3/account/balance" \
  -H "Authorization: Bearer YOUR_KEY"
```

```json
{"code":0,"status":"success","data":{"balance":900.00,"frozen":0.00}}
```

A key that authenticates but reports `balance: 0.00` belongs to a different
account or workspace than the one you funded.

---

## 9. API reference

**Request**

```http
POST /createProductWithParts
Authorization: Bearer <token>
Content-Type: application/json

{
  "userId":  "<uuid>",
  "name":    "Wooden Chair",
  "imageURL": "https://.../photo.jpg",
  "mesh_asset_id": 9          // optional, defaults to 9
}
```

No generation options are accepted by design — see §5.

**Response** — `201`, immediately:

```json
{
  "success": true,
  "data": {
    "id": "<product uuid>",
    "status": "draft",
    "imageURL": "https://.../photo.jpg",
    "gpu": {
      "estimated_time": "7 min",
      "estimated_seconds": 420,
      "is_measured": false
    }
  }
}
```

The `gpu` block is a **seed** estimate only. Once generation starts, the
WebSocket carries Tripo's real progress, which supersedes it.

**Then watch:**

```http
GET /products/{id}/status      draft → queue → processing → ready
GET /products/{id}/assets      asset_id 9 = the GLB
```

---

## 10. Setup checklist

- [ ] `TRIPO_API_KEY` in `.env` — the **secret** (`tsk_…`), not the Client ID
- [ ] Balance check returns non-zero
- [ ] Migration applied: `sql/add_generation_task_tracking.sql`
- [ ] `npm install` in `scripts/glb_compress/` — otherwise Draco silently no-ops
      and you store 12.8 MB instead of ~4 MB
- [ ] Test user is on Pro or Enterprise
- [ ] Server restarted after any `.env` change

Related config in `app/core/config.py`: `TRIPO_API_BASE_URL`,
`TRIPO_GEOMETRY_MODEL`, `TRIPO_TEXTURE_MODEL`, `TRIPO_MAX_WAIT_SECONDS`.
