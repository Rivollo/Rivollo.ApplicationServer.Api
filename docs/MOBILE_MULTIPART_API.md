# Multi-Part 3D Generation — Mobile API Guide

Everything a mobile client needs to reproduce the web portal's **Multi-part**
creation tab: a 3D model whose mesh is split into separately colourable parts,
generated from a single product photo.

For *why* the pipeline is built this way (two chained Tripo tasks, why
`texture: false` on stage 1), see [`MULTIPART_3D_GENERATION.md`](./MULTIPART_3D_GENERATION.md).
This document is the wire contract only.

---

## Contents

1. [The flow at a glance](#1-the-flow-at-a-glance)
2. [Base URLs and conventions](#2-base-urls-and-conventions)
3. [Step 0 — Authenticate](#3-step-0--authenticate)
4. [Step 1 — Upload the product photo](#4-step-1--upload-the-product-photo)
5. [Step 2 — Create the product](#5-step-2--create-the-product)
6. [Step 3 — Track progress over WebSocket](#6-step-3--track-progress-over-websocket)
7. [Step 3b — Polling fallback](#7-step-3b--polling-fallback)
8. [Step 4 — Fetch the finished model](#8-step-4--fetch-the-finished-model)
9. [Status lifecycle](#9-status-lifecycle)
10. [Credits and plan gating](#10-credits-and-plan-gating)
11. [Complete error reference](#11-complete-error-reference)
12. [Building the mobile UI](#12-building-the-mobile-ui)
13. [End-to-end smoke test](#13-end-to-end-smoke-test)

---

## 1. The flow at a glance

```
┌──────────────┐
│  Pick photo  │  JPEG / PNG / WebP  ≤ 10 MB
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│ POST /uploads/content        │  multipart/form-data
│   → { imageURL }             │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ POST /createProductWithParts │  charges 60 AI credits
│   → 201 { id, gpu, ... }     │  returns IMMEDIATELY, status "draft"
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ WS /ws/products/{id}/status                  │
│   status_update  progress   0 →  65  (parts) │  ~4 min
│   status_update  progress  65 → 100 (texture)│  ~3 min
│   done                                       │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ GET /products/{id}/assets    │
│   → mesh[].url  (the GLB)    │
└──────────────────────────────┘
```

**Total wall time ≈ 7 minutes.** Design the screen to be left alone — the user
will background the app. See [§12](#12-building-the-mobile-ui).

---

## 2. Base URLs and conventions

| | Value |
|---|---|
| REST base | `https://<your-api-host>` — same host the web portal uses |
| WebSocket base | Same host, `https://` → `wss://` |
| Auth | `Authorization: Bearer <token>` on **every** REST call |

### Response envelope

Every **2xx** response from these endpoints is wrapped:

```json
{ "success": true, "data": { ... } }
```

Some failures arrive as **2xx with `success: false`** rather than an HTTP error:

```json
{ "success": false, "data": null, "error": { "code": "...", "message": "..." } }
```

FastAPI-level failures (validation, auth, quota) use the plain HTTP error shape:

```json
{ "detail": "..." }
```

> ⚠️ **You must handle all three.** Checking only `response.ok` will silently
> treat a `success: false` body as a successful creation. The web portal's
> [`creationErrors.ts`](../../Rivollo.Web.Portal/lib/product/creationErrors.ts)
> exists precisely because of this; mirror it on mobile.

Note also that `422` returns `detail` as an **array** of objects, not a string:

```json
{ "detail": [ { "loc": ["body","imageURL"], "msg": "...", "type": "..." } ] }
```

---

## 3. Step 0 — Authenticate

```bash
curl -X POST "$API/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "seller@example.com",
    "password": "••••••••",
    "remember_me": true
  }'
```

### `200 OK`

```json
{
  "success": true,
  "data": {
    "user": {
      "id": "3f9a1c22-7e04-4b8d-9a11-2c6d5e8f0a31",
      "email": "seller@example.com",
      "first_name": "Priya",
      "last_name": "Sharma",
      "created_at": "2026-01-14T09:22:10.441Z",
      "updated_at": "2026-07-30T11:02:55.108Z"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

Keep **both**:

- `data.token` → the `Authorization` header
- `data.user.id` → the `userId` body field in [step 2](#5-step-2--create-the-product)

### `401 Unauthorized`

```json
{ "detail": "Invalid email or password" }
```

---

## 4. Step 1 — Upload the product photo

The generation endpoint takes a **publicly reachable URL**, not raw bytes. Upload
first.

```bash
curl -X POST "$API/uploads/content" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/sneaker.jpg"
```

### `200 OK`

```json
{
  "success": true,
  "data": {
    "uploadId": "b81f2c7d9e0a4c15",
    "url": "https://cdn.rivollo.com/u/3f9a1c22/b81f2c7d9e0a4c15.jpg",
    "imageURL": "https://cdn.rivollo.com/u/3f9a1c22/b81f2c7d9e0a4c15.jpg",
    "publicURL": "https://blob.rivollo.com/content/b81f2c7d9e0a4c15.jpg",
    "contentType": "image/jpeg",
    "sizeBytes": 842113,
    "formats": null,
    "blobUrls": null
  }
}
```

Use **`data.imageURL`**, falling back to `data.url` (both carry the same value
for images; the portal reads `imageURL || url`).

### Accepted formats — read this before writing the picker

The server accepts only:

```
.jpg  .jpeg  .png  .webp  .glb  .gltf  .usdz
```

**`.heic` / `.heif` are NOT accepted.** This matters a lot on iOS, where the
camera roll returns HEIC by default. The web portal transcodes to PNG/JPEG in
the browser before uploading
([`image-normalization.ts`](../../Rivollo.Web.Portal/lib/utils/image-normalization.ts)),
and **mobile must do the same** or every iPhone photo will fail with a 400.

- **iOS:** request `UTType.jpeg` from `PHPickerViewController`, or re-encode
  via `UIImage.jpegData(compressionQuality:)`.
- **Android:** already JPEG in practice; re-encode anyway for consistency.

Also enforce the client-side ceiling the portal uses: **10 MB**. Downscale large
photos before upload — it costs nothing in quality at Tripo's input resolution
and saves the user's data.

### Errors

| HTTP | Body | Cause |
|---|---|---|
| `400` | `{"detail":"Filename is required"}` | No filename on the multipart part |
| `400` | `{"detail":"Unsupported file type"}` | Extension outside the list above (**HEIC lands here**) |
| `401` | `{"detail":"Not authenticated"}` | Missing or expired token |
| `500` | `{"detail":"Unable to upload file"}` | Storage backend failure — safe to retry |

---

## 5. Step 2 — Create the product

```bash
curl -X POST "$API/createProductWithParts" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "3f9a1c22-7e04-4b8d-9a11-2c6d5e8f0a31",
    "name": "Trail Runner GTX",
    "imageURL": "https://cdn.rivollo.com/u/3f9a1c22/b81f2c7d9e0a4c15.jpg"
  }'
```

### Request body

| Field | Type | Required | Notes |
|---|---|---|---|
| `userId` | string (UUID) | ✅ | From the login response. Non-UUID → `400` |
| `name` | string | ✅ | 1–200 chars. Must be unique per seller |
| `imageURL` | string (URL) | ✅ | From step 1. Must be publicly reachable — Tripo fetches it server-side |
| `mesh_asset_id` | int | ❌ | Defaults to `9`. **Do not send it.** |

> **There is no `model` field — by design.** Unlike `/createProductFal`, this
> path is not model-selectable. Only Tripo exposes `generate_parts`, and the
> mesh settings are chosen so the colour configurator can read the result.
> Sending `model` is ignored.

### `201 Created`

```json
{
  "success": true,
  "data": {
    "id": "d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3",
    "name": "Trail Runner GTX",
    "accent_color": "#2563EB",
    "tags": [],
    "status": "draft",
    "created_at": "2026-08-07T10:14:22.918Z",
    "updated_at": "2026-08-07T10:14:22.918Z",
    "imageURL": "https://cdn.rivollo.com/u/3f9a1c22/b81f2c7d9e0a4c15.jpg",
    "gpu": {
      "estimated_time": "7 minutes",
      "estimated_seconds": 420,
      "gpu_status": "warm",
      "message": "Generating your 3D model. This usually takes about 7 minutes.",
      "model": "tripo-parts",
      "is_measured": false,
      "sample_count": 0
    }
  }
}
```

**This returns in well under a second. Generation has not started yet.**

Three fields drive the next screen:

- **`data.id`** — the product id. Open the WebSocket with it.
- **`data.gpu.estimated_seconds`** — seed for a countdown, shown until the first
  real `progress` arrives.
- **`data.gpu.is_measured`** — `false` means `estimated_seconds` is the hardcoded
  420s baseline; `true` means it is the median of the last 20 real runs. Consider
  softening the wording ("about 7 minutes") while this is `false`.

`glbURL` is **absent** on creation and appears only once generation succeeds.

### Errors

<details>
<summary><code>400</code> — invalid userId</summary>

```json
{ "detail": "Invalid userId format. Expected UUID string." }
```
</details>

<details>
<summary><code>400</code> — not enough AI credits</summary>

```json
{ "detail": "Not enough AI credits. 60 credits are required to create a multi-part product." }
```

Detect with a lowercase substring match on `"not enough ai credits"` and route
the user to the plans screen. **No credits are consumed.**
</details>

<details>
<summary><code>403</code> — plan gate</summary>

```json
{ "detail": "Creating products requires a Pro or Enterprise plan. Please subscribe to continue." }
```

Free and weekly plans cannot use this endpoint. Gate the tab in the UI too, so
the user does not spend an upload discovering this.
</details>

<details>
<summary><code>503</code> — provider not configured</summary>

```json
{ "detail": "Multi-part 3D generation is not configured on this server (TRIPO_API_KEY is missing)." }
```

Server misconfiguration, not user error. Do **not** offer a retry or an upgrade
prompt — show a generic "temporarily unavailable" and report it.
</details>

<details>
<summary><code>200</code> with <code>success: false</code> — duplicate name</summary>

```json
{
  "success": false,
  "data": null,
  "error": { "code": "DUPLICATE_PRODUCT_NAME", "message": "A product with this name already exists." }
}
```

⚠️ This is **HTTP 200**. Inspect the body, not just the status code.
</details>

<details>
<summary><code>422</code> — validation</summary>

```json
{
  "detail": [
    { "loc": ["body", "imageURL"], "msg": "Input should be a valid URL", "type": "url_parsing" }
  ]
}
```

`detail` is an **array** here. Join the `msg` values for display.
</details>

---

## 6. Step 3 — Track progress over WebSocket

```
wss://<your-api-host>/ws/products/{product_id}/status
```

```bash
# npm i -g wscat
wscat -c "wss://<your-api-host>/ws/products/d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3/status"
```

> **No authentication.** The socket is keyed by product UUID alone — no token,
> no header. Knowing the UUID is enough to watch a product's status. It leaks
> only status and progress, never asset URLs, but be aware of it.

The server pushes JSON; the client sends nothing. Four message types:

### `status_update` — on connect

```json
{
  "type": "status_update",
  "product_id": "d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3",
  "status": "draft",
  "updated_date": "2026-08-07T10:14:22.918Z",
  "source": "initial_query"
}
```

Sent immediately so a client that reconnects mid-run sees the current state
without waiting. `source` is `"initial_query"` here and `"pg_notify"` on live
updates.

### `status_update` — live progress

```json
{
  "type": "status_update",
  "product_id": "d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3",
  "status": "processing",
  "old_status": "queue",
  "updated_date": "2026-08-07T10:18:41.223Z",
  "source": "pg_notify",
  "message": "Generating parts",
  "progress": 42
}
```

**`progress` is a real 0–100 percentage reported by Tripo**, not a timer. It is
unique to this endpoint — the fal-based paths send `estimated_time` instead. The
two stages are mapped onto one continuous scale:

| Range | Stage | `message` |
|---|---|---|
| `0 → 65` | Segmented geometry | `"Generating parts"` |
| `65 → 100` | Texturing | `"Texturing"` |

`progress` is optional — treat its absence as "no change", never as zero.
It is monotonic and clamped server-side to 0–100.

### `keepalive` — every 30s of silence

```json
{
  "type": "keepalive",
  "product_id": "d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3",
  "status": "processing",
  "message": "Still crafting your 3D model — quality takes a little time.",
  "progress": 42
}
```

Sent to stay under the load balancer's ~240s idle timeout. `message` rotates
through friendly copy — safe to display verbatim. Carries the **last known**
`progress`, so a client that connects between updates still gets a number.

### `done` — terminal

```json
{ "type": "done", "status": "ready" }
```

The server closes the socket after this. **Now** call
[`/products/{id}/assets`](#8-step-4--fetch-the-finished-model).

### `error`

```json
{ "type": "error", "message": "Product not found" }
```

---

### Critical: failure does not send `done`

Only `status === "ready"` is terminal. **When generation fails, the product is
reset to `draft` and no `done` message is ever sent** — the socket simply goes
quiet, then keeps emitting keepalives at `status: "draft"` forever.

Your client must detect this itself:

```
if (status_update.status == "draft" AND we have previously seen "processing")
    → generation failed
```

Also apply a hard ceiling — **15 minutes** — after which you show a failure and
stop waiting, regardless of messages. Credits are **not** refunded on failure.

---

## 7. Step 3b — Polling fallback

Mobile sockets die: backgrounding, network handover, doze. Poll as a safety net.

```bash
curl "$API/products/d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3/assets" \
  -H "Authorization: Bearer $TOKEN"
```

Read `data.status`; when it is `"ready"`, read `data.mesh[0].url`.

**Recommended strategy:**

- WebSocket as the primary channel while the app is foregrounded.
- On socket close **before** `done`, reconnect with exponential backoff
  (2s → 4s → 8s, capped at 30s).
- Poll every **15s** whenever the socket is not open.
- On app foreground, poll **once immediately**, then reopen the socket.

Do not poll faster than 10s — it will not make generation finish sooner.

---

## 8. Step 4 — Fetch the finished model

```bash
curl "$API/products/d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3/assets" \
  -H "Authorization: Bearer $TOKEN"
```

### `200 OK`

```json
{
  "success": true,
  "data": {
    "id": "d4e77b91-3a5c-4f28-8b6e-91c0a7f2d5e3",
    "name": "Trail Runner GTX",
    "description": null,
    "price": null,
    "currency_type": null,
    "status": "ready",
    "created_at": "2026-08-07T10:14:22.918Z",
    "updated_at": "2026-08-07T10:21:47.552Z",
    "mesh": [
      { "asset_id": 9,  "url": "https://cdn.rivollo.com/m/d4e77b91/model.glb" },
      { "asset_id": 11, "url": "https://cdn.rivollo.com/m/d4e77b91/model.usdz" }
    ],
    "images": [
      { "asset_id": 1, "url": "https://cdn.rivollo.com/u/.../b81f2c7d9e0a4c15.jpg", "type": "original" }
    ],
    "masks": [],
    "background": null,
    "links": null,
    "hotspots": [],
    "model": { "dimensions": null },
    "public_id": null
  }
}
```

### Asset ids

| `asset_id` | Meaning |
|---|---|
| `1` | Original uploaded photo |
| `2` | Segmentation mask (empty for this path — no mask step) |
| `9` | **GLB — the 3D model** |
| `11` | USDZ (iOS AR Quick Look) |

Select by id, never by array position:

```
glb  = mesh.first { it.asset_id == 9  }?.url
usdz = mesh.first { it.asset_id == 11 }?.url
```

`asset_id 11` is produced by a separate conversion job and **may lag behind the
GLB by a few seconds**. If you need USDZ for AR Quick Look and it is absent,
re-fetch assets once after a short delay rather than treating it as an error.

### What makes this model different

The GLB contains **~19 separate meshes, each with its own material** (measured on
a reference run: 19 meshes / 19 nodes / 19 materials / 12.84 MB). Parts are named
`tripo_part_0` … `tripo_part_18`.

Any standard glTF viewer renders it correctly with no special handling. The
per-part split only matters if you build a colour configurator — then each
material is independently recolourable.

> **Size warning.** These GLBs currently ship **uncompressed at ~12.8 MB** —
> Draco compression is not running on the server (a known pending fix; ~4 MB once
> enabled). On cellular that is a slow download. Show a determinate progress
> indicator, cache aggressively to disk, and consider a Wi-Fi-only prefetch.

### Errors

| HTTP | Body | Cause |
|---|---|---|
| `400` | `{"detail":"Invalid productId format. Expected UUID string."}` | Malformed UUID |
| `401` | `{"detail":"Not authenticated"}` | Missing/expired token |
| `404` | `{"detail":"Product not found."}` | Wrong id, or product deleted |

---

## 9. Status lifecycle

```
draft ──▶ queue ──▶ processing ──▶ ready
  ▲                      │
  └──────────────────────┘
     failure resets to draft (no "done" message)
```

| Status | Meaning | UI |
|---|---|---|
| `draft` | Created, or **generation failed** | Ambiguous — see below |
| `queue` | Accepted, awaiting the provider | Spinner |
| `processing` | Tripo is generating | Progress bar from `progress` |
| `ready` | GLB stored and available | Render the model |
| `published` / `archived` | Post-pipeline user actions | Not seen during creation (the socket discards them) |

> **`draft` is ambiguous.** It means both "just created" and "generation
> failed". Disambiguate by tracking whether you have already observed
> `processing` on this product — that is exactly what the server-side handler
> does, and there is no dedicated `failed` status to key off.

---

## 10. Credits and plan gating

| | |
|---|---|
| Cost | **60 AI credits** per product |
| Charged | On successful `201`, **before** generation runs |
| Refunded on failure | **No** |
| Plan required | `pro` (route also accepts `enterprise`) |

Priced 1:1 with Tripo's own billing — 40 credits for segmented geometry + 20 for
texturing.

### Plan headroom — check this before designing the screen

| Plan | `max_ai_credits_month` | Multi-part products/month |
|---|---|---|
| weekly | 5 | **0** |
| free | 10 | **0** |
| pro | 100 | **1** |

A Pro seller can create **one multi-part product per month**, leaving 40 credits.
Surface the remaining balance prominently *before* the generate button — running
out mid-flow after a 7-minute wait is a bad experience, and there is no refund.

There is also **no `enterprise` row** in `tbl_mstr_plans` (only `free`, `pro`,
`weekly`), so in practice only Pro reaches this endpoint.

Compared with the other creation paths:

| Endpoint | AI credits | Multi-part? |
|---|---|---|
| `/createProduct` (segmented) | 2 | ✗ |
| `/createProductFal` (direct image) | 10–20 | ✗ |
| **`/createProductWithParts`** | **60** | **✓** |
| `/createProductFromGlb` (upload) | 0 | n/a |

---

## 11. Complete error reference

| Step | HTTP | Code / detail | Meaning | Mobile action |
|---|---|---|---|---|
| Login | `401` | `Invalid email or password` | Bad credentials | Inline field error |
| Upload | `400` | `Unsupported file type` | HEIC or other rejected format | **Transcode to JPEG and retry** |
| Upload | `400` | `Filename is required` | Multipart part had no filename | Client bug — always set it |
| Upload | `500` | `Unable to upload file` | Storage failure | Retry with backoff |
| Create | `400` | `Invalid userId format...` | Not a UUID | Client bug |
| Create | `400` | `Not enough AI credits. 60 credits...` | Quota exhausted | Show balance → plans screen |
| Create | `403` | `Creating products requires a Pro...` | Free/weekly plan | Upgrade screen |
| Create | `422` | `detail` is an **array** | Validation | Join `msg` values |
| Create | `503` | `...TRIPO_API_KEY is missing` | Server misconfigured | Generic error; do not retry |
| Create | `200` | `error.code = DUPLICATE_PRODUCT_NAME` | Name taken | Inline error on the name field |
| WS | — | `type: "error"` | Product not found | Fall back to polling |
| WS | — | silence, `status` back to `draft` | **Generation failed** | Failure state — credits not refunded |
| Assets | `404` | `Product not found.` | Deleted or wrong id | Return to list |

**Token expiry** (`401` on any authenticated call) should trigger a refresh and a
single automatic retry before showing the login screen.

---

## 12. Building the mobile UI

### State machine

```
IDLE
  └─ photo picked ──▶ UPLOADING
                        ├─ success ──▶ READY_TO_GENERATE
                        └─ failure ──▶ IDLE (toast)

READY_TO_GENERATE
  └─ name valid && credits ok && tap Generate ──▶ CREATING

CREATING
  ├─ 201 ──▶ GENERATING (open socket)
  └─ error ──▶ READY_TO_GENERATE (or PLAN_GATE / OUT_OF_CREDITS)

GENERATING
  ├─ done                      ──▶ FETCHING_ASSETS ──▶ COMPLETE
  ├─ draft after processing    ──▶ FAILED
  └─ 15 min elapsed            ──▶ FAILED (timeout)
```

Enable Generate only when **all** hold — mirroring the portal so behaviour
matches across clients:

- name trimmed length ≥ 2
- upload finished and `imageURL` is non-null
- remaining credits ≥ 60
- not already generating

### Match the portal's copy

| State | Text |
|---|---|
| No image | `Upload an image to continue` |
| Uploading | `Uploading image…` |
| No name | `Enter a product name to continue` |
| Ready | `Ready to generate` |
| Generating | `Generating parts, then texturing…` |
| Done | `3D model ready` |
| Over limit | `Need 60 AI credits to continue` |

Button label: **`Generate parts`**, with the credit cost (`💎 60`) beside it.

### Seven minutes is a long time

This is the single biggest difference from the other creation paths, and the
main thing to design around.

- **Never block the screen.** Let the user navigate away; keep generation alive.
- **Send a local notification on completion.** They will not sit and watch.
- **Persist `product_id` to disk immediately** on `201`. If the app is killed,
  resume tracking on next launch by polling assets — generation continues
  server-side regardless of the client.
- **Show the real `progress` bar** once it arrives; use `gpu.estimated_seconds`
  only as a placeholder before the first update.
- **Label the stage** from `message` ("Generating parts" → "Texturing"), so the
  bar reaching 65% and slowing down reads as progress rather than a stall.

### Don't repeat these

- Treating HTTP 200 as success without reading `success` → duplicate-name errors
  silently look like successful creations.
- Assuming a `failed` status exists → there isn't one; failure looks like
  `draft`.
- Indexing `mesh[0]` for the GLB → order is not guaranteed; match `asset_id == 9`.
- Uploading HEIC → every iPhone photo 400s.
- Polling every 2s → generation takes 7 minutes; you will do ~200 pointless
  round trips.

---

## 13. End-to-end smoke test

```bash
API="https://<your-api-host>"

# 1. Log in
TOKEN=$(curl -s -X POST "$API/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"seller@example.com","password":"••••••••"}' \
  | jq -r '.data.token')

USER_ID=$(curl -s -X POST "$API/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"seller@example.com","password":"••••••••"}' \
  | jq -r '.data.user.id')

# 2. Upload the photo
IMAGE_URL=$(curl -s -X POST "$API/uploads/content" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./sneaker.jpg" \
  | jq -r '.data.imageURL')

echo "Uploaded: $IMAGE_URL"

# 3. Create the product  (⚠️ spends 60 AI credits)
PRODUCT_ID=$(curl -s -X POST "$API/createProductWithParts" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"userId\":\"$USER_ID\",\"name\":\"Smoke Test $(date +%s)\",\"imageURL\":\"$IMAGE_URL\"}" \
  | jq -r '.data.id')

echo "Product: $PRODUCT_ID"

# 4. Watch progress  (~7 minutes)
wscat -c "${API/https:/wss:}/ws/products/$PRODUCT_ID/status"

# 5. Fetch the GLB
curl -s "$API/products/$PRODUCT_ID/assets" \
  -H "Authorization: Bearer $TOKEN" \
  | jq -r '.data.mesh[] | select(.asset_id == 9) | .url'
```

> Each full run costs **60 of your AI credits and 60 Tripo credits**. On a Pro
> plan that is the entire month's allowance — test against a seeded account or
> raise the limit first.

---

## Appendix — endpoint summary

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/auth/login` | — | Obtain token + userId |
| `POST` | `/uploads/content` | Bearer | Upload photo → `imageURL` |
| `POST` | `/createProductWithParts` | Bearer | Start generation (60 credits) |
| `WS` | `/ws/products/{id}/status` | **none** | Live status + `progress` |
| `GET` | `/products/{id}/assets` | Bearer | Poll status; fetch GLB/USDZ |
