# Rivollo API — Shopify App Integration Reference

For the team building the Rivollo 3D Shopify app. Everything the app needs from the Rivollo
backend: authentication, the flow in order, and every endpoint with its request, response and
errors. Sections 9, 9b and 10 are for the portal and viewer teams.

**Base URL (dev):** `https://dev-api-f7e16734.rivollo.com` — no `/api/v1` prefix.

---

## 1. Authentication

1. The merchant logs into the Rivollo portal → **Settings → API Keys → Generate key**, and copies
   the key (shown once): `riv_live_3f9a1c07e4b2d85a6c1f0e9b7a2d4c6e8f0a1b3c5d7e9f02`
2. They paste it into the Shopify app's settings. Store it **per shop**, server-side (e.g. in the
   app's Prisma DB), never in the browser and never in env vars shared by every shop.
3. Send it on **every** request:

```http
Authorization: Bearer riv_live_3f9a1c07e4b2d85a6c1f0e9b7a2d4c6e8f0a1b3c5d7e9f02
Content-Type: application/json
```

The key is tied to one Rivollo account: everything the app creates belongs to that account and
spends its AI credits. It does not expire unless the merchant set an expiry, and it stops
working the moment they revoke it.

**Scopes** — a key may be created with fewer than all three:

| Scope | Needed for |
|---|---|
| `read` | connect, connection, models, product list/state |
| `write` | sync, options, accept/discard, unlink, redact |
| `convert` | the two `…/glb` calls (they spend AI credits) |

---

## 2. Response format

Success:

```json
{ "success": true, "data": { … } }
```

Errors — the HTTP status is the signal; read `detail` for the message to show:

```json
{ "detail": "This product already has a 3D model." }
```

Validation errors are `422` with a list:

```json
{ "detail": [ { "loc": ["body", "variants", 0, "price"], "msg": "price must be a decimal string such as \"499.00\"", "type": "value_error" } ] }
```

An unexpected server error is `500 { "success": false, "data": null, "error": { "code": "INTERNAL_SERVER_ERROR", "message": "…" } }`.

### Status codes to handle everywhere

| Status | Meaning | What the app should do |
|---|---|---|
| `401` | No key, or the key is invalid / revoked / expired | Show "Reconnect Rivollo" and ask for a new key |
| `403` | Key lacks the scope, **or** the key is not connected to this shop yet, or the Rivollo account is deactivated / pending deletion | Show `detail`; if it mentions connect, call `/connect` |
| `404` | Not found — or the Shopify integration is switched off on this server | Show `detail` |
| `409` | Not allowed in the current state (e.g. already generating) | Show `detail`; usually refresh state |
| `422` | Request failed validation | Fix the payload (a bug on the app side) |
| `429` | Too many failed authentications from this IP | Wait `Retry-After` seconds |
| `502` | A download or storage step failed | Retry later |

---

## 3. The flow

```
ONCE PER SHOP
  merchant pastes key ─► GET  /api-keys/current                (show "Connected as …", credits)
                     ─► POST /integrations/shopify/connect      { shop_domain }

PER PRODUCT
  1. Sync            POST /integrations/shopify/products/sync   → draft Rivollo product created
                     save data.rivollo_product_id in the rivollo_product_id metafield
  2. Option roles    PUT  /integrations/shopify/products/{id}/options    (only if it has a layout option)
  3. Main 3D model   POST /integrations/shopify/products/{id}/glb        { image_url }
  4. Layout models   POST /integrations/shopify/products/{id}/layouts/{layout_id}/glb   (each other layout)
                     then preview → POST …/generations/{gid}/accept  or  DELETE …/generations/{gid}
  5. Poll            GET  /integrations/shopify/products/{id}    every 5–10 s while anything is generating
  6. When rivollo_product.main_glb_state == "ready":
                     download rivollo_product.glb_url → upload to Shopify media (existing staged-upload code)
                     publish (section 8) → save viewer_url in a metafield

WEBHOOKS
  products/update  → POST   /integrations/shopify/products/sync  (same payload)
  products/delete  → DELETE /integrations/shopify/products/{id}
  app/uninstalled  → POST   /integrations/shopify/uninstall
  shop/redact      → POST   /integrations/shopify/shop/redact
  customers/*      → acknowledge only (Rivollo stores no Shopify customer data)
```

`{id}` in every product path is the **numeric** Shopify product id: strip
`gid://shopify/Product/` (`gid://shopify/Product/10575538127127` → `10575538127127`). A GID does
not work in a URL path. Request bodies accept either form.

---

## 4. Account and connection

### `GET /api-keys/current` — who the key belongs to

Any scope. Use it right after the merchant pastes a key.

```json
{ "success": true, "data": {
  "api_key": { "id": "7c0e…", "name": "Shopify - my-store", "key_prefix": "riv_live_3f9a1c07",
               "scopes": ["read","write","convert"], "status": "active",
               "created_at": "2026-09-29T10:00:00Z", "last_used_at": "2026-09-29T10:05:00Z",
               "expires_at": null, "revoked_at": null },
  "user":    { "id": "0151…", "name": "Uday Satpute", "email": "uday@example.com", "avatar_url": null },
  "credits": { "limit": 1500, "used": 100, "remaining": 1400 }
}}
```

`credits.limit` and `remaining` are `null` on an unlimited plan.

### `POST /integrations/shopify/connect` — bind the key to this shop

Scope `read`. Call once after the key is saved (safe to call again).

```json
{ "shop_domain": "my-store.myshopify.com" }
```

→ `200`

```json
{ "success": true, "data": {
  "id": "c1…", "shop_domain": "my-store.myshopify.com", "api_key_id": "7c0e…",
  "isactive": true, "connected_at": "2026-09-29T10:01:00Z", "disconnected_at": null
}}
```

- Every other Shopify call acts for **this** shop; the shop is never sent again.
- `409` if the shop is connected to a **different** Rivollo account.
- Connecting a new key of the same account to the shop replaces the old binding.
- `422` if `shop_domain` is not `*.myshopify.com`.

### `GET /integrations/shopify/connection`

Scope `read`. The current binding (same object). `403` if not connected.

### `POST /integrations/shopify/uninstall` — `app/uninstalled` webhook

Any scope. No body. Unbinds the shop; the key stays valid; data is kept until `shop/redact`.

```json
{ "success": true, "data": { "disconnected": true } }
```

### `POST /integrations/shopify/shop/redact` — `shop/redact` webhook

Scope `write`. No body. Deletes every Shopify row Rivollo holds for the shop, then unbinds.
Rivollo products created for it are **kept** (they belong to the Rivollo account).

```json
{ "success": true, "data": { "shop_domain": "my-store.myshopify.com", "products_deleted": 12 } }
```

### `GET /integrations/shopify/models` — model picker and credit cost

Scope `read`.

```json
{ "success": true, "data": [
  { "key": "tripo-h3.1", "label": "Tripo H3.1", "description": "…", "credit_cost": 10,
    "is_default": true, "estimated_seconds": 95, "estimated_time": "1 min 35 sec",
    "estimate_is_measured": true, "free_plan_eligible": false }
]}
```

Show `credit_cost` on every "Create 3D" button. Pass `key` as `model` to the `glb` calls (or omit
it for the default). A paid model on a Free plan returns `403`.

---

## 5. Products

### `POST /integrations/shopify/products/sync` — create or update

Scope `write`. **First sync → `201`** and creates a draft Rivollo product (thumbnail = the first
image). **Later syncs → `200`** and update only prices, variants, images and options — never the
product's name/description/thumbnail in Rivollo, which the merchant may have edited there.

```json
{
  "shopify_product_id": "gid://shopify/Product/10575538127127",
  "title": "Sofa with variant",
  "handle": "sofa-with-variant",
  "description": "<p>A modular sofa.</p>",
  "vendor": "Rivollo",
  "product_type": "Furniture",
  "tags": ["sofa", "modular"],
  "status": "ACTIVE",
  "currency": "INR",
  "images": [
    { "id": "gid://shopify/ProductImage/1001", "url": "https://cdn.shopify.com/s/files/1/sofa-main.jpg", "alt_text": "Sofa" }
  ],
  "variants": [
    {
      "shopify_variant_id": "gid://shopify/ProductVariant/44001111",
      "title": "4 Seater-corner / Red",
      "sku": "SOF-C-R",
      "price": "499.00",
      "compare_at_price": "599.00",
      "inventory": 10,
      "available": true,
      "image_urls": ["https://cdn.shopify.com/s/files/1/corner-red.jpg"],
      "options": [
        { "name": "Layout", "value": "4 Seater-corner" },
        { "name": "Color", "value": "Red" }
      ]
    }
  ]
}
```

| Field | Rules |
|---|---|
| `shopify_product_id`, `shopify_variant_id` | GID or number |
| `title`, `handle` | required, ≤ 255 chars |
| `status` | Shopify status, e.g. `ACTIVE`, `DRAFT`, `ARCHIVED` |
| `currency` | **required**, ISO 4217 (`shop.currencyCode`) |
| `images[].url`, `variants[].image_urls[]` | **must be `https://cdn.shopify.com/…`** — anything else is `422` |
| `price`, `compare_at_price` | decimal **strings** (`"499.00"`), not numbers |
| `variants` | 1–100, no repeated ids; send **all** selected variants — ones missing from a later sync are removed |
| `variants[].image_urls` | every media image of the variant (up to 20). A single legacy `image_url` is also accepted |
| `options` | `[]` for a simple product (`Default Title`) |

Unknown fields are ignored.

```json
{ "success": true, "data": {
  "id": "9a…", "shopify_product_id": "10575538127127",
  "rivollo_product_id": "5d…", "rivollo_status": "draft",
  "synced_at": "2026-09-29T10:02:00Z", "variants_synced": 2, "created": true
}}
```

`409` if this Shopify product is linked to another Rivollo account.

### `GET /integrations/shopify/products`

Scope `read`. The shop's synced products, newest first:
`[ { "id", "shopify_product_id", "title", "rivollo_product_id", "synced_at" } ]`.

### `GET /integrations/shopify/products/{id}` — full state (poll this)

Scope `read`.

```json
{ "success": true, "data": {
  "id": "9a…", "shopify_product_id": "10575538127127", "shop_domain": "my-store.myshopify.com",
  "title": "Sofa with variant", "handle": "sofa-with-variant", "currency": "INR",
  "shopify_status": "ACTIVE", "synced_at": "…",
  "images": [ { "id": "…", "url": "https://cdn.shopify.com/…", "alt_text": "Sofa" } ],
  "options": [ { "name": "Layout", "values": ["4 Seater-corner", "4 Seater-lounge"] },
               { "name": "Color",  "values": ["Red", "Blue"] } ],
  "option_roles": { "Layout": "layout", "Color": "info" },
  "variants": [ { "shopify_variant_id": "44001111", "title": "4 Seater-corner / Red", "sku": "SOF-C-R",
                  "price": "499.00", "compare_at_price": "599.00", "inventory": 10, "available": true,
                  "image_urls": ["…"], "options": { "Layout": "4 Seater-corner", "Color": "Red" } } ],
  "rivollo_product": {
    "id": "5d…", "status": "ready",
    "main_glb_state": "ready",
    "glb_url": "https://…/model.glb", "usdz_url": null, "thumbnail_url": "https://…",
    "public_id": "Ab3xY9kLm2Qp", "viewer_url": "https://view.rivollo.com/Ab3xY9kLm2Qp"
  },
  "layouts": [
    { "id": "e4…", "option_value": "4 Seater-lounge", "is_original": true, "position": 1,
      "state": "ready",
      "model": { "id": "original", "name": "Sofa with variant", "glb_url": "https://…/model.glb", "thumbnail_url": null },
      "candidate_image_urls": ["https://cdn.shopify.com/…"], "generations": [] },
    { "id": "f7…", "option_value": "4 Seater-corner", "is_original": false, "position": 0,
      "state": "ready_for_review", "model": null,
      "candidate_image_urls": ["https://cdn.shopify.com/…/corner-red.jpg", "https://cdn.shopify.com/…/sofa-main.jpg"],
      "generations": [ { "id": "g1…", "status": "ready", "candidate_glb_url": "https://…/candidate.glb",
                         "credit_cost": 10, "error": null, "…": "…" } ] }
  ]
}}
```

| Field | Values |
|---|---|
| `rivollo_product` | `null` if the merchant deleted it in the portal — sync again to recreate it |
| `rivollo_product.main_glb_state` | `none` · `generating` · `stalled` (stuck; offer retry) · `ready` |
| `rivollo_product.status` | `draft` · `queue` · `processing` · `ready` · `published` |
| `layouts[].state` | `none` · `generating` · `ready_for_review` (preview `candidate_glb_url`, then accept) · `ready` · `failed` |
| `layouts[].candidate_image_urls` | images to offer for that layout: its variants' images first, then the product gallery |
| `public_id` / `viewer_url` | set once published; `viewer_url` is the full shopper link |

### `PUT /integrations/shopify/products/{id}/options` — which option is the 3D layout

Scope `write`. Needed only when a product's variants differ in **shape** (e.g. "Layout").

```json
{ "roles": { "Layout": "layout", "Color": "info" }, "original_layout_value": "4 Seater-lounge" }
```

- `layout` — each value is a separate 3D model. At most one option.
- `info` — a normal selector (price and cart only). Unlisted options default to `info`.
- `colour` — not available yet (`400`); use `info` for colour options for now.
- `original_layout_value` — required with a `layout` option: that value uses the product's
  **main** 3D model; the others get their own (section 6).

→ `200` with the full state (as `GET`). `400` for an unknown option or a value not in the option.

### `DELETE /integrations/shopify/products/{id}` — `products/delete` webhook

Scope `write`. Removes the link and Shopify data; the Rivollo product is kept.
`{ "success": true, "data": { "unlinked": true } }`.

---

## 6. 3D models

**Images:** `image_url` must be one of the product's **synced** Shopify images (from `images` or
any variant's `image_urls`); anything else is `400`. Rivollo copies the image into its own storage
before generating.

**Credits** are charged when the request is accepted and are **not refunded** if generation fails.

### `POST /integrations/shopify/products/{id}/glb` — the main 3D model

Scope `convert`.

```json
{ "image_url": "https://cdn.shopify.com/s/files/1/sofa-main.jpg", "model": null, "retry": false }
```

→ `202`

```json
{ "success": true, "data": {
  "rivollo_product_id": "5d…", "status": "queue",
  "estimate": { "estimated_time": "1 min 35 sec", "estimated_seconds": 95, "gpu_status": "warm",
                "message": "Generating your 3D model. This usually takes about 1 min 35 sec.",
                "model": "tripo-h3.1", "is_measured": true, "sample_count": 12 }
}}
```

Poll `GET …/products/{id}` until `main_glb_state` is `ready` (then use `glb_url`) or back to
`none` (generation failed; the product returned to draft — offer "Try again").

| `409` detail | Meaning |
|---|---|
| "already being generated" | wait, keep polling |
| "already has a 3D model" | done — regenerating is not supported yet |
| "linked Rivollo product was deleted" | sync again first |

`retry: true` restarts a generation shown as `stalled` (charged again).

### `POST /integrations/shopify/products/{id}/layouts/{layout_id}/glb` — a layout's model

Scope `convert`. For any layout with `is_original: false`. Can run at the same time as the main
model.

```json
{ "image_url": "https://cdn.shopify.com/s/files/1/corner-red.jpg", "model": null, "auto_accept": false }
```

→ `202` with a generation:

```json
{ "success": true, "data": {
  "id": "g1…", "product_id": "5d…", "name": "4 Seater-corner", "source_image_url": "…",
  "model": "tripo-h3.1", "credit_cost": 10, "status": "queued", "error": null,
  "candidate_glb_url": null, "accepted_variant_id": null, "auto_accept": false,
  "client_ref": "shopify-layout:f7…", "started_at": null, "completed_at": null,
  "created_at": "…", "estimate": { … }
}}
```

The generation moves `queued → generating → ready` (or `failed`). When the layout shows
`ready_for_review`, preview `generations[].candidate_glb_url` in a model viewer, then accept or
discard. Several attempts per layout are allowed (each is charged).

`auto_accept: true` skips the preview: the model is added as soon as it is ready. `409` when the
layout already has a model, or for the original layout (use `…/glb` instead).

### `POST /integrations/shopify/products/{id}/generations/{generation_id}/accept`

Scope `write`. No body. `201` the first time, `200` if already accepted:

```json
{ "success": true, "data": {
  "generation": { "id": "g1…", "status": "accepted", "accepted_variant_id": "mv…", "…": "…" },
  "model_variant": { "id": "mv…", "name": "4 Seater-corner", "glb_url": "https://…", "thumbnail_url": "https://…",
                     "order_index": 1, "compression_status": "compressed", "…": "…" }
}}
```

`409` unless the generation is `ready`, or if the layout already has a model.

### `DELETE /integrations/shopify/products/{id}/generations/{generation_id}`

Scope `write`. Discards a candidate (or cancels one still generating). Returns the generation with
`status: "discarded"`. Credits are not refunded.

---

## 7. Uploading the model to Shopify

When `main_glb_state` is `ready`, download `rivollo_product.glb_url` (a public CDN URL) and upload
it through Shopify's staged upload as today. Layout models (`layouts[].model.glb_url`) are
Rivollo-viewer features and do not need to go to Shopify.

---

## 8. Publishing — **Rivollo.Viewer.Api**, not this API

Publishing creates the public viewer link. It lives in `Rivollo.Viewer.Api`:

```http
POST {VIEWER_API}/api/products/{rivollo_product_id}/publish
{ "publish": true }
```

→ `{ "published": true, "publishedAt": "…", "publicId": "Ab3xY9kLm2Qp" }`

Requires `rivollo_product.status == "ready"`. Call it when polling first sees `ready` ("Publish
when ready"). Afterwards `GET …/products/{id}` returns `public_id` and the full `viewer_url`.
Layout models accepted later appear in the viewer automatically; no re-publish needed.

> **Pending in Viewer.Api:** accepting this `riv_live_…` API key on the publish call and
> returning `viewerUrl` directly. Until then, read `viewer_url` from `GET …/products/{id}`.

---

## 9. Portal team — the API Keys screen (portal JWT)

| Method | Path | Body / result |
|---|---|---|
| `POST` | `/api-keys` | `{ "name": "Shopify - my-store", "scopes": ["read","write","convert"], "expires_in_days": null }` → `201`, **`data.key` shown once** |
| `GET` | `/api-keys` | the user's keys, newest first, revoked included (`status`: `active` / `revoked` / `expired`) |
| `GET` | `/api-keys/{id}` | one key |
| `DELETE` | `/api-keys/{id}` | revoke (idempotent); the key stops working immediately |

Key object: `id, name, key_prefix, scopes, status, created_at, last_used_at, expires_at, revoked_at`.
`409` above 10 active keys. UI: list with name, prefix, last used, Revoke; "Generate key" modal
showing the full key once with a copy button and *"Copy this key now. It won't be shown again."*
Full detail: `docs/api_keys.md`.

---

## 9b. Portal team — Shopify products in the product editor (portal JWT)

The same operations as sections 5–6, with the seller's **portal JWT** instead of the API key, and
addressed by the **Rivollo product id** instead of the Shopify id. Same services, so the bodies,
responses, `409`s and image rules are exactly those of sections 5–6.

**Is this a Shopify product?** `source` on the product responses:

| Endpoint | Field |
|---|---|
| `GET /products/{id}` | `data.source` |
| `GET /v2/me/products` (product list, recent products) | `data.items[].source` |

`"shopify"` means linked to a Shopify store the seller still has connected — exactly when
`GET /products/{id}/shopify` answers `200`. `"rivollo"` (or absent/null on other endpoints)
otherwise. After the merchant uninstalls the app it goes back to `"rivollo"`.

| Method | Path | Same as |
|---|---|---|
| `GET` | `/products/{product_id}/shopify` | `GET /integrations/shopify/products/{id}` — full state, poll it (5–10 s) while anything generates |
| `PUT` | `/products/{product_id}/shopify/options` | `PUT …/options` — body `{ "roles": { "Layout": "layout" }, "original_layout_value": "…" }` |
| `POST` | `/products/{product_id}/shopify/glb` | `POST …/glb` — main model, `202` |
| `POST` | `/products/{product_id}/shopify/layouts/{layout_id}/glb` | `POST …/layouts/{layout_id}/glb` — layout candidate, `202` |
| `POST` | `/products/{product_id}/shopify/generations/{generation_id}/accept` | `POST …/generations/{id}/accept` |
| `DELETE` | `/products/{product_id}/shopify/generations/{generation_id}` | `DELETE …/generations/{id}` |

Rules:

- **`404`** when the product is not the caller's, is not linked to Shopify (`"This product is not
  linked to a connected Shopify store."`), or its store's app was uninstalled. Never `403`.
  `400` for a malformed id.
- **No layouts until a layout option is chosen.** Sync never sets option roles, so `layouts` is
  `[]` on a freshly synced product. Let the merchant pick which option changes the shape (e.g.
  "Layout") and which value is the main model, then call `PUT …/options`; the layouts appear in
  the response. Products whose variants do not differ in shape skip this and only need the main
  model.
- **Main model first.** The `original` layout uses the main model: create it with `…/glb`
  (`…/layouts/{original}/glb` is `409`). A second `…/glb` while one is in flight is `409` and is
  not charged, even straight after the first.
- **Layout candidates are charged per request** — several per layout are allowed by design, so
  disable the button while a request is pending.
- **Accept with this route, not `POST /configurator/model-variant-generations/{id}/accept`.**
  Both create the model variant, but only this one refuses a second model for a layout that
  already has one. Discarding through either is equivalent.
- **Model picker and cost:** `GET /ai/3d-models` (portal JWT) — the same list as
  `GET /integrations/shopify/models`.
- **Credits are not refunded** when a generation fails.
- `GET /products/{id}/configurator/model-variants` answers `200` for a product with no GLB yet:
  the list holds the `Original` entry with `glb_url: null` (and no other entries).

---

## 10. Viewer team — the shopper commerce payload (no auth)

```http
GET /public/products/{rivollo_product_id}/shopify
```

`404` unless the product is published and linked to a connected Shopify shop — treat `404` as
"no commerce panel".

```json
{ "success": true, "data": {
  "title": "Sofa with variant", "currency": "INR",
  "product_url": "https://my-store.myshopify.com/products/sofa-with-variant",
  "layout_option": "Layout",
  "options": [ { "name": "Layout", "role": "layout", "values": ["4 Seater-corner", "4 Seater-lounge"] },
               { "name": "Color",  "role": "info",   "values": ["Red", "Blue"] } ],
  "layouts": [ { "value": "4 Seater-lounge", "model": "original" },
               { "value": "4 Seater-corner", "model": "mv…" } ],
  "variants": [ { "id": "44001111", "title": "4 Seater-corner / Red",
                  "options": { "Layout": "4 Seater-corner", "Color": "Red" },
                  "price": "499.00", "compare_at_price": "599.00", "available": true,
                  "image_url": "https://cdn.shopify.com/…",
                  "add_to_cart_url": "https://my-store.myshopify.com/cart/44001111:1" } ]
}}
```

- `layouts[].model` uses the same ids as the configurator payload's models (`"original"` or the
  model-variant id): picking a layout tile selects that value.
- Picking layout + options narrows `variants` to one: show `price` / `compare_at_price` /
  `available`, and link **Add to cart** to `add_to_cart_url`.
- Layouts without an accepted model are omitted; their variants still exist and show the original
  model.
- **Configured products (Capacity × Layout, `configurator/api-spec.md` §9c):** each variant also
  carries `"model": "original" | "<model-variant id>" | null`, resolved through the seller's
  dimension mapping. When the shopper picks options, show the variant's price and switch to its
  `model`. `null` means no model has that combination: keep the current shape, never substitute.
  `model` is `null` on products without a mapping; use `layouts` there.
- Never contains stock counts, SKUs or Shopify ids beyond the variant number.

---

## 11. Server configuration (Rivollo side)

The Shopify routes are on by default; an environment with `ENABLE_SHOPIFY_INTEGRATION=false` answers `404` on all of them.
`VIEWER_BASE_URL` must be set for `viewer_url` to be filled.
