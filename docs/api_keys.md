# API Keys

Long-lived, revocable keys that let a third-party integration (the Shopify app first) call
Rivollo on behalf of a seller, without holding a 60-minute portal JWT it cannot refresh.

- Created and revoked by the seller in the portal (JWT).
- Sent by the integration on every call as `Authorization: Bearer riv_live_…`.
- Accepted **only** by routes that opt in (integration endpoints). A key never works on a
  portal route, and a JWT never works where a key is required.

Code: `app/models/api_key.py`, `app/schemas/api_keys.py`, `app/database/api_key_repo.py`,
`app/services/api_key_service.py`, `app/api/routes/api_keys.py`, `get_api_key_principal` /
`require_api_key_scope` in `app/api/deps.py`. Migration `a5c1e9d4b7f2`.

---

## Key format

```
riv_live_3f9a1c07e4b2d85a6c1f0e9b7a2d4c6e8f0a1b3c5d7e9f02
└ prefix ┘└──────────── 48 hex chars (24 random bytes) ────────────┘
key_prefix (display only) = first 17 chars, e.g. "riv_live_3f9a1c07"
```

- Only the **SHA-256** of the key is stored (`key_hash`, unique index). The raw key is returned
  **once** by `POST /api-keys` and is not stored or logged anywhere.
- A fast hash is correct here, unlike for passwords: the key has 192 random bits, so there is no
  dictionary to slow down, and authentication stays one indexed lookup.

## Scopes

| Scope | Meant for |
|---|---|
| `read` | reading products, status, account |
| `write` | creating or updating data (e.g. product sync) |
| `convert` | anything that spends AI credits (3D generation) |

New keys get all three unless the request narrows them. A key-authenticated route declares
the scope it needs with `require_api_key_scope("…")`; a key without it gets **403**.

---

## Endpoints

All responses use the standard envelope `{"success": true, "data": …}`. Errors are
`{"detail": "…"}` with the HTTP status (FastAPI validation errors are `422 {"detail": [...]}`).
Paths sit at `API_PREFIX` (default empty).

### `POST /api-keys` · JWT · 201

Create a key.

```json
{ "name": "Shopify - my-store", "scopes": ["read", "write", "convert"], "expires_in_days": null }
```

| Field | Rules |
|---|---|
| `name` | required, 1–100 chars, trimmed |
| `scopes` | optional, non-empty subset of `read`, `write`, `convert`; default all |
| `expires_in_days` | optional, 1–365; omit or `null` for a key that never expires |

Unknown fields are rejected (422).

```json
{ "success": true, "data": {
  "id": "7c0e…", "name": "Shopify - my-store",
  "key": "riv_live_3f9a1c07e4b2d85a6c1f0e9b7a2d4c6e8f0a1b3c5d7e9f02",
  "key_prefix": "riv_live_3f9a1c07",
  "scopes": ["read", "write", "convert"], "status": "active",
  "created_at": "2026-09-29T10:00:00Z", "last_used_at": null,
  "expires_at": null, "revoked_at": null
}}
```

`key` appears in this response only. **409** when the user already has
`API_KEY_MAX_ACTIVE_PER_USER` (default 10) active keys.

### `GET /api-keys` · JWT · 200

The caller's keys, newest first, **including revoked and expired ones** (`status` tells
them apart). Same object as above without `key`.

### `GET /api-keys/{key_id}` · JWT · 200

One key. Another user's key, or an unknown id, is **404**. A malformed id is **400**.

### `DELETE /api-keys/{key_id}` · JWT · 200

Revoke. The key stops working on the next request; the row is kept (`status: "revoked"`,
`revoked_at` set). Idempotent: revoking again returns the same object. Another user's key is
**404**.

### `GET /api-keys/current` · **API key** · 200

Who the key belongs to, for an integration's "Connected as …" screen. Authenticated **by the
key**, so it can only describe its caller.

```json
{ "success": true, "data": {
  "api_key": { "id": "7c0e…", "name": "Shopify - my-store", "key_prefix": "riv_live_3f9a1c07",
               "scopes": ["read","write","convert"], "status": "active", "created_at": "…",
               "last_used_at": "…", "expires_at": null, "revoked_at": null },
  "user":    { "id": "0151…", "name": "Uday Satpute", "email": "…", "avatar_url": "…" },
  "credits": { "limit": 1500, "used": 100, "remaining": 1400 }
}}
```

`credits.limit` / `remaining` are `null` for an unlimited plan and `0` when there is no active
licence.

---

## Authentication errors (any key-authenticated route)

| Situation | Status | `detail` |
|---|---|---|
| No `Authorization` header | 401 | `An API key is required.` |
| Malformed, unknown, revoked or expired key | 401 | `Invalid, expired or revoked API key.` (one message for all) |
| Owner no longer exists | 401 | same |
| Owner's account pending deletion | 403 | same text as the portal gives |
| Owner's account deactivated | 403 | same text as the portal gives |
| Key lacks the route's scope | 403 | `This API key does not have the '<scope>' scope.` |
| Too many failed attempts from one IP this minute | 429 | `Retry-After: 60` |

401 responses carry `WWW-Authenticate: Bearer`.

---

## Security rules (enforced in code, covered by tests)

- Raw key: returned once, never stored, never logged, never in the audit log.
- Every management call is scoped to the caller in the SQL `WHERE`; another user's key is 404.
- Revocation is `isactive = false` + `revoked_at`; rows are never deleted by the API.
- Authentication re-reads the owner every time: a deleted, deactivated or purged account's keys
  stop working immediately.
- `last_used_at` is written at most once a minute per key.
- Failed authentications are throttled per IP (`API_KEY_AUTH_FAILURES_PER_IP_PER_MINUTE`,
  default 30). Per-process, so with N replicas the effective budget is about N×; it is
  defence in depth, not the primary control.
- Audit rows: `apikey.created`, `apikey.revoked` in `tbl_activity_logs` (id, name, prefix,
  scopes only).
- `tbl_api_keys.user_id → tbl_users` **ON DELETE CASCADE**: keys die with their owner, and the
  account purge removes them when it deletes the user row. 🔴 `Rivollo.AccountPurge.Job` must
  allow-list this FK before the table reaches production — see
  [account-purge-job-changes.md](account-purge-job-changes.md). Audit columns carry no FK.

---

## Configuration

| Setting | Default |
|---|---|
| `API_KEY_MAX_ACTIVE_PER_USER` | 10 |
| `API_KEY_LAST_USED_RESOLUTION_SECONDS` | 60 |
| `API_KEY_AUTH_FAILURES_PER_IP_PER_MINUTE` | 30 |

## Deployment

Apply migration `a5c1e9d4b7f2` (`alembic upgrade head`), or the equivalent SQL. It only creates
`tbl_api_keys`, its FK and two indexes. Deploy the purge-job change first (above).

---

## Portal UI (Settings → API Keys)

- List: name, `key_prefix`, status, last used, created, **Revoke** button (hidden for revoked).
- **Generate key** modal: name input (+ optional scopes and expiry). On success, show
  `data.key` once with a copy button and the warning *"Copy this key now. It won't be shown
  again."* Never store it in local storage or state beyond the modal.

## Integration usage

```http
GET /api-keys/current
Authorization: Bearer riv_live_3f9a1c07e4b2d85a6c1f0e9b7a2d4c6e8f0a1b3c5d7e9f02
```

Treat **401** as "key invalid or revoked, ask the merchant to reconnect", and **403** on a
scope as "this key was created without that permission".

---

## Differences from the original request, and why

| Requested | Built | Reason |
|---|---|---|
| `POST /auth/apikey/generate`, `GET /auth/apikey/list`, `DELETE /auth/apikey/:id` | `POST /api-keys`, `GET /api-keys`, `DELETE /api-keys/{id}` (+ `GET /api-keys/{id}`) | Resource-oriented REST paths |
| `POST /auth/apikey/validate` (no auth, returns email/credits) | `GET /api-keys/current`, authenticated by the key | An unauthenticated endpoint returning account data for any submitted key is a data leak and needs its own rate limit; the key is instead sent on every request |
| bcrypt hash + prefix scan | SHA-256 + unique index | See "Key format"; the repo already uses SHA-256 for app tokens (`hash_token`) |
| FK `user_id → tbl_users` (no rule given) | FK with `ON DELETE CASCADE` | Keys must go when the account is purged; the purge job allow-lists it ([account-purge-job-changes.md](account-purge-job-changes.md)) |
| Hand-run `CREATE TABLE` | Alembic migration `a5c1e9d4b7f2` | All DDL goes through Alembic |
| `{"success": false, "error": …}` errors | `{"detail": …}` + HTTP status | Repo convention |
| 32-char key, 14-char prefix (inconsistent) | 48 hex chars, 17-char prefix | Matches `randomBytes(24)`; longer prefix tells keys apart |
