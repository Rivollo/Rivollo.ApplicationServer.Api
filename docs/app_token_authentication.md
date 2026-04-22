# App Token Authentication

## Overview

App Token Authentication is a server-to-server security layer that sits in front of the
`/auth/login` and `/auth/signup` endpoints. Before any user can log in or sign up, the
client application must present a valid app token. This ensures that only registered,
authorized client applications can interact with these endpoints.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Database Table](#database-table)
3. [Token Lifecycle](#token-lifecycle)
4. [API Endpoints](#api-endpoints)
5. [Verification Flow](#verification-flow)
6. [Security Design](#security-design)
7. [Activity Logging](#activity-logging)
8. [Managing Tokens](#managing-tokens)
9. [Error Responses](#error-responses)
10. [Configuration](#configuration)

---

## Architecture

```
Client Application
      │
      │  Step 1 — Get app token (once at startup)
      ▼
POST /auth/apptoken { clientKey }
      │
      │  Step 2 — Use app token on every login/signup
      ▼
POST /auth/login  { Authorization: Bearer <appToken>, email, password }
POST /auth/signup { Authorization: Bearer <appToken>, ...fields }
      │
      ▼
verify_app_token dependency (runs before route logic)
      │
      ├── Layer 1: JWT decode (signature + expiry + type check)
      └── Layer 2: DB lookup (hash match + isactive + expires_at)
```

---

## Database Table

**Table name:** `tbl_app_tokens`

| Column | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `client_key` | VARCHAR UNIQUE | Identifies the client application |
| `token` | TEXT | SHA-256 hash of the issued JWT |
| `expires_at` | TIMESTAMPTZ | When the token expires |
| `isactive` | BOOLEAN | Manual on/off switch (default: true) |
| `created_by` | UUID | Audit — who created (NULL for machine calls) |
| `created_date` | TIMESTAMPTZ | Audit — when first created |
| `updated_by` | UUID | Audit — who last updated |
| `updated_date` | TIMESTAMPTZ | Audit — when last updated |

**Design rule:** Exactly one row per `client_key` at all times.
The token is updated in place via upsert — no duplicate rows, no table bloat.

**SQL to create the table:**
```sql
CREATE TABLE IF NOT EXISTS tbl_app_tokens (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    client_key    VARCHAR     NOT NULL,
    token         TEXT        NOT NULL,
    expires_at    TIMESTAMPTZ NOT NULL,
    isactive      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_by    UUID,
    created_date  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by    UUID,
    updated_date  TIMESTAMPTZ,
    CONSTRAINT uq_app_tokens_client_key UNIQUE (client_key)
);
```

---

## Token Lifecycle

```
Client calls /auth/apptoken
         │
         ▼
First call for this client_key?
         │
    ┌────┴────┐
   YES        NO (row exists)
    │          │
    │     Token still valid?
    │          │
    │     ┌────┴────┐
    │    YES        NO
    │     │    (expired or isactive=false)
    │  Return        │
    │  existing   Generate new JWT
    │  token      Hash it
    │             Upsert into DB
    │             (updates same row)
    │                  │
    └──────────────────┘
                  │
             Return JWT to client
             (valid for 24 hours)
```

**Key points:**
- The raw JWT is **returned to the client** but never stored in the DB
- Only the **SHA-256 hash** is stored — a compromised DB cannot yield working tokens
- The same row is always updated — no accumulation of old entries

---

## API Endpoints

### POST /auth/apptoken

Issues a JWT for a registered client application.

**Request:**
```json
{
  "clientKey": "your-secret-client-key"
}
```

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "client_key": "your-secret-client-key",
    "expires_in_minutes": 1440
  }
}
```

**Response (401) — invalid client key:**
```json
{
  "detail": "Invalid client key"
}
```

---

### POST /auth/login _(requires app token)_

**Request:**
```
Headers:
  Authorization: Bearer <app_token>

Body:
{
  "email": "user@example.com",
  "password": "password123"
}
```

---

### POST /auth/signup _(requires app token)_

**Request:**
```
Headers:
  Authorization: Bearer <app_token>

Body:
{
  "email": "user@example.com",
  "password": "password123",
  "name": "John Doe",
  "signup_token": "<from verify-signup-otp>"
}
```

---

## Verification Flow

When `/auth/login` or `/auth/signup` is called, the `verify_app_token` dependency
runs automatically before the route logic:

```
Request arrives
      │
      ▼
Read Authorization: Bearer <token> from header
      │
      ▼
LAYER 1 — JWT Decode
  decode_access_token(token)
      │
      ├── Signature invalid?     → 401
      ├── Token expired?         → 401
      └── type != "app_token"?   → 401
            (prevents user JWTs being used as app tokens)
      │
      ▼
LAYER 2 — Database Lookup
  hash_token(token) → SHA-256 digest
  SELECT * FROM tbl_app_tokens
  WHERE token     = <hash>
    AND isactive  = true
    AND expires_at > NOW()
      │
      ├── No row found?   → 401
      └── Row found?      → proceed to login/signup ✅
```

---

## Security Design

| Threat | Defence |
|---|---|
| Forged/tampered token | JWT signature verification (HS256 + secret key) |
| Expired token | `exp` claim in JWT + `expires_at` in DB |
| User JWT used as app token | `type == "app_token"` payload check |
| Token stolen from database | SHA-256 hash stored — irreversible |
| Unauthorized client | `clientKey` validated against `APP_CLIENT_KEYS` env var |
| Disabled client | `isactive = false` check in DB |
| Token never issued by server | DB lookup must find matching hash |

**Why hash the token?**
The JWT is a signed string that any holder can use. If the database were
compromised and raw tokens were stored, an attacker could immediately use them.
By storing only the SHA-256 hash, the raw JWT is never recoverable from the DB.
On each request, the incoming token is hashed and compared — same security,
zero exposure.

---

## Activity Logging

All token events are recorded in `tbl_activity_logs` with IP address and User-Agent.

| Event | `action` value | `metadata` |
|---|---|---|
| Token issued | `apptoken.issued` | `{ client_key }` |
| Bad/expired JWT or wrong type | `apptoken.validation_failed` | `{ reason: "invalid_jwt" }` |
| Token not in DB or inactive | `apptoken.validation_failed` | `{ reason: "token_not_found_or_inactive", client_key }` |

**Query to see all token activity:**
```sql
SELECT action, activity_metadata, ip, user_agent, occurred_at
FROM tbl_activity_logs
WHERE action LIKE 'apptoken.%'
ORDER BY occurred_at DESC;
```

---

## Managing Tokens

### Disable a client (immediate effect)
```sql
UPDATE tbl_app_tokens
SET    isactive     = false,
       updated_date = NOW()
WHERE  client_key   = 'your-client-key';
```

After this, any login/signup call using that client's token returns `401` immediately.
The client can recover by calling `/auth/apptoken` again — a new token will be issued
and the row updated with `isactive = true`.

### Re-enable a client
```sql
UPDATE tbl_app_tokens
SET    isactive     = true,
       updated_date = NOW()
WHERE  client_key   = 'your-client-key';
```

### View all active tokens
```sql
SELECT client_key, expires_at, isactive, created_date, updated_date
FROM   tbl_app_tokens
ORDER  BY created_date DESC;
```

### Force token refresh for a client
Set `isactive = false`. The next call to `/auth/apptoken` will issue a fresh token.

---

## Error Responses

| Scenario | HTTP Status | Detail |
|---|---|---|
| Missing `Authorization` header | 403 | Handled by FastAPI automatically |
| Invalid JWT signature | 401 | `"Invalid or missing app token"` |
| JWT expired | 401 | `"Invalid or missing app token"` |
| Token type is not `app_token` | 401 | `"Invalid or missing app token"` |
| Token not found in DB | 401 | `"Invalid or missing app token"` |
| Token disabled (`isactive=false`) | 401 | `"Invalid or missing app token"` |
| Invalid `clientKey` on issuance | 401 | `"Invalid client key"` |

All validation failures return the same generic message to avoid leaking
information about why verification failed.

---

## Configuration

| Setting | Env Variable | Default | Description |
|---|---|---|---|
| Allowed client keys | `APP_CLIENT_KEYS` | `""` | Comma-separated list of valid client keys |
| Token expiry | `APP_TOKEN_EXPIRES_MINUTES` | `1440` | Token lifetime in minutes (24 hours) |
| JWT secret | `JWT_SECRET` | `dev-change-me` | Secret used to sign tokens — **must be changed in production** |
| JWT algorithm | `JWT_ALGORITHM` | `HS256` | Signing algorithm |

**Example `.env`:**
```
APP_CLIENT_KEYS=mobile-app,web-app,admin-panel
APP_TOKEN_EXPIRES_MINUTES=1440
JWT_SECRET=your-strong-random-secret-here
```

---

## Files Changed

| File | Change |
|---|---|
| `app/models/models.py` | Added `AppToken` model |
| `app/core/security.py` | Added `hash_token()` helper |
| `app/services/auth_service.py` | Added `generate_app_token()` and `validate_app_token()` |
| `app/services/activity_service.py` | No changes — used as-is |
| `app/api/deps.py` | Added `verify_app_token` dependency and `AppTokenVerified` alias |
| `app/api/routes/auth.py` | Updated `app_token` route; added `AppTokenVerified` to `login` and `signup` |
| `sql/add_app_tokens_table.sql` | SQL script to create the table manually |
