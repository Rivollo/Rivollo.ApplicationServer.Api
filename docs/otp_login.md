# Email OTP Login

## Overview

A fourth authentication entry point: an **existing** user proves control of
their mailbox with a 6-digit code and receives the same JWT that
`POST /auth/login` returns.

This flow is additive and isolated. Signup, the signup OTP flow, password
login, Google login and password reset are unchanged and remain fully
functional — the only change to an existing code file is two lines in
`app/main.py` registering the router.

**OTP login never creates a user.** Signup remains the only way an account
comes into existence.

```
Signup OTP  -> creates a new user
Login OTP   -> authenticates an existing user
```

---

## Table of Contents

1. [Endpoints](#endpoints)
2. [Resend policy](#resend-policy)
3. [The 30-minute lockout](#the-30-minute-lockout)
4. [Verification attempts](#verification-attempts)
5. [Database table](#database-table)
6. [Security controls](#security-controls)
7. [Configuration](#configuration)
8. [Error responses](#error-responses)
9. [Operations](#operations)

---

## Endpoints

Both sit behind the same `AppTokenVerified` gate as every other `/auth/*`
route: the client application must present a valid app token from
`POST /auth/apptoken`.

The paths are nested because **`/auth/verify-otp` is already taken** by
password-reset verification. `/auth/account/restore` is the existing
precedent for nesting under `/auth`.

### `POST /auth/otp/request`

Sends a sign-in code. **This is also the resend endpoint** — calling it again
either reports the remaining cooldown or issues the next code. There is no
separate resend endpoint.

```json
{ "email": "user@example.com" }
```

```json
{
  "success": true,
  "data": {
    "message": "If an account exists for this email, we've sent a sign-in code.",
    "expires_in_minutes": 5,
    "resends_remaining": 2,
    "resend_available_in_seconds": 60
  }
}
```

The response is **identical** whether or not the address belongs to an
account, whether the account is active, deactivated or pending deletion. Use
`resends_remaining` and `resend_available_in_seconds` to drive the resend
button and its countdown.

### `POST /auth/otp/verify`

```json
{ "email": "user@example.com", "otp": "123456", "remember_me": false }
```

Returns exactly the `AuthResponse` body that `POST /auth/login` returns —
same keys, same token semantics — so a client that handles password login
handles this with no new response handling.

---

## Resend policy

**Three sends per challenge series: one initial, plus two resends.**

| Event | `resend_count` | `resends_remaining` | Effect |
|---|---|---|---|
| Initial request | 0 | 2 | OTP #1 sent |
| Resend #1 | 1 | 1 | OTP #1 invalidated, OTP #2 sent |
| Resend #2 | 2 | 0 | OTP #2 invalidated, OTP #3 sent |
| Next request | — | — | **Locked for 30 minutes** |

Rules:

- **Only an actual send counts.** A request refused by the 60-second cooldown
  or by the lock does not consume a resend — a double-clicking client must not
  spend the user's budget.
- **Each send invalidates the previous code** by overwriting `otp_hash`. At
  most one code per address is ever valid.
- **Each send resets `verification_attempts` to zero.** A new code gets a
  fresh attempt budget.
- **The series resets** on a successful login, when a lock expires, or when
  the last send is older than the 30-minute series window. Any of these starts
  a fresh budget.

---

## The 30-minute lockout

The lockout is **two nullable columns on one row of `tbl_login_otps`** and
nothing else:

| Column | Meaning |
|---|---|
| `locked_until` | NULL = not locked. A future timestamp = both endpoints answer 429. |
| `lock_reason` | `resend_limit` today. |

Expiry is passive: nothing clears the column. The next request simply sees it
is in the past and starts a fresh series.

**What the lockout does NOT do.** It never writes to `tbl_users`, so it does
not set `is_active = false`, does not set `deleted_at`, and does not disable
the account. A locked user can still sign in with **Google** and with their
**password**, and any session token already issued keeps working. Neither
`/auth/login` nor `/auth/google` reads this table.

Repeated requests during a lock do **not** extend it — otherwise it could be
made effectively permanent.

> **Known trade-off.** Because the lock is keyed by email and anyone can
> request a code for any address, a third party can lock a known address out
> of *OTP login* for 30 minutes. This is inherent to any email-keyed send
> budget. It is acceptable while password and Google login remain available —
> an OTP lock is not an account lock — and **must be re-evaluated before
> password login is removed.**

---

## Verification attempts

`resend_count` and `verification_attempts` are two separate controls and are
never read as one another.

| | `resend_count` | `verification_attempts` |
|---|---|---|
| Counts | codes issued in the series | wrong codes against the **current** code |
| Scope | the whole series | one code |
| Maximum | 2 resends (3 sends) | 5 |
| Reset by | a new series | **every send**, and a new series |
| On limit | **30-minute lockout** | **code invalidated, no lockout** |

At 5 failed attempts the current code is invalidated (`otp_hash` set to NULL,
`invalidated_reason = 'verification_attempts'`) but the series survives, so
the user may resend if budget remains. The lockout arrives only through the
resend limit.

The effective ceiling is therefore **3 codes x 5 attempts = 15 guesses per
address per 30 minutes**, against a space of 10<sup>6</sup>.

Attempt exhaustion deliberately does not lock: locking at five guesses would
make an OTP denial-of-service cheaper than it already is, for no security
gain, and would punish ordinary mistyping.

---

## Database table

**`tbl_login_otps`** — created by migration `b8e2f4a10c73`, mirrored in
`sql/add_login_otps_table.sql` for environments where Alembic is not run.

**One row per `(email, purpose)`.** A row is a challenge *series*, not a
single code: the resend budget spans up to three codes and the lockout
outlives all of them. The row is maintained with
`pg_insert(...).on_conflict_do_update(...)` against
`uq_login_otps_email_purpose`, the same shape `tbl_app_tokens` uses for
`client_key`.

| Column | Type | Purpose |
|---|---|---|
| `id` | UUID | Primary key |
| `email` | CITEXT | Challenge identity — the address, not the account |
| `purpose` | VARCHAR(32) | `login` |
| `otp_hash` | TEXT NULL | Peppered SHA-256 of the current code. NULL = no live code |
| `expires_at` | TIMESTAMPTZ | Expiry of the current code |
| `resend_count` / `max_resends` | SMALLINT | Series-scoped send budget |
| `verification_attempts` / `max_verification_attempts` | SMALLINT | Code-scoped guess budget |
| `last_sent_at` | TIMESTAMPTZ | Drives the cooldown and the series window |
| `locked_until` / `lock_reason` | TIMESTAMPTZ / VARCHAR(32) | The entire lockout state |
| `consumed_at` | TIMESTAMPTZ | Single-use marker and atomic guard |
| `invalidated_reason` | VARCHAR(32) | `superseded` / `verification_attempts` / `expired` |
| `request_ip` | TEXT | Abuse forensics |
| `created_at` / `updated_at` | TIMESTAMPTZ | |

### No foreign key to `tbl_users` — deliberate

1. The account purge job runs a schema contract check whose **assertion 9
   aborts the entire run** if an unexpected new FK references `tbl_users`.
2. Rows are written for **any** valid address, whether or not an account
   exists — which is what keeps the lockout response free of an
   account-enumeration oracle. A FK could not express that.

**The purge job needs an email-keyed `DELETE FROM tbl_login_otps WHERE
email = :email`**, alongside the one it already has for `tbl_signup_otps`.

There is also **no unique index on `otp_hash`**. `tbl_password_resets.token`
carries exactly that over a 6-digit space, which makes two concurrent resets
collide into a 500. Constrain the subject, never the secret.

---

## Security controls

| Control | Implementation |
|---|---|
| Generation | `secrets.randbelow(1_000_000)`, zero-padded — a CSPRNG, not `random.randint` |
| Storage | Peppered SHA-256 via `hash_token`. Plaintext never reaches the database |
| Expiration | 5 minutes |
| Single use | `UPDATE ... WHERE id = :id AND consumed_at IS NULL` with a rowcount check |
| Previous-code invalidation | Structural — a resend overwrites `otp_hash` |
| Max attempts | 5 per code, committed before the error is returned |
| Resend cooldown | 60 seconds, derived from `last_sent_at` in the database |
| Concurrent verification | `SELECT ... FOR UPDATE` plus the consume guard |
| Concurrent requests | `UNIQUE (email, purpose)` with `on_conflict_do_update` |
| Enumeration | Identical status, body and latency; email queued via `BackgroundTasks` |
| Logging | The code, its hash, the pepper and the email never appear in any log line |

### Why SHA-256 and not Argon2

Argon2 would add 50–100 ms of CPU to **every** attempt, including
attacker-driven ones — a self-inflicted denial-of-service lever on a public
endpoint. But bare SHA-256 over a 6-digit space is reversible in milliseconds
from a leaked row. **The pepper is what does the work:** a server-side secret
held in configuration and never in the database, so a database-only compromise
yields nothing to compute against.

---

## Configuration

All settings live in `app/core/login_otp_config.py` as an independent
`LoginOtpSettings` class — deliberately *not* on `app.core.config.Settings`,
so that shared file stays out of this feature's change set.

| Setting | Default | Purpose |
|---|---|---|
| `LOGIN_OTP_ENABLED` | `false` | Kill switch. **Off by default** |
| `LOGIN_OTP_EXPIRES_MINUTES` | 5 | Code lifetime |
| `LOGIN_OTP_MAX_RESENDS` | 2 | Resends per series (3 total sends) |
| `LOGIN_OTP_RESEND_COOLDOWN_SECONDS` | 60 | Minimum gap between sends |
| `LOGIN_OTP_MAX_VERIFICATION_ATTEMPTS` | 5 | Wrong codes per code |
| `LOGIN_OTP_LOCKOUT_MINUTES` | 30 | Lock duration |
| `LOGIN_OTP_SERIES_WINDOW_MINUTES` | 30 | How long a series stays current |
| `LOGIN_OTP_REQUESTS_PER_IP_PER_MINUTE` | 5 | Per-IP request budget |
| `LOGIN_OTP_VERIFY_PER_IP_PER_MINUTE` | 10 | Per-IP verify budget |
| `LOGIN_OTP_PEPPER` | *(empty)* | **Secret.** Required when enabled |

The application **refuses to start** if `LOGIN_OTP_ENABLED` is true and
`LOGIN_OTP_PEPPER` is empty. Generate one with:

```
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

> **Rotating the pepper** invalidates every code in flight. The blast radius
> is up to five minutes of unverifiable codes; users simply request again.
> Treat it as a secret alongside `JWT_SECRET`.

---

## Error responses

### `POST /auth/otp/request`

| Status | Condition |
|---|---|
| 200 | Code sent, cooldown active, unregistered address, deactivated account — **all identical** |
| 422 | Malformed email |
| 429 | Address locked, or per-IP budget exceeded (`Retry-After` set) |
| 401 | Missing or invalid app token |
| 503 | `LOGIN_OTP_ENABLED` is false |

### `POST /auth/otp/verify`

| Status | Condition |
|---|---|
| 200 | Success — standard `AuthResponse` |
| **400** | **All of:** no challenge, wrong code, expired, attempts exhausted, superseded by a resend, already consumed, user gone. **One identical message** — distinct messages would reveal whether a live challenge exists |
| 403 | Account deactivated (`ACCOUNT_DEACTIVATED_DETAIL`) |
| 429 | Address locked, or per-IP verify budget exceeded |
| 503 | `LOGIN_OTP_ENABLED` is false |

An account **pending deletion** receives no code (the user lookup filters
`deleted_at IS NULL`) and cannot sign in via OTP. Password login, Google
login and `POST /auth/account/restore` remain the routes back in.

---

## Operations

### Deployment order

1. Apply the DDL **before** deploying the code — the table is inert without
   the code; the code errors without the table.
2. Deploy with `LOGIN_OTP_ENABLED=false`, then enable.

### Monitoring

From `tbl_activity_logs`:

- `auth.otp.requested` vs `user.login.otp` — the completion ratio, and the
  practical email-delivery health signal, since the send is backgrounded and
  a failure never reaches the HTTP response.
- Lockout volume via `lock_reason` on `tbl_login_otps`.

### Rollback

| Level | Action | Time |
|---|---|---|
| 1 | `LOGIN_OTP_ENABLED=false` + restart | Minutes, no deploy |
| 2 | Comment out the `include_router` line in `app/main.py` | One deploy |
| 3 | Revert the branch | One deploy |

**Leave the table.** No other flow reads it, and dropping it is not
reversible. Tokens already issued via OTP keep working after any rollback —
nobody is logged out.

### Cleanup

```
python -m scripts.cleanup_login_otps --dry-run
python -m scripts.cleanup_login_otps --days 7
```

Housekeeping only: the unique constraint already bounds the table to one row
per address that has ever attempted OTP login. Rows whose lock is still in
force are never deleted.
