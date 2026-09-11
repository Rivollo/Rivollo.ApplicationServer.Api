# Rivollo Application Server API

Backend for the Rivollo seller portal, shopper viewer and mobile app — FastAPI +
SQLAlchemy 2.0 (async, asyncpg) + Alembic + PostgreSQL. Deployed to Azure Container Apps;
assets live in Azure Blob Storage behind a CDN.

- **Python** ≥ 3.11, managed with [`uv`](https://docs.astral.sh/uv/)
- **Node/npm** — only for Draco GLB compression (`scripts/glb_compress`)
- **PostgreSQL** — required; the app refuses to start without `DATABASE_URL`

---

## Quickstart

```bash
# 1. Install dependencies (add the extras you need)
uv sync --extra test --extra dev

# 2. Configure — see "Configuration" below
#    create .env in the repo root

# 3. Apply the schema
uv run alembic upgrade head

# 4. Run
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive API reference. `GET /health` is a
quick liveness check (also `/health/ready`, `/health/live`).

Use `--host 0.0.0.0` only when you deliberately want the API reachable from other devices.

**Draco compression.** `ENABLE_DRACO_COMPRESSION` defaults to `true`, and the compression
service shells out to Node. Either install the toolchain once —
`cd scripts/glb_compress && npm install --omit=dev` — or set
`ENABLE_DRACO_COMPRESSION=false` locally.

**Optional USD/USDZ conversion:** `uv sync --extra conversion` (`usd-core`, `trimesh`).
`app/services/model_converter.py` imports these lazily and degrades without them.

---

## Configuration

Settings are declared in [app/core/config.py](app/core/config.py) and read from the
environment or a `.env` file. [.env.example](.env.example) is the complete list of **names**
(never values), grouped by area and marked by how each is deployed.

Minimum for local development:

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/rivollo
JWT_SECRET=dev-change-me
CDN_BASE_URL=https://cdn.example     # no trailing slash
API_PREFIX=                          # must not end with '/'; default is the bare root
```

Things that bite:

- **`GOOGLE_CLIENT_ID`** defaults to `""`, which **rejects every Google token** rather
  than skipping the check. It must equal the Portal's `NEXT_PUBLIC_GOOGLE_CLIENT_ID`
  exactly — no trimming — or `POST /auth/google` returns 401.
- **`PUBLIC_API_USERNAME` / `PUBLIC_API_PASSWORD`** default to literals committed to this
  repo. Override them anywhere that isn't your machine.
- **An empty value is not the same as unset.** Never blank a variable to "turn it off".
- `postgres://` and `postgresql://` URLs are normalised automatically —
  to `postgresql+asyncpg://` for the app and `postgresql+psycopg://` for Alembic.
- `USE_MANAGED_IDENTITY=true` requires `MANAGED_IDENTITY_CLIENT_ID` (validated at startup).

Never commit a real `.env`, and never log tokens, connection strings, SAS URLs or email
addresses.

---

## Project layout

```
app/
  main.py                       app, middleware, router registration, lifespan tasks
  core/                         config, DB engine/session, security
  api/routes/<domain>.py        thin: parse ids, call service, return api_success(...)
  api/deps.py                   CurrentUser / DB dependencies
  schemas/<domain>.py           Pydantic v2 contracts
  services/<domain>_service.py  business rules; raise HTTPException here
  database/<domain>_repo.py     SQLAlchemy statements only
  models/models.py              ORM tables
  middleware/cdn.py             BlobToCdnMiddleware
migrations/                     Alembic (hand-written revisions)
tests/                          pytest suite
scripts/                        seed_data.py, glb_compress/, maintenance scripts
docs/                           feature and frontend-integration guides
```

New code follows the route → service → repository layering above. `app/repositories/` holds
one legacy module — add repositories to `app/database/`.

---

## API conventions

- **Paths** sit at `settings.API_PREFIX` (default `""`). There is no `/api/v1`; the only
  versioned router is `products v2`, mounted at `/v2`.
- **Success** is always `{"success": true, "data": ...}` via `api_success()`
  ([app/utils/envelopes.py](app/utils/envelopes.py)).
- **Errors:** business errors raise `HTTPException` → `{"detail": ...}`; request validation
  gives FastAPI's `422`; only unhandled 500s use the `api_error` envelope.
- **Field casing:** new schemas are snake_case with no aliases. The camelCase aliases in
  `uploads`, `analytics`, `branding`, `dashboard` and `galleries` are legacy.
- **Auth tiers:**
  - seller/app routes — `Authorization: Bearer <jwt>`
  - public product routes — HTTP Basic (`verify_public_basic_auth`)
  - a small no-auth public tier, which includes the shopper Configurator endpoint
- **Blob → CDN:** `BlobToCdnMiddleware` rewrites Azure Blob URLs in JSON responses to
  `CDN_BASE_URL` (no-op when `AZURE_STORAGE_ACCOUNT` or `CDN_BASE_URL` is unset).
- **WebSocket:** `/ws/products/{product_id}/status` streams product processing status.

### Background work in the process

Started in the `lifespan` handler in [app/main.py](app/main.py):

| Task | Cadence |
|---|---|
| Subscription deactivation (expired periods → canceled, licences revoked) | every 5 min |
| Configurator stale-bake recovery | once at startup, then every `CONFIGURATOR_BAKE_SWEEP_INTERVAL_SECONDS` |
| Managed Identity DB token refresh | when `USE_MANAGED_IDENTITY=true` |
| Product-status broadcaster | always |

Logs go to the console **and** `.server.log` in the working directory.

---

## Database and migrations

Alembic owns the schema. `migrations/env.py` reads `DATABASE_URL` from settings — the
`sqlalchemy.url` in `alembic.ini` is overridden.

```bash
uv run alembic upgrade head        # apply
uv run alembic current             # what the DB is at
uv run alembic history             # the chain
```

**Write revisions by hand.** The database has 43 `created_by`/`updated_by` foreign keys the
ORM deliberately doesn't declare, so autogenerate would emit DDL dropping them all —
`env.py` now blocks foreign keys from autogenerate entirely. The schema has also drifted
from the chain (columns no revision created, modelled tables that don't exist), so treat
any autogenerate output as untrustworthy.

**Don't add hand-run `.sql` scripts.** The files in `sql/` and the repo root predate this
rule and sit **outside** the migration chain — e.g. `sql/create_color_variants.sql`, so a
fresh database built with `alembic upgrade head` has no colour-variant tables.
`docs/configurator/create_configurator_tables.sql` is a generated, read-only reference;
don't run it.

Deploy workflows do **not** run migrations. How each environment gets migrated needs
confirming with whoever owns the database before you ship a revision.

Seed plans and demo users: `uv run python -m scripts.seed_data`.

---

## Testing

```bash
uv run pytest                      # full suite
uv run pytest tests/test_configurator_api.py -q
uv run ruff check app tests
```

- `asyncio_mode = "auto"` — async tests need no decorator.
- **Nothing touches a real database.** `tests/conftest.py` sets a dummy `DATABASE_URL` so
  the app can import; every route test overrides `get_db`.
- Cover ownership: a second user must get `404` on every seller endpoint.

---

## Dependencies

`pyproject.toml` + `uv.lock` are the source of truth; the Docker image installs from
`uv export --frozen`. To add a runtime dependency:

```bash
uv add <package>        # updates pyproject.toml and uv.lock
```

`requirements.txt` is for plain-venv installs only — a package added **only** there will
not ship to dev or prod.

---

## Deployment

| Environment | Trigger | Workflow |
|---|---|---|
| Dev | push to `main` | [.github/workflows/dev-deploy.yml](.github/workflows/dev-deploy.yml) |
| Prod | manual `workflow_dispatch` | [.github/workflows/deploy-prod.yml](.github/workflows/deploy-prod.yml) |

Both build the [Dockerfile](Dockerfile) (Python 3.11-slim, uvicorn on `$PORT`, default
`8080`), push to ACR, and update the Container App. Prod accepts an existing `image_tag`
for rollback.

Environment variables come from GitHub **secrets** (credentials) and **variables**
(everything else) on the `dev` / `prod` environments. A name missing from GitHub is
skipped, never written as empty. `.env.example` records which is which.

---

## Documentation

| Topic | Where |
|---|---|
| Product Configurator — start here | [docs/configurator/architecture.md](docs/configurator/architecture.md), then `decisions.md`, `data-model.md`, `api-spec.md`, `baking.md` |
| Configurator — frontend guide | [docs/configurator/frontend-integration.md](docs/configurator/frontend-integration.md) |
| 3D product creation (frontend) | [docs/create-product-3d-frontend-integration.md](docs/create-product-3d-frontend-integration.md) |
| Multipart 3D generation | [docs/MULTIPART_3D_GENERATION.md](docs/MULTIPART_3D_GENERATION.md), [docs/MOBILE_MULTIPART_API.md](docs/MOBILE_MULTIPART_API.md) |
| Generation time estimates | [docs/estimate-time-feature.md](docs/estimate-time-feature.md), [docs/estimate-time-frontend-integration.md](docs/estimate-time-frontend-integration.md), [docs/websocket-gpu-estimate-frontend-integration.md](docs/websocket-gpu-estimate-frontend-integration.md) |
| OTP login | [docs/otp_login.md](docs/otp_login.md) |
| App-token authentication | [docs/app_token_authentication.md](docs/app_token_authentication.md) |
| Account purge job contract | [ACCOUNT_PURGE_JOB_HANDOFF.md](ACCOUNT_PURGE_JOB_HANDOFF.md) |
| AI credits (frontend) | [FRONTEND_AI_CREDITS_INTEGRATION.md](FRONTEND_AI_CREDITS_INTEGRATION.md) |
| Working rules for AI-assisted changes | [CLAUDE.md](CLAUDE.md) |

`/docs` on a running server is generated from the code and is the live contract.
[openapi.yaml](openapi.yaml) is a checked-in snapshot and may lag behind it.

### Product Configurator, in brief

Sellers split a product's GLB into **Parts** by glTF material index and give each Part
**Options**. Each Option bakes one texture per affected material — never a whole GLB — and
the shopper viewer swaps those textures onto the seller's original model. Code lives only in
`app/api/routes/configurator.py`, `app/schemas/configurator.py`,
`app/services/configurator/` and `app/database/configurator_repo.py`.

Before changing it, read the non-negotiable rules and open questions in
[CLAUDE.md](CLAUDE.md#product-configurator). Two block shipping:

- **Q7** — can the viewer swap textures by material index?
- **Q8** — has `Rivollo.AccountPurge.Job` been updated for the new foreign key to
  `tbl_products`? It must be, before the migration reaches production.
