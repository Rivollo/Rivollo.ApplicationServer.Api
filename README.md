## Rivollo Applicationserver API (FastAPI)

FastAPI implementation for the mobile app, aligned with `API docs/openapi.yaml`.

### Quickstart

1. Create and activate a virtual env with `uv`, then install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

2. Create a `.env` file (or export env vars) with at least:

```bash
APP_NAME=Rivollo API
DEBUG=true
# API_PREFIX must not end with '/'. Use empty for root or like '/v1'
API_PREFIX=
# Provide a Postgres URL, e.g. postgresql+asyncpg://user:pass@localhost:5432/rivollo
DATABASE_URL=
JWT_SECRET=dev-change-me
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRES_MINUTES=60
# CDN base used for returned file URLs (no trailing slash)
CDN_BASE_URL=https://cdn.example
# Storage placeholders (implement real Azure wiring later)
AZURE_STORAGE_ACCOUNT=
AZURE_STORAGE_KEY=
AZURE_STORAGE_CONN_STRING=
STORAGE_CONTAINER_UPLOADS=uploads
# Model service endpoint (GPU service); use mock to simulate
MODEL_SERVICE_URL=mock://local
# fal.ai API key for SAM2 image segmentation
FAL_KEY=
```

3. Run the server:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open docs at http://localhost:8000/docs

Use `--host 0.0.0.0` only when you intentionally want the API reachable from other devices on your network.

### Notes
- The app auto-normalizes common Postgres URLs: `postgres://` and `postgresql://` are upgraded to `postgresql+asyncpg://` for the async engine. Alembic migrations normalize to `postgresql+psycopg://`.
- Presigned uploads are stubbed: we return a temporary `uploadUrl` and a final `fileUrl` under `CDN_BASE_URL`. No binary is proxied through the API.
- The 3D processing call is stubbed via `MODEL_SERVICE_URL=mock://...` and updates the job to `ready` with placeholder asset parts.
- Database integration uses SQLAlchemy; provide a real `DATABASE_URL` for Postgres. For production, apply the schema in `API docs/database.txt` or use migrations.
