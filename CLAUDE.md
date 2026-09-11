# CLAUDE.md

Operating rules for Claude Code sessions in this repository.

---

## Repository

Rivollo Application Server API — FastAPI + SQLAlchemy 2.0 (async, asyncpg) + Alembic +
PostgreSQL, deployed to Azure Container Apps, storing assets in Azure Blob behind a CDN.

**Layering** (follow it; the newest domains already do):

```
app/api/routes/<domain>.py   thin: parse ids, call service, return api_success(...)
app/schemas/<domain>.py      Pydantic v2 contracts
app/services/<domain>_service.py   business rules; raise HTTPException here
app/database/<domain>_repo.py      SQLAlchemy statements only
app/models/models.py               ORM tables
```

- Repositories live in `app/database/*_repo.py`. (`app/repositories/` holds one legacy module —
  do not add to it.)
- Responses use `api_success(data)` from `app/utils/envelopes.py`; routes declare
  `response_model=dict`.
- Business errors raise `HTTPException`. Only unhandled 500s use `api_error`. This is
  inconsistent but repo-wide — match neighbours, don't invent a third convention.
- Schemas use **snake_case with no aliases**. The camelCase-alias style in `uploads.py`,
  `analytics.py`, `branding.py`, `dashboard.py`, `galleries.py` is legacy; the four newest
  schema modules contain zero `alias=`.
- Auth: `CurrentUser` / `DB` from `app/api/deps.py`. Public tier uses
  `verify_public_basic_auth`.
- Tests: pytest with `asyncio_mode = "auto"` (`pyproject.toml`). Override `get_db`; nothing
  connects to a real database.

---

## Product Configurator

A new domain, currently **specified but not implemented**. If you are asked to work on it:

### Read first, in order

1. `docs/configurator/architecture.md` — boundary, reuse map, seller/shopper split
2. `docs/configurator/decisions.md` — ADRs and the six open questions
3. `docs/configurator/data-model.md` — tables, constraints, GLB versioning
4. `docs/configurator/api-spec.md` — endpoints, auth, schemas, errors
5. `docs/configurator/baking.md` — preview vs bake, recolouring spec, status lifecycle

### Non-negotiable rules

**Domain boundary.** Configurator code lives in `app/api/routes/configurator*.py`,
`app/schemas/configurator.py`, `app/services/configurator/`,
`app/database/configurator_repo.py`. Never add Configurator logic to
`app/api/routes/products.py`. There is **no `/api/v1` convention in this repo** — paths sit at
`settings.API_PREFIX` (default `""`) under a `/configurator` segment:
`/products/{id}/configurator/parts`, `/configurator/parts/{id}`,
`/public/products/{id}/configurator`. That namespace is what keeps them clear of the
colour-variant routes, which own `/products/{id}/materials` and
`/products/{id}/color-variants` — never modify those. See ADR-001.

**One original GLB.** The seller's uploaded GLB (`asset_id == 9`) is canonical and read-only.
Never rewrite its geometry. Configurator data attaches by glTF **material index**. ADR-002.

**Texture baking, never per-option GLBs.** A Part Option bakes one texture per affected
material index. Do **not** generate a complete GLB per colour/material combination — that is
what the existing colour-variant feature does and what this design replaces. ADR-003.

**Parts are seller-owned and persisted.** `group_id` from `glb_recolor._compute_groups` is a
recomputed, unstable heuristic — surface it as `similarity_group_hint`, never store it as a
part identity. ADR-004.

**Exactly three tables.** `tbl_product_parts`, `tbl_part_options`,
`tbl_part_option_textures`. No `tbl_product_part_materials`, no separate texture-option table,
no `option_type` / `source_type` column.

**Material-index uniqueness is service-enforced, not DB-enforced.** `material_indices` stays
JSONB. No trigger, no GIN index, no JSONB CHECK constraints. `PartService` takes a
`SELECT ... FOR UPDATE` on the product row — in the same query as the ownership check — then
loads siblings, rejects any intersection, and writes. ADR-012.

**Uploaded textures are `recipe.method = "image"`.** Not an option type, not a new table, not
a new endpoint. `image_url` must be validated server-side as belonging to the caller's own
`users/{user_id}/uploads/…` namespace (SSRF + cross-tenant hotlink vector), and the file is
**copied** into the Configurator namespace before it is recorded or purged. `swatch_hex` is
required for image options. Sellers upload via the existing `POST /uploads/content`. ADR-013.

**The default option lives on the option.** `is_default BOOLEAN` + partial unique index
`ux_part_options_one_default ON (part_id) WHERE is_default` — mirroring
`ux_color_variants_one_default`. There is **no `default_option_id` column**; the API exposes
it as a computed field. `set_as_default` is **not** accepted on create; a part's default comes
from the first bake that reaches `completed` (never displacing an existing default), or from
an explicit `PATCH`. `set_as_default: false` is ignored. ADR-011.

**`auto` ambiguity is an error, not a guess.** When a part's materials suggest different
methods there is no specified tie-break, so the write is rejected with `400` naming the
candidates. Do not add first-wins or majority-wins behaviour.

**🔴 Configurator tables add a new FK to `tbl_products`.** `ACCOUNT_PURGE_JOB_HANDOFF.md` §25
assertion 9 aborts every production purge run on an unrecognised FK to `tbl_users` or
`tbl_products`. Therefore: declare **no** `created_by`/`updated_by` FK to `tbl_users` (plain
UUIDs, as `AuditMixin` already does — see `app/models/login_otp.py:19-33` for precedent);
**keep** the `product_id` CASCADE FK; and get `Rivollo.AccountPurge.Job` updated before
deploying the migration. Do not "solve" this by dropping the product FK. ADR-010.

**Preview ≠ bake.** Preview is client-side and requires no backend call per interaction. Bake
is asynchronous and persists CDN textures. Both implement one recipe specification; a
divergence is a bug. `auto` is resolved server-side at save time and never persisted. ADR-005.

**Reuse, don't duplicate.** Auth, DB session, `storage_service`, `POST /uploads/content`,
`BlobToCdnMiddleware`, `model_cache`, and the colour maths in `app/services/color/` all
already exist. The recolour algorithm has exactly one home — extract it, never copy it.

**Ownership on every endpoint.** Resolve products through
`created_by == current_user.id AND deleted_at IS NULL`; return **404** (not 403) on failure.
The neighbouring hotspot and colour-variant services skip this check — do not copy them.
ADR-008.

**Alembic owns all DDL.** Write migrations **by hand** (`env.py` blocks FK autogenerate and
the schema has drifted from the chain). Never add a hand-run `.sql` script — that is how the
colour-variant tables ended up outside the migration chain. ADR-009.

**Bake durability from day one.** `bake_started_at` + a startup sweep + a periodic sweep.
Without them a recycled replica leaves rows stuck in `baking` forever. Keep execution behind
`bake_runner.enqueue()` so it can move to a worker later without touching the API. ADR-007.

### Testing requirements

Do not merge Configurator code without:

- **Unit** — part validation, material membership and overlap rules, option/recipe validation,
  bake state transitions, GLB version validation, recolour maths against golden fixtures.
- **Service** — part and option CRUD, bake lifecycle, authorization rules.
- **Repository** — relationships, cascades, uniqueness, foreign keys.
- **API** — request validation, response contracts, auth, **ownership (a second user gets
  404 everywhere)**, the public shopper payload, error shapes.
- **Baking** — each recolour method, failure handling, idempotency at the same `recipe_hash`.

The existing colour-variant feature has zero tests. Do not repeat that.

### Security requirements

Enforce in services, never in routes, never only in the frontend: authenticated user →
product ownership → part belongs to product → option belongs to part → material index valid
for that product's GLB. The shopper payload uses a **separate response schema**, not the
seller schema with fields omitted, and never exposes `recipe`, `bake_*`, `glb_version`,
`blob_url`, or audit columns.

### Decisions requiring review — do not resolve these silently

| # | Question | Blocks |
|---|---|---|
| Q1 | Is CDN CORS configured? Nothing in this repo configures it. | client-side preview |
| Q2 | Should Configurator errors use the `api_error` envelope? | error contract |
| Q3 | Extract source textures server-side, or read them in the browser? | materials response |
| Q4 | Are triangle counts readable from a Draco-compressed GLB? Unverified. | materials response |
| ~~Q5~~ | ~~Trigger or child table for material-index uniqueness?~~ ✅ resolved by ADR-012 | — |
| Q6 | Fate of the existing colour-variant feature — and the `/products/{id}/materials` path collision. | routing |
| **Q7** | 🔴 **Can the viewer swap textures by glTF material index at runtime?** ADR-003 rests on it entirely. | **the whole texture-baking approach** |
| **Q8** | 🔴 **Has `Rivollo.AccountPurge.Job` been updated?** ADR-010. | **production deployment** |

Ask **Q7 before writing any Configurator code** — a negative answer invalidates ADR-003.
Open **Q8 on day one** — it is cross-repository and has the longest lead time.

**Never describe ADR-005 or ADR-006 as Accepted.** Both are Proposed / Needs Verification.
ADR-006's `glb_version` **column** is safe to build; its **value semantics** are not settled —
use the `asset:` / `sha256:` prefix discriminator and hardcode neither.

Also unresolved: the **`remap` algorithm mismatch** — the backend uses 2nd/98th percentiles
(`glb_recolor.py:383`); the frontend reportedly uses absolute min/max. Do not silently pick
one. See `docs/configurator/baking.md` §5.3.

**If you change the architecture, update the decision docs in the same change.** An ADR that
no longer describes the code is worse than no ADR. Do not promote an ADR from
*Proposed / Needs Verification* to *Accepted* without the evidence in hand.

---

## General rules

- Do not claim something works in production without repository or database evidence.
- Distinguish **CONFIRMED** (observed in code) from **PROPOSED** (recommendation) from
  **NEEDS VERIFICATION** (runtime, Azure, database, or another repository) in any analysis.
- Prefer consistency with existing patterns over introducing new ones. No microservices, no
  event buses, no queues, no caching layers without a stated reason.
- Never log tokens, connection strings, SAS URLs, or user email addresses.
