# GPU Estimate Time — WebSocket Feature

## What it does

When a user creates a product, a background task checks whether the 3D generation GPU is warm or cold and broadcasts an estimated wait time to any connected WebSocket clients. This lets the frontend show a meaningful progress message instead of a generic spinner.

---

## Files involved

| File | Role |
|------|------|
| `app/models/gpu_activation.py` | DB model — stores when a cold-start began |
| `app/database/gpu_activation_repo.py` | `get_activation` / `upsert_activation` queries |
| `app/core/config.py` | `GPU_COLD_START_SECONDS` env var (default 720 = 12 min) |
| `app/integrations/threed_model_client.py` | `quick_warmth_check()` — probes `/health` to detect warm vs cold |
| `app/services/product_service.py` | `_get_cold_start_estimate()` and `_run_3d_generation_background()` |
| `app/api/websocket/product_status.py` | WebSocket handler — receives and forwards estimate to browser |

---

## Flow — step by step

```
POST /products  (create product)
        │
        ▼
Product saved as DRAFT → HTTP response returned immediately
        │
        ▼ (FastAPI BackgroundTask)
_run_3d_generation_background()
        │
        ├─ 1. Flip product status → QUEUE  (fires pg_notify)
        │
        ├─ 2. quick_warmth_check()
        │       └─ Hits GET /health on the 3D service
        │          Up to 2 attempts × 15s timeout each (~32s max)
        │
        ├─ 3a. GPU warm (200 response)
        │       estimated_time = "~20 seconds"
        │       message = "GPU is ready — your 3D model is being generated now."
        │
        └─ 3b. GPU cold (no 200 response)
                │
                ▼
        _get_cold_start_estimate()
                ├─ Query tbl_gpu_activation for service = "3d_model"
                ├─ Record found AND not stale (< cold_seconds + 120s):
                │       elapsed = now - activation_started_at
                │       remaining = max(30, cold_seconds - elapsed)
                │       label = "~{ceil(remaining/60)} minutes"    ← e.g. "~11 minutes"
                │
                └─ No record OR stale:
                        upsert_activation(now)
                        label = "~{round(cold_seconds/60)} minutes"  ← e.g. "~12 minutes"
        │
        ▼
broadcaster.broadcast_to_product(product_id, {
    "new_status": "queue",
    "estimated_time": "~12 minutes",   ← or "~11 minutes" for 2nd user
    "gpu_status": "cold",              ← or "warm"
    "message": "GPU is loading up..."
})
        │
        ▼ (if cold)
wait_until_ready()  — polls /health until 200 (up to 20 min)
        │
        ▼ (once ready)
broadcast "~20 seconds / GPU is now ready" update
        │
        ▼
generate_3d_and_finalize() → status PROCESSING → READY
```

---

## WebSocket messages the browser receives

### 1. `initial_query` — on connect
```json
{
    "type": "status_update",
    "product_id": "<uuid>",
    "status": "queue",
    "updated_date": "2026-05-09T...",
    "source": "initial_query"
}
```
> No `estimated_time` here currently — see Known Gap below.

### 2. `status_update` — when broadcast fires
```json
{
    "type": "status_update",
    "product_id": "<uuid>",
    "status": "queue",
    "estimated_time": "~12 minutes",
    "gpu_status": "cold",
    "message": "GPU is loading up. This usually takes around 12 minutes.",
    "source": "pg_notify"
}
```

### 3. `keepalive` — every 30s while waiting
```json
{
    "type": "keepalive",
    "product_id": "<uuid>",
    "status": "queue",
    "message": "Your 3D generation request is queued — you'll be next up shortly.",
    "estimated_time": "~12 minutes"   ← only present if broadcast was received
}
```

### 4. GPU ready update (after cold-start completes)
```json
{
    "type": "status_update",
    "status": "queue",
    "estimated_time": "~20 seconds",
    "gpu_status": "warm",
    "message": "GPU is now ready — your 3D model generation is starting."
}
```

### 5. `done` — when product reaches READY
```json
{
    "type": "done",
    "status": "ready"
}
```

---

## Config

```env
# .env
GPU_COLD_START_SECONDS=720   # 12 minutes — adjust per environment
```

Defined in `app/core/config.py`:
```python
GPU_COLD_START_SECONDS: int = Field(default=720)
```

---

## DB table — `tbl_gpu_activation`

One row per service. Upserted each time a fresh cold-start is detected.

| Column | Type | Description |
|--------|------|-------------|
| `service` | varchar(50) PK | Fixed value `"3d_model"` |
| `activation_started_at` | timestamptz | When the cold-start began |

### How subsequent users get a shorter estimate

- User 1 creates product → no activation record → upsert with `now` → returns "~12 minutes"
- User 2 creates product 1 min later → finds record → elapsed = 60s → remaining = 660s → returns "~11 minutes"
- Stale threshold = `GPU_COLD_START_SECONDS + 120s` — records older than this are treated as a fresh cold-start

---

## Known Gap — late WebSocket connectors miss the estimate

`broadcast_to_product` is an **in-memory pub/sub** — it only delivers to clients connected at the exact moment it fires. The broadcast fires once, ~15-32 seconds after product creation (after `quick_warmth_check` completes).

If the user connects to the WebSocket after the broadcast has already fired:
- `initial_query` returns the current status with no `estimated_time`
- `keepalive` messages also have no `estimated_time` (it was never stored in the handler's local variable)

**Affected scenario:** Any user who connects to the WebSocket more than ~32 seconds after creating their product, or a second concurrent user watching their own product.

**Fix needed:** Query `tbl_gpu_activation` in the WebSocket handler and calculate remaining time dynamically for `initial_query` and `keepalive` — same formula already used in `_get_cold_start_estimate()`.

---

## Commit history

| Commit | Description |
|--------|-------------|
| `af8f4f9` | add estimate time to web socket for 3d generation |
| `02ccaaa` | bug fix |
