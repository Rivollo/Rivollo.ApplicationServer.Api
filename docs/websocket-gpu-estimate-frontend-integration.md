# WebSocket — GPU Estimate Frontend Integration

## Connection

```
ws://<host>/ws/products/<product_uuid>/status
```

Connect immediately after the `/createProduct` response is received. No auth token required.

---

## Message Types

All messages are JSON. The `type` field identifies the message.

### `status_update`

Sent when the product status changes, or when the browser connects (initial state).

```json
{
  "type": "status_update",
  "product_id": "3f7a9e12-...",
  "status": "queue",
  "old_status": "draft",
  "updated_date": "2024-01-15T10:23:45.123456",
  "source": "pg_notify",

  "estimated_seconds": 720,
  "gpu_status": "cold",
  "message": "GPU is loading up. This usually takes around 12 minutes."
}
```

| Field | Always present | Description |
|---|---|---|
| `type` | yes | Always `"status_update"` |
| `product_id` | yes | UUID of the product |
| `status` | yes | Current status — see status values below |
| `old_status` | no | Previous status (absent on `initial_query`) |
| `updated_date` | no | ISO timestamp of the last DB update |
| `source` | yes | `"initial_query"` / `"pg_notify"` / `"recovery_poll"` |
| `estimated_seconds` | no | Estimated seconds until ready — present when GPU state is known |
| `gpu_status` | no | `"warm"` or `"cold"` — only on the estimate broadcast |
| `message` | no | Human-readable GPU status message |

---

### `keepalive`

Sent every 30 seconds while waiting. Confirms the connection is alive.

```json
{
  "type": "keepalive",
  "product_id": "3f7a9e12-...",
  "status": "queue",
  "message": "Still in the queue — your model is almost up next.",
  "estimated_seconds": 720
}
```

| Field | Always present | Description |
|---|---|---|
| `estimated_seconds` | no | Repeats the last known estimate so UI can keep displaying it |

---

### `done`

Sent once when the product reaches `ready` status. Close the WebSocket after receiving this.

```json
{
  "type": "done",
  "status": "ready"
}
```

---

### `error`

Sent if the product is not found or an internal error occurs. Close the WebSocket after receiving this.

```json
{
  "type": "error",
  "message": "Product <uuid> not found"
}
```

---

## Status Values

| Status | Meaning |
|---|---|
| `draft` | Product created, background task not yet started |
| `queue` | Background task running — GPU warming up or request queued |
| `processing` | GPU is ready, 3D generation actively running |
| `ready` | 3D model generated successfully — assets are available |

---

## GPU Estimate Fields

`estimated_seconds` and `message` arrive on the **first `status_update` with `status: "queue"`** after the background task does its warm/cold check (within ~5 seconds of the product being created).

| `gpu_status` | `estimated_seconds` | Meaning |
|---|---|---|
| `"warm"` | `20` | GPU container was already running |
| `"cold"` | `720` | GPU container was off, loading now (~12 min) |

These same values are repeated on every subsequent `keepalive` so the UI never loses the estimate between status changes.

**If the browser connects after the estimate was already broadcast** (late connection), the `initial_query` response for `status: "queue"` will include `estimated_seconds: 720` as a conservative default.

---

## Recommended UI Logic

```
on status_update / keepalive:
  if estimated_seconds present → show countdown or progress bar
  if gpu_status == "cold"      → show "GPU is loading..." state
  if gpu_status == "warm"      → show "Generating 3D model..." state

on status == "processing"      → hide GPU loading state, show generation progress
on status == "ready"           → hide progress, load 3D viewer
on type == "done"              → close WebSocket, fetch product assets
on type == "error"             → show error message, close WebSocket
```

---

## JavaScript Example

```js
function watchProductStatus(productId, onReady) {
  const ws = new WebSocket(`wss://<host>/ws/products/${productId}/status`);

  let estimatedSeconds = null;

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.estimated_seconds != null) {
      estimatedSeconds = msg.estimated_seconds;
      showEstimate(estimatedSeconds, msg.gpu_status, msg.message);
    }

    switch (msg.type) {
      case "status_update":
        updateStatusUI(msg.status);
        break;

      case "keepalive":
        // Connection is alive — optionally tick down a countdown using estimatedSeconds
        break;

      case "done":
        ws.close();
        onReady();
        break;

      case "error":
        ws.close();
        showError(msg.message);
        break;
    }
  };

  ws.onerror = () => showError("Connection lost — please refresh.");
}

function showEstimate(seconds, gpuStatus, message) {
  const minutes = Math.round(seconds / 60);
  const label = gpuStatus === "cold"
    ? `GPU is loading — estimated ${minutes} min`
    : `GPU ready — generating your 3D model (~${seconds}s)`;

  document.getElementById("gpu-status").textContent = message ?? label;
  document.getElementById("eta").textContent = `~${minutes} min`;
}
```

---

## Message Sequence — Cold Start

```
[browser connects]

← status_update  status=draft   source=initial_query
← status_update  status=queue   source=pg_notify                    (DB trigger)
← status_update  status=queue   estimated_seconds=720  gpu_status=cold   (broadcaster)
← keepalive      status=queue   estimated_seconds=720
← keepalive      status=queue   estimated_seconds=720
  ... (every 30s for ~12 min) ...
← status_update  status=processing  source=pg_notify
← status_update  status=ready       source=pg_notify
← done           status=ready
```

## Message Sequence — Warm GPU

```
[browser connects]

← status_update  status=draft   source=initial_query
← status_update  status=queue   source=pg_notify
← status_update  status=queue   estimated_seconds=20  gpu_status=warm
← status_update  status=processing  source=pg_notify
← status_update  status=ready       source=pg_notify
← done           status=ready
```
