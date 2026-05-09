# GPU Estimate Time — Frontend Integration Guide

## WebSocket endpoint

```
ws://<host>/ws/products/{product_id}/status
```

Connect immediately after receiving the `product_id` from the create product API response.

---

## Full message reference

All messages are JSON. The `type` field tells you what to do with each message.

| `type` | When it arrives | Action |
|--------|----------------|--------|
| `status_update` | On connect + on every status change | Update UI status and estimate |
| `keepalive` | Every 30s while waiting | Keep connection alive, refresh estimate if present |
| `done` | When product is ready | Hide spinner, show result |
| `error` | On server error | Show error message, close connection |

---

## Step 1 — Create product, then connect WebSocket

```js
// 1. Call create product API
const res = await fetch('/products', { method: 'POST', body: formData });
const { id: productId } = await res.json();

// 2. Connect WebSocket immediately with that product ID
const ws = new WebSocket(`ws://your-api-host/ws/products/${productId}/status`);
```

> Connect as soon as you have the `product_id`. The GPU estimate broadcast fires ~15-32 seconds after creation — connecting early ensures you receive it.

---

## Step 2 — Handle incoming messages

```js
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    switch (msg.type) {

        case 'status_update':
            updateStatus(msg.status);
            if (msg.estimated_time) showEstimate(msg.estimated_time);
            if (msg.message)        showGpuMessage(msg.message);
            break;

        case 'keepalive':
            // Connection is alive — refresh estimate if present
            if (msg.estimated_time) showEstimate(msg.estimated_time);
            break;

        case 'done':
            hideSpinner();
            showSuccess();
            ws.close();
            break;

        case 'error':
            showError(msg.message);
            ws.close();
            break;
    }
};
```

---

## Step 3 — What each status means

| `status` | What to show the user |
|----------|-----------------------|
| `draft` | "Preparing your product..." |
| `queue` | "In queue — waiting for GPU" + estimate |
| `processing` | "Generating your 3D model..." + estimate |
| `ready` | Done — show 3D model |

The `status_update` message includes `estimated_time` and `message` when the GPU state is known.

---

## Message shapes

### `status_update` — GPU cold (waiting for GPU to start)
```json
{
    "type": "status_update",
    "product_id": "988a0463-...",
    "status": "queue",
    "estimated_time": "~12 minutes",
    "gpu_status": "cold",
    "message": "GPU is loading up. This usually takes around 12 minutes.",
    "source": "pg_notify"
}
```

### `status_update` — GPU warm (generating now)
```json
{
    "type": "status_update",
    "product_id": "988a0463-...",
    "status": "queue",
    "estimated_time": "~20 seconds",
    "gpu_status": "warm",
    "message": "GPU is ready — your 3D model is being generated now.",
    "source": "pg_notify"
}
```

### `status_update` — initial state on connect (no estimate yet)
```json
{
    "type": "status_update",
    "product_id": "988a0463-...",
    "status": "queue",
    "updated_date": "2026-05-09T08:51:10.317543+00:00",
    "source": "initial_query"
}
```
> `estimated_time` is absent here — show a generic "please wait" until the next `status_update` arrives with the estimate.

### `keepalive` — every 30 seconds
```json
{
    "type": "keepalive",
    "product_id": "988a0463-...",
    "status": "queue",
    "message": "Your 3D generation request is queued — you'll be next up shortly.",
    "estimated_time": "~11 minutes"
}
```
> `estimated_time` may be absent if the client connected after the broadcast fired. Show last known estimate if you have one stored.

### `done`
```json
{
    "type": "done",
    "status": "ready"
}
```

### `error`
```json
{
    "type": "error",
    "message": "Product 988a0463-... not found"
}
```

---

## Recommended UI states

```
Connect
    │
    ▼
source = "initial_query"
    ├─ status = draft/queue/processing  →  Show spinner + "Please wait..."
    └─ status = ready                   →  Show result immediately, close WS

First status_update with estimated_time
    └─ Show: "Estimated time: ~12 minutes"
       Show: gpu_message (e.g. "GPU is loading up...")

Every keepalive (30s)
    └─ If estimated_time present → refresh the displayed estimate
       (the time counts down: ~12 min → ~11 min → ~10 min...)

GPU ready broadcast (estimated_time = "~20 seconds")
    └─ Update to: "Almost done — generating your 3D model now"

done
    └─ Hide spinner, fetch final product data, show 3D viewer
```

---

## Complete example (vanilla JS)

```js
async function createProductAndTrack(formData) {
    // 1. Create product
    const res = await fetch('/products', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
    });
    const product = await res.json();

    showSpinner('Preparing your product...');

    // 2. Open WebSocket
    const ws = new WebSocket(
        `ws://${location.host}/ws/products/${product.id}/status`
    );

    let lastEstimate = null;

    ws.onopen = () => {
        console.log('WebSocket connected for product', product.id);
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'status_update' || msg.type === 'keepalive') {
            updateStatusLabel(msg.status);

            if (msg.estimated_time) {
                lastEstimate = msg.estimated_time;
                showEstimate(lastEstimate);
            } else if (lastEstimate) {
                showEstimate(lastEstimate); // keep showing last known
            }

            if (msg.message) {
                showGpuMessage(msg.message);
            }
        }

        if (msg.type === 'done') {
            hideSpinner();
            showSuccess('Your 3D model is ready!');
            ws.close();
            loadProduct(product.id); // fetch final product data
        }

        if (msg.type === 'error') {
            hideSpinner();
            showError(msg.message);
            ws.close();
        }
    };

    ws.onerror = () => {
        showError('Connection lost. Please refresh the page.');
    };

    ws.onclose = () => {
        console.log('WebSocket closed for product', product.id);
    };
}
```

---

## Edge cases to handle

| Scenario | What happens | How to handle |
|----------|-------------|---------------|
| User connects after product is already `ready` | `initial_query` returns `status: ready`, then `done` fires | `done` closes the WS — just show result |
| User connects after GPU broadcast already fired | `estimated_time` absent from `initial_query` and early `keepalive` | Show "Please wait..." until estimate arrives, or show last known |
| GPU is warm (fast path) | `estimated_time = "~20 seconds"`, done within ~30s | Same flow — estimate still arrives via `status_update` |
| Network drops mid-wait | `ws.onerror` / `ws.onclose` fires | Reconnect with same `product_id` — `initial_query` will return current status |
| Tab hidden for a while | Keepalives continue server-side for up to 20 min | On tab focus, check if WS is still open; reconnect if needed |

---

## Reconnect on disconnect

```js
function connectWithRetry(productId, attempt = 1) {
    const ws = new WebSocket(`ws://${location.host}/ws/products/${productId}/status`);

    ws.onclose = (e) => {
        if (!e.wasClean && attempt < 5) {
            setTimeout(
                () => connectWithRetry(productId, attempt + 1),
                Math.min(1000 * attempt, 10000) // exponential backoff, max 10s
            );
        }
    };

    return ws;
}
```
