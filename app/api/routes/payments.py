"""Razorpay payment routes.

Architecture:
    - Routes are thin HTTP layer: validate input → call service → format response.
    - All business logic (Razorpay API calls, DB saves, signature verification,
      subscription activation) lives in payment_service.py and its dependencies.
    - /orders and /verify require authentication (CurrentUser).
    - /webhook does NOT require authentication (called by Razorpay, not the user).

Endpoints:
    POST /payments/orders   — Create a Razorpay order (step 1 of payment flow)
    POST /payments/verify   — Verify + activate after Razorpay checkout (step 2)
    POST /payments/webhook  — Razorpay webhook: reliable fallback activation
"""

from fastapi import APIRouter, Request, status

from app.api.deps import CurrentUser, DB
from app.schemas.payments import (
    CreateOrderRequest,
    VerifyPaymentRequest,
    VerifyPaymentResponse,
)
from app.services.payment_service import (
    create_razorpay_order,
    handle_razorpay_webhook,
    verify_and_activate_payment,
)
from app.utils.envelopes import api_success

router = APIRouter(prefix="/payments", tags=["payments"])


@router.post(
    "/orders",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Razorpay order",
    description="""
Create a new Razorpay order — the **first step** of the payment flow.

### How it works
1. Send `planCode` (e.g. `"pro"`) — the backend looks up the correct price server-side.
2. Receive back `keyId` and `orderId`.
3. Pass `keyId` and `orderId` to the [Razorpay Checkout JS](https://razorpay.com/docs/payments/payment-gateway/web-integration/standard/) widget.
4. After the user completes payment, Razorpay returns three values in the handler callback.
5. Send those values + `planCode` to `POST /payments/verify`.

### Plan prices (server-side, cannot be overridden)
| Plan | Price |
|------|-------|
| `pro` | ₹1,999/month |
| `enterprise` | Contact sales — not available via self-serve payment |

### Error codes
| Code | Meaning |
|------|---------|
| `400` | Unknown plan_code or Razorpay rejected request |
| `401` | Not authenticated |
| `502` | Razorpay API unreachable |
| `503` | Razorpay credentials not configured |
""",
    responses={
        201: {"description": "Order created successfully"},
        400: {"description": "Invalid plan_code or Razorpay error"},
        401: {"description": "Authentication required"},
        502: {"description": "Razorpay API error"},
        503: {"description": "Payment gateway not configured"},
    },
)
async def create_order(
    body: CreateOrderRequest,
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Create a Razorpay order for the authenticated user.

    The amount is derived server-side from planCode — never from the request body.
    A payment row (status='created') is saved to tbl_payments immediately.
    """
    order = await create_razorpay_order(
        db,
        user_id=current_user.id,
        plan_code=body.plan_code,
        currency=body.currency,
        receipt=body.receipt,
        notes=body.notes,
        promo_code=body.promo_code   # NEW
    )
    return api_success(order)


@router.post(
    "/verify",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Verify Razorpay payment and activate subscription",
    description="""
Verify a completed Razorpay payment and activate the subscription — the **second step** of the payment flow.

### What to send
After the user completes checkout, the Razorpay JS handler callback gives you:
```js
handler: function(response) {
    // POST these 4 values to /payments/verify:
    response.razorpay_order_id
    response.razorpay_payment_id
    response.razorpay_signature
    // + the planCode you used when creating the order
}
```

### What happens
1. **Signature verification** — HMAC-SHA256 check proves the payment is genuine.
2. **Subscription activation** — creates or extends `tbl_subscriptions` + `tbl_license_assignments`.
3. **Idempotency** — sending the same request twice returns the same result without duplicate rows.

### Error codes
| Code | Meaning |
|------|---------|
| `400` | Signature verification failed (payment is not genuine) |
| `401` | Not authenticated |
| `404` | Order not found (must call POST /orders first) |
| `503` | Payment gateway not configured |
""",
    responses={
        200: {"description": "Payment verified and subscription activated"},
        400: {"description": "Invalid signature — payment not genuine"},
        401: {"description": "Authentication required"},
        404: {"description": "Payment record not found for this order ID"},
        503: {"description": "Payment gateway not configured"},
    },
)
async def verify_payment(
    body: VerifyPaymentRequest,
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Verify Razorpay payment and activate the subscription for the authenticated user."""
    result = await verify_and_activate_payment(
        db,
        user_id=current_user.id,
        plan_code=body.plan_code,
        razorpay_order_id=body.razorpay_order_id,
        razorpay_payment_id=body.razorpay_payment_id,
        razorpay_signature=body.razorpay_signature,
    )

    response = VerifyPaymentResponse(
        verified=result["verified"],
        message=result["message"],
        razorpay_payment_id=result.get("razorpayPaymentId"),
        razorpay_order_id=result.get("razorpayOrderId"),
        plan=result.get("plan"),
        subscription_id=result.get("subscriptionId"),
        period_end=result.get("periodEnd"),
    )
    return api_success(response.model_dump(by_alias=True, exclude_none=True))


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    summary="Razorpay payment webhook (internal)",
    description="""
Receives `payment.captured` events directly from Razorpay's servers.

### Important
- This endpoint is called **by Razorpay**, not by the frontend.
- **Do not** call this endpoint from your app — it will silently ignore
  requests with an invalid `X-Razorpay-Signature` header.
- Every response is HTTP 200 regardless of outcome (Razorpay retries on non-200).

### Security
The `X-Razorpay-Signature` header is verified using HMAC-SHA256 with
`RAZORPAY_WEBHOOK_SECRET`. Any request with a missing or invalid signature
is silently discarded.

### Idempotency
If the payment was already captured by `POST /payments/verify`, this endpoint
skips activation and returns immediately without duplicating any DB rows.
""",
    include_in_schema=True,
)
async def razorpay_webhook(
    request: Request,
    db: DB,
) -> dict:
    """Receive and process Razorpay webhook events.

    NO authentication required — Razorpay calls this directly.
    Request authenticity is verified via HMAC-SHA256 signature.
    """
    # Must read the raw bytes BEFORE any JSON parsing.
    # Parsing JSON first alters the byte representation and breaks the HMAC check.
    payload_bytes: bytes = await request.body()
    signature_header: str = request.headers.get("x-razorpay-signature", "")

    result = await handle_razorpay_webhook(
        db,
        payload_bytes=payload_bytes,
        signature_header=signature_header,
    )
    return result
