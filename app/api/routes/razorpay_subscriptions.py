"""Razorpay subscription routes — create, verify, cancel subscriptions.

Architecture:
    - Routes are thin HTTP layer: validate input -> call service -> format response.
    - All business logic lives in razorpay_subscription_service.py.
    - /create, /verify, /cancel require authentication (CurrentUser).
    - /webhook does NOT require authentication (called by Razorpay).
"""

from fastapi import APIRouter, Request, status

from app.api.deps import CurrentUser, DB
from app.schemas.razorpay_subscriptions import (
    CancelSubscriptionRequest,
    CancelSubscriptionResponse,
    CreateSubscriptionRequest,
    CreateSubscriptionResponse,
    VerifySubscriptionRequest,
    VerifySubscriptionResponse,
)
from app.services.razorpay_subscription_service import (
    cancel_subscription,
    create_subscription,
    verify_subscription,
)
from app.services.subscription_webhook_service import handle_subscription_webhook
from app.utils.envelopes import api_success

router = APIRouter(prefix="/razorpay-subscriptions", tags=["razorpay-subscriptions"])


@router.post(
    "/create",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Razorpay recurring subscription",
    description="""
Create a new Razorpay subscription — the **first step** of the subscription flow.

### How it works
1. Send `planCode` (e.g. `"pro"`), `billingInterval` (`"monthly"` or `"yearly"`), and optionally an `offerId`.
2. Backend picks the correct Razorpay plan ID based on the billing interval.
3. Receive back `subscriptionId` and `keyId`.
4. Pass `subscriptionId` and `keyId` to the Razorpay Checkout JS widget.
5. After the user completes checkout, send the callback values to `POST /razorpay-subscriptions/verify`.

### Billing interval
- `monthly` (default) — billed every month.
- `yearly` — billed once per year.

### With offer
If a valid Razorpay offer ID is provided, the corresponding discount is applied.
Razorpay handles the discounted billing automatically for the specified number of cycles.
If the offer ID is invalid, Razorpay will return an error.
""",
)
async def create_sub(
    body: CreateSubscriptionRequest,
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Create a Razorpay subscription for the authenticated user."""
    result = await create_subscription(
        db,
        user_id=current_user.id,
        plan_code=body.plan_code,
        billing_interval=body.billing_interval,
        offer_id=body.offer_id,
    )
    return api_success(result)


@router.post(
    "/verify",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Verify Razorpay subscription payment",
    description="""
Verify a completed Razorpay subscription payment — the **second step** of the flow.

### What to send
After the user completes checkout, the Razorpay JS handler callback gives you:
```js
handler: function(response) {
    // POST these 3 values to /razorpay-subscriptions/verify:
    response.razorpay_payment_id
    response.razorpay_subscription_id
    response.razorpay_signature
}
```

### What happens
1. **Signature verification** — HMAC-SHA256 proves the payment is genuine.
   Message format: `{payment_id}|{subscription_id}` (different from one-time payments).
2. **Subscription confirmed** — status set to ACTIVE in database.
3. **Idempotent** — safe to call multiple times.
""",
)
async def verify_sub(
    body: VerifySubscriptionRequest,
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Verify Razorpay subscription payment for the authenticated user."""
    result = await verify_subscription(
        db,
        user_id=current_user.id,
        razorpay_payment_id=body.razorpay_payment_id,
        razorpay_subscription_id=body.razorpay_subscription_id,
        razorpay_signature=body.razorpay_signature,
    )
    response = VerifySubscriptionResponse(
        verified=result["verified"],
        message=result["message"],
        plan=result.get("plan"),
        subscription_id=result.get("subscriptionId"),
        period_end=result.get("periodEnd"),
    )
    return api_success(response.model_dump(by_alias=True, exclude_none=True))


@router.post(
    "/cancel",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Cancel Razorpay subscription",
    description="""
Cancel the user's active Razorpay subscription.

### Options
- `cancelAtCycleEnd: true` (default) — user keeps access until current period ends,
  then Razorpay stops billing. No more charges.
- `cancelAtCycleEnd: false` — cancel immediately. Access revoked right away.
""",
)
async def cancel_sub(
    body: CancelSubscriptionRequest,
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Cancel the authenticated user's Razorpay subscription."""
    result = await cancel_subscription(
        db,
        user_id=current_user.id,
        cancel_at_cycle_end=body.cancel_at_cycle_end,
    )
    response = CancelSubscriptionResponse(
        cancelled=result["cancelled"],
        message=result["message"],
        access_until=result.get("accessUntil"),
    )
    return api_success(response.model_dump(by_alias=True, exclude_none=True))


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    summary="Razorpay subscription webhook (internal)",
    description="""
Receives subscription lifecycle events from Razorpay's servers.

### Handled events
| Event | Action |
|-------|--------|
| `subscription.authenticated` | Confirm checkout completed |
| `subscription.activated` | First payment — set status ACTIVE |
| `subscription.charged` | Monthly payment — extend period, reset quotas |
| `subscription.pending` | Payment failed — set status PAST_DUE |
| `subscription.halted` | All retries failed — CANCEL + revoke license |
| `subscription.cancelled` | Subscription cancelled — revoke license |

### Important
- Called **by Razorpay**, not by the frontend.
- Always returns HTTP 200 regardless of outcome.
- Signature verified via HMAC-SHA256 with `RAZORPAY_WEBHOOK_SECRET`.
""",
    include_in_schema=True,
)
async def subscription_webhook(
    request: Request,
    db: DB,
) -> dict:
    """Receive and process Razorpay subscription webhook events."""
    payload_bytes: bytes = await request.body()
    signature_header: str = request.headers.get("x-razorpay-signature", "")

    result = await handle_subscription_webhook(
        db,
        payload_bytes=payload_bytes,
        signature_header=signature_header,
    )

    return result
