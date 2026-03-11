"""Payment service — Razorpay order creation, payment verification + activation, and webhooks.

Architecture:
    - Uses httpx directly to call Razorpay REST API (no SDK needed).
    - create_razorpay_order: calls Razorpay API, then saves payment row to DB.
    - verify_and_activate_payment: verifies signature, handles idempotency,
      marks payment captured, and triggers subscription activation.
    - handle_razorpay_webhook: called by Razorpay's servers (no user auth).
      Verifies webhook signature with RAZORPAY_WEBHOOK_SECRET, extracts order/
      payment IDs from the JSON payload, and runs the same activation path.
    - All database operations are committed in the route layer (not here),
      so a single transaction spans the full verify + activate flow.

Security:
    - Amount is NEVER taken from the frontend — always derived from plan_code
      via SubscriptionActivationService.get_plan_price_paise().
    - Payment authenticity is guaranteed by HMAC-SHA256 signature verification.
    - Idempotency: if the same order_id is verified twice, we return the
      already-captured result without re-activating the subscription.
    - Webhook: always returns {"status": "ok"} (HTTP 200) even on errors so
      Razorpay does not keep retrying with bad payloads. Failures are logged.
"""

import hashlib
import hmac
import logging
import uuid
from typing import Any, Optional

import httpx
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.payment_repo import PaymentRepository
from app.models.payment import PaymentStatus
from app.services.subscription_activation_service import SubscriptionActivationService

_logger = logging.getLogger("rivollo.payment_service")

_RAZORPAY_BASE_URL = "https://api.razorpay.com/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_credentials() -> None:
    """Raise 503 if Razorpay credentials are not configured."""
    if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
        _logger.error("Razorpay credentials are not configured in settings.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment gateway is not configured. Contact support.",
        )


def _verify_signature(
    razorpay_order_id: str,
    razorpay_payment_id: str,
    razorpay_signature: str,
) -> bool:
    """Verify Razorpay HMAC-SHA256 signature.

    Razorpay's signing scheme:
        HMAC_SHA256(key=KEY_SECRET, msg="{order_id}|{payment_id}")

    Returns:
        True if valid, False if tampered.
    """
    msg = f"{razorpay_order_id}|{razorpay_payment_id}"
    expected = hmac.new(
        key=settings.RAZORPAY_KEY_SECRET.encode("utf-8"),
        msg=msg.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, razorpay_signature)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def create_razorpay_order(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    plan_code: str,
    currency: str = "INR",
    receipt: Optional[str] = None,
    notes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a Razorpay order and save a payment record to the database.

    Amount is derived server-side from plan_code — never trusted from the frontend.

    Flow:
        1. Validate plan_code and look up server-side price
        2. Check Razorpay credentials
        3. Call Razorpay API to create order
        4. Save payment row to DB (status='created')
        5. Return order details to frontend

    Args:
        db:        Async database session
        user_id:   UUID of the authenticated user
        plan_code: "pro" or "enterprise"
        currency:  ISO currency code (default: INR)
        receipt:   Optional internal receipt reference (max 40 chars)
        notes:     Optional arbitrary key-value notes

    Returns:
        Dict with orderId, amount, currency, receipt, status, keyId, planCode.

    Raises:
        HTTPException 400: Unknown plan_code or Razorpay rejected request
        HTTPException 502: Razorpay API unreachable
        HTTPException 503: Credentials not configured
    """
    # ── 1. Server-side amount — never trust frontend ─────────────────────────
    amount = await SubscriptionActivationService.get_plan_price_paise(db, plan_code)

    # ── 2. Check credentials ─────────────────────────────────────────────────
    _check_credentials()

    # ── 3. Build payload ─────────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "amount": amount,
        "currency": currency,
        "payment_capture": 1,  # auto-capture on payment success
    }
    if receipt:
        payload["receipt"] = receipt
    if notes:
        payload["notes"] = notes

    # ── 4. Call Razorpay API ─────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{_RAZORPAY_BASE_URL}/orders",
                json=payload,
                auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET),
            )

        if response.status_code == 400:
            error_detail = response.json().get("error", {}).get("description", "Bad request")
            _logger.warning("Razorpay bad request: %s", error_detail)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Razorpay rejected the request: {error_detail}",
            )

        if response.status_code == 401:
            _logger.error("Razorpay auth failed — check KEY_ID and KEY_SECRET.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Payment gateway authentication failed. Contact support.",
            )

        response.raise_for_status()
        order: dict[str, Any] = response.json()

    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        _logger.exception("Razorpay API timed out: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment gateway timed out. Please try again.",
        ) from exc
    except Exception as exc:
        _logger.exception("Razorpay order creation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment gateway error. Please try again later.",
        ) from exc

    # ── 5. Save payment record to DB ─────────────────────────────────────────
    razorpay_order_id: str = order["id"]
    await PaymentRepository.create_payment(
        db,
        user_id=user_id,
        razorpay_order_id=razorpay_order_id,
        amount=order["amount"],
        currency=order["currency"],
        plan_code=plan_code,
    )
    # Commit so the payment row is durably saved even if verify never arrives
    await db.commit()

    _logger.info(
        "Razorpay order created and payment record saved",
        extra={"order_id": razorpay_order_id, "amount": amount, "plan": plan_code},
    )

    return {
        "orderId": razorpay_order_id,
        "amount": order["amount"],
        "currency": order["currency"],
        "receipt": order.get("receipt"),
        "status": order["status"],
        "keyId": settings.RAZORPAY_KEY_ID,
        "planCode": plan_code,
    }


async def verify_and_activate_payment(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    plan_code: str,
    razorpay_order_id: str,
    razorpay_payment_id: str,
    razorpay_signature: str,
) -> dict[str, Any]:
    """Verify payment signature and activate the subscription.

    Flow:
        1. Fetch the payment record by order_id (idempotency check)
        2. If already captured → return existing result without re-activating
        3. Verify HMAC-SHA256 signature
        4. If invalid → mark payment FAILED, raise 400
        5. If valid → activate subscription (extend or create)
        6. Mark payment CAPTURED with subscription_id
        7. Commit transaction
        8. Return verification + subscription result

    Args:
        db:                   Async database session
        user_id:              UUID of the authenticated user
        plan_code:            "pro" or "enterprise"
        razorpay_order_id:    Order ID from Razorpay
        razorpay_payment_id:  Payment ID from Razorpay checkout
        razorpay_signature:   HMAC signature from Razorpay checkout handler

    Returns:
        Dict with verified=True, plan, subscriptionId, periodEnd, etc.

    Raises:
        HTTPException 400: Signature invalid or payment already has a different status
        HTTPException 404: Payment record not found (order_id never created via our API)
        HTTPException 503: Credentials not configured
    """
    _check_credentials()

    # ── 1. Fetch payment record ──────────────────────────────────────────────
    payment = await PaymentRepository.get_by_order_id(db, razorpay_order_id)
    if payment is None:
        _logger.warning(
            "Verify called for unknown order_id: %s", razorpay_order_id
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Payment record not found for this order ID. "
                "Make sure to create an order via POST /payments/orders first."
            ),
        )

    # ── 2. Idempotency — already captured → return existing result ───────────
    if payment.status == PaymentStatus.CAPTURED:
        _logger.info(
            "Duplicate verify call for already-captured order_id: %s", razorpay_order_id
        )
        subscription = payment.subscription
        return {
            "verified": True,
            "message": "Payment already verified and subscription is active.",
            "razorpayPaymentId": payment.razorpay_payment_id,
            "razorpayOrderId": razorpay_order_id,
            "plan": plan_code,
            "subscriptionId": str(payment.subscription_id) if payment.subscription_id else None,
            "periodEnd": (
                subscription.current_period_end if subscription else None
            ),
        }

    # ── 3. Verify HMAC-SHA256 signature ──────────────────────────────────────
    is_valid = _verify_signature(
        razorpay_order_id=razorpay_order_id,
        razorpay_payment_id=razorpay_payment_id,
        razorpay_signature=razorpay_signature,
    )

    if not is_valid:
        _logger.warning(
            "Signature verification FAILED for order_id: %s", razorpay_order_id
        )
        # ── 4. Mark as FAILED ────────────────────────────────────────────────
        await PaymentRepository.mark_failed(
            db,
            payment_id=payment.id,
            reason="HMAC-SHA256 signature mismatch — possible data tampering.",
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment signature verification failed. The payment may not be genuine.",
        )

    # ── 5. Activate / extend subscription ────────────────────────────────────
    subscription = await SubscriptionActivationService.activate(
        db,
        user_id=user_id,
        plan_code=plan_code,
        razorpay_order_id=razorpay_order_id,
        razorpay_payment_id=razorpay_payment_id,
    )

    # ── 6. Mark payment as CAPTURED ──────────────────────────────────────────
    await PaymentRepository.mark_captured(
        db,
        payment_id=payment.id,
        razorpay_payment_id=razorpay_payment_id,
        razorpay_signature=razorpay_signature,
        subscription_id=subscription.id,
    )

    # ── 7. Commit the full transaction ────────────────────────────────────────
    await db.commit()

    _logger.info(
        "Payment captured and subscription activated",
        extra={
            "order_id": razorpay_order_id,
            "payment_id": razorpay_payment_id,
            "subscription_id": str(subscription.id),
            "plan": plan_code,
            "period_end": subscription.current_period_end.isoformat()
            if subscription.current_period_end else None,
        },
    )

    return {
        "verified": True,
        "message": f"Payment verified. Your {plan_code.capitalize()} subscription is now active! 🎉",
        "razorpayPaymentId": razorpay_payment_id,
        "razorpayOrderId": razorpay_order_id,
        "plan": plan_code,
        "subscriptionId": str(subscription.id),
        "periodEnd": subscription.current_period_end,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Webhook handler — called by Razorpay, NOT by the user / frontend
# ─────────────────────────────────────────────────────────────────────────────

def _verify_webhook_signature(payload_bytes: bytes, signature_header: str) -> bool:
    """Verify Razorpay webhook HMAC-SHA256 signature.

    Razorpay's webhook signing scheme:
        HMAC_SHA256(key=WEBHOOK_SECRET, msg=raw_request_body_bytes)

    This is DIFFERENT from the payment signature (which uses KEY_SECRET and
    combines order_id + payment_id). The webhook secret is a separate value
    configured in the Razorpay Dashboard → Settings → Webhooks.

    Args:
        payload_bytes:    The raw, unmodified request body bytes.
        signature_header: Value of the X-Razorpay-Signature request header.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not settings.RAZORPAY_WEBHOOK_SECRET:
        _logger.error(
            "RAZORPAY_WEBHOOK_SECRET is not set — cannot verify webhook signature."
        )
        return False

    expected = hmac.new(
        key=settings.RAZORPAY_WEBHOOK_SECRET.encode("utf-8"),
        msg=payload_bytes,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


async def handle_razorpay_webhook(
    db: AsyncSession,
    *,
    payload_bytes: bytes,
    signature_header: str,
) -> dict[str, str]:
    """Handle an incoming Razorpay webhook event.

    This function is called by the webhook route (POST /payments/webhook).
    It is the **reliable fallback** for subscription activation — it runs even
    if the user closed their browser before the frontend called /payments/verify.

    Design principles:
        - ALWAYS return {"status": "ok"} — never raise an exception.
          If we return non-200, Razorpay retries the webhook, causing duplicate
          activations. All errors are logged and silently discarded.
        - Idempotent — if /payments/verify already captured the payment, we skip
          activation and return silently.
        - plan_code and user_id come from the DB payment row (stored at order
          creation time), not from the webhook payload, so there is no risk of
          a malicious payload injecting an incorrect plan.
        - Only handles the `payment.captured` event. All others are ignored.

    Args:
        db:               Async database session (transaction managed here).
        payload_bytes:    Raw request body bytes (needed for HMAC verification).
        signature_header: Value of X-Razorpay-Signature header.

    Returns:
        Always {"status": "ok"}.
    """
    # ── 1. Verify webhook signature ──────────────────────────────────────────
    if not _verify_webhook_signature(payload_bytes, signature_header):
        # Do NOT return an error status — log and return 200 silently.
        # Returning non-200 would cause Razorpay to retry, flooding our server.
        _logger.warning(
            "Webhook signature verification FAILED — request may be forged."
        )
        return {"status": "ok"}

    # ── 2. Parse JSON payload ────────────────────────────────────────────────
    try:
        payload: dict = __import__("json").loads(payload_bytes)
    except Exception as exc:
        _logger.error("Webhook payload JSON parse failed: %s", exc)
        return {"status": "ok"}

    event: str = payload.get("event", "")
    _logger.info("Razorpay webhook received: event=%s", event)

    # ── 3. Only process payment.captured events ──────────────────────────────
    if event != "payment.captured":
        _logger.info("Webhook event '%s' ignored — not payment.captured.", event)
        return {"status": "ok"}

    # ── 4. Extract order_id and payment_id from payload ──────────────────────
    try:
        payment_entity: dict = payload["payload"]["payment"]["entity"]
        razorpay_order_id: str = payment_entity["order_id"]
        razorpay_payment_id: str = payment_entity["id"]
    except (KeyError, TypeError) as exc:
        _logger.error("Webhook payload missing expected fields: %s", exc)
        return {"status": "ok"}

    _logger.info(
        "Webhook payment.captured: order_id=%s payment_id=%s",
        razorpay_order_id,
        razorpay_payment_id,
    )

    # ── 5. Load payment record from DB ───────────────────────────────────────
    payment = await PaymentRepository.get_by_order_id(db, razorpay_order_id)
    if payment is None:
        # Order wasn't created via our API — possibly a manual test order.
        _logger.warning(
            "Webhook: no payment row found for order_id=%s — ignoring.",
            razorpay_order_id,
        )
        return {"status": "ok"}

    # ── 6. Idempotency — already captured by /payments/verify ────────────────
    if payment.status == PaymentStatus.CAPTURED:
        _logger.info(
            "Webhook: order_id=%s already captured — skipping re-activation.",
            razorpay_order_id,
        )
        return {"status": "ok"}

    # ── 7. Read plan_code and user_id from the stored payment row ─────────────
    # These were saved when the order was created — we NEVER trust the webhook
    # payload for these values (security: no plan injection possible).
    plan_code: str = payment.plan_code
    user_id = payment.user_id

    # ── 8. Activate / extend subscription ────────────────────────────────────
    try:
        subscription = await SubscriptionActivationService.activate(
            db,
            user_id=user_id,
            plan_code=plan_code,
            razorpay_order_id=razorpay_order_id,
            razorpay_payment_id=razorpay_payment_id,
        )
    except Exception as exc:
        _logger.exception(
            "Webhook: subscription activation failed for order_id=%s: %s",
            razorpay_order_id,
            exc,
        )
        # Roll back and return 200 (don't let Razorpay retry indefinitely)
        await db.rollback()
        return {"status": "ok"}

    # ── 9. Mark payment as CAPTURED ───────────────────────────────────────────
    # Webhook signature from Razorpay doesn't include the payment signature
    # (that's only in the checkout callback). Store empty string as sentinel.
    await PaymentRepository.mark_captured(
        db,
        payment_id=payment.id,
        razorpay_payment_id=razorpay_payment_id,
        razorpay_signature="webhook",  # sentinel: captured via webhook, not verify
        subscription_id=subscription.id,
    )

    # ── 10. Commit the full transaction ───────────────────────────────────────
    await db.commit()

    _logger.info(
        "Webhook: subscription activated via webhook — "
        "order_id=%s payment_id=%s subscription_id=%s plan=%s",
        razorpay_order_id,
        razorpay_payment_id,
        subscription.id,
        plan_code,
    )

    return {"status": "ok"}

