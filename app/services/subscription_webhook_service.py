"""Subscription Webhook Service — handles Razorpay subscription lifecycle events.

Razorpay sends webhook events for subscription state changes.
This service processes each event and updates the database accordingly.

Supported events:
    subscription.authenticated — user completed checkout (card verified)
    subscription.activated     — first payment charged after creation
    subscription.charged       — recurring monthly payment succeeded
    subscription.pending       — payment failed, Razorpay is retrying
    subscription.halted        — all retries failed
    subscription.cancelled     — subscription cancelled

Design principles:
    - Always return {"status": "ok"} — never raise exceptions.
    - Idempotent — processing the same event twice has no side effects.
    - plan_code and user_id come from DB, never from the webhook payload.
"""

import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.license_assignment import LicenseAssignment
from app.models.payment import Payment, PaymentStatus
from app.models.plan import Plan, PlanFeature
from app.models.subscription import Subscription
from app.models.subscription_enums import LicenseStatus, SubscriptionStatus
from app.models.webhook_event import WebhookEvent

_logger = logging.getLogger("rivollo.subscription_webhook_service")




def _verify_webhook_signature(payload_bytes: bytes, signature_header: str) -> bool:
    """Verify Razorpay webhook HMAC-SHA256 signature.

    Uses RAZORPAY_WEBHOOK_SECRET (separate from KEY_SECRET).
    """
    if not settings.RAZORPAY_WEBHOOK_SECRET:
        _logger.error("RAZORPAY_WEBHOOK_SECRET is not set.")
        return False

    expected = hmac.new(
        key=settings.RAZORPAY_WEBHOOK_SECRET.encode("utf-8"),
        msg=payload_bytes,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


async def _get_subscription_by_rz_id(
    db: AsyncSession, rz_subscription_id: str
) -> Optional[Subscription]:
    """Find subscription by Razorpay subscription ID."""
    result = await db.execute(
        select(Subscription)
        .where(Subscription.razorpay_subscription_id == rz_subscription_id)
        .options(selectinload(Subscription.plan))
    )
    return result.scalar_one_or_none()


async def _get_plan_limits(db: AsyncSession, plan_id: uuid.UUID) -> dict:
    """Get feature limits for a plan."""
    result = await db.execute(
        select(Plan)
        .where(Plan.id == plan_id)
        .options(selectinload(Plan.plan_features).selectinload(PlanFeature.feature))
    )
    plan = result.scalar_one_or_none()
    if not plan:
        return {}

    limits = {}
    for pf in plan.plan_features:
        if pf.feature and pf.limit_value is not None:
            limits[pf.feature.code] = pf.limit_value
    return limits



async def _save_payment_from_webhook(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    subscription_id: uuid.UUID,
    rz_subscription_id: str,
    rz_payment_id: str,
    amount: int,
    plan_code: str,
) -> None:
    """Save a payment record from webhook data."""
    # Use a composite key for razorpay_order_id to maintain uniqueness
    order_id_key = f"webhook_{rz_subscription_id}_{rz_payment_id}"

    # Check if this payment already exists (idempotency)
    existing = await db.execute(
        select(Payment).where(Payment.razorpay_order_id == order_id_key)
    )
    if existing.scalar_one_or_none() is not None:
        return  # already saved

    payment = Payment(
        user_id=user_id,
        subscription_id=subscription_id,
        razorpay_order_id=order_id_key,
        razorpay_payment_id=rz_payment_id,
        razorpay_signature="webhook",
        razorpay_subscription_id=rz_subscription_id,
        amount=amount,
        currency="INR",
        plan_code=plan_code,
        status=PaymentStatus.CAPTURED,
    )
    db.add(payment)
    await db.flush()


async def _revoke_license(
    db: AsyncSession, subscription_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Revoke the user's license when subscription ends."""
    result = await db.execute(
        select(LicenseAssignment).where(
            LicenseAssignment.subscription_id == subscription_id,
            LicenseAssignment.user_id == user_id,
        )
    )
    license_obj = result.scalar_one_or_none()
    if license_obj:
        license_obj.status = LicenseStatus.REVOKED
        await db.flush()


async def _upsert_license(
    db: AsyncSession,
    *,
    subscription: Subscription,
    user_id: uuid.UUID,
    limits: dict,
    reset_usage: bool = False,
) -> None:
    """Create or update license assignment for a subscription."""
    result = await db.execute(
        select(LicenseAssignment).where(
            LicenseAssignment.subscription_id == subscription.id,
            LicenseAssignment.user_id == user_id,
        )
    )
    existing = result.scalar_one_or_none()

    if existing is not None:
        existing.status = LicenseStatus.ACTIVE
        existing.limit_max_products = limits.get("max_products", 0)
        existing.limit_max_ai_credits = limits.get("max_ai_credits_month", 0)
        existing.limit_max_public_views = limits.get("max_public_views", 0)
        existing.limit_max_galleries = limits.get("max_galleries", 0)
        if reset_usage:
            existing.usage_ai_credits = 0
            existing.usage_public_views = 0
    else:
        db.add(LicenseAssignment(
            subscription_id=subscription.id,
            user_id=user_id,
            status=LicenseStatus.ACTIVE,
            limit_max_products=limits.get("max_products", 0),
            limit_max_ai_credits=limits.get("max_ai_credits_month", 0),
            limit_max_public_views=limits.get("max_public_views", 0),
            limit_max_galleries=limits.get("max_galleries", 0),
            usage_products=0,
            usage_ai_credits=0,
            usage_public_views=0,
            usage_galleries=0,
        ))

    await db.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Event handlers
# ─────────────────────────────────────────────────────────────────────────────


async def _handle_subscription_authenticated(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.authenticated — user completed checkout, card verified.

    This event means the card was verified, NOT that payment was charged.
    Subscription stays PENDING — activation happens in subscription.activated
    or via verify_subscription() after the frontend callback.
    """
    subscription = await _get_subscription_by_rz_id(db, rz_subscription_id)
    if not subscription:
        _logger.warning(
            "Webhook authenticated: no subscription found for rz_sub_id=%s",
            rz_subscription_id,
        )
        return

    # Do NOT activate here — payment has not been charged yet.
    _logger.info(
        "Webhook authenticated: rz_sub_id=%s card verified, awaiting payment.",
        rz_subscription_id,
    )


async def _handle_subscription_activated(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.activated — first payment charged successfully.

    Source of truth for period dates, license and payment record.
    Safe to run multiple times — idempotent.
    """
    subscription = await _get_subscription_by_rz_id(db, rz_subscription_id)
    if not subscription:
        _logger.warning(
            "Webhook activated: no subscription found for rz_sub_id=%s",
            rz_subscription_id,
        )
        return

    subscription_entity = payload_entity.get("subscription", {}).get("entity", {})

    # ── Update customer_id if not yet stored ────────────────────────────────
    customer_id = subscription_entity.get("customer_id")
    if customer_id and not subscription.razorpay_customer_id:
        subscription.razorpay_customer_id = customer_id

    # ── Set period dates from Razorpay (not computed locally) ────────────────
    current_start = subscription_entity.get("current_start")
    current_end = subscription_entity.get("current_end")
    now = datetime.now(timezone.utc)

    subscription.status = SubscriptionStatus.ACTIVE
    subscription.current_period_start = datetime.fromtimestamp(current_start, tz=timezone.utc) if current_start else now
    subscription.current_period_end = datetime.fromtimestamp(current_end, tz=timezone.utc) if current_end else None
    subscription.updated_date = now

    # ── Create/update license ────────────────────────────────────────────────
    limits = await _get_plan_limits(db, subscription.plan_id)
    if limits:
        await _upsert_license(
            db,
            subscription=subscription,
            user_id=subscription.user_id,
            limits=limits,
            reset_usage=True,
        )

    # ── Save payment record ──────────────────────────────────────────────────
    payment_entity = payload_entity.get("payment", {}).get("entity", {})
    rz_payment_id = payment_entity.get("id", "")
    amount = payment_entity.get("amount", 0)
    plan_code = subscription.plan.code if subscription.plan else "unknown"

    if rz_payment_id:
        await _save_payment_from_webhook(
            db,
            user_id=subscription.user_id,
            subscription_id=subscription.id,
            rz_subscription_id=rz_subscription_id,
            rz_payment_id=rz_payment_id,
            amount=amount,
            plan_code=plan_code,
        )

    _logger.info("Webhook activated: rz_sub_id=%s status=active", rz_subscription_id)


async def _handle_subscription_charged(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.charged — recurring payment succeeded."""
    subscription = await _get_subscription_by_rz_id(db, rz_subscription_id)
    if not subscription:
        _logger.warning(
            "Webhook charged: no subscription found for rz_sub_id=%s",
            rz_subscription_id,
        )
        return

    subscription_entity = payload_entity.get("subscription", {}).get("entity", {})

    # ── Update customer_id if not yet stored ────────────────────────────────
    customer_id = subscription_entity.get("customer_id")
    if customer_id and not subscription.razorpay_customer_id:
        subscription.razorpay_customer_id = customer_id

    # ── Extend period from Razorpay payload ──────────────────────────────────
    current_start = subscription_entity.get("current_start")
    current_end = subscription_entity.get("current_end")
    now = datetime.now(timezone.utc)

    subscription.status = SubscriptionStatus.ACTIVE
    subscription.current_period_start = datetime.fromtimestamp(current_start, tz=timezone.utc) if current_start else now
    subscription.current_period_end = datetime.fromtimestamp(current_end, tz=timezone.utc) if current_end else None
    subscription.updated_date = now

    # ── Reset monthly quotas and refresh limits ───────────────────────────────
    limits = await _get_plan_limits(db, subscription.plan_id)
    if limits:
        await _upsert_license(
            db,
            subscription=subscription,
            user_id=subscription.user_id,
            limits=limits,
            reset_usage=True,
        )

    # ── Save payment record ──────────────────────────────────────────────────
    payment_entity = payload_entity.get("payment", {}).get("entity", {})
    rz_payment_id = payment_entity.get("id", "")
    amount = payment_entity.get("amount", 0)
    plan_code = subscription.plan.code if subscription.plan else "unknown"

    if rz_payment_id:
        await _save_payment_from_webhook(
            db,
            user_id=subscription.user_id,
            subscription_id=subscription.id,
            rz_subscription_id=rz_subscription_id,
            rz_payment_id=rz_payment_id,
            amount=amount,
            plan_code=plan_code,
        )

    _logger.info(
        "Webhook charged: rz_sub_id=%s period extended to %s",
        rz_subscription_id,
        subscription.current_period_end,
    )


async def _handle_subscription_pending(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.pending — payment failed, Razorpay is retrying."""
    subscription = await _get_subscription_by_rz_id(db, rz_subscription_id)
    if not subscription:
        return

    subscription.status = SubscriptionStatus.PAST_DUE
    subscription.updated_date = datetime.now(timezone.utc)

    _logger.warning(
        "Webhook pending: rz_sub_id=%s status=past_due (payment failed, retrying)",
        rz_subscription_id,
    )


async def _handle_subscription_halted(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.halted — all retry attempts failed."""
    subscription = await _get_subscription_by_rz_id(db, rz_subscription_id)
    if not subscription:
        return

    now = datetime.now(timezone.utc)
    subscription.status = SubscriptionStatus.CANCELED
    subscription.current_period_end = now
    subscription.updated_date = now

    # Revoke license
    await _revoke_license(db, subscription.id, subscription.user_id)

    _logger.warning(
        "Webhook halted: rz_sub_id=%s status=canceled (all retries failed)",
        rz_subscription_id,
    )


async def _handle_subscription_cancelled(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.cancelled — subscription cancelled by user or system."""
    subscription = await _get_subscription_by_rz_id(db, rz_subscription_id)
    if not subscription:
        return

    now = datetime.now(timezone.utc)
    subscription.status = SubscriptionStatus.CANCELED
    subscription.updated_date = now

    # Revoke license
    await _revoke_license(db, subscription.id, subscription.user_id)

    _logger.info(
        "Webhook cancelled: rz_sub_id=%s status=canceled", rz_subscription_id
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main webhook handler
# ─────────────────────────────────────────────────────────────────────────────

# Map of event names to handler functions
_EVENT_HANDLERS = {
    "subscription.authenticated": _handle_subscription_authenticated,
    "subscription.activated": _handle_subscription_activated,
    "subscription.charged": _handle_subscription_charged,
    "subscription.pending": _handle_subscription_pending,
    "subscription.halted": _handle_subscription_halted,
    "subscription.cancelled": _handle_subscription_cancelled,
}


async def handle_subscription_webhook(
    db: AsyncSession,
    *,
    payload_bytes: bytes,
    signature_header: str,
) -> dict[str, str]:
    """Handle an incoming Razorpay subscription webhook event.

    Always returns {"status": "ok"} — errors are logged, never raised.
    """
    # ── 1. Verify webhook signature ──────────────────────────────────────────
    if not _verify_webhook_signature(payload_bytes, signature_header):
        _logger.warning("Subscription webhook signature verification FAILED.")
        return {"status": "ok"}

    # ── 2. Parse JSON payload ────────────────────────────────────────────────
    try:
        payload: dict = json.loads(payload_bytes)
    except Exception as exc:
        _logger.error("Webhook payload JSON parse failed: %s", exc)
        return {"status": "ok"}

    event: str = payload.get("event", "")
    event_id: str = payload.get("id", "")
    _logger.info("Subscription webhook received: event=%s id=%s", event, event_id)

    # ── 3. Route to appropriate handler ──────────────────────────────────────
    handler = _EVENT_HANDLERS.get(event)

    if handler is None:
        _logger.info("Webhook event '%s' not a subscription event — skipping.", event)
        return {"status": "ok", "skipped": True}

    # ── 4. Extract subscription ID from payload ──────────────────────────────
    try:
        payload_data = payload.get("payload", {})
        subscription_entity = payload_data.get("subscription", {}).get("entity", {})
        rz_subscription_id = subscription_entity.get("id", "")

        if not rz_subscription_id:
            _logger.error("Webhook payload missing subscription.entity.id")
            return {"status": "ok"}

    except (KeyError, TypeError) as exc:
        _logger.error("Webhook payload structure error: %s", exc)
        return {"status": "ok"}

    # ── 5. Save event log (idempotency — skip if already processed) ──────────
    if event_id:
        stmt = (
            pg_insert(WebhookEvent)
            .values(
                event_id=event_id,
                event=event,
                rz_sub_id=rz_subscription_id,
                payload=payload,
                processed=False,
            )
            .on_conflict_do_nothing(index_elements=["event_id"])
        )
        result = await db.execute(stmt)
        await db.flush()

        # If nothing was inserted, this event was already processed — skip
        if result.rowcount == 0:
            _logger.info("Webhook event_id=%s already processed — skipping.", event_id)
            return {"status": "ok"}

    # ── 6. Process the event ─────────────────────────────────────────────────
    error_message: Optional[str] = None
    try:
        await handler(db, rz_subscription_id, payload_data)

        # Mark event as successfully processed
        if event_id:
            await db.execute(
                update(WebhookEvent)
                .where(WebhookEvent.event_id == event_id)
                .values(processed=True)
            )

        await db.commit()

    except Exception as exc:
        error_message = str(exc)
        _logger.exception(
            "Webhook handler failed for event=%s rz_sub_id=%s: %s",
            event,
            rz_subscription_id,
            exc,
        )
        await db.rollback()

        # Save error on the event log row in a new transaction
        if event_id:
            try:
                await db.execute(
                    update(WebhookEvent)
                    .where(WebhookEvent.event_id == event_id)
                    .values(error=error_message)
                )
                await db.commit()
            except Exception:
                pass

    return {"status": "ok"}
