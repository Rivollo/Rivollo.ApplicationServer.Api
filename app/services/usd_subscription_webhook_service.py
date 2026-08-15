"""USD subscription webhook handling.

Reached from exactly one guard in subscription_webhook_service, placed after that
module verifies the signature and extracts the subscription ID. Everything below
that guard is the existing INR logic and is not touched; an INR payload never
enters this module.

The one behaviour that genuinely differs from INR, and the reason this file
exists rather than a currency flag on the existing handlers:

    A monthly USD subscription is created with a start date one month out, so
    Razorpay holds it in `authenticated` — not `active` — until the first
    full-price charge. The customer has already paid the upfront amount and is
    inside the period they paid for. A handler that grants entitlement only on
    `activated` or `charged`, as the INR handlers do, would leave every promo
    customer paid-up and locked out for a full month.

So `subscription.authenticated` grants entitlement here. That is the difference.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.geo import USD
from app.models.payment import Payment, PaymentStatus
from app.models.subscription import Subscription
from app.models.subscription_enums import SubscriptionStatus
from app.services import usd_promo_service, webhook_inbox
from app.services.usd_entitlement import (
    feature_limits,
    get_subscription,
    upsert_usd_license,
)

# Reused unchanged from the INR handler: these three only flip a status and
# revoke a licence. Nothing in them reads an amount or a currency, so there is
# nothing to fork.
from app.services.subscription_webhook_service import (
    _handle_subscription_cancelled,
    _handle_subscription_halted,
    _handle_subscription_pending,
)

_logger = logging.getLogger("rivollo.usd_subscription_webhook_service")


async def _save_usd_payment(
    db: AsyncSession,
    *,
    subscription: Subscription,
    rz_subscription_id: str,
    payment_entity: dict,
) -> None:
    """Record a USD payment. Idempotent on the Razorpay payment ID."""
    rz_payment_id = payment_entity.get("id", "")
    if not rz_payment_id:
        return

    order_id_key = f"webhook_{rz_subscription_id}_{rz_payment_id}"
    existing = await db.execute(
        select(Payment).where(Payment.razorpay_order_id == order_id_key)
    )
    if existing.scalar_one_or_none() is not None:
        return

    db.add(
        Payment(
            user_id=subscription.user_id,
            subscription_id=subscription.id,
            razorpay_order_id=order_id_key,
            razorpay_payment_id=rz_payment_id,
            razorpay_signature="webhook",
            razorpay_subscription_id=rz_subscription_id,
            amount=payment_entity.get("amount", 0),
            # The whole reason this is not _save_payment_from_webhook: that one
            # hardcodes INR.
            currency=payment_entity.get("currency") or USD,
            plan_code=subscription.plan.code if subscription.plan else "unknown",
            status=PaymentStatus.CAPTURED,
        )
    )
    await db.flush()


def _period_dates(subscription_entity: dict) -> tuple[Optional[datetime], Optional[datetime]]:
    start = subscription_entity.get("current_start")
    end = subscription_entity.get("current_end")
    return (
        datetime.fromtimestamp(start, tz=timezone.utc) if start else None,
        datetime.fromtimestamp(end, tz=timezone.utc) if end else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers
# ─────────────────────────────────────────────────────────────────────────────


async def _handle_usd_authenticated(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.authenticated — the upfront amount has been captured.

    This is where a USD customer's access begins. With a future start date
    Razorpay leaves the subscription in `authenticated` until the first
    full-price charge, which for monthly is a month away, so waiting for
    `activated` would lock out a customer who has already paid.
    """
    subscription = await get_subscription(db, rz_subscription_id)
    if not subscription:
        _logger.warning(
            "USD webhook authenticated: no subscription for rz_sub_id=%s", rz_subscription_id
        )
        return

    # Annual is authorised and charged in one go, so `activated` follows within
    # seconds and is the event that proves payment. Granting here instead would
    # hand out a year of Pro on mandate authorisation alone — and if the charge
    # then failed, `pending` only marks the subscription PAST_DUE without
    # revoking, leaving permanent unpaid access.
    if subscription.start_at is None:
        _logger.info(
            "USD webhook authenticated: rz_sub_id=%s has no trial period; "
            "deferring entitlement to activated/charged.",
            rz_subscription_id,
        )
        return

    # Monthly carries an upfront addon charged at this transaction. Grant only
    # against evidence that money actually moved: a captured payment entity.
    # If Razorpay ever stops including it here, the customer is still granted by
    # `charged` when the first real payment lands — late access is recoverable,
    # unpaid access is not.
    payment_entity = payload_entity.get("payment", {}).get("entity", {})
    captured_amount = payment_entity.get("amount")
    payment_status = payment_entity.get("status")

    if not payment_entity or payment_status != "captured" or not captured_amount:
        _logger.warning(
            "USD webhook authenticated: rz_sub_id=%s has no captured payment "
            "(status=%s amount=%s) — withholding entitlement until charged.",
            rz_subscription_id,
            payment_status,
            captured_amount,
        )
        return

    if subscription.upfront_amount is not None and int(captured_amount) != int(
        subscription.upfront_amount
    ):
        # The card WAS charged, just not the amount expected. Withholding access
        # over a reconciliation discrepancy turns it into a support ticket and a
        # chargeback, so grant and alert loudly instead.
        _logger.error(
            "USD upfront mismatch rz_sub_id=%s captured=%s expected=%s — "
            "granting entitlement and flagging for reconciliation.",
            rz_subscription_id,
            captured_amount,
            subscription.upfront_amount,
        )

    now = datetime.now(timezone.utc)
    subscription.status = SubscriptionStatus.ACTIVE
    subscription.updated_date = now

    # The paid-for period runs from now until the first full-price charge.
    subscription.current_period_start = now
    subscription.current_period_end = subscription.start_at
    # Only a discounted first period is a promo period. Monthly without a promo
    # also starts a period out, but at full price — flagging that as promotional
    # would misreport it.
    subscription.promo_period_active = subscription.promo_code is not None

    # Count the redemption now that the money is actually captured. This handler
    # is idempotent on the Razorpay event ID, so a replayed event cannot
    # double-count it.
    if subscription.promo_code:
        await usd_promo_service.record_redemption_by_code(db, subscription.promo_code)

    await upsert_usd_license(
        db, subscription=subscription, limits=feature_limits(subscription), reset_usage=True
    )

    if payment_entity:
        await _save_usd_payment(
            db,
            subscription=subscription,
            rz_subscription_id=rz_subscription_id,
            payment_entity=payment_entity,
        )

    _logger.info(
        "USD webhook authenticated: rz_sub_id=%s entitlement granted, promo_period=%s",
        rz_subscription_id,
        subscription.promo_period_active,
    )


async def _handle_usd_activated(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.activated — the first full-price cycle has begun."""
    subscription = await get_subscription(db, rz_subscription_id)
    if not subscription:
        _logger.warning(
            "USD webhook activated: no subscription for rz_sub_id=%s", rz_subscription_id
        )
        return

    subscription_entity = payload_entity.get("subscription", {}).get("entity", {})
    customer_id = subscription_entity.get("customer_id")
    if customer_id and not subscription.razorpay_customer_id:
        subscription.razorpay_customer_id = customer_id

    now = datetime.now(timezone.utc)
    start, end = _period_dates(subscription_entity)

    subscription.status = SubscriptionStatus.ACTIVE
    subscription.current_period_start = start or now
    subscription.current_period_end = end
    # The discounted period is over; the customer is now on list price.
    subscription.promo_period_active = False
    subscription.updated_date = now

    await upsert_usd_license(
        db, subscription=subscription, limits=feature_limits(subscription), reset_usage=True
    )
    await _save_usd_payment(
        db,
        subscription=subscription,
        rz_subscription_id=rz_subscription_id,
        payment_entity=payload_entity.get("payment", {}).get("entity", {}),
    )

    _logger.info("USD webhook activated: rz_sub_id=%s promo_period cleared", rz_subscription_id)


async def _handle_usd_charged(
    db: AsyncSession, rz_subscription_id: str, payload_entity: dict
) -> None:
    """subscription.charged — a recurring full-price payment succeeded."""
    subscription = await get_subscription(db, rz_subscription_id)
    if not subscription:
        _logger.warning(
            "USD webhook charged: no subscription for rz_sub_id=%s", rz_subscription_id
        )
        return

    subscription_entity = payload_entity.get("subscription", {}).get("entity", {})
    customer_id = subscription_entity.get("customer_id")
    if customer_id and not subscription.razorpay_customer_id:
        subscription.razorpay_customer_id = customer_id

    now = datetime.now(timezone.utc)
    start, end = _period_dates(subscription_entity)

    subscription.status = SubscriptionStatus.ACTIVE
    subscription.current_period_start = start or now
    subscription.current_period_end = end
    subscription.promo_period_active = False
    subscription.updated_date = now

    await upsert_usd_license(
        db, subscription=subscription, limits=feature_limits(subscription), reset_usage=True
    )
    await _save_usd_payment(
        db,
        subscription=subscription,
        rz_subscription_id=rz_subscription_id,
        payment_entity=payload_entity.get("payment", {}).get("entity", {}),
    )

    _logger.info(
        "USD webhook charged: rz_sub_id=%s period extended to %s",
        rz_subscription_id,
        subscription.current_period_end,
    )


_USD_EVENT_HANDLERS = {
    "subscription.authenticated": _handle_usd_authenticated,
    "subscription.activated": _handle_usd_activated,
    "subscription.charged": _handle_usd_charged,
    # Status-only transitions, reused from the INR handler unchanged.
    "subscription.pending": _handle_subscription_pending,
    "subscription.halted": _handle_subscription_halted,
    "subscription.cancelled": _handle_subscription_cancelled,
}


async def handle_usd_subscription_event(
    db: AsyncSession,
    *,
    event: str,
    event_id: str,
    rz_subscription_id: str,
    payload: dict,
    payload_data: dict,
) -> dict[str, str]:
    """Process a USD subscription webhook event.

    Mirrors the INR handler's contract: always returns {"status": "ok"}, never
    raises, and is idempotent — a replayed event is recognised by its Razorpay
    event ID and skipped, so entitlement is never granted twice.
    """
    handler = _USD_EVENT_HANDLERS.get(event)
    if handler is None:
        _logger.info("USD webhook event '%s' has no handler — skipping.", event)
        return {"status": "ok", "skipped": True}

    effective_event_id = event_id or f"{event}_{rz_subscription_id}"

    # Committed before processing, so a handler failure cannot roll away the
    # record that this event arrived. See webhook_inbox.
    event_row = await webhook_inbox.claim_event(
        db,
        event_id=effective_event_id,
        event=event,
        rz_subscription_id=rz_subscription_id,
        payload=payload,
    )
    if event_row is None:
        return {"status": "ok"}

    try:
        await handler(db, rz_subscription_id, payload_data)
        await webhook_inbox.mark_processed(db, event_row)
    except Exception as exc:
        _logger.exception(
            "USD webhook handler failed for event=%s rz_sub_id=%s: %s",
            event,
            rz_subscription_id,
            exc,
        )
        await webhook_inbox.record_failure(
            db, event_id=effective_event_id, error=str(exc)
        )

    return {"status": "ok"}
