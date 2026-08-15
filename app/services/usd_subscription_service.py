"""USD Subscription Service — create recurring subscriptions for customers
outside India.

This runs alongside razorpay_subscription_service, not through it. The INR path
is frozen: an Indian customer's request executes exactly the same code before and
after this module existed. Verification and cancellation are genuinely
currency-agnostic and are shared with the INR routes; only creation differs, and
it differs enough to deserve its own file.

How the two USD discounts differ — they are not the same kind of thing:

    Monthly   50% off the first month.  Promotional and temporary.
              Built from a future start_at plus an upfront amount.

    Annual    Two months off.  Permanent, and already inside the list price:
              annual = 10x monthly, in perpetuity. Nothing to build.

So annual creates a plain subscription starting immediately, with no promo, no
upfront amount and no future start date. Setting annual at 12x monthly and
layering a 2-month promo on top would instead create a first-year discount that
silently expires, hitting a foreign card with a ~20% increase a year later with
no human in the loop — the highest-risk renewal event available.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.geo import INR, USD, is_india
from app.models.plan import Plan
from app.models.plan_price_usd import PlanPriceUsd
from app.models.promo_usd import PromoCodeUsd
from app.models.subscription import Subscription
from app.models.subscription_enums import SubscriptionStatus
from app.services import usd_promo_service
from app.services.billing_currency import get_locked_currency
# Pure HMAC over "{payment_id}|{subscription_id}" — no currency, no DB, nothing
# to fork. Reused rather than reimplemented so the two paths can never disagree
# about what a valid signature is.
from app.services.razorpay_subscription_service import _verify_subscription_signature
from app.services.usd_entitlement import (
    feature_limits,
    get_subscription,
    upsert_usd_license,
)
from app.services.usd_promo_service import PromoRejected
from app.utils.billing_dates import next_period_start, to_razorpay_start_at

_logger = logging.getLogger("rivollo.usd_subscription_service")

# Razorpay caps notes at 15 keys; we stay well under.
_MAX_NOTE_LEN = 250


def _check_credentials() -> None:
    """Raise 503 if Razorpay credentials are not configured."""
    if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
        _logger.error("Razorpay credentials are not configured in settings.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment gateway is not configured. Contact support.",
        )


async def _load_usd_plan(
    db: AsyncSession, plan_code: str, billing_interval: str
) -> tuple[Plan, PlanPriceUsd]:
    """Resolve the tier and its USD price, or fail.

    Never falls through to a default. An unknown tier is a 400, not a silent
    downgrade to Pro monthly — a customer must never be charged for a plan they
    did not pick.
    """
    plan_result = await db.execute(
        select(Plan).where(Plan.code == plan_code, Plan.isactive == True)  # noqa: E712
    )
    plan = plan_result.scalar_one_or_none()
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' not found or is no longer active.",
        )

    price_result = await db.execute(
        select(PlanPriceUsd).where(
            PlanPriceUsd.plan_id == plan.id,
            PlanPriceUsd.billing_interval == billing_interval,
            PlanPriceUsd.isactive.is_(True),
        )
    )
    plan_price = price_result.scalar_one_or_none()
    if plan_price is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' is not available in USD for {billing_interval} billing.",
        )

    if not plan_price.razorpay_plan_id_usd:
        # Never fall back to an INR plan — that would charge a foreign customer
        # in rupees.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' is not yet configured for {billing_interval} USD billing.",
        )

    if plan_price.price_usd <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' is not a paid plan.",
        )

    return plan, plan_price


async def _assert_currency_not_locked_to_inr(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Stop an existing INR customer being moved onto USD rails.

    Currency is locked at first subscription, in both directions. An abandoned
    checkout does not count as a lock — see billing_currency.
    """
    if await get_locked_currency(db, user_id) == INR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account already bills in INR and cannot be switched to USD. "
            "Contact support if you need to change billing currency.",
        )


def _build_notes(
    *,
    user_id: uuid.UUID,
    plan_code: str,
    billing_interval: str,
    promo_code: Optional[str],
    full_amount: int,
    upfront_amount: Optional[int],
) -> dict[str, str]:
    notes = {
        "user_id": str(user_id),
        "plan_code": plan_code,
        "billing_interval": billing_interval,
        "currency": USD,
        "full_amount": str(full_amount),
    }
    if promo_code:
        notes["promo_code"] = promo_code[:_MAX_NOTE_LEN]
    if upfront_amount is not None:
        notes["upfront_amount"] = str(upfront_amount)
    return notes


async def _call_razorpay(payload: dict[str, Any]) -> dict[str, Any]:
    """POST a subscription to Razorpay, mapping failures to useful errors."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{settings.RAZORPAY_BASE_URL}/subscriptions",
                json=payload,
                auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET),
            )

        if response.status_code == 400:
            error_detail = response.json().get("error", {}).get("description", "Bad request")
            _logger.warning("Razorpay USD subscription creation bad request: %s", error_detail)
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
        return response.json()

    except HTTPException:
        raise
    except Exception as exc:
        _logger.exception("Razorpay USD subscription creation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment gateway error. Please try again later.",
        )


async def create_usd_subscription(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    plan_code: str,
    billing_interval: str,
    promo_code: Optional[str] = None,
    checkout_country: Optional[str] = None,
) -> dict[str, Any]:
    """Create a USD Razorpay subscription for the user.

    ``checkout_country`` must come from Cloudflare's own header, never from a
    header a caller could set.
    """
    _check_credentials()

    # ── 1. India never bills in USD ──────────────────────────────────────────
    # Unreachable through the normal UI, since an Indian visitor is never shown
    # the USD path. Enforced anyway: RBI requires domestic transactions in INR,
    # and the server does not trust the client to have got that right.
    if is_india(checkout_country):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customers in India are billed in INR.",
        )

    await _assert_currency_not_locked_to_inr(db, user_id)

    # ── 2. Resolve the price server-side ─────────────────────────────────────
    plan, plan_price = await _load_usd_plan(db, plan_code, billing_interval)
    full_amount = plan_price.price_usd

    promo: Optional[PromoCodeUsd] = None
    upfront_amount: Optional[int] = None
    start_at_dt: Optional[datetime] = None
    now = datetime.now(timezone.utc)

    if billing_interval == "yearly":
        # Annual carries no promo mechanism at all: its discount is permanent
        # and already in the list price. A code submitted against annual is
        # rejected loudly rather than ignored — a customer who types a code,
        # sees it accepted and is then charged full price has a real complaint.
        if promo_code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Promo codes do not apply to annual plans — the annual price "
                "already includes two months free.",
            )
    else:
        try:
            promo = await usd_promo_service.resolve_promo_for_checkout(
                db,
                user_id=user_id,
                plan_code=plan_code,
                billing_interval=billing_interval,
                submitted_code=promo_code,
            )
        except PromoRejected as exc:
            if promo_code:
                # The customer typed this code. Tell them why it failed.
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=exc.reason
                )
            # We tried to auto-apply the advertised promo and could not. Charge
            # full price, but never silently — this is a pricing-page/checkout
            # mismatch and someone needs to see it.
            _logger.error(
                "Public USD promo could not be applied for user=%s plan=%s: %s",
                user_id,
                plan_code,
                exc.reason,
            )
            promo = None

        # compute_upfront_amount raises PromoRejected on a misconfigured
        # discount_type, so it stays inside a handler — outside one it would
        # surface as a 500 with no reason the customer can act on.
        try:
            upfront_amount = usd_promo_service.compute_upfront_amount(full_amount, promo)
        except PromoRejected as exc:
            _logger.error(
                "USD promo %s is misconfigured: %s",
                promo.code if promo else "<none>",
                exc.reason,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=exc.reason
            )

        usd_promo_service.assert_within_guard_rails(
            list_amount_minor=full_amount,
            upfront_amount_minor=upfront_amount,
            billing_interval=billing_interval,
        )

        # Razorpay treats the gap before start_at as a trial: the customer is
        # charged the upfront amount now, gets a full period, and the plan
        # amount begins at start_at. Promo and non-promo monthly are created
        # identically — only the upfront number differs.
        start_at_dt = next_period_start(now, billing_interval)

    # ── 3. Create the subscription at Razorpay ───────────────────────────────
    payload: dict[str, Any] = {
        "plan_id": plan_price.razorpay_plan_id_usd,
        "total_count": plan_price.total_count,
        "customer_notify": 1,
        "notes": _build_notes(
            user_id=user_id,
            plan_code=plan_code,
            billing_interval=billing_interval,
            promo_code=promo.code if promo else None,
            full_amount=full_amount,
            upfront_amount=upfront_amount,
        ),
    }

    if start_at_dt is not None:
        payload["start_at"] = to_razorpay_start_at(start_at_dt)
        payload["addons"] = [
            {
                "item": {
                    "name": f"{plan.name} — first month",
                    "amount": upfront_amount,
                    # Inherits nothing from the plan: state it explicitly.
                    "currency": USD,
                }
            }
        ]

    rz_sub = await _call_razorpay(payload)

    rz_subscription_id = rz_sub["id"]
    rz_customer_id = rz_sub.get("customer_id")
    rz_status = rz_sub.get("status", "created")
    rz_short_url = rz_sub.get("short_url")

    # ── 4. Persist locally BEFORE the customer reaches checkout ──────────────
    # Writing only on webhook receipt would leave a paying customer with no
    # account state for as long as the webhook is delayed.
    # Only an abandoned attempt is recycled. Overwriting a row that was actually
    # paid for would repoint razorpay_subscription_id at the new subscription and
    # orphan the live one — which keeps charging the card while its webhooks no
    # longer match any row. It would also flip the row to PENDING, which silently
    # releases the currency lock and makes the customer "new" again, re-earning
    # the first-month discount on every abandoned checkout.
    existing_result = await db.execute(
        select(Subscription)
        .where(
            Subscription.user_id == user_id,
            Subscription.status == SubscriptionStatus.PENDING,
        )
        .order_by(Subscription.created_date.desc())
        .limit(1)
    )
    existing_sub = existing_result.scalars().first()

    if existing_sub is not None:
        subscription = existing_sub
        subscription.plan_id = plan.id
        subscription.status = SubscriptionStatus.PENDING
        subscription.razorpay_subscription_id = rz_subscription_id
        # Only overwrite when Razorpay actually returned one: it omits
        # customer_id for subscriptions created without a customer, and blindly
        # assigning that would wipe a customer ID we already hold.
        if rz_customer_id:
            subscription.razorpay_customer_id = rz_customer_id
        subscription.billing_interval = billing_interval
        # Period dates belong to the abandoned attempt, not this one.
        subscription.current_period_start = None
        subscription.current_period_end = None
        subscription.updated_by = user_id
        subscription.updated_date = now
    else:
        subscription = Subscription(
            user_id=user_id,
            plan_id=plan.id,
            status=SubscriptionStatus.PENDING,
            seats_purchased=1,
            razorpay_subscription_id=rz_subscription_id,
            razorpay_customer_id=rz_customer_id,
            billing_interval=billing_interval,
            created_by=user_id,
        )
        db.add(subscription)

    # Every USD-relevant field is set explicitly, so reusing a row left over
    # from an abandoned attempt cannot carry stale values forward. offer_id is
    # cleared because it belongs to the INR Razorpay-Offer flow, which the USD
    # path never uses — leaving it set would misreport this subscription.
    subscription.currency = USD
    subscription.offer_id = None
    subscription.billing_country = checkout_country
    subscription.promo_code = promo.code if promo else None
    subscription.full_amount = full_amount
    # Stored, never recomputed from the percentage later: a future price change
    # would otherwise make historical records lie about what was charged.
    subscription.upfront_amount = upfront_amount
    subscription.start_at = start_at_dt
    subscription.promo_period_active = False

    # The redemption is NOT counted here. A subscription row at this point only
    # means checkout was opened, and counting it now would let abandoned
    # checkouts burn a promo's max_redemptions without anyone ever paying. It is
    # counted when the upfront payment is actually captured, in the webhook's
    # subscription.authenticated handler.

    await db.commit()

    _logger.info(
        "USD subscription created: rz_sub_id=%s user=%s plan=%s interval=%s "
        "full=%s upfront=%s promo=%s start_at=%s",
        rz_subscription_id,
        user_id,
        plan_code,
        billing_interval,
        full_amount,
        upfront_amount,
        promo.code if promo else None,
        start_at_dt,
    )

    return {
        "subscriptionId": rz_subscription_id,
        "planCode": plan_code,
        "keyId": settings.RAZORPAY_KEY_ID,
        "status": rz_status,
        "shortUrl": rz_short_url,
        "currency": USD,
        "billingInterval": billing_interval,
        "fullAmount": full_amount,
        "upfrontAmount": upfront_amount,
        "promoCode": promo.code if promo else None,
        "firstChargeAt": start_at_dt,
    }


async def verify_usd_subscription(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    razorpay_payment_id: str,
    razorpay_subscription_id: str,
    razorpay_signature: str,
) -> dict[str, Any]:
    """Verify a completed USD checkout and activate the subscription.

    Separate from the INR verify rather than shared, despite doing the same job,
    because the INR one resolves entitlements through tbl_plan_prices. For a USD
    subscription that would apply INR credit limits, and would raise a 400 —
    *after* the customer's card had been charged — for any plan without an
    active INR row.

    Usage counters are deliberately NOT reset here. This endpoint is driven by
    checkout-callback values the customer holds and has no replay protection, so
    resetting quotas would let anyone refill their own AI credits on demand. New
    periods are the webhook's job.
    """
    _check_credentials()

    if not _verify_subscription_signature(
        razorpay_payment_id=razorpay_payment_id,
        razorpay_subscription_id=razorpay_subscription_id,
        razorpay_signature=razorpay_signature,
    ):
        _logger.warning(
            "USD subscription signature verification FAILED for rz_sub_id=%s",
            razorpay_subscription_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment signature verification failed. The payment may not be genuine.",
        )

    subscription = await get_subscription(
        db, razorpay_subscription_id, user_id=user_id
    )
    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found for this user.",
        )

    subscription.status = SubscriptionStatus.ACTIVE
    subscription.updated_date = datetime.now(timezone.utc)

    await upsert_usd_license(
        db,
        subscription=subscription,
        limits=feature_limits(subscription),
        reset_usage=False,
    )
    await db.commit()

    _logger.info(
        "USD subscription verified: rz_sub_id=%s user=%s",
        razorpay_subscription_id,
        user_id,
    )

    return {
        "verified": True,
        "message": "Payment verified. Your subscription is now active!",
        "plan": subscription.plan.code if subscription.plan else "pro",
        "subscriptionId": str(subscription.id),
        "periodEnd": subscription.current_period_end,
    }


async def validate_usd_promo_code(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    plan_code: str,
    billing_interval: str,
    code: str,
) -> dict[str, Any]:
    """Check a promo code before checkout, for the code-entry UI.

    Uses the same resolution as checkout, so a code that validates here is a
    code that applies there.
    """
    _, plan_price = await _load_usd_plan(db, plan_code, billing_interval)

    try:
        promo = await usd_promo_service.resolve_promo_for_checkout(
            db,
            user_id=user_id,
            plan_code=plan_code,
            billing_interval=billing_interval,
            submitted_code=code,
        )
    except PromoRejected as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.reason)

    if promo is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This promo code was not recognised.",
        )

    # Misconfigured discount_type raises PromoRejected; surface it as a 400 with
    # a reason rather than a bare 500.
    try:
        upfront_amount = usd_promo_service.compute_upfront_amount(
            plan_price.price_usd, promo
        )
    except PromoRejected as exc:
        _logger.error("USD promo %s is misconfigured: %s", promo.code, exc.reason)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.reason)

    usd_promo_service.assert_within_guard_rails(
        list_amount_minor=plan_price.price_usd,
        upfront_amount_minor=upfront_amount,
        billing_interval=billing_interval,
    )

    return {
        "valid": True,
        "code": promo.code,
        "currency": USD,
        "fullAmount": plan_price.price_usd,
        "upfrontAmount": upfront_amount,
        "description": promo.description,
    }
