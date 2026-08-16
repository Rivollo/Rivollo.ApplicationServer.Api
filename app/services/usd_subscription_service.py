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
from app.models.plan import Plan, PlanPrice
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
# tbl_plan_prices.price_inr holds WHOLE units of the row's currency — rupees on
# an INR row, dollars on a USD one. Razorpay is told amounts in minor units, so
# every read of it converts here rather than storing cents in a column the INR
# path reads as rupees.
from app.utils.money import to_minor_units

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
) -> tuple[Plan, PlanPrice]:
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
        select(PlanPrice).where(
            PlanPrice.plan_id == plan.id,
            PlanPrice.billing_interval == billing_interval,
            # The row that is priced in dollars, never the rupee row for the
            # same plan and interval.
            PlanPrice.currency == USD,
            PlanPrice.isactive.is_(True),
        )
    )
    plan_price = price_result.scalar_one_or_none()
    if plan_price is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' is not available in USD for {billing_interval} billing.",
        )

    if not plan_price.razorpay_plan_id:
        # Never fall back to an INR plan — that would charge a foreign customer
        # in rupees.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' is not yet configured for {billing_interval} USD billing.",
        )

    if plan_price.price_inr <= 0:
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
    full_amount = to_minor_units(plan_price.price_inr, USD)

    now = datetime.now(timezone.utc)

    # ── 2a. No introductory pricing, on any interval ─────────────────────────
    # The USD promo mechanism is gone, and with it the architecture it required.
    # Monthly used to be created with a future start_at and an upfront addon:
    # Razorpay treats the gap before start_at as a trial, so the customer was
    # charged a discounted amount immediately and the plan amount began a month
    # later. That was built to deliver a first-month discount, and it was
    # applied to promo and non-promo checkouts alike — every USD monthly
    # subscription carried it, whether or not anything was actually discounted.
    #
    # The cost of that was not theoretical. A subscription with a future
    # start_at sits in Razorpay's `authenticated` state with no billing cycle
    # running, which meant it could not be cancelled at cycle end — Razorpay
    # rejects the request outright — and it forced entitlement to be granted
    # from the `authenticated` webhook rather than the ordinary `activated` /
    # `charged` path, because the customer had paid but the subscription would
    # not be active for a month.
    #
    # Monthly is now created exactly like annual: one price, charged
    # immediately, plan active from the first payment. No start_at, no addon,
    # no upfront amount, no trial gap, and no state where paid and active
    # disagree.
    if promo_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Promo codes are not available on this plan.",
        )

    # ── 3. Create the subscription at Razorpay ───────────────────────────────
    payload: dict[str, Any] = {
        "plan_id": plan_price.razorpay_plan_id,
        "total_count": plan_price.total_count,
        "customer_notify": 1,
        "notes": _build_notes(
            user_id=user_id,
            plan_code=plan_code,
            billing_interval=billing_interval,
            promo_code=None,
            full_amount=full_amount,
            upfront_amount=None,
        ),
    }

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
    subscription.full_amount = full_amount
    # Explicitly cleared rather than left alone. This row may be a recycled
    # abandoned attempt from before the intro was removed, and a stale promo
    # code or upfront amount carried forward would misreport what was charged.
    subscription.promo_code = None
    subscription.upfront_amount = None
    subscription.start_at = None
    subscription.promo_period_active = False

    await db.commit()

    _logger.info(
        "USD subscription created: rz_sub_id=%s user=%s plan=%s interval=%s "
        "amount=%s",
        rz_subscription_id,
        user_id,
        plan_code,
        billing_interval,
        full_amount,
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
            to_minor_units(plan_price.price_inr, USD), promo
        )
    except PromoRejected as exc:
        _logger.error("USD promo %s is misconfigured: %s", promo.code, exc.reason)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.reason)

    usd_promo_service.assert_within_guard_rails(
        list_amount_minor=to_minor_units(plan_price.price_inr, USD),
        upfront_amount_minor=upfront_amount,
        billing_interval=billing_interval,
    )

    return {
        "valid": True,
        "code": promo.code,
        "currency": USD,
        "fullAmount": to_minor_units(plan_price.price_inr, USD),
        "upfrontAmount": upfront_amount,
        "description": promo.description,
    }
