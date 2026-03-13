"""Razorpay Subscription Service — create, verify, cancel recurring subscriptions.

Architecture:
    - Uses httpx directly to call Razorpay REST API (no SDK needed).
    - create_subscription: creates a Razorpay subscription with optional offer_id.
    - verify_subscription: verifies HMAC signature after checkout.
    - cancel_subscription: cancels via Razorpay API.
    - All database operations committed in the route layer.

Security:
    - Amount is NEVER taken from frontend — plan price is on Razorpay's side.
    - HMAC-SHA256 signature verification for payment authenticity.
    - Promo/offer handling is fully managed by Razorpay — no DB storage needed.
"""

import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.license_assignment import LicenseAssignment
from app.models.payment import Payment, PaymentStatus
from app.models.plan import Plan, PlanFeature
from app.models.subscription import Subscription
from app.models.subscription_enums import LicenseStatus, SubscriptionStatus

_logger = logging.getLogger("rivollo.razorpay_subscription_service")

_PERIOD_DAYS = {"monthly": 30, "yearly": 365}
_TOTAL_COUNT = {"monthly": 12, "yearly": 5}


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


def _verify_subscription_signature(
    razorpay_payment_id: str,
    razorpay_subscription_id: str,
    razorpay_signature: str,
) -> bool:
    """Verify Razorpay subscription HMAC-SHA256 signature.

    For subscriptions, Razorpay signs: "{payment_id}|{subscription_id}"
    (different from orders which sign: "{order_id}|{payment_id}")
    """
    msg = f"{razorpay_payment_id}|{razorpay_subscription_id}"
    expected = hmac.new(
        key=settings.RAZORPAY_KEY_SECRET.encode("utf-8"),
        msg=msg.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, razorpay_signature)


async def _get_plan_with_features(
    db: AsyncSession, plan_code: str
) -> tuple[Plan, dict]:
    """Fetch plan by code with its feature limits."""
    result = await db.execute(
        select(Plan)
        .where(Plan.code == plan_code, Plan.isactive == True)
        .options(selectinload(Plan.plan_features).selectinload(PlanFeature.feature))
    )
    plan = result.scalar_one_or_none()

    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' not found or is no longer active.",
        )

    limits = {}
    for pf in plan.plan_features:
        if pf.feature and pf.limit_value is not None:
            limits[pf.feature.code] = pf.limit_value

    return plan, limits


async def _upsert_license(
    db: AsyncSession,
    *,
    subscription: Subscription,
    user_id: uuid.UUID,
    limits: dict,
    reset_usage: bool = False,
) -> LicenseAssignment:
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
        license_obj = existing
    else:
        license_obj = LicenseAssignment(
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
        )
        db.add(license_obj)

    await db.flush()
    return license_obj


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


async def create_subscription(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    plan_code: str,
    billing_interval: str = "monthly",
    offer_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a Razorpay subscription for the user.

    Flow:
        1. Validate plan_code and get razorpay_plan_id from DB
        2. Call Razorpay API to create subscription (with offer_id if provided)
        3. Razorpay validates the offer — if invalid, returns an error
        4. Save subscription row in DB
        5. Return subscription_id and key_id for frontend checkout
    """
    _check_credentials()

    # ── 1. Get plan from DB ──────────────────────────────────────────────────
    plan, limits = await _get_plan_with_features(db, plan_code)

    # ── Pick the correct Razorpay plan ID based on billing interval ─────────
    if billing_interval == "yearly":
        razorpay_plan_id = plan.razorpay_plan_id_yearly
    else:
        razorpay_plan_id = plan.razorpay_plan_id

    if not razorpay_plan_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan '{plan_code}' is not configured for {billing_interval} billing.",
        )

    period_days = _PERIOD_DAYS[billing_interval]
    total_count = _TOTAL_COUNT[billing_interval]

    # ── 2. Call Razorpay API — create subscription ───────────────────────────
    payload: dict[str, Any] = {
        "plan_id": razorpay_plan_id,
        "total_count": total_count,
        "customer_notify": 1,
    }

    if offer_id:
        payload["offer_id"] = offer_id

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{settings.RAZORPAY_BASE_URL}/subscriptions",
                json=payload,
                auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET),
            )

        if response.status_code == 400:
            error_detail = (
                response.json().get("error", {}).get("description", "Bad request")
            )
            _logger.warning("Razorpay subscription creation bad request: %s", error_detail)
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
        rz_sub = response.json()

    except HTTPException:
        raise
    except Exception as exc:
        _logger.exception("Razorpay subscription creation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment gateway error. Please try again later.",
        )

    rz_subscription_id = rz_sub["id"]
    rz_customer_id = rz_sub.get("customer_id")
    rz_status = rz_sub.get("status", "created")
    rz_short_url = rz_sub.get("short_url")

    # ── 3. Save subscription row in DB ────────────────────────────────────────
    now = datetime.now(timezone.utc)

    # Check for existing subscription for this user
    existing_result = await db.execute(
        select(Subscription)
        .where(Subscription.user_id == user_id)
        .order_by(Subscription.created_date.desc())
        .limit(1)
    )
    existing_sub = existing_result.scalar_one_or_none()

    billing_data = {
        "razorpay_subscription_id": rz_subscription_id,
        "plan_code": plan_code,
        "billing_interval": billing_interval,
        "offer_id": offer_id,
        "created_at": now.isoformat(),
    }

    if existing_sub is not None:
        # Update existing subscription row
        existing_sub.plan_id = plan.id
        existing_sub.status = SubscriptionStatus.ACTIVE
        existing_sub.razorpay_subscription_id = rz_subscription_id
        existing_sub.razorpay_customer_id = rz_customer_id
        existing_sub.current_period_start = now
        existing_sub.current_period_end = now + timedelta(days=period_days)
        existing_sub.billing = json.dumps(billing_data)
        existing_sub.updated_by = user_id
        existing_sub.updated_date = now
        subscription = existing_sub
    else:
        # Create new subscription row
        subscription = Subscription(
            user_id=user_id,
            plan_id=plan.id,
            status=SubscriptionStatus.ACTIVE,
            seats_purchased=1,
            razorpay_subscription_id=rz_subscription_id,
            razorpay_customer_id=rz_customer_id,
            current_period_start=now,
            current_period_end=now + timedelta(days=period_days),
            billing=json.dumps(billing_data),
            created_by=user_id,
        )
        db.add(subscription)
        await db.flush()
        await db.refresh(subscription)

    # ── 4. Create/update license assignment ───────────────────────────────────
    await _upsert_license(
        db,
        subscription=subscription,
        user_id=user_id,
        limits=limits,
        reset_usage=(existing_sub is None),
    )

    await db.commit()

    _logger.info(
        "Razorpay subscription created: rz_sub_id=%s user=%s plan=%s",
        rz_subscription_id,
        user_id,
        plan_code,
    )

    return {
        "subscriptionId": rz_subscription_id,
        "planCode": plan_code,
        "keyId": settings.RAZORPAY_KEY_ID,
        "status": rz_status,
        "shortUrl": rz_short_url,
    }


async def verify_subscription(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    razorpay_payment_id: str,
    razorpay_subscription_id: str,
    razorpay_signature: str,
) -> dict[str, Any]:
    """Verify Razorpay subscription payment signature and confirm activation.

    Flow:
        1. Verify HMAC-SHA256 signature (msg = payment_id|subscription_id)
        2. Find subscription in DB by razorpay_subscription_id
        3. Confirm subscription is active
        4. Save payment record
        5. Return verification result
    """
    _check_credentials()

    # ── 1. Verify signature ──────────────────────────────────────────────────
    is_valid = _verify_subscription_signature(
        razorpay_payment_id=razorpay_payment_id,
        razorpay_subscription_id=razorpay_subscription_id,
        razorpay_signature=razorpay_signature,
    )

    if not is_valid:
        _logger.warning(
            "Subscription signature verification FAILED for rz_sub_id: %s",
            razorpay_subscription_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment signature verification failed. The payment may not be genuine.",
        )

    # ── 2. Find subscription in DB ───────────────────────────────────────────
    result = await db.execute(
        select(Subscription).where(
            Subscription.razorpay_subscription_id == razorpay_subscription_id,
            Subscription.user_id == user_id,
        )
    )
    subscription = result.scalar_one_or_none()

    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found for this user.",
        )

    # ── 3. Confirm subscription is active ────────────────────────────────────
    now = datetime.now(timezone.utc)
    billing_info = json.loads(subscription.billing) if subscription.billing else {}
    billing_interval = billing_info.get("billing_interval", "monthly")
    period_days = _PERIOD_DAYS.get(billing_interval, 30)

    subscription.status = SubscriptionStatus.ACTIVE
    subscription.current_period_start = now
    subscription.current_period_end = now + timedelta(days=period_days)
    subscription.updated_date = now

    # ── 4. Save payment record ───────────────────────────────────────────────
    payment = Payment(
        user_id=user_id,
        subscription_id=subscription.id,
        razorpay_order_id=f"sub_{razorpay_subscription_id}_{razorpay_payment_id}",
        razorpay_payment_id=razorpay_payment_id,
        razorpay_signature=razorpay_signature,
        razorpay_subscription_id=razorpay_subscription_id,
        amount=0,  # actual amount comes from webhook
        currency="INR",
        plan_code=subscription.billing and json.loads(subscription.billing).get("plan_code", "pro") or "pro",
        status=PaymentStatus.CAPTURED,
    )
    db.add(payment)

    await db.commit()

    _logger.info(
        "Subscription verified: rz_sub_id=%s user=%s",
        razorpay_subscription_id,
        user_id,
    )

    return {
        "verified": True,
        "message": "Payment verified. Your subscription is now active!",
        "plan": subscription.billing and json.loads(subscription.billing).get("plan_code", "pro") or "pro",
        "subscriptionId": str(subscription.id),
        "periodEnd": subscription.current_period_end,
    }


async def cancel_subscription(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    cancel_at_cycle_end: bool = True,
) -> dict[str, Any]:
    """Cancel a user's Razorpay subscription.

    Flow:
        1. Find the user's active subscription with razorpay_subscription_id
        2. Call Razorpay API to cancel
        3. Update DB status
    """
    _check_credentials()

    # ── 1. Find active subscription ──────────────────────────────────────────
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.status.in_([
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.PAST_DUE,
            ]),
            Subscription.razorpay_subscription_id.isnot(None),
        )
    )
    subscription = result.scalar_one_or_none()

    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found to cancel.",
        )

    rz_subscription_id = subscription.razorpay_subscription_id

    # ── 2. Call Razorpay cancel API ──────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{settings.RAZORPAY_BASE_URL}/subscriptions/{rz_subscription_id}/cancel",
                json={"cancel_at_cycle_end": 1 if cancel_at_cycle_end else 0},
                auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET),
            )

        if response.status_code >= 400:
            error_detail = (
                response.json().get("error", {}).get("description", "Cancel failed")
            )
            _logger.warning(
                "Razorpay subscription cancel failed: %s (rz_sub_id=%s)",
                error_detail,
                rz_subscription_id,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Razorpay cancel failed: {error_detail}",
            )

    except HTTPException:
        raise
    except Exception as exc:
        _logger.exception(
            "Razorpay subscription cancel error: %s (rz_sub_id=%s)",
            exc,
            rz_subscription_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment gateway error. Please try again later.",
        )

    # ── 3. Update DB ─────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    access_until = subscription.current_period_end if cancel_at_cycle_end else now

    if not cancel_at_cycle_end:
        subscription.status = SubscriptionStatus.CANCELED
        subscription.current_period_end = now
    # If cancel_at_cycle_end=True, keep status as ACTIVE until webhook confirms

    subscription.updated_date = now

    await db.commit()

    _logger.info(
        "Subscription cancelled: rz_sub_id=%s user=%s cancel_at_cycle_end=%s",
        rz_subscription_id,
        user_id,
        cancel_at_cycle_end,
    )

    return {
        "cancelled": True,
        "message": (
            "Subscription will be cancelled at the end of the current billing period."
            if cancel_at_cycle_end
            else "Subscription cancelled immediately."
        ),
        "accessUntil": access_until,
    }
