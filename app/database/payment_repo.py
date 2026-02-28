"""Payment repository — database operations for tbl_payments.

This module contains ONLY database access logic. No business rules here.
All methods are static and receive an AsyncSession from the caller.

Responsibilities:
    - Create a payment record when Razorpay order is created
    - Fetch a payment by its Razorpay order ID (idempotency check)
    - Mark a payment as captured (success) or failed
    - Link a subscription_id to a payment after activation
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.payment import Payment, PaymentStatus


class PaymentRepository:
    """Repository for payment database operations."""

    @staticmethod
    async def create_payment(
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        razorpay_order_id: str,
        amount: int,
        currency: str,
        plan_code: str,
    ) -> Payment:
        """Insert a new payment row with status = 'created'.

        Call this immediately after Razorpay order creation succeeds.

        Args:
            db: Async database session
            user_id: ID of the user creating the payment
            razorpay_order_id: The order ID returned by Razorpay API
            amount: Amount in paise (e.g. 49900 for ₹499)
            currency: ISO currency code (e.g. "INR")
            plan_code: Plan being purchased ("pro" or "enterprise")

        Returns:
            The newly created Payment row (already added to session, not yet committed)
        """
        payment = Payment(
            user_id=user_id,
            razorpay_order_id=razorpay_order_id,
            amount=amount,
            currency=currency,
            plan_code=plan_code,
            status=PaymentStatus.CREATED,
        )
        db.add(payment)
        await db.flush()   # flush to DB so we get the id, but don't commit yet
        await db.refresh(payment)
        return payment

    @staticmethod
    async def get_by_order_id(
        db: AsyncSession, razorpay_order_id: str
    ) -> Optional[Payment]:
        """Fetch a payment record by its Razorpay order ID.

        Used for idempotency — if this returns a captured payment then we
        skip re-activation and return the existing result.

        Args:
            db: Async database session
            razorpay_order_id: The Razorpay order ID to look up

        Returns:
            Payment row if found, None otherwise
        """
        result = await db.execute(
            select(Payment).where(Payment.razorpay_order_id == razorpay_order_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_captured(
        db: AsyncSession,
        *,
        payment_id: uuid.UUID,
        razorpay_payment_id: str,
        razorpay_signature: str,
        subscription_id: uuid.UUID,
    ) -> Payment:
        """Update payment status to 'captured' after successful verification.

        Call this after:
          1. Signature verification passes
          2. Subscription is activated

        Args:
            db: Async database session
            payment_id: UUID of the payment row to update
            razorpay_payment_id: Payment ID from Razorpay after user paid
            razorpay_signature: HMAC signature from Razorpay
            subscription_id: UUID of the activated subscription

        Returns:
            Updated Payment row
        """
        result = await db.execute(
            select(Payment).where(Payment.id == payment_id)
        )
        payment = result.scalar_one()
        payment.status = PaymentStatus.CAPTURED
        payment.razorpay_payment_id = razorpay_payment_id
        payment.razorpay_signature = razorpay_signature
        payment.subscription_id = subscription_id
        payment.updated_date = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(payment)
        return payment

    @staticmethod
    async def mark_failed(
        db: AsyncSession,
        *,
        payment_id: uuid.UUID,
        reason: str,
    ) -> Payment:
        """Update payment status to 'failed'.

        Call this when signature verification fails.

        Args:
            db: Async database session
            payment_id: UUID of the payment row to update
            reason: Human-readable reason for failure

        Returns:
            Updated Payment row
        """
        result = await db.execute(
            select(Payment).where(Payment.id == payment_id)
        )
        payment = result.scalar_one()
        payment.status = PaymentStatus.FAILED
        payment.failure_reason = reason
        payment.updated_date = datetime.now(timezone.utc)
        await db.flush()
        await db.refresh(payment)
        return payment
