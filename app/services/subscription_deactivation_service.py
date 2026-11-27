"""Subscription Deactivation Service.

Automatically marks expired subscriptions as 'canceled' and revokes their
licenses, returning users to the free plan.

Design decisions:
    - Runs as a background job — NEVER raises exceptions outward.
      Any failure is logged, rolled back, and silently swallowed so the server
      never crashes because of a failed scheduled task.
    - Status is MATCHED without regard to case and WRITTEN through the models.
      The status columns are plain text holding the enum NAMES ('ACTIVE',
      'CANCELED', 'REVOKED'), because the models declare them
      ``Enum(..., native_enum=False)``. The raw SQL this job used to run
      compared against lowercase 'active', so it matched no row — nothing ever
      expired — and wrote lowercase 'canceled' / 'revoked', which the ORM
      cannot read back. Now the job finds a row whichever case it was stored
      in (``_status_is``), reads plain columns rather than model objects so an
      oddly-cased row cannot break the run, and writes the value the rest of
      the app expects.
    - Asks Razorpay before ending a subscription it is still billing. A missed
      or late renewal webhook leaves current_period_end in the past for a
      customer who has paid; cancelling on our clock alone would take their
      plan away. See ``decide_expiry``.
    - Idempotent — a subscription this job cancels is no longer ACTIVE, so the
      next run skips it.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from sqlalchemy import String, cast, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.license_assignment import LicenseAssignment
from app.models.models import Notification, User
from app.models.subscription import Subscription
from app.models.subscription_enums import LicenseStatus, SubscriptionStatus
from app.services.licensing_service import carry_over_ai_credit_overdraft

logger = logging.getLogger("rivollo.subscription_deactivation")


# Subscriptions examined per run, so one run cannot make an unbounded number of
# gateway calls. The rest are picked up by the next run (every 5 minutes).
_MAX_EXPIRED_PER_RUN = 50

# How long past current_period_end a subscription Razorpay still reports as
# running (charge due, or being retried) keeps its plan while we wait for the
# charge. After that it is ended whatever the gateway says.
_RENEWAL_GRACE = timedelta(hours=72)

# Razorpay subscription statuses that mean "still being billed".
_GATEWAY_RUNNING = ("active", "authenticated", "pending")

EXPIRY_CANCEL = "cancel"
EXPIRY_EXTEND = "extend"
EXPIRY_WAIT = "wait"


def _status_is(column, member) -> Any:
    """``column`` holds ``member``, whatever case it was stored in ('ACTIVE',
    'active', ...). Compared as text, so it works for the stored enum name and
    for the enum's lowercase value alike."""
    return func.lower(cast(column, String)) == member.name.lower()


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _gateway_time(value: Any) -> Optional[datetime]:
    """A Razorpay unix timestamp as an aware datetime (None when absent)."""
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc) if value else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def decide_expiry(
    *,
    period_end: datetime,
    has_gateway_subscription: bool,
    cancel_at_period_end: bool,
    gateway: Optional[dict],
    gateway_lookup_failed: bool,
    now: datetime,
) -> str:
    """What to do with an ACTIVE subscription whose period end has passed.

    cancel  end it and revoke its licence
    extend  Razorpay shows a paid period we never recorded (missed webhook):
            take the period from Razorpay instead of ending it
    wait    Razorpay is still billing it (charge due or retrying), or could
            not be asked: leave it for the next run, up to _RENEWAL_GRACE

    Pure: no database, no network.
    """
    # Nothing at the gateway (e.g. a manually granted plan), or the customer
    # cancelled and was promised access until exactly this date.
    if not has_gateway_subscription or cancel_at_period_end:
        return EXPIRY_CANCEL

    within_grace = now - _as_utc(period_end) < _RENEWAL_GRACE

    if gateway is None:
        # Could not ask Razorpay. Do not end a possibly-paid plan on a network
        # error — but do not keep it forever either.
        return EXPIRY_WAIT if gateway_lookup_failed and within_grace else EXPIRY_CANCEL

    if gateway.get("status") not in _GATEWAY_RUNNING:
        # cancelled / halted / completed / expired
        return EXPIRY_CANCEL

    gateway_end = _gateway_time(gateway.get("current_end"))
    if gateway.get("status") == "active" and gateway_end and gateway_end > now:
        return EXPIRY_EXTEND
    return EXPIRY_WAIT if within_grace else EXPIRY_CANCEL


async def fetch_gateway_subscription(rz_subscription_id: str) -> tuple[Optional[dict], bool]:
    """Razorpay's view of a subscription: ``(entity, lookup_failed)``.

    ``(None, False)`` when Razorpay has no such subscription or no credentials
    are configured (nothing to wait for); ``(None, True)`` on a network or
    server error (worth retrying on the next run). Never raises.
    """
    if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
        return None, False
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{settings.RAZORPAY_BASE_URL}/subscriptions/{rz_subscription_id}",
                auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET),
            )
        if response.status_code in (400, 404):
            return None, False
        response.raise_for_status()
        body = response.json()
        return (body, False) if isinstance(body, dict) else (None, True)
    except Exception:
        logger.warning(
            "Deactivation job: could not read rz_sub_id=%s from Razorpay",
            rz_subscription_id,
            exc_info=True,
        )
        return None, True


class SubscriptionDeactivationService:
    """Service that deactivates expired subscriptions and revokes their licenses."""

    @staticmethod
    async def _subscription_expiry_notification_exists(
        db: AsyncSession,
        user_id,
        notification_type: str,
    ) -> bool:
        result = await db.execute(
            select(Notification.id)
            .where(
                Notification.user_id == user_id,
                Notification.type == notification_type,
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    @staticmethod
    async def _send_subscription_expiry_reminders(db: AsyncSession, now: datetime) -> None:
        reminder_days = settings.subscription_expiry_reminder_days()
        if not reminder_days:
            return

        # Deleted accounts are excluded. Their owner cannot log in, cannot renew,
        # and cannot turn the reminders off — but their FCM tokens are still
        # registered, so without this join a user who deleted their account would
        # keep getting "your subscription expires in 5 days" push notifications
        # for up to 30 days. The subscription table has no owner-state of its own,
        # hence the join rather than a filter.
        result = await db.execute(
            select(Subscription)
            .join(User, User.id == Subscription.user_id)
            .where(
                User.deleted_at.is_(None),
                Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]),
                Subscription.current_period_end.is_not(None),
            )
        )
        subscriptions = result.scalars().all()

        from app.services.notification_service import NotificationService

        for subscription in subscriptions:
            if subscription.current_period_end is None:
                continue

            days_remaining = (subscription.current_period_end.date() - now.date()).days
            if days_remaining not in reminder_days:
                continue

            expiry_day = subscription.current_period_end.date().isoformat().replace("-", "")
            notification_type = f"subscription.expiry.{days_remaining}.{expiry_day}"
            if await SubscriptionDeactivationService._subscription_expiry_notification_exists(
                db,
                subscription.user_id,
                notification_type,
            ):
                continue

            if days_remaining == 0:
                title = "Subscription Expires Today"
                body = "Your subscription expires today. Renew now to keep your access active."
            elif days_remaining == 1:
                title = "Subscription Expires Tomorrow"
                body = "Your subscription expires in 1 day. Renew now to avoid interruption."
            else:
                title = f"Subscription Expires in {days_remaining} Days"
                body = f"Your subscription expires in {days_remaining} days. Renew now to avoid interruption."

            try:
                await NotificationService.create_and_push_notification(
                    db=db,
                    user_id=subscription.user_id,
                    notification_type=notification_type,
                    title=title,
                    body=body,
                    data={
                        "subscription_id": str(subscription.id),
                        "days_remaining": days_remaining,
                        "expires_at": subscription.current_period_end.isoformat(),
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to send subscription expiry reminder for subscription %s",
                    subscription.id,
                    exc_info=True,
                )
                with suppress(Exception):
                    await db.rollback()

    @staticmethod
    async def _extend_from_gateway(
        db: AsyncSession, subscription_id, user_id, gateway: dict, now: datetime
    ) -> Optional[datetime]:
        """Record a paid period Razorpay has and we do not — what the missed
        subscription.charged webhook would have done: move the period on and
        start the licence's monthly counters again (an AI-credit overdraft is
        carried over, as on any renewal). Limits are left as they are.
        Returns the new period end."""
        period_end = _gateway_time(gateway.get("current_end"))
        await db.execute(
            update(Subscription)
            .where(Subscription.id == subscription_id)
            .values(
                current_period_start=_gateway_time(gateway.get("current_start")) or now,
                current_period_end=period_end,
                updated_date=now,
            )
            .execution_options(synchronize_session=False)
        )

        result = await db.execute(
            select(
                LicenseAssignment.id,
                LicenseAssignment.usage_ai_credits,
                LicenseAssignment.limit_max_ai_credits,
            ).where(
                LicenseAssignment.subscription_id == subscription_id,
                LicenseAssignment.user_id == user_id,
                _status_is(LicenseAssignment.status, LicenseStatus.ACTIVE),
            )
        )
        for license_id, usage, limit in result.all():
            await db.execute(
                update(LicenseAssignment)
                .where(LicenseAssignment.id == license_id)
                .values(
                    usage_ai_credits=carry_over_ai_credit_overdraft(usage, limit),
                    usage_public_views=0,
                )
                .execution_options(synchronize_session=False)
            )
        return period_end

    @staticmethod
    async def run_deactivation_job(db: AsyncSession) -> None:
        """Find ACTIVE subscriptions whose period has ended; cancel them and
        revoke their licences — unless Razorpay shows the customer has paid
        for a period we never recorded, or is still being charged for it
        (``decide_expiry``).

        This is the main entry point for the background scheduler. All of a
        run's changes are committed together.

        Args:
            db: An async database session (obtained from the session factory,
                NOT from a request — this runs outside the HTTP lifecycle).
        """
        try:
            now = datetime.now(timezone.utc)

            await SubscriptionDeactivationService._send_subscription_expiry_reminders(db, now)

            # Plain columns, not Subscription objects: loading a model whose
            # status is stored in an unexpected case raises inside the ORM.
            result = await db.execute(
                select(
                    Subscription.id,
                    Subscription.user_id,
                    Subscription.razorpay_subscription_id,
                    Subscription.cancel_at_period_end,
                    Subscription.current_period_end,
                )
                .where(
                    _status_is(Subscription.status, SubscriptionStatus.ACTIVE),
                    Subscription.current_period_end.is_not(None),
                    Subscription.current_period_end <= now,
                )
                .order_by(Subscription.current_period_end.asc())
                .limit(_MAX_EXPIRED_PER_RUN)
            )
            expired = result.all()

            if not expired:
                logger.debug("Deactivation job: no expired subscriptions found.")
                await db.commit()
                return

            canceled_ids = []
            extended = waiting = 0
            for (
                subscription_id,
                user_id,
                rz_subscription_id,
                cancel_at_period_end,
                period_end,
            ) in expired:
                gateway: Optional[dict] = None
                lookup_failed = False
                if rz_subscription_id and not cancel_at_period_end:
                    gateway, lookup_failed = await fetch_gateway_subscription(
                        rz_subscription_id
                    )

                decision = decide_expiry(
                    period_end=period_end,
                    has_gateway_subscription=bool(rz_subscription_id),
                    cancel_at_period_end=bool(cancel_at_period_end),
                    gateway=gateway,
                    gateway_lookup_failed=lookup_failed,
                    now=now,
                )

                if decision == EXPIRY_EXTEND:
                    new_end = await SubscriptionDeactivationService._extend_from_gateway(
                        db, subscription_id, user_id, gateway or {}, now
                    )
                    extended += 1
                    logger.warning(
                        "Deactivation job: subscription %s (rz_sub_id=%s) was past its "
                        "period end but Razorpay shows it paid until %s — period taken "
                        "from Razorpay (a renewal webhook was missed).",
                        subscription_id,
                        rz_subscription_id,
                        new_end,
                    )
                elif decision == EXPIRY_WAIT:
                    waiting += 1
                else:
                    canceled_ids.append(subscription_id)

            license_count = 0
            if canceled_ids:
                await db.execute(
                    update(Subscription)
                    .where(Subscription.id.in_(canceled_ids))
                    .values(status=SubscriptionStatus.CANCELED, updated_date=now)
                    .execution_options(synchronize_session=False)
                )
                license_result = await db.execute(
                    update(LicenseAssignment)
                    .where(
                        LicenseAssignment.subscription_id.in_(canceled_ids),
                        _status_is(LicenseAssignment.status, LicenseStatus.ACTIVE),
                    )
                    .values(status=LicenseStatus.REVOKED)
                    .execution_options(synchronize_session=False)
                )
                license_count = license_result.rowcount

            await db.commit()

            logger.info(
                "Deactivation job: %d subscription(s) expired → canceled, "
                "%d license(s) revoked, %d extended from Razorpay, "
                "%d awaiting their renewal charge.",
                len(canceled_ids),
                license_count,
                extended,
                waiting,
            )

        except Exception as exc:
            # Never propagate — a background task crash would stop the loop.
            logger.exception(
                "Deactivation job failed — rolling back. Error: %s", exc
            )
            with suppress(Exception):
                await db.rollback()
