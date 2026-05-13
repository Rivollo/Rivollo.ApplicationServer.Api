"""Subscription Deactivation Service.

Automatically marks expired subscriptions as 'canceled' and revokes their
licenses, returning users to the free plan.

Design decisions:
    - Runs as a background job — NEVER raises exceptions outward.
      Any failure is logged, rolled back, and silently swallowed so the server
      never crashes because of a failed scheduled task.
    - Uses bulk UPDATE ... RETURNING for efficiency — one round-trip to find AND
      cancel expired subscriptions, a second to revoke their licenses.
    - Uses raw SQL from app.queries.subscription_queries (pure SQL constants,
      no ORM overhead for a batch update).
    - Idempotent — running it twice in quick succession is safe because the
      first run marks subscriptions 'canceled', so the second run's WHERE clause
      (status = 'active') skips them.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import Notification
from app.models.subscription import Subscription
from app.models.subscription_enums import SubscriptionStatus
from app.queries.subscription_queries import (
    DEACTIVATE_EXPIRED_SUBSCRIPTIONS,
    REVOKE_LICENSES_FOR_SUBSCRIPTIONS,
)

logger = logging.getLogger("rivollo.subscription_deactivation")


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

        result = await db.execute(
            select(Subscription).where(
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
    async def run_deactivation_job(db: AsyncSession) -> None:
        """Find all expired active subscriptions, cancel them, and revoke licenses.

        This is the main entry point for the background scheduler.

        It performs exactly two SQL statements:
            1. UPDATE tbl_subscriptions WHERE period_end <= now → CANCELED
               (RETURNING the list of IDs updated)
            2. UPDATE tbl_license_assignments WHERE subscription_id IN (above IDs)
               → REVOKED

        A single transaction wraps both updates so they succeed or fail together.

        Args:
            db: An async database session (obtained from the session factory,
                NOT from a request — this runs outside the HTTP lifecycle).
        """
        try:
            now = datetime.now(timezone.utc)

            await SubscriptionDeactivationService._send_subscription_expiry_reminders(db, now)

            # ── Step 1: Cancel all expired subscriptions ──────────────────────
            # RETURNING gives us the IDs immediately without a follow-up SELECT.
            result = await db.execute(
                text(DEACTIVATE_EXPIRED_SUBSCRIPTIONS),
                {"now": now},
            )
            canceled_ids = [row[0] for row in result.fetchall()]
            sub_count = len(canceled_ids)

            if sub_count == 0:
                # Nothing to do — skip license revocation and commit.
                logger.debug("Deactivation job: no expired subscriptions found.")
                await db.commit()
                return

            # ── Step 2: Revoke licenses for the canceled subscriptions ────────
            # ANY(:subscription_ids) requires a list — asyncpg handles it natively.
            license_result = await db.execute(
                text(REVOKE_LICENSES_FOR_SUBSCRIPTIONS),
                {"now": now, "subscription_ids": canceled_ids},
            )
            license_count = license_result.rowcount

            # ── Step 3: Commit both updates together ──────────────────────────
            await db.commit()

            logger.info(
                "Deactivation job: %d subscription(s) expired → canceled, "
                "%d license(s) revoked.",
                sub_count,
                license_count,
            )

        except Exception as exc:
            # Never propagate — a background task crash would stop the loop.
            logger.exception(
                "Deactivation job failed — rolling back. Error: %s", exc
            )
            with suppress(Exception):
                await db.rollback()
