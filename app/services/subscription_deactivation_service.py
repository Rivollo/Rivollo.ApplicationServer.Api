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

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.queries.subscription_queries import (
    DEACTIVATE_EXPIRED_SUBSCRIPTIONS,
    REVOKE_LICENSES_FOR_SUBSCRIPTIONS,
)

logger = logging.getLogger("rivollo.subscription_deactivation")


class SubscriptionDeactivationService:
    """Service that deactivates expired subscriptions and revokes their licenses."""

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
