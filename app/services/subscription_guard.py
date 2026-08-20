"""Does this user still have a paid subscription running?

Account deletion is refused while one does. The customer cancels first, keeps the
period they have already paid for, and deletes once it ends — so the account is
never torn down underneath a live billing relationship, and we never have to
reach for the payment gateway during deletion.

Nothing here calls Razorpay. It is a read of local state only.

Why these three conditions, and not others
------------------------------------------
Each one is taken from business logic that already exists, so this check and the
rest of the product cannot drift apart:

``razorpay_subscription_id IS NOT NULL``
    The free-vs-paid discriminator ``cancel_subscription`` uses. Every account
    gets a free-plan subscription row at signup with no gateway id, so without
    this the rule would lock every free user out of deleting their account.

``status IN (ACTIVE, PAST_DUE)``
    Exactly the statuses ``cancel_subscription`` will act on. That matters
    because the error we return tells the customer to cancel first: for these two
    statuses that instruction is actionable, and for PENDING or CANCELED it would
    be a dead end. PAST_DUE counts as running — the mandate is live and Razorpay
    is still retrying the charge.

``not _is_expired(current_period_end, now)``
    ``SubscriptionService``'s own realtime expiry check, imported rather than
    re-written. A subscription whose period has ended no longer blocks anything,
    which is what lets a cancelled customer delete their account once the paid
    period runs out. Note it treats a NULL period end as *not* expired, so an
    unterminated subscription keeps blocking rather than slipping through.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.subscription import Subscription
from app.models.subscription_enums import SubscriptionStatus
from app.services.subscription_service import SubscriptionService

# Statuses in which a gateway-backed subscription is still running. Mirrors
# cancel_subscription's selector so "you must cancel first" is always advice the
# customer can actually act on.
PAID_ACTIVE_STATUSES = (SubscriptionStatus.ACTIVE, SubscriptionStatus.PAST_DUE)

# Shown when deletion is refused. Phrased as the next step rather than a refusal,
# because the customer is one action away from being able to proceed.
ACTIVE_SUBSCRIPTION_BLOCK_DETAIL = (
    "Please cancel your subscription first. You can delete your account after "
    "the current billing period ends."
)


async def get_blocking_subscription(
    db: AsyncSession, user_id: uuid.UUID
) -> Optional[Subscription]:
    """The paid subscription preventing deletion, or None if there isn't one.

    Returns the row rather than a bool so callers can report *when* the block
    lifts (``current_period_end``) without running a second query.
    """
    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.status.in_(PAID_ACTIVE_STATUSES),
            Subscription.razorpay_subscription_id.is_not(None),
        )
    )

    # Expiry is evaluated in Python, not SQL, so it goes through exactly the same
    # predicate as GET /subscriptions/me. Duplicating it as a WHERE clause would
    # be faster and would be the thing that quietly disagrees with the rest of the
    # product the next time that logic changes.
    for subscription in result.scalars().all():
        if not SubscriptionService._is_expired(subscription.current_period_end, now):
            return subscription

    return None


async def has_active_paid_subscription(db: AsyncSession, user_id: uuid.UUID) -> bool:
    """True while a paid subscription is still running for this user."""
    return await get_blocking_subscription(db, user_id) is not None
