"""Which currency a customer is committed to.

Currency is locked at first subscription and never changes: a USD subscriber
travelling to India keeps paying USD, and an Indian customer abroad keeps paying
INR. Both the pricing display and the USD checkout guard read the lock from here
so they can never disagree about it.

"First subscription" deliberately means the first one that was actually paid
for. A row that never got past PENDING is an abandoned checkout — treating that
as a currency commitment would let a visitor who opened the wrong checkout once
be locked out of the correct currency permanently.
"""

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.subscription import Subscription
from app.models.subscription_enums import SubscriptionStatus

PAID_STATUSES = (
    SubscriptionStatus.ACTIVE,
    SubscriptionStatus.PAST_DUE,
    SubscriptionStatus.CANCELED,
    SubscriptionStatus.TRIALING,
)


async def get_locked_currency(
    db: AsyncSession, user_id: uuid.UUID
) -> Optional[str]:
    """The currency this user's billing is locked to, or None if never charged."""
    result = await db.execute(
        select(Subscription.currency)
        .where(
            Subscription.user_id == user_id,
            Subscription.status.in_(PAID_STATUSES),
        )
        .order_by(Subscription.created_date.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()
