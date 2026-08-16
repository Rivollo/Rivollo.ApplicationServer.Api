"""Which currency a customer is committed to.

Currency is locked at first subscription and never changes: a USD subscriber
travelling to India keeps paying USD, and an Indian customer abroad keeps paying
INR. Both the pricing display and the USD checkout guard read the lock from here
so they can never disagree about it.

"First subscription" deliberately means the first one that was actually paid
for. Two kinds of row look like subscriptions but are not commitments:

  * A row that never got past PENDING is an abandoned checkout. Treating that
    as a currency commitment would let a visitor who opened the wrong checkout
    once be locked out of the correct currency permanently.

  * A free-plan row. Every account gets one at signup — licensing_service
    creates it with status ACTIVE — and it takes the column default
    currency='INR' because nothing sets a currency for a plan that is not
    charged for. It is ACTIVE and it is not PENDING, so status alone does not
    exclude it.

The second one is not hypothetical: it silently disabled the entire USD path.
Every registered account was locked to rupees the moment it was created, the
pricing page quoted INR to visitors anywhere in the world, and
usd_subscription_service rejected checkout with "this account already bills in
INR" for a user who had never been charged anything.

What separates the two is a gateway subscription id. Every path that takes
money sets razorpay_subscription_id; free rows and abandoned checkouts leave it
NULL. That is the condition below, and it is the one to preserve if this query
is ever rewritten.
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
            # The free-plan row is ACTIVE and carries the default INR. Without
            # this, every account is locked to rupees at signup — see above.
            Subscription.razorpay_subscription_id.isnot(None),
        )
        .order_by(Subscription.created_date.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()
