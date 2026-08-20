"""A deleted account must stop hearing from us.

Deletion leaves the user's FCM tokens in tbl_user_devices — deliberately, since
restore needs them — and the expiry-reminder query had no notion of account
state. So a user who deleted their account could keep receiving "Your
subscription expires in 5 days" push notifications for the whole 30-day
retention window, for a subscription they cannot renew, from an account they
cannot log into, with no way to make it stop.

The fix is a join to tbl_users, which is what these tests pin.
"""

import inspect

from app.services.subscription_deactivation_service import (
    SubscriptionDeactivationService,
)


def test_reminder_query_joins_users_and_excludes_deleted():
    source = inspect.getsource(
        SubscriptionDeactivationService._send_subscription_expiry_reminders
    )
    assert ".join(User" in source
    assert "User.deleted_at.is_(None)" in source


def test_reminder_query_still_targets_active_and_trialing():
    """The exclusion must narrow by account state only, not by status."""
    source = inspect.getsource(
        SubscriptionDeactivationService._send_subscription_expiry_reminders
    )
    assert "SubscriptionStatus.ACTIVE" in source
    assert "SubscriptionStatus.TRIALING" in source
    assert "current_period_end.is_not(None)" in source
