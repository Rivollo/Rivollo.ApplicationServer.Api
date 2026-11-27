"""When the expiry job may end a subscription.

The job once ran raw SQL comparing status to lowercase 'active' while the
column holds 'ACTIVE', so it never ended anything. Now that it works, it must
not end a plan the customer has paid for: a missed or late renewal webhook
leaves current_period_end in the past, and Razorpay is asked before cancelling.
"""

import inspect
from datetime import datetime, timedelta, timezone

from app.services import subscription_deactivation_service as job
from app.services.subscription_deactivation_service import (
    EXPIRY_CANCEL,
    EXPIRY_EXTEND,
    EXPIRY_WAIT,
    decide_expiry,
)

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)


def _decide(*, ended_ago=timedelta(hours=1), has_gateway=True, cancel_at_end=False,
            gateway=None, lookup_failed=False):
    return decide_expiry(
        period_end=NOW - ended_ago,
        has_gateway_subscription=has_gateway,
        cancel_at_period_end=cancel_at_end,
        gateway=gateway,
        gateway_lookup_failed=lookup_failed,
        now=NOW,
    )


def _ts(delta):
    return int((NOW + delta).timestamp())


def test_no_gateway_subscription_is_cancelled():
    assert _decide(has_gateway=False) == EXPIRY_CANCEL


def test_customer_cancellation_ends_on_the_promised_date():
    paid_on = {"status": "active", "current_end": _ts(timedelta(days=20))}
    assert _decide(cancel_at_end=True, gateway=paid_on) == EXPIRY_CANCEL


def test_paid_period_we_never_recorded_is_extended_not_cancelled():
    gateway = {"status": "active", "current_end": _ts(timedelta(days=20))}
    assert _decide(gateway=gateway) == EXPIRY_EXTEND
    # however long ago our own date passed
    assert _decide(gateway=gateway, ended_ago=timedelta(days=60)) == EXPIRY_EXTEND


def test_charge_due_or_retrying_waits_then_gives_up():
    for status in ("active", "authenticated", "pending"):
        gateway = {"status": status, "current_end": _ts(-timedelta(hours=1))}
        assert _decide(gateway=gateway) == EXPIRY_WAIT
        assert _decide(gateway=gateway, ended_ago=timedelta(days=4)) == EXPIRY_CANCEL


def test_ended_at_the_gateway_is_cancelled():
    for status in ("cancelled", "halted", "completed", "expired"):
        assert _decide(gateway={"status": status}) == EXPIRY_CANCEL


def test_a_failed_lookup_does_not_end_a_plan_until_the_grace_runs_out():
    assert _decide(lookup_failed=True) == EXPIRY_WAIT
    assert _decide(lookup_failed=True, ended_ago=timedelta(days=4)) == EXPIRY_CANCEL


def test_unknown_at_the_gateway_is_cancelled():
    assert _decide(gateway=None, lookup_failed=False) == EXPIRY_CANCEL


def test_naive_period_end_is_read_as_utc():
    assert decide_expiry(
        period_end=(NOW - timedelta(hours=1)).replace(tzinfo=None),
        has_gateway_subscription=False,
        cancel_at_period_end=False,
        gateway=None,
        gateway_lookup_failed=False,
        now=NOW,
    ) == EXPIRY_CANCEL


def test_job_matches_status_in_any_case_and_writes_through_the_models():
    """Rows are found whether status is stored 'ACTIVE' or 'active'; what the
    job writes is the value the ORM reads back."""
    from sqlalchemy.dialects import postgresql

    from app.models.license_assignment import LicenseAssignment
    from app.models.subscription import Subscription
    from app.models.subscription_enums import LicenseStatus, SubscriptionStatus

    for column, member in (
        (Subscription.status, SubscriptionStatus.ACTIVE),
        (LicenseAssignment.status, LicenseStatus.ACTIVE),
    ):
        sql = str(
            job._status_is(column, member).compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )
        assert sql.startswith("lower(CAST(")
        assert sql.endswith("= 'active'")

    source = inspect.getsource(job.SubscriptionDeactivationService.run_deactivation_job)
    assert "_status_is(Subscription.status, SubscriptionStatus.ACTIVE)" in source
    assert "_status_is(LicenseAssignment.status, LicenseStatus.ACTIVE)" in source
    assert "status=SubscriptionStatus.CANCELED" in source
    assert "status=LicenseStatus.REVOKED" in source
    assert "text(" not in source
