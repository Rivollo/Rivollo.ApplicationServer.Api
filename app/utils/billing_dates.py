"""Calendar arithmetic for billing periods.

Billing dates are calendar arithmetic, not day counts. "+30 days" drifts a month
out of alignment over a year and "+365 days" breaks on leap years, so both would
eventually charge a customer on the wrong day.

Stdlib only — ``calendar.monthrange`` gives the clamping behaviour that dateutil
or Luxon would, and adding a dependency for one function is not worth it.
"""

import calendar
from datetime import datetime, timedelta, timezone

# Razorpay rejects a start_at in the past. Nothing here should ever produce one
# — the dates are a month or a year out — but a clock skew between this
# container and Razorpay's servers could, so every start_at is floored to at
# least this far ahead.
_MIN_LEAD = timedelta(hours=1)


def add_calendar_months(moment: datetime, months: int) -> datetime:
    """Add whole calendar months, clamping the day to the target month's length.

    31 Jan + 1 month is 28 Feb (29 in a leap year); 31 Mar + 1 month is 30 Apr.
    Time of day is preserved.
    """
    zero_based = moment.month - 1 + months
    year = moment.year + zero_based // 12
    month = zero_based % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def next_period_start(moment: datetime, billing_interval: str) -> datetime:
    """The date one billing period after ``moment``.

    29 Feb 2028 + 1 year clamps to 28 Feb 2029.
    """
    months = 12 if billing_interval == "yearly" else 1
    return add_calendar_months(moment, months)


def to_razorpay_start_at(moment: datetime) -> int:
    """Convert a start date to the Unix seconds (UTC) Razorpay expects.

    Floored to at least an hour from now so clock skew can never make Razorpay
    reject the subscription for a start date in the past.
    """
    floor = datetime.now(timezone.utc) + _MIN_LEAD
    effective = max(moment, floor)
    return int(effective.timestamp())
