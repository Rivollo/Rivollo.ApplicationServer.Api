"""Per-IP request throttling for the OTP login endpoints.

This is the same minute-bucket counter pattern used by _RateLimiter in
app/api/routes/ai.py, re-expressed here rather than imported. Two reasons: that
one is keyed by authenticated user id and OTP has no authenticated user yet,
and importing from a route module would couple two unrelated features so that
a change to AI rate limiting could alter authentication behaviour.

KNOWN LIMITATION, inherited from that pattern: the counter is per-process. On
Azure Container Apps with N replicas a caller effectively gets
``limit x N`` calls per minute. That is why this is only the FIRST line of
defence — the controls that actually bound abuse are in the database and are
correct under any replica count:

    resend budget      -> tbl_login_otps.resend_count + locked_until
    code guessing      -> tbl_login_otps.verification_attempts

To make this exact across replicas, replace check() with a Redis INCR/EXPIRE;
the call sites do not change.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, Request, status

from app.core.login_otp_config import otp_settings

logger = logging.getLogger(__name__)


def client_ip(request: Optional[Request]) -> str:
    """Best-effort caller IP.

    Resolved exactly as ActivityService resolves it — x-forwarded-for first
    (taking the leftmost entry), then the socket peer — so a rate-limit bucket
    and the audit row for the same request agree on who the caller was.

    Falls back to a constant when neither is available, which buckets all such
    callers together. That is the safe direction: it over-throttles an
    anonymous group rather than letting unattributable traffic through
    unlimited.
    """
    if request is None:
        return "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


class _IpRateLimiter:
    """Per-IP, per-minute call counter for one OTP endpoint.

    ``limit_attr`` is read from settings on every check so an environment
    override takes effect without rebuilding the limiter, and ``label`` names
    the budget in the log line — a verify 429 should not tell the caller it hit
    the request cap.
    """

    def __init__(self, limit_attr: str, label: str) -> None:
        self._limit_attr = limit_attr
        self._label = label
        self._counter: dict[tuple[str, str], int] = defaultdict(int)

    def check(self, ip: str) -> None:
        """Raise HTTP 429 if this IP has exceeded the budget this minute."""
        limit = getattr(otp_settings, self._limit_attr)
        bucket = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

        # Drop other minutes' keys so the dict cannot grow without bound.
        stale = [k for k in self._counter if k[1] != bucket]
        for k in stale:
            del self._counter[k]

        key = (ip, bucket)
        self._counter[key] += 1

        if self._counter[key] > limit:
            # IP only. Never the email and never the code — this line is
            # written on a path where both are in scope.
            logger.warning(
                "OTP %s rate limit exceeded for ip=%s (count=%d, limit=%d)",
                self._label,
                ip,
                self._counter[key],
                limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please wait a moment and try again.",
                headers={"Retry-After": "60"},
            )

    def reset(self) -> None:
        """Clear all buckets. For tests only."""
        self._counter.clear()


request_limiter = _IpRateLimiter("LOGIN_OTP_REQUESTS_PER_IP_PER_MINUTE", "request")
verify_limiter = _IpRateLimiter("LOGIN_OTP_VERIFY_PER_IP_PER_MINUTE", "verify")
