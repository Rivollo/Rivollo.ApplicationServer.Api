"""Delete stale login OTP challenge rows.

Standalone on purpose. Adding a sweep to app/main.py's lifespan would mean
changing application startup behaviour, and this feature's change to that file
is deliberately limited to registering its router. Run this by hand, from cron,
or from a scheduled Container Apps job.

This is housekeeping, not correctness. UNIQUE (email, purpose) already bounds
tbl_login_otps to one row per address that has ever attempted OTP login, so the
table grows with distinct addresses rather than with attempts — unlike
tbl_signup_otps, which stores a row per code and had accumulated 98 stale rows
in dev with no sweeper at all.

Rows whose lock is still in force are never deleted: removing one would hand
the address a fresh resend budget and silently cancel its 30-minute lockout.

Usage:
    python -m scripts.cleanup_login_otps            # default retention
    python -m scripts.cleanup_login_otps --days 14
    python -m scripts.cleanup_login_otps --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

# Keeping a week means a support question about a sign-in that failed a few
# days ago still has a row to look at. There is no correctness reason to retain
# anything: a series older than the 30-minute window is inert.
DEFAULT_RETENTION_DAYS = 7

logger = logging.getLogger("rivollo.cleanup_login_otps")


async def _run(days: int, dry_run: bool) -> int:
    from sqlalchemy import func, or_, select

    from app.core.db import dispose_engine, init_engine_and_session, new_session
    from app.database.login_otp_repo import LoginOtpRepository
    from app.models.login_otp import LoginOtp

    now = datetime.now(timezone.utc)
    older_than = now - timedelta(days=days)

    init_engine_and_session()
    try:
        session = new_session()
        async with session:
            if dry_run:
                result = await session.execute(
                    select(func.count())
                    .select_from(LoginOtp)
                    .where(
                        LoginOtp.last_sent_at < older_than,
                        or_(
                            LoginOtp.locked_until.is_(None),
                            LoginOtp.locked_until <= now,
                        ),
                    )
                )
                count = int(result.scalar_one())
                logger.info(
                    "Dry run: %d row(s) older than %s would be deleted.",
                    count,
                    older_than.isoformat(),
                )
                return count

            deleted = await LoginOtpRepository.delete_stale(session, older_than, now)
            await session.commit()
            logger.info(
                "Deleted %d login OTP row(s) last used before %s.",
                deleted,
                older_than.isoformat(),
            )
            return deleted
    finally:
        await dispose_engine()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Delete rows last used more than this many days ago "
        f"(default: {DEFAULT_RETENTION_DAYS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would be deleted, and delete nothing.",
    )
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1.")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s"
    )
    asyncio.run(_run(args.days, args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
