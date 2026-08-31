"""ORM model for the email OTP login challenge.

Lives in its own module rather than in ``models/models.py`` so that the file
defining User, SignupOtp, PasswordReset and AuthIdentity stays out of this
feature's change set. Follows the precedent set by plan.py, subscription.py,
link_share_log.py and user_device.py.

ONE ROW PER (email, purpose) — NOT one row per code
---------------------------------------------------
The resend budget spans up to three codes and the lockout outlives all of
them, so neither can live on a per-code row. A row here is a challenge
*series*: it carries the series state and whichever code is currently live.
Issuing a code overwrites ``otp_hash``, which is what invalidates the previous
one — invalidation is structural rather than something the caller has to
remember.

The upsert-in-place shape mirrors tbl_app_tokens, which holds one row per
client_key and is maintained with pg_insert(...).on_conflict_do_update(...) in
AuthService.generate_app_token. Same idiom, already used in three places in
this project.

NO FOREIGN KEY TO tbl_users — deliberate, and load-bearing twice over
---------------------------------------------------------------------
1. ACCOUNT_PURGE_JOB_HANDOFF.md section 25 specifies a schema contract check
   run before every purge, whose assertion 9 aborts the entire run if an
   unexpected new foreign key references tbl_users. A user_id FK here would
   break a production job in another repository.
2. Rows are created for ANY syntactically valid address, whether or not an
   account exists — that is what lets the lockout be reported honestly without
   becoming an account-enumeration oracle (see LoginOtpService). A FK could
   not express that, and a nullable user_id would simply be NULL for exactly
   the rows that need it most.

The purge job therefore needs an email-keyed DELETE for this table, alongside
the one it already has for tbl_signup_otps.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import SmallInteger, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from app.models.base import Base
from app.models.models import CreatedAtMixin, UUIDMixin

# The only purpose in use today. Part of the unique key so a future
# login-adjacent challenge (step-up auth, email change) can have its own
# independent series per address without a fourth OTP implementation.
PURPOSE_LOGIN = "login"


class LoginOtp(UUIDMixin, CreatedAtMixin, Base):
    """A login OTP challenge series for one email address."""

    __tablename__ = "tbl_login_otps"
    __table_args__ = (
        UniqueConstraint("email", "purpose", name="uq_login_otps_email_purpose"),
    )

    # Plain TEXT, normalised to lowercase by the service on every path in and
    # out (LoginOtpService lowercases before both the request and the verify
    # lookup), which is the same approach tbl_signup_otps takes.
    #
    # CITEXT would match tbl_users.email and give case-insensitivity at the
    # column level, but Azure Database for PostgreSQL refuses CREATE EXTENSION
    # for anything not on its `azure.extensions` allow-list, and citext is not
    # on it for this server. Since every read and write is already lowercased,
    # TEXT behaves identically here — the only thing given up is a safety net
    # for a future query that bypasses the service and forgets to normalise.
    email: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'login'"), default=PURPOSE_LOGIN
    )

    # Peppered SHA-256 of the CURRENT code. NULL means no live code: set on
    # successful consumption and on attempt exhaustion. Nullable is
    # load-bearing — it kills a code without killing the series, so a resend
    # is still possible afterwards.
    otp_hash: Mapped[Optional[str]] = mapped_column(Text)
    expires_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # Resends used in this series. 0 immediately after the initial send, so
    # total sends == resend_count + 1. Named for the business rule rather than
    # for the arithmetic.
    resend_count: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0"), default=0
    )
    # Policy snapshotted when the series starts, so tightening the configured
    # limit cannot retroactively strand a user who is mid-series.
    max_resends: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("2"), default=2
    )

    # Wrong codes submitted against the CURRENT code. Reset to zero on every
    # send. Kept strictly separate from resend_count: different scope,
    # different reset rule, different consequence.
    verification_attempts: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0"), default=0
    )
    max_verification_attempts: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("5"), default=5
    )

    # Drives both the resend cooldown and the series window.
    last_sent_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))

    # The ENTIRE lockout state. NULL means not locked. Expiry is passive —
    # nothing clears this; the next request simply observes it is in the past
    # and starts a fresh series.
    locked_until: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    lock_reason: Mapped[Optional[str]] = mapped_column(String(32))

    # Successful verification. The atomic guard column for single use.
    consumed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    invalidated_reason: Mapped[Optional[str]] = mapped_column(String(32))

    # Abuse forensics. Text rather than INET to match tbl_activity_logs.ip.
    request_ip: Mapped[Optional[str]] = mapped_column(Text)

    # The row mutates through a series, so this is genuinely informative here
    # in a way it is not on append-only tables.
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        # Deliberately omits email and otp_hash. This object is passed around
        # request handlers and must never render a credential or an address
        # into a log line or a traceback.
        return (
            f"<LoginOtp id={self.id} purpose={self.purpose} "
            f"resend_count={self.resend_count} "
            f"verification_attempts={self.verification_attempts}>"
        )
