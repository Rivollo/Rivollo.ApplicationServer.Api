"""Data access for the email OTP login challenge.

Every statement the OTP flow issues lives here. Concentrating them in one
module is what makes the concurrency-sensitive ones — the FOR UPDATE load, the
upsert, and the guarded consume — reviewable and testable against a
statement-recording fake session, which is how this project tests SQL (see
tests/test_account_restore_on_sign_in.py).

None of these functions commit. The caller owns the transaction, matching the
convention in app/database/*_repo.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.login_otp import PURPOSE_LOGIN, LoginOtp


class LoginOtpRepository:
    """Repository for tbl_login_otps."""

    @staticmethod
    async def get_for_update(
        db: AsyncSession, email: str, purpose: str = PURPOSE_LOGIN
    ) -> Optional[LoginOtp]:
        """Load the challenge series for an address, locking the row.

        FOR UPDATE serialises the two races that matter: two requests arriving
        together (which must not both spend a resend or both issue a code) and
        two verifications of the same code. Without the lock, read-then-write
        sequences in the service would interleave.

        Returns None when the address has never started a series.

        Must be called inside a transaction, which the request-scoped session
        provides.
        """
        result = await db.execute(
            select(LoginOtp)
            .where(LoginOtp.email == email, LoginOtp.purpose == purpose)
            .with_for_update()
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def upsert_challenge(
        db: AsyncSession,
        *,
        email: str,
        otp_hash: str,
        expires_at: datetime,
        now: datetime,
        resend_count: int,
        max_resends: int,
        max_verification_attempts: int,
        request_ip: Optional[str],
        purpose: str = PURPOSE_LOGIN,
    ) -> None:
        """Write a newly issued code onto the series, creating it if needed.

        One statement, so a concurrent request cannot land between a SELECT and
        an INSERT. Targets uq_login_otps_email_purpose by name, exactly as
        AuthService.generate_app_token targets uq_app_tokens_client_key.

        Writing otp_hash is what invalidates the previous code — there is no
        separate invalidation step, and therefore no way to forget it.

        Every send also clears the whole per-code state: verification_attempts
        back to zero (a new code deserves a fresh attempt budget, which is what
        keeps that counter independent of resend_count), and consumed_at,
        locked_until, lock_reason and invalidated_reason back to NULL, because
        the caller only reaches this point having decided the series is live or
        should be restarted.
        """
        stmt = (
            pg_insert(LoginOtp)
            .values(
                email=email,
                purpose=purpose,
                otp_hash=otp_hash,
                expires_at=expires_at,
                resend_count=resend_count,
                max_resends=max_resends,
                verification_attempts=0,
                max_verification_attempts=max_verification_attempts,
                last_sent_at=now,
                locked_until=None,
                lock_reason=None,
                consumed_at=None,
                invalidated_reason=None,
                request_ip=request_ip,
                updated_at=now,
            )
            .on_conflict_do_update(
                constraint="uq_login_otps_email_purpose",
                set_={
                    "otp_hash": otp_hash,
                    "expires_at": expires_at,
                    "resend_count": resend_count,
                    "max_resends": max_resends,
                    "verification_attempts": 0,
                    "max_verification_attempts": max_verification_attempts,
                    "last_sent_at": now,
                    "locked_until": None,
                    "lock_reason": None,
                    "consumed_at": None,
                    "invalidated_reason": None,
                    "request_ip": request_ip,
                    "updated_at": now,
                },
            )
        )
        await db.execute(stmt)

    @staticmethod
    async def apply_lock(
        db: AsyncSession,
        row_id: uuid.UUID,
        locked_until: datetime,
        reason: str,
        now: datetime,
    ) -> None:
        """Lock the OTP flow for this address until ``locked_until``.

        Writes only to tbl_login_otps. Nothing here touches tbl_users, so the
        lock cannot deactivate an account, cannot affect Google or password
        login, and cannot invalidate an already-issued session token.

        The live code is deliberately left alone: someone who hit the resend
        limit may still hold a code that was legitimately delivered, and there
        is no reason to void it. The lock gates issuing more codes.
        """
        await db.execute(
            update(LoginOtp)
            .where(LoginOtp.id == row_id)
            .values(locked_until=locked_until, lock_reason=reason, updated_at=now)
        )

    @staticmethod
    async def increment_attempts(
        db: AsyncSession, row_id: uuid.UUID, now: datetime
    ) -> int:
        """Record one failed verification and return the new count.

        Incremented in the database rather than in Python so the value cannot
        be lost to a concurrent verification reading a stale object. The caller
        commits before returning its error — an attempt counter that rolls back
        with the failure response is not a counter.
        """
        result = await db.execute(
            update(LoginOtp)
            .where(LoginOtp.id == row_id)
            .values(
                verification_attempts=LoginOtp.verification_attempts + 1,
                updated_at=now,
            )
            .returning(LoginOtp.verification_attempts)
        )
        return int(result.scalar_one())

    @staticmethod
    async def invalidate_code(
        db: AsyncSession, row_id: uuid.UUID, reason: str, now: datetime
    ) -> None:
        """Kill the current code without ending the series.

        Setting otp_hash to NULL means no submitted value can ever match, while
        leaving the series available for a resend if budget remains. This is
        what lets attempt exhaustion invalidate a code without locking the
        address.
        """
        await db.execute(
            update(LoginOtp)
            .where(LoginOtp.id == row_id)
            .values(otp_hash=None, invalidated_reason=reason, updated_at=now)
        )

    @staticmethod
    async def consume(db: AsyncSession, row_id: uuid.UUID, now: datetime) -> bool:
        """Atomically mark the code used. Returns False if it already was.

        ``consumed_at IS NULL`` in the WHERE clause is the mutex, not the
        earlier SELECT: two verifications submitting the same correct code
        concurrently both pass the comparison, and exactly one of them updates
        a row here. The loser gets False and is answered with the same generic
        error as any other failure.

        otp_hash is nulled in the same statement so the used code cannot be
        replayed even if consumed_at were somehow cleared.
        """
        result = await db.execute(
            update(LoginOtp)
            .where(LoginOtp.id == row_id, LoginOtp.consumed_at.is_(None))
            .values(consumed_at=now, otp_hash=None, updated_at=now)
        )
        return result.rowcount == 1

    @staticmethod
    async def delete_stale(
        db: AsyncSession, older_than: datetime, now: datetime
    ) -> int:
        """Delete finished series last used before ``older_than``.

        Rows whose lock is still in force are kept regardless of age: deleting
        one would hand the address a fresh resend budget and silently cancel
        its lockout. A lock that has already expired is no longer protecting
        anything, so those rows are collectable. Returns the rows removed.

        Housekeeping only — the unique constraint already bounds this table to
        one row per address, so growth is proportional to the number of
        addresses that have ever attempted OTP login, not to attempts.
        """
        result = await db.execute(
            delete(LoginOtp).where(
                LoginOtp.last_sent_at < older_than,
                or_(
                    LoginOtp.locked_until.is_(None),
                    LoginOtp.locked_until <= now,
                ),
            )
        )
        return int(result.rowcount or 0)
