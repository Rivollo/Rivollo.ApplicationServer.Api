"""Email OTP login.

Authenticates an EXISTING user who proves control of their mailbox. This flow
never creates a user — signup remains the only way an account comes into
existence — and it never modifies the existing signup, password-login,
Google-login or password-reset implementations. It calls two of their building
blocks (AuthService's email lookups and AuthService.generate_token) without
changing either, and reuses AccountService.restore_on_sign_in unchanged.

THE SERIES MODEL
----------------
A row in tbl_login_otps is a challenge SERIES for one address, not one code.
The resend budget spans up to three codes and the lockout outlives all of them,
so neither fits on a per-code row. Issuing a code overwrites otp_hash, which is
what invalidates the previous one.

    initial request -> OTP #1   resend_count = 0, resends_remaining = 2
    resend #1       -> OTP #2   resend_count = 1, OTP #1 dead
    resend #2       -> OTP #3   resend_count = 2, OTP #2 dead
    next request    -> LOCKED   locked_until = now + 30 min

TWO INDEPENDENT COUNTERS
------------------------
``resend_count`` is series-scoped and caps how many codes are issued; hitting
its limit locks the flow for 30 minutes. ``verification_attempts`` is
code-scoped, caps wrong guesses at 5, resets to zero on every send, and on its
limit only invalidates the current code — it does NOT lock. A single counter
could not do both jobs: a resend would either wipe the guessing budget (making
resend a brute-force reset) or preserve it (so a fresh code arrives with
attempts already spent).

The ceiling is therefore 3 codes x 5 attempts = 15 guesses per address per 30
minutes, against a space of 10^6.

SIGNING IN HERE RESTORES AN ACCOUNT PENDING DELETION
----------------------------------------------------
A correct code inside the 30-day recovery window brings a deleted account back,
exactly as a correct password does on /auth/login. With passwordless sign-in
this flow is the ONLY way back for most accounts: an OTP-only user has no
password for /auth/login and no Google identity for /auth/google, and
/auth/account/restore accepts nothing else - so without this the recovery
window would exist with no door.

Two lookups therefore have to see deleted rows - request, to decide whether to
send, and verify, to find the account afterwards - which is why both call
get_user_by_email_including_deleted rather than get_user_by_email.

WHY SERIES ROWS EXIST FOR UNREGISTERED ADDRESSES
------------------------------------------------
Counters, cooldown, budget and lockout advance for ANY syntactically valid
address. The account lookup decides one thing only: whether an email is
actually sent. If rows existed only for real users, a locked real address would
answer 429 while an unregistered one answered 200 — a clean enumeration oracle
the moment a lockout exists. This way every response, including the lockout, is
identical for both.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Request, status

from app.api.deps import ACCOUNT_DEACTIVATED_DETAIL
from app.core.login_otp_config import otp_settings
from app.core.security import hash_token
from app.database.login_otp_repo import LoginOtpRepository
from app.models.models import User
from app.schemas.auth import AuthResponse, UserResponse
from app.services.account_service import AccountRestoreResult, AccountService
from app.services.activity_service import ActivityService
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)

# Every verification failure answers with this one string: no live challenge,
# wrong code, expired code, attempts exhausted, code superseded by a resend,
# code already consumed, or the account having vanished between request and
# verify. Distinct messages would tell a caller whether a live challenge exists
# for an address, and which of their guesses was closest to a real state.
INVALID_OTP_DETAIL = "Invalid or expired code."

# Returned by request for every outcome that is not a lockout — registered,
# unregistered, deactivated or pending deletion alike.
GENERIC_REQUEST_MESSAGE = (
    "If an account exists for this email, we've sent a sign-in code."
)

LOCK_REASON_RESEND_LIMIT = "resend_limit"
INVALIDATED_BY_ATTEMPTS = "verification_attempts"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _generate_otp() -> str:
    """A 6-digit code from the system CSPRNG.

    secrets.randbelow, NOT random.randint — which is what the existing signup
    and password-reset OTP flows use. Mersenne Twister is reconstructible from
    observed output, and OTPs are observable by anyone who can request one.
    Zero-padded so every code is exactly six characters.
    """
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_otp(otp: str) -> str:
    """Peppered SHA-256 of a code.

    The pepper is what makes SHA-256 sound here. Argon2 would add 50-100ms of
    CPU to every attempt including attacker-driven ones, which is a
    self-inflicted denial-of-service lever on a public endpoint; but bare
    SHA-256 over a 6-digit space is reversible in milliseconds from a leaked
    row. A server-side secret that never enters the database closes that,
    because a database-only compromise yields nothing to compute against.
    """
    return hash_token(f"{otp_settings.LOGIN_OTP_PEPPER}:{otp}")


@dataclass(frozen=True)
class OtpRequestResult:
    """Outcome of a request, and the delivery instruction if there is one."""

    locked: bool
    resends_remaining: int
    resend_available_in_seconds: int
    expires_in_minutes: int
    retry_after_seconds: int = 0
    # Set only when an email should actually go out. Carries the plaintext code
    # to the route, which hands it straight to a background task. It is never
    # logged, never stored, and never returned to the caller.
    deliver_otp: Optional[str] = None
    deliver_to_email: Optional[str] = None
    deliver_to_name: Optional[str] = None

    @property
    def should_send(self) -> bool:
        return self.deliver_otp is not None


class LoginOtpService:
    """Request and verify login OTPs."""

    @staticmethod
    def _invalid() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_OTP_DETAIL
        )

    @staticmethod
    def _locked(seconds: int) -> HTTPException:
        minutes = max(1, round(seconds / 60))
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Too many sign-in code requests for this email. "
                f"Please try again in about {minutes} minutes."
            ),
            headers={"Retry-After": str(max(1, seconds))},
        )

    # ------------------------------------------------------------------
    # Request
    # ------------------------------------------------------------------

    @staticmethod
    async def request_otp(
        db,
        email: str,
        request_ip: Optional[str] = None,
    ) -> OtpRequestResult:
        """Issue or resend a login code.

        Returns an OtpRequestResult in every non-lockout case; raises 429 only
        when the address is locked. The caller is responsible for actually
        sending the email (through BackgroundTasks) and for the audit row.
        """
        now = _utcnow()
        normalized = email.strip().lower()
        cooldown = otp_settings.LOGIN_OTP_RESEND_COOLDOWN_SECONDS
        expires_minutes = otp_settings.LOGIN_OTP_EXPIRES_MINUTES

        row = await LoginOtpRepository.get_for_update(db, normalized)

        # 1. Locked. Refuse without touching any counter — otherwise repeated
        #    attempts during a lock would extend it, and an attacker could keep
        #    an address locked indefinitely.
        if row is not None and row.locked_until is not None and row.locked_until > now:
            raise LoginOtpService._locked(
                int((row.locked_until - now).total_seconds())
            )

        # A series stays current while it is unconsumed, its lock (if any) has
        # not been set, and its last send is inside the series window. Anything
        # else means the next send starts a fresh series with a fresh budget:
        # a consumed series (successful login), an expired lock, or an
        # abandoned attempt older than the window.
        window = timedelta(minutes=otp_settings.LOGIN_OTP_SERIES_WINDOW_MINUTES)
        series_live = (
            row is not None
            and row.consumed_at is None
            and row.locked_until is None
            and row.last_sent_at is not None
            and row.last_sent_at > now - window
        )

        if series_live:
            assert row is not None and row.last_sent_at is not None  # for type checkers
            elapsed = (now - row.last_sent_at).total_seconds()

            # 2. Cooldown. Nothing is sent and the budget is NOT spent — a
            #    double-clicking client must not burn the user's resends.
            if elapsed < cooldown:
                return OtpRequestResult(
                    locked=False,
                    resends_remaining=max(0, row.max_resends - row.resend_count),
                    resend_available_in_seconds=int(cooldown - elapsed) + 1,
                    expires_in_minutes=expires_minutes,
                )

            # 3. Budget spent -> this is the fourth send attempt. Lock.
            if row.resend_count >= row.max_resends:
                locked_until = now + timedelta(
                    minutes=otp_settings.LOGIN_OTP_LOCKOUT_MINUTES
                )
                await LoginOtpRepository.apply_lock(
                    db, row.id, locked_until, LOCK_REASON_RESEND_LIMIT, now
                )
                await db.commit()
                logger.info(
                    "Login OTP locked after resend limit | ip=%s | until=%s",
                    request_ip,
                    locked_until.isoformat(),
                )
                raise LoginOtpService._locked(
                    int((locked_until - now).total_seconds())
                )

            resend_count = row.resend_count + 1
            max_resends = row.max_resends
            max_attempts = row.max_verification_attempts
        else:
            resend_count = 0
            max_resends = otp_settings.LOGIN_OTP_MAX_RESENDS
            max_attempts = otp_settings.LOGIN_OTP_MAX_VERIFICATION_ATTEMPTS

        otp = _generate_otp()
        await LoginOtpRepository.upsert_challenge(
            db,
            email=normalized,
            otp_hash=_hash_otp(otp),
            expires_at=now + timedelta(minutes=expires_minutes),
            now=now,
            resend_count=resend_count,
            max_resends=max_resends,
            max_verification_attempts=max_attempts,
            request_ip=request_ip,
        )
        await db.commit()

        # Only NOW does account existence enter the picture, and only to decide
        # whether mail goes out. Everything above ran identically for an
        # unregistered address, which is what keeps the responses symmetric.
        #
        # The lookup deliberately INCLUDES accounts pending deletion: signing in
        # inside the recovery window is how an account comes back, and this flow
        # is the only way left to do it for an account with no password and no
        # Google identity. Filtering here would leave those users a recovery
        # window with no door. This joins authenticate_email and
        # get_or_create_google_user on the short list of lookups that must see
        # deleted rows — see tests/test_sign_in_lookups_see_deleted_accounts.py.
        user = await AuthService.get_user_by_email_including_deleted(db, normalized)

        # is_active is false for two unrelated reasons: an account pending
        # deletion, which we want to reach so it can be restored, and one we
        # deactivated, which we do not. deleted_at is what tells them apart.
        #
        # Past the recovery window a code still goes out and verify answers with
        # the honest 403 — the same thing password sign-in does today, and
        # better than silence the user cannot distinguish from broken mail.
        deliver = user is not None and (user.is_active or user.deleted_at is not None)

        return OtpRequestResult(
            locked=False,
            resends_remaining=max(0, max_resends - resend_count),
            resend_available_in_seconds=cooldown,
            expires_in_minutes=expires_minutes,
            deliver_otp=otp if deliver else None,
            deliver_to_email=user.email if deliver and user else None,
            deliver_to_name=(user.name or user.email) if deliver and user else None,
        )

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    @staticmethod
    async def verify_otp(
        db,
        email: str,
        otp: str,
        remember_me: bool = False,
        request: Optional[Request] = None,
    ) -> AuthResponse:
        """Consume a code and authenticate the user behind the address.

        Raises 400 with one generic message for every failure, 403 if the
        account is deactivated, and 429 if the address is locked.
        """
        now = _utcnow()
        normalized = email.strip().lower()

        row = await LoginOtpRepository.get_for_update(db, normalized)
        if row is None:
            raise LoginOtpService._invalid()

        if row.locked_until is not None and row.locked_until > now:
            raise LoginOtpService._locked(
                int((row.locked_until - now).total_seconds())
            )

        # Consumed series, no live code (used, or killed by attempt
        # exhaustion), or an expired code. All indistinguishable to the caller.
        if row.consumed_at is not None or row.otp_hash is None:
            raise LoginOtpService._invalid()
        if row.expires_at is None or row.expires_at <= now:
            raise LoginOtpService._invalid()

        if not secrets.compare_digest(row.otp_hash, _hash_otp(otp)):
            attempts = await LoginOtpRepository.increment_attempts(db, row.id, now)
            if attempts >= row.max_verification_attempts:
                # Kill the code, keep the series: a resend is still allowed if
                # budget remains. Exhausting attempts deliberately does NOT
                # lock the address — locking here would let anyone lock a known
                # address out with five guesses, a cheaper denial-of-service
                # for no security gain, and would punish ordinary mistyping.
                await LoginOtpRepository.invalidate_code(
                    db, row.id, INVALIDATED_BY_ATTEMPTS, now
                )
            # Committed before the error is raised. A counter that rolls back
            # with the failure response is not a counter.
            await db.commit()
            raise LoginOtpService._invalid()

        # Correct code. The guarded UPDATE is the mutex: two concurrent
        # verifications both reach here, and exactly one updates a row.
        if not await LoginOtpRepository.consume(db, row.id, now):
            await db.commit()
            raise LoginOtpService._invalid()
        await db.commit()

        user = await AuthService.get_user_by_email_including_deleted(db, normalized)
        if user is None:
            # Reachable only if the account was purged between request and
            # verify. Generic, so it cannot be used to probe addresses.
            raise LoginOtpService._invalid()

        # A correct code inside the recovery window brings a deleted account
        # back, and sign-in then continues exactly as it would for anyone else.
        # Control of the mailbox is the same proof /auth/account/restore asks
        # for, so demanding a second, separate step would only strand people who
        # deleted an account they turned out to still want.
        #
        # Placed after the code check so it cannot be used to probe which
        # addresses have deleted accounts, and before the is_active check
        # because deletion sets is_active = false — reaching that check first
        # would answer "contact support" for something the user can undo
        # themselves. Raises 403 past the window.
        restored = await AccountService.restore_on_sign_in(db, user)
        if restored is not None:
            await ActivityService.log_auth_action(
                db=db,
                action="user.account.restored",
                user_id=user.id,
                request=request,
                metadata={
                    "products_restored": restored.products_restored,
                    "via": "otp",
                },
            )
            # _apply_restore wrote through Core UPDATE, so the ORM object still
            # carries the pre-restore is_active = false.
            await db.refresh(user)

        # Ordered after the code check for the same reason /auth/login orders
        # it after the password check: it discloses account state only to
        # someone who has already proven control of the mailbox.
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ACCOUNT_DEACTIVATED_DETAIL,
            )

        # The existing token mechanism, called and not modified. The result is
        # byte-compatible with a token from /auth/login: same payload shape, no
        # type claim, same secret and algorithm — so get_current_user accepts
        # it and every protected route works with it unchanged.
        token = AuthService.generate_token(user.id, remember_me)

        await ActivityService.log_auth_action(
            db=db,
            action="user.login.otp",
            user_id=user.id,
            request=request,
            metadata={"method": "email_otp"},
        )

        return LoginOtpService._build_auth_response(user, token, restored)

    @staticmethod
    def _build_auth_response(
        user: User,
        token: str,
        restored: Optional[AccountRestoreResult] = None,
    ) -> AuthResponse:
        """Construct exactly the body /auth/login returns.

        account_restored and products_restored are reported here the same way
        the password and Google paths report them, so a client can say "welcome
        back, your account has been restored" instead of dropping the user into
        an account they had asked us to delete with no acknowledgement that
        anything happened. Both stay at their defaults on an ordinary sign-in,
        which keeps the body key-for-key identical to the password-login
        response — a client can still handle both with no new code.
        """
        return AuthResponse(
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                created_at=user.created_at,
                updated_at=user.updated_at,
            ),
            token=token,
            account_restored=restored is not None,
            products_restored=restored.products_restored if restored else None,
        )
