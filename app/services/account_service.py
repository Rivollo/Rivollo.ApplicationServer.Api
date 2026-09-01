"""Account management service — handles account deletion and related operations."""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_password
from app.database.users_repo import UserRepository
from app.models.models import AuthIdentity, AuthProvider, Product, User
from app.services.subscription_guard import (
    ACTIVE_SUBSCRIPTION_BLOCK_DETAIL,
    get_blocking_subscription,
)

logger = logging.getLogger(__name__)

# How long a deleted account stays recoverable before the purge job may erase it.
# Stored per-account in tbl_users.purge_after rather than applied as a constant at
# purge time, so an individual account can be put on hold or extended without a
# code change and without this value having to stay stable forever.
ACCOUNT_RETENTION_DAYS = 30

# What the user types to confirm a deletion. A constant rather than a literal
# because it appears in the check, in the error message the client shows, and in
# the tests — three places that must never disagree about what we accept.
DELETE_CONFIRMATION_PHRASE = "confirm"


@dataclass(frozen=True)
class AccountDeletionResult:
    """Outcome of a deletion request.

    ``already_pending`` distinguishes a fresh deletion from a repeat request that
    was intentionally ignored, so the caller can tell "we just scheduled this"
    from "this was already scheduled" without comparing timestamps.
    """

    deleted_at: datetime
    purge_after: datetime
    already_pending: bool


@dataclass(frozen=True)
class AccountRestoreResult:
    """Outcome of a successful restore.

    ``products_restored`` counts only products carrying the deletion fingerprint,
    so it is the number brought back — not the number the account owns.
    """

    user_id: uuid.UUID
    restored_at: datetime
    products_restored: int


class AccountService:

    @staticmethod
    async def delete_account(
        db: AsyncSession,
        user: User,
        confirmation: Optional[str],
    ) -> AccountDeletionResult:
        """Schedule an account for deletion, recoverable for 30 days.

        Every account, however it signs in, supplies the same
        confirmation = "confirm" (DELETE_CONFIRMATION_PHRASE).

        This is a check of INTENT, not of ownership. Sign-in moved to email OTP,
        so there is no password left to re-ask for here and an OTP-only account
        has no second credential to present. Ownership is established once, at
        sign-in; the bearer of a valid token is taken to be the owner. The typed
        phrase is here to stop an accidental or mis-clicked deletion.

        Refused with 409 while a paid subscription is still running — the customer
        cancels first and deletes once the billing period ends. Nothing here talks
        to Razorpay; deletion neither cancels a subscription nor touches payment
        records, which stay with the gateway for financial compliance.

        What this does:
        1. Verifies the typed confirmation.
        2. Refuses if a gateway-backed subscription is still running.
        3. Marks the user pending deletion: is_active = false (access kill-switch),
           deleted_at = now, purge_after = now + ACCOUNT_RETENTION_DAYS.
        4. Soft-deletes all products owned by the user (created_by = user.id).
        5. Commits — single atomic transaction; any failure rolls everything back.

        Nothing is erased here. AuthIdentity, subscriptions, products, assets and
        blobs all survive untouched, because every one of them is needed to put the
        account back. Permanent erasure is the purge job's job, and it may not run
        before purge_after.

        After this call:
        - All JWTs for this user return 401 (deps.py filters deleted_at IS NULL).
        - Login with the same email returns 401.
        - All published product links go dark (products are soft-deleted).
        - The email stays claimed, so it cannot be re-registered while the original
          account is still restorable.
        """
        _verify_confirmation(confirmation)

        # A running paid subscription blocks deletion outright. The customer
        # cancels first, keeps the period they paid for, and deletes once it ends.
        #
        # Enforced here rather than in the route so it cannot be bypassed by
        # calling the API directly — the frontend greying out the button is a
        # convenience, not the control.
        #
        # Checked after the confirmation so a caller who has not typed the phrase
        # is answered with the confirmation error rather than with the account's
        # billing state.
        blocking = await get_blocking_subscription(db, user.id)
        if blocking is not None:
            logger.info(
                "Account deletion refused, paid subscription still running | "
                "user_id=%s | subscription_id=%s | period_end=%s",
                user.id,
                blocking.id,
                blocking.current_period_end,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=ACTIVE_SUBSCRIPTION_BLOCK_DETAIL,
            )

        now = datetime.now(timezone.utc)
        purge_after = now + timedelta(days=ACCOUNT_RETENTION_DAYS)

        # The `deleted_at IS NULL` guard is what makes this idempotent, and it has
        # to live in the WHERE clause rather than in a prior read: two concurrent
        # requests carrying the same still-valid token would both pass an
        # if-already-deleted check in Python and the second would push purge_after
        # another 30 days out. As a WHERE clause only one statement can match, so a
        # repeat request cannot extend the recovery window.
        #
        # RETURNING tells us which of the two happened without a second round-trip.
        result = await db.execute(
            update(User)
            .where(
                User.id == user.id,
                User.deleted_at.is_(None),
            )
            .values(
                is_active=False,
                deleted_at=now,
                purge_after=purge_after,
                updated_date=now,
            )
            .returning(User.deleted_at, User.purge_after)
        )
        row = result.first()

        if row is None:
            # Already pending deletion. Report the dates that are actually in
            # force — not the ones we just computed and threw away — and leave the
            # products alone, since they were soft-deleted by the first request.
            existing = await db.execute(
                select(User.deleted_at, User.purge_after).where(User.id == user.id)
            )
            prior = existing.first()
            # Defensive: the row is guaranteed to exist (the caller is holding it),
            # so this only fires if it vanished mid-request.
            prior_deleted_at, prior_purge_after = (
                prior if prior is not None else (now, purge_after)
            )
            await db.commit()
            logger.info(
                "Account deletion re-requested, window unchanged | user_id=%s | "
                "deleted_at=%s | purge_after=%s",
                user.id,
                prior_deleted_at,
                prior_purge_after,
            )
            return AccountDeletionResult(
                deleted_at=prior_deleted_at,
                purge_after=prior_purge_after,
                already_pending=True,
            )

        deleted_at, scheduled_purge_after = row

        # Soft-delete the user's products, stamping them with the *same* instant
        # written to the user row. That shared timestamp is the fingerprint restore
        # uses to tell these products apart from ones the user had already deleted
        # themselves, so `now` must stay a single value bound into both statements
        # — recomputing it here would silently break restore.
        #
        # Only deleted_at is written. updated_date/updated_by are deliberately left
        # alone: deleted_at already records when the cascade ran, and overwriting
        # the update metadata would destroy the products' real edit history and
        # flatten the list ordering (products_repo sorts on updated_date) with no
        # way to recover it on restore.
        await db.execute(
            update(Product)
            .where(
                Product.created_by == user.id,
                Product.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )

        await db.commit()

        logger.info(
            "Account scheduled for deletion | user_id=%s | email=%s | purge_after=%s",
            user.id,
            user.email,
            scheduled_purge_after,
        )
        return AccountDeletionResult(
            deleted_at=deleted_at,
            purge_after=scheduled_purge_after,
            already_pending=False,
        )

    @staticmethod
    async def restore_account(
        db: AsyncSession,
        email: str,
        password: Optional[str] = None,
        google_sub: Optional[str] = None,
    ) -> AccountRestoreResult:
        """Undo a pending deletion, returning the account to its pre-deletion state.

        ``google_sub`` is the verified subject of a Google ID token, already
        checked against Google by the caller. It arrives pre-verified because that
        check is a network round-trip and this method holds a row lock.

        Order of operations matters:
        1. Lock the user row (soft-deleted rows included — every other lookup in
           the codebase hides exactly the rows restore needs).
        2. Verify ownership, before disclosing anything about the account's state.
        3. Check the account is pending deletion and still inside its window.
        4. Capture deleted_at *before* clearing it — it is the fingerprint that
           identifies which products this deletion took down.
        5. Clear the user's deletion state, then un-delete only the fingerprinted
           products.
        6. Commit once.

        Restore writes deleted_at and nothing else on products. status is left
        untouched, so a product the user had archived before deleting their
        account comes back archived rather than silently republished.
        """
        now = datetime.now(timezone.utc)

        user = await UserRepository.get_for_restore(db, email)
        if user is None:
            # Same 401 as bad credentials: a distinct "no such account" would let
            # anyone probe which addresses have deleted accounts.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials.",
            )

        await _verify_restore_identity(db, user, password, google_sub)

        if user.deleted_at is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This account is not pending deletion.",
            )

        if user.purge_after is None or user.purge_after <= now:
            # Past the window the data may already be partly erased, so a
            # "restored" account could come back pointing at deleted files. Refuse
            # cleanly instead.
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=(
                    "The recovery period for this account has ended and it can no "
                    "longer be restored."
                ),
            )

        products_restored = await _apply_restore(db, user, now)
        if products_restored is None:
            # Another request restored this account between our lock and here.
            # Nothing was written; report success is wrong, so surface the race.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This account is not pending deletion.",
            )

        await db.commit()

        logger.info(
            "Account restored | user_id=%s | email=%s | products_restored=%d",
            user.id,
            user.email,
            products_restored,
        )
        return AccountRestoreResult(
            user_id=user.id,
            restored_at=now,
            products_restored=products_restored,
        )

    @staticmethod
    async def restore_on_sign_in(
        db: AsyncSession,
        user: User,
    ) -> Optional[AccountRestoreResult]:
        """Undo a pending deletion because the owner just signed in.

        This is the ordinary way an account comes back. POST /auth/account/restore
        still exists for clients that want to ask explicitly, but nobody has to
        call it: signing in is already proof of ownership — the same proof restore
        asks for — so demanding a second, separate step would only strand people
        who deleted an account they turned out to still want.

        Call only once the credential has been verified. Returns None when the
        account was not pending deletion, which is every ordinary sign-in, so
        callers can read None as "carry on".

        Two things differ from restore_account, and both follow from the caller:

        * Ownership is already proven, so there is no identity check here. The
          caller must not skip its own.
        * Losing the race is not an error. restore_account raises 409 because a
          client that explicitly asked to restore deserves to know its request
          did nothing; a sign-in only needs the account live, and whoever won the
          race left it exactly that way. It signs the user in either way.

        Past the window this raises 403 rather than restoring. The purge job may
        already have erased data by then, so a "restored" account could come back
        pointing at files that are gone.
        """
        if user.deleted_at is None:
            return None  # fast path: ordinary sign-in, no lock taken

        now = datetime.now(timezone.utc)

        # The same locked lookup restore_account uses, for the same reasons:
        # every other user lookup in this codebase hides exactly the rows this
        # needs, and FOR UPDATE makes a concurrent restore or delete queue rather
        # than interleave. It also re-reads the row, so the check above — made
        # against a possibly stale object — is not the one that decides.
        locked = await UserRepository.get_for_restore(db, user.email)
        if locked is None or locked.deleted_at is None:
            # Already live: another request won, or the object we were handed was
            # stale. Either way the account is in the state this caller wanted.
            return None

        if locked.purge_after is None or locked.purge_after <= now:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "This account was deleted and its recovery period has ended. "
                    "It can no longer be restored."
                ),
            )

        products_restored = await _apply_restore(db, locked, now)
        if products_restored is None:
            return None

        await db.commit()

        logger.info(
            "Account restored on sign-in | user_id=%s | email=%s | products_restored=%d",
            locked.id,
            locked.email,
            products_restored,
        )
        return AccountRestoreResult(
            user_id=locked.id,
            restored_at=now,
            products_restored=products_restored,
        )


async def _apply_restore(
    db: AsyncSession, user: User, now: datetime
) -> Optional[int]:
    """Clear a pending deletion and bring back the products it took down.

    The caller has verified ownership, holds the row lock, and has checked the
    window. Returns the number of products restored, or None if the guarded
    UPDATE matched nothing — meaning another request got there first and this
    one wrote nothing at all. Does not commit; the caller owns the transaction.

    Shared by restore_account and restore_on_sign_in so the two entry points
    cannot drift. They differ in how they answer their caller, not in what they
    write, and what they write is the part that must stay identical.
    """
    # Captured before the clear below — this is the deletion fingerprint.
    original_deleted_at = user.deleted_at

    # Guarded exactly like deletion: the row lock already serialises callers,
    # and this WHERE clause means that even if one slipped through, only the
    # first restore can match.
    result = await db.execute(
        update(User)
        .where(
            User.id == user.id,
            User.deleted_at.is_not(None),
            User.purge_after > now,
        )
        .values(
            is_active=True,
            deleted_at=None,
            purge_after=None,
            updated_date=now,
        )
        .returning(User.id)
    )
    if result.first() is None:
        return None

    # Only products the cascade took down. A product the user deleted
    # themselves carries a different deleted_at and stays deleted — restoring
    # it would resurrect something they had chosen to remove, and if it was
    # published, put it back online.
    restored = await db.execute(
        update(Product)
        .where(
            Product.created_by == user.id,
            Product.deleted_at == original_deleted_at,
        )
        .values(deleted_at=None)
        .returning(Product.id)
    )
    return len(restored.fetchall())


async def _verify_restore_identity(
    db: AsyncSession,
    user: User,
    password: Optional[str],
    google_sub: Optional[str],
) -> None:
    """Raise HTTP 401 unless the caller proves they own this account.

    Deletion asks for proof of *intent* (typing the confirmation phrase); restore asks
    for proof of *ownership*. A typed phrase is worthless here — anyone who knows
    a deleted address could otherwise resurrect a stranger's account, along with
    every product it publishes.

    For Google accounts the proof is the AuthIdentity row that deletion now
    deliberately keeps: the credential's ``sub`` must match a stored
    provider_user_id for this user. That row is the restore credential, which is
    why permanent deletion is the only thing allowed to remove it.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials.",
    )

    if user.password_hash:
        if not password or not verify_password(password, user.password_hash):
            raise unauthorized
        return

    # Google account — no password to check against.
    if not google_sub:
        raise unauthorized

    result = await db.execute(
        select(AuthIdentity.id).where(
            AuthIdentity.user_id == user.id,
            AuthIdentity.provider == AuthProvider.GOOGLE,
            AuthIdentity.provider_user_id == google_sub,
        )
    )
    if result.first() is None:
        raise unauthorized


def _verify_confirmation(confirmation: Optional[str]) -> None:
    """Raise HTTP 400 unless the caller typed the confirmation phrase.

    One rule for every account. The old branch on ``password_hash`` is gone:
    with passwordless sign-in an OTP-only account has no password to check, so
    that branch would have silently handed exactly those users the weaker of
    the two gates while password holders kept the stronger one — a difference
    in security that nothing in the request would have explained.

    Compared after stripping and case-folding so a phone keyboard's
    autocapitalisation or a trailing space cannot block a deletion the user
    plainly meant. The phrase still has to be typed in full.

    Note this proves intent, not ownership — see delete_account.
    """
    if (
        not confirmation
        or confirmation.strip().casefold() != DELETE_CONFIRMATION_PHRASE.casefold()
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Please type "{DELETE_CONFIRMATION_PHRASE}" to delete your account.',
        )
