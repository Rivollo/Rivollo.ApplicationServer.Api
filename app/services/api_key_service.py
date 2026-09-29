"""API key management and authentication (docs/api_keys.md).

Key format: ``riv_live_`` + 48 lowercase hex characters (24 random bytes from
``secrets``). Only ``hash_token(raw_key)`` (SHA-256) is stored; the raw key is
returned once, by ``create``, and never logged.

Ownership: every management call is scoped to the caller in the repository's
WHERE clause, and a key that is not the caller's is reported as 404, never 403
(the same rule as ADR-008), so key ids cannot be probed.

Authentication (``authenticate``) settles only the KEY: well-formed, known,
active, not expired. The OWNER's state (missing, deactivated, pending deletion)
is checked by the dependency in app/api/deps.py, so API-key callers get exactly
the messages a JWT caller gets from get_current_user.
"""

from __future__ import annotations

import logging
import re
import secrets
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import hash_token
from app.database.api_key_repo import api_key_repository as repo
from app.models.api_key import ALL_SCOPES, ApiKey
from app.models.models import User
from app.services.activity_service import ActivityService
from app.services.licensing_service import LicensingService

logger = logging.getLogger(__name__)

KEY_PREFIX = "riv_live_"
_SECRET_BYTES = 24
KEY_LENGTH = len(KEY_PREFIX) + _SECRET_BYTES * 2
# What the portal shows: "riv_live_" + 8 hex digits. Not secret.
DISPLAY_PREFIX_LENGTH = len(KEY_PREFIX) + 8
_KEY_PATTERN = re.compile(r"^riv_live_[0-9a-f]{48}$")

# One message for every key failure (unknown, malformed, revoked, expired), so a
# caller learns nothing about which keys exist.
INVALID_API_KEY_DETAIL = "Invalid, expired or revoked API key."
API_KEY_NOT_FOUND_DETAIL = "API key not found."

STATUS_ACTIVE = "active"
STATUS_REVOKED = "revoked"
STATUS_EXPIRED = "expired"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: Optional[datetime]) -> Optional[datetime]:
    """Treat a naive timestamp as UTC so comparisons never raise."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class CreatedApiKey:
    api_key: ApiKey
    raw_key: str


@dataclass(frozen=True)
class ApiKeyPrincipal:
    """Who is calling, when a route is authenticated by API key."""

    user: User
    api_key: ApiKey

    @property
    def user_id(self) -> uuid.UUID:
        return self.user.id

    def has_scope(self, scope: str) -> bool:
        return scope in (self.api_key.scopes or [])


class _AuthFailureLimiter:
    """Per-IP, per-minute count of FAILED key authentications.

    The same minute-bucket pattern as app/services/login_otp_rate_limit.py,
    restated rather than imported so API-key auth and OTP login cannot change
    each other's behaviour. Per-process: with N replicas a caller gets roughly
    N x the budget. That is acceptable because this is defence in depth — a key
    has 192 random bits, so guessing one is not a practical attack.
    """

    def __init__(self) -> None:
        self._counter: dict[tuple[str, str], int] = defaultdict(int)

    @staticmethod
    def _bucket() -> str:
        return _utcnow().strftime("%Y-%m-%dT%H:%M")

    def _prune(self, bucket: str) -> None:
        for key in [k for k in self._counter if k[1] != bucket]:
            del self._counter[key]

    def check(self, ip: str) -> None:
        """Raise 429 if this IP has already failed too often this minute."""
        bucket = self._bucket()
        self._prune(bucket)
        if self._counter[(ip, bucket)] >= settings.API_KEY_AUTH_FAILURES_PER_IP_PER_MINUTE:
            logger.warning("API key auth failure limit reached for ip=%s", ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts. Please wait a minute and try again.",
                headers={"Retry-After": "60"},
            )

    def record_failure(self, ip: str) -> None:
        bucket = self._bucket()
        self._prune(bucket)
        self._counter[(ip, bucket)] += 1

    def reset(self) -> None:
        """Clear all buckets. For tests only."""
        self._counter.clear()


auth_failure_limiter = _AuthFailureLimiter()


def client_ip(request: Optional[Request]) -> str:
    """Caller IP, resolved the way ActivityService resolves it."""
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


class ApiKeyService:
    """Create, list, revoke and authenticate API keys."""

    # ------------------------------------------------------------------ #
    # Key material
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_raw_key() -> str:
        return KEY_PREFIX + secrets.token_hex(_SECRET_BYTES)

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hash_token(raw_key)

    @staticmethod
    def display_prefix(raw_key: str) -> str:
        return raw_key[:DISPLAY_PREFIX_LENGTH]

    @staticmethod
    def looks_like_api_key(token: Optional[str]) -> bool:
        """Cheap shape check, so a JWT or junk never costs a database round trip."""
        return bool(token) and len(token) == KEY_LENGTH and bool(_KEY_PATTERN.match(token))

    @staticmethod
    def status_of(api_key: ApiKey, now: Optional[datetime] = None) -> str:
        if not api_key.isactive:
            return STATUS_REVOKED
        expires_at = _as_aware(api_key.expires_at)
        if expires_at is not None and expires_at <= (now or _utcnow()):
            return STATUS_EXPIRED
        return STATUS_ACTIVE

    # ------------------------------------------------------------------ #
    # Management (JWT-authenticated, from the portal)
    # ------------------------------------------------------------------ #
    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        name: str,
        scopes: Optional[list[str]] = None,
        expires_in_days: Optional[int] = None,
        request: Optional[Request] = None,
    ) -> CreatedApiKey:
        """Create a key. The returned ``raw_key`` must be shown once and dropped."""
        now = _utcnow()
        limit = settings.API_KEY_MAX_ACTIVE_PER_USER
        if await repo.count_usable_for_user(db, user_id, now) >= limit:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"You already have {limit} active API keys. "
                    "Revoke one before creating another."
                ),
            )

        clean_scopes = [s for s in ALL_SCOPES if s in set(scopes or ALL_SCOPES)]
        if not clean_scopes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one scope is required.",
            )

        raw_key = ApiKeyService.generate_raw_key()
        api_key = ApiKey(
            id=uuid.uuid4(),
            user_id=user_id,
            name=name,
            key_hash=ApiKeyService.hash_key(raw_key),
            key_prefix=ApiKeyService.display_prefix(raw_key),
            scopes=clean_scopes,
            expires_at=(now + timedelta(days=expires_in_days)) if expires_in_days else None,
            isactive=True,
            created_by=user_id,
            created_date=now,
        )
        repo.add(db, api_key)

        # log_activity commits, so the key and its audit row land together. The
        # prefix is safe to record; the raw key and the hash never are.
        try:
            await ActivityService.log_activity(
                db=db,
                action="apikey.created",
                user_id=user_id,
                target_type="api_key",
                target_id=str(api_key.id),
                metadata={"name": name, "key_prefix": api_key.key_prefix, "scopes": clean_scopes},
                request=request,
            )
        except Exception:
            await db.rollback()
            logger.exception("Could not save API key for user %s", user_id)
            raise

        logger.info("API key %s (%s) created for user %s", api_key.id, api_key.key_prefix, user_id)
        return CreatedApiKey(api_key=api_key, raw_key=raw_key)

    @staticmethod
    async def list_for_user(db: AsyncSession, user_id: uuid.UUID) -> list[ApiKey]:
        return await repo.list_for_user(db, user_id)

    @staticmethod
    async def get_for_user(db: AsyncSession, user_id: uuid.UUID, key_id: uuid.UUID) -> ApiKey:
        api_key = await repo.get_for_user(db, key_id, user_id)
        if api_key is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=API_KEY_NOT_FOUND_DETAIL)
        return api_key

    @staticmethod
    async def revoke(
        db: AsyncSession,
        user_id: uuid.UUID,
        key_id: uuid.UUID,
        *,
        request: Optional[Request] = None,
    ) -> ApiKey:
        """Switch a key off for good. Idempotent; the row is kept, never deleted."""
        api_key = await ApiKeyService.get_for_user(db, user_id, key_id)
        if not api_key.isactive:
            return api_key

        now = _utcnow()
        api_key.isactive = False
        api_key.revoked_at = now
        api_key.updated_by = user_id
        api_key.updated_date = now
        try:
            await ActivityService.log_activity(
                db=db,
                action="apikey.revoked",
                user_id=user_id,
                target_type="api_key",
                target_id=str(api_key.id),
                metadata={"key_prefix": api_key.key_prefix},
                request=request,
            )
        except Exception:
            await db.rollback()
            logger.exception("Could not revoke API key %s", key_id)
            raise

        logger.info("API key %s (%s) revoked by user %s", api_key.id, api_key.key_prefix, user_id)
        return api_key

    # ------------------------------------------------------------------ #
    # Authentication (called by app.api.deps for key-authenticated routes)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _invalid() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INVALID_API_KEY_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @staticmethod
    async def authenticate(
        db: AsyncSession, raw_key: Optional[str], *, ip: str = "unknown"
    ) -> tuple[ApiKey, Optional[User]]:
        """Resolve a presented key to (key, owner).

        Raises 401 for any key problem and 429 when this IP has failed too
        often. The owner is returned as loaded — possibly None, deactivated or
        pending deletion — for the caller to judge.
        """
        auth_failure_limiter.check(ip)

        if not ApiKeyService.looks_like_api_key(raw_key):
            auth_failure_limiter.record_failure(ip)
            raise ApiKeyService._invalid()

        api_key = await repo.get_by_hash(db, ApiKeyService.hash_key(raw_key))
        now = _utcnow()
        if api_key is None or ApiKeyService.status_of(api_key, now) != STATUS_ACTIVE:
            auth_failure_limiter.record_failure(ip)
            # Prefix only, and only when the key is known: never the raw value.
            logger.info(
                "API key authentication refused (%s)",
                api_key.key_prefix if api_key is not None else "unknown key",
            )
            raise ApiKeyService._invalid()

        await ApiKeyService._record_use(db, api_key, now)
        user = await repo.get_user(db, api_key.user_id)
        return api_key, user

    @staticmethod
    async def _record_use(db: AsyncSession, api_key: ApiKey, now: datetime) -> None:
        """Update last_used_at at most once per resolution window. Never fails a request."""
        resolution = timedelta(seconds=settings.API_KEY_LAST_USED_RESOLUTION_SECONDS)
        last_used = _as_aware(api_key.last_used_at)
        if last_used is not None and now - last_used < resolution:
            return
        try:
            await repo.touch_last_used(db, api_key.id, now, now - resolution)
            await db.commit()
            api_key.last_used_at = now
        except Exception:  # noqa: BLE001 - bookkeeping must not reject a valid key
            logger.warning("Could not record use of API key %s", api_key.id, exc_info=True)
            try:
                await db.rollback()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # Account summary for GET /api-keys/current
    # ------------------------------------------------------------------ #
    @staticmethod
    async def credits_for(db: AsyncSession, user_id: uuid.UUID) -> dict[str, Optional[int]]:
        """AI credits on the active licence. No licence = nothing available."""
        license_obj = await LicensingService.get_active_license(db, user_id)
        if license_obj is None:
            return {"limit": 0, "used": 0, "remaining": 0}
        limit = getattr(license_obj, "limit_max_ai_credits", None)
        used = int(getattr(license_obj, "usage_ai_credits", 0) or 0)
        if limit is None:
            return {"limit": None, "used": used, "remaining": None}
        return {"limit": int(limit), "used": used, "remaining": max(int(limit) - used, 0)}


api_key_service = ApiKeyService()
