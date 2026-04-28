from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

_logger = logging.getLogger("rivollo.db")

_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker[AsyncSession]] = None

# Cached Azure AD token — written once at startup, then kept fresh by
# token_refresh_loop(). The do_connect listener reads this without any I/O.
_cached_token: Optional[str] = None
_token_expires_on: float = 0.0
_credential = None  # ManagedIdentityCredential instance


def _ensure_async_url(url: str) -> str:
	if url.startswith("postgresql+asyncpg://"):
		return url
	if url.startswith("postgres://"):
		return url.replace("postgres://", "postgresql+asyncpg://", 1)
	if url.startswith("postgresql://"):
		return url.replace("postgresql://", "postgresql+asyncpg://", 1)
	if url.startswith("postgresql+psycopg2://"):
		return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
	if url.startswith("postgresql+psycopg://"):
		return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
	return url


def _attach_managed_identity_token_refresh(engine: AsyncEngine) -> None:
	global _cached_token, _token_expires_on, _credential
	from azure.identity import ManagedIdentityCredential

	_credential = ManagedIdentityCredential(client_id=settings.MANAGED_IDENTITY_CLIENT_ID)

	# Blocking fetch once at startup — before the server accepts any requests.
	token = _credential.get_token(settings.MANAGED_IDENTITY_TOKEN_SCOPE)
	_cached_token = token.token
	_token_expires_on = float(token.expires_on)

	@event.listens_for(engine.sync_engine, "do_connect")
	def _inject_token(dialect, conn_rec, cargs, cparams):  # noqa: ANN001
		if _cached_token is None:
			raise RuntimeError("Managed Identity token not available — startup token fetch may have failed")
		cparams["password"] = _cached_token


async def token_refresh_loop() -> None:
	"""Background task: refresh the cached Azure AD token before it expires.

	Sleeps until 15 minutes before the token's actual expiry, then fetches a
	fresh token via a thread (get_token is blocking). On failure it logs a
	warning and retries after 60 seconds — the loop never exits.
	"""
	global _cached_token, _token_expires_on
	while True:
		# Sleep until 15 min before expiry. min=60 prevents a tight retry loop
		# if the token is already expired or the refresh repeatedly fails.
		sleep_seconds = max(60.0, _token_expires_on - time.time() - 15 * 60)
		await asyncio.sleep(sleep_seconds)
		try:
			token = await asyncio.to_thread(
				_credential.get_token, settings.MANAGED_IDENTITY_TOKEN_SCOPE
			)
			_cached_token = token.token
			_token_expires_on = float(token.expires_on)
		except Exception:
			_logger.warning("Managed Identity token refresh failed, retrying next cycle")


def init_engine_and_session() -> None:
	global _engine, _SessionLocal
	if _engine is not None:
		return
	if not settings.DATABASE_URL:
		raise RuntimeError("DATABASE_URL is not configured. Set it in the environment or .env file.")
	database_url = _ensure_async_url(settings.DATABASE_URL)
	_engine = create_async_engine(database_url, pool_size=20, max_overflow=10, pool_recycle=1800, pool_pre_ping=False, future=True)

	if settings.USE_MANAGED_IDENTITY:
		_attach_managed_identity_token_refresh(_engine)

	_SessionLocal = async_sessionmaker(bind=_engine, expire_on_commit=False)


async def dispose_engine() -> None:
	if _engine is not None:
		await _engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
	if _SessionLocal is None:
		init_engine_and_session()
	assert _SessionLocal is not None
	async with _SessionLocal() as session:
		yield session
