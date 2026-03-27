from __future__ import annotations

from typing import AsyncGenerator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker[AsyncSession]] = None


def _ensure_async_url(url: str) -> str:
	"""Ensure the SQLAlchemy URL uses the asyncpg driver.

	Handles common provider formats like:
	- postgresql://...
	- postgres://...
	- postgresql+psycopg://... (or +psycopg2)

	Returns the URL unchanged if it's already asyncpg.
	"""
	# Already async
	if url.startswith("postgresql+asyncpg://"):
		return url

	# Normalize short scheme used by some providers (e.g. "postgres://")
	if url.startswith("postgres://"):
		return url.replace("postgres://", "postgresql+asyncpg://", 1)

	# Upgrade plain postgresql to asyncpg
	if url.startswith("postgresql://"):
		return url.replace("postgresql://", "postgresql+asyncpg://", 1)

	# Migrate psycopg/psycopg2 URLs to asyncpg for async engine
	if url.startswith("postgresql+psycopg2://"):
		return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
	if url.startswith("postgresql+psycopg://"):
		return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)

	return url


def _attach_managed_identity_token_refresh(engine: AsyncEngine) -> None:
	"""Register a SQLAlchemy event that injects a fresh Azure AD token as the
	password before every new database connection is opened.

	DefaultAzureCredential caches the token internally and only calls Azure when
	the token is close to expiry (~5 min before the 1-hour lifetime ends), so
	this is cheap to call on every new connection.
	"""
	from azure.identity import ManagedIdentityCredential

	_credential = ManagedIdentityCredential(client_id=settings.MANAGED_IDENTITY_CLIENT_ID)

	@event.listens_for(engine.sync_engine, "do_connect")
	def _inject_token(dialect, conn_rec, cargs, cparams):  # noqa: ANN001
		token = _credential.get_token(settings.MANAGED_IDENTITY_TOKEN_SCOPE)
		cparams["password"] = token.token


def init_engine_and_session() -> None:
	global _engine, _SessionLocal
	if _engine is not None:
		return
	if not settings.DATABASE_URL:
		raise RuntimeError("DATABASE_URL is not configured. Set it in the environment or .env file.")
	database_url = _ensure_async_url(settings.DATABASE_URL)
	_engine = create_async_engine(database_url, pool_pre_ping=True, future=True)

	if settings.USE_MANAGED_IDENTITY:
		# Attach the event listener that refreshes the Azure AD token before
		# each new connection.  No password is needed in DATABASE_URL.
		_attach_managed_identity_token_refresh(_engine)

	_SessionLocal = async_sessionmaker(bind=_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
	if _SessionLocal is None:
		init_engine_and_session()
	assert _SessionLocal is not None
	async with _SessionLocal() as session:
		yield session
