"""
Raw asyncpg connection factory for PostgreSQL LISTEN/NOTIFY.

Used exclusively by the WebSocket broadcaster. Never used for
regular application queries — those go through the SQLAlchemy
session (get_db) as always.

Why raw asyncpg and not SQLAlchemy:
  SQLAlchemy pools and recycles connections. A LISTEN connection
  must stay permanently open with a registered callback and can
  never be returned to a pool. Raw asyncpg gives full control.

Managed Identity:
  Reads _cached_token from app.core.db at call time (module import,
  not value import) so the broadcaster always sees the latest token
  that token_refresh_loop() keeps fresh. Never reads it at import time.

Azure specifics:
  - ssl="require" is mandatory for Azure PostgreSQL
  - TCP keepalives prevent Azure LB from silently dropping idle connections
    (Azure LB idle timeout ~4 minutes; keepalive starts at 60s idle)
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse, unquote

import asyncpg

from app.core.config import settings

_logger = logging.getLogger("rivollo.db_listen")


async def create_listen_connection() -> asyncpg.Connection:
    """
    Open a dedicated asyncpg connection for LISTEN/NOTIFY.

    Called by the broadcaster on startup and on every reconnect.
    Each call reads the Managed Identity token fresh so token
    refreshes performed by token_refresh_loop() are always used.
    """
    host, port, user, database = _parse_database_url()

    connect_kwargs: dict = {
        "host":     host,
        "port":     port,
        "user":     user,
        "database": database,
        "ssl":      "require",
        "server_settings": {
            # TCP keepalives — prevents Azure LB silent idle drop
            "tcp_keepalives_idle":     "60",  # probe after 60s idle
            "tcp_keepalives_interval": "10",  # retry probe every 10s
            "tcp_keepalives_count":    "5",   # drop after 5 failures
            # Visible in pg_stat_activity for monitoring/debugging
            "application_name": "rivollo_listen",
        },
        "command_timeout": 30,
    }

    if settings.USE_MANAGED_IDENTITY:
        # Import module, not value — reads _cached_token at call time.
        # token_refresh_loop() in db.py reassigns _cached_token in place.
        # A direct `from app.core.db import _cached_token` would capture
        # the value at import time and never see refreshed tokens.
        import app.core.db as _db_core  # noqa: PLC0415

        token = _db_core._cached_token
        if token is None:
            raise RuntimeError(
                "Managed Identity token not yet available. "
                "Ensure token_refresh_loop() started before broadcaster."
            )
        connect_kwargs["password"] = token
    else:
        # Password auth — read from DATABASE_URL
        parsed = urlparse(_normalise_url(settings.DATABASE_URL))
        if parsed.password:
            connect_kwargs["password"] = unquote(parsed.password)

    _logger.debug(
        "Opening LISTEN connection → %s:%s/%s as %s", host, port, database, user
    )
    return await asyncpg.connect(**connect_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    """Strip SQLAlchemy driver prefix so urlparse works correctly."""
    for prefix in (
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
        "postgresql+psycopg://",
    ):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


def _parse_database_url() -> tuple[str, int, str, str]:
    """Return (host, port, user, database) parsed from settings.DATABASE_URL."""
    parsed = urlparse(_normalise_url(settings.DATABASE_URL))
    host     = parsed.hostname or ""
    port     = parsed.port or 5432
    user     = parsed.username or ""
    database = parsed.path.lstrip("/")
    return host, port, user, database
