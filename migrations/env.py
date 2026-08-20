from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from typing import Dict, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import settings
from app.models import Base

config = context.config


def _psycopg_ssl_param(url: str) -> str:
    """asyncpg spells the TLS option `ssl=require`; psycopg only accepts `sslmode`."""
    parts = urlsplit(url)
    if not parts.query:
        return url
    params = parse_qsl(parts.query, keep_blank_values=True)
    converted = [("sslmode", v) if k == "ssl" else (k, v) for k, v in params]
    if converted == params:
        return url
    return urlunsplit(parts._replace(query=urlencode(converted)))


def _get_database_url() -> str:
    url = settings.DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL is not configured; cannot run migrations.")
    # If app uses asyncpg, switch to psycopg for Alembic (sync)
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    # Normalize short scheme and plain postgresql to psycopg
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    # Avoid needing psycopg2 in environments; prefer psycopg v3
    elif url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    return _psycopg_ssl_param(url)


if config.config_file_name is not None:
	fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Escape % so configparser does not read it as interpolation syntax — URL-encoded
# passwords (e.g. %23 for "#") would otherwise raise ValueError at import time.
config.set_main_option("sqlalchemy.url", _get_database_url().replace("%", "%%"))


def include_object(object_, name, type_, reflected, compare_to) -> bool:
    """Keep autogenerate away from foreign keys entirely.

    43 foreign keys run from created_by / updated_by columns to tbl_users(id),
    and NONE of them are declared in the ORM — AuditMixin defines those columns
    as plain UUIDs with no ForeignKey (see app/models/models.py). Autogenerate
    compares the model against the database and emits DDL for whatever it finds
    only in the database, so it reads all 43 as constraints to DROP.

    They were converted to ON DELETE SET NULL by e7a15c93f0b2 so that erasing a
    user anonymises their audit trail instead of failing. Dropping them would
    hand the account purge job a schema that passes its contract check and then
    leaves dangling references to users that no longer exist.

    A comment cannot prevent that; this hook can. Foreign keys in this project
    are owned by migrations, never by autogenerate, so refusing the whole
    category is both the safe answer and an honest description of reality.

    Note that autogenerate is discouraged here regardless — the schema has
    drifted from the migration chain (columns exist that no revision created,
    and three modelled tables do not exist at all), so its output always needs
    reading line by line. This hook removes the single most destructive class of
    false positive; it does not make autogenerate trustworthy.
    """
    if type_ == "foreign_key_constraint":
        return False
    return True


def run_migrations_offline() -> None:
	context.configure(
		url=config.get_main_option("sqlalchemy.url"),
		target_metadata=target_metadata,
		literal_binds=True,
		dialect_opts={"paramstyle": "named"},
		include_object=include_object,
	)

	with context.begin_transaction():
		context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
	context.configure(
		connection=connection,
		target_metadata=target_metadata,
		include_object=include_object,
	)

	with context.begin_transaction():
		context.run_migrations()


async def run_migrations_online() -> None:
	configuration: Dict[str, Any] = config.get_section(config.config_ini_section, {})
	configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")

	connectable = async_engine_from_config(
		configuration,
		prefix="sqlalchemy.",
		poolclass=pool.NullPool,
	)

	async with connectable.connect() as connection:
		await connection.run_sync(do_run_migrations)

	await connectable.dispose()


if context.is_offline_mode():
	run_migrations_offline()
elif sys.platform == "win32":
	# psycopg's async mode cannot run on Windows' default ProactorEventLoop
	if sys.version_info >= (3, 12):
		asyncio.run(run_migrations_online(), loop_factory=asyncio.SelectorEventLoop)
	else:
		asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
		asyncio.run(run_migrations_online())
else:
	asyncio.run(run_migrations_online())
