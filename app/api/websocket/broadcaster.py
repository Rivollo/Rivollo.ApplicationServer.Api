"""
ProductStatusBroadcaster — central LISTEN/NOTIFY fan-out hub.

Architecture
────────────
Maintains exactly ONE raw asyncpg LISTEN connection for the entire
FastAPI process lifetime. All active WebSocket clients share it.

When pg_notify fires on channel 'tbl_product_status', asyncpg calls
_on_notify() synchronously in the event loop. The callback extracts
the product_id from the payload and puts the notification into every
asyncio.Queue that is currently subscribed to that product_id.

                   broadcaster (singleton)
                   ┌──────────────────────────┐
  pg_notify ──────►│  _conn (1 LISTEN conn)   │
                   │                          │
                   │  _subscribers = {        │
                   │   "uuid-A": {Q1, Q2}     │ ← 2 users, same product
                   │   "uuid-B": {Q3}         │ ← 1 user, other product
                   │  }                       │
                   └────┬──────────┬──────────┘
                        │          │
                  Queue Q1    Queue Q2
                        │          │
                  WS handler  WS handler

Scale benefit: 100 concurrent WebSocket clients = 1 DB connection,
not 100.

Lifecycle
─────────
  broadcaster.start() → called once in lifespan startup
  broadcaster.stop()  → called once in lifespan shutdown

Resilience
──────────
  - Keepalive SELECT 1 every 25s prevents Azure LB idle drops
  - Auto-reconnects on failure with exponential backoff (1s→60s)
  - Subscriber queues preserved across reconnects
  - _connect_lock prevents simultaneous connect attempts
  - _on_notify snapshots subscriber set before iterating (safe copy)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Optional

from app.core.config import settings

_logger = logging.getLogger("rivollo.ws.broadcaster")

CHANNEL                 = settings.WS_NOTIFY_CHANNEL
KEEPALIVE_INTERVAL_SECS = 25    # well under Azure LB 4-minute idle timeout
QUEUE_MAX_SIZE          = 50    # per-subscriber; status changes are infrequent


class ProductStatusBroadcaster:

    def __init__(self) -> None:
        self._conn: Optional[object] = None           # asyncpg.Connection
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._running:       bool                     = False
        self._keepalive_task: Optional[asyncio.Task] = None
        # Two locks with distinct responsibilities:
        #   _subscribers_lock — protects subscribe/unsubscribe dict mutations
        #   _connect_lock     — prevents simultaneous connection attempts
        self._subscribers_lock = asyncio.Lock()
        self._connect_lock     = asyncio.Lock()

    # -----------------------------------------------------------------------
    # Public API — called by application code
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """
        Called once at app startup from lifespan.

        Launches both the keepalive loop and initial connection as
        background tasks so app startup is never blocked — even if
        the database is temporarily unreachable at boot time.
        """
        self._running = True
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name="ws_broadcaster_keepalive",
        )
        # Fire initial connection in background; keepalive retries if it fails.
        asyncio.create_task(
            self._connect(),
            name="ws_broadcaster_initial_connect",
        )
        _logger.info("ProductStatusBroadcaster started (channel=%s)", CHANNEL)

    async def stop(self) -> None:
        """Called once at app shutdown from lifespan."""
        self._running = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
        await self._close_connection()
        _logger.info("ProductStatusBroadcaster stopped")

    async def subscribe(self, product_id: str) -> asyncio.Queue:
        """
        Register a WebSocket client as a subscriber for product_id.

        Returns a dedicated asyncio.Queue that will receive pg_notify
        payloads for this product. Must be called BEFORE the caller
        queries the current status from DB — this eliminates the race
        condition where a notification fires between query and subscribe.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        async with self._subscribers_lock:
            self._subscribers[product_id].add(queue)
        _logger.debug(
            "Subscribed to product %s (subscribers for this product: %d)",
            product_id,
            len(self._subscribers[product_id]),
        )
        return queue

    async def broadcast_to_product(self, product_id: str, payload: dict) -> None:
        """
        Push a custom payload directly to all WebSocket subscribers for product_id.

        Used by background tasks (e.g. GPU warmth estimation) to send messages
        that don't originate from a PostgreSQL NOTIFY. Mirrors the snapshot
        pattern used by _on_notify — no lock needed for a defensive copy.
        """
        queues = list(self._subscribers.get(product_id, set()))
        for queue in queues:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                _logger.warning(
                    "Subscriber queue full for product %s — broadcast dropped", product_id
                )

    async def unsubscribe(self, product_id: str, queue: asyncio.Queue) -> None:
        """
        Remove a WebSocket client's queue.

        Called in the finally block of the handler — guaranteed to run
        on disconnect, error, or normal termination. If this is the last
        subscriber for a product, the product key is removed from the dict.
        """
        async with self._subscribers_lock:
            self._subscribers[product_id].discard(queue)
            if not self._subscribers[product_id]:
                del self._subscribers[product_id]
        _logger.debug("Unsubscribed from product %s", product_id)

    # -----------------------------------------------------------------------
    # Notification callback — called by asyncpg synchronously
    # -----------------------------------------------------------------------

    def _on_notify(
        self, _conn: object, _pid: int, _channel: str, payload: str
    ) -> None:
        """
        asyncpg calls this synchronously in the event loop when a pg_notify
        arrives on CHANNEL. Must be fast and non-blocking — no awaits.

        Snapshots the subscriber set before iterating so that concurrent
        subscribe/unsubscribe operations (which run between event loop ticks)
        do not cause 'dict changed size during iteration' errors.
        """
        try:
            data = json.loads(payload)
            product_id = str(data.get("product_id", ""))
            if not product_id:
                _logger.warning(
                    "pg_notify payload missing product_id: %s", payload[:200]
                )
                return

            # Snapshot — defensive copy before iterating.
            queues = list(self._subscribers.get(product_id, set()))
            if not queues:
                return  # nobody is watching this product right now

            for queue in queues:
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    _logger.warning(
                        "Subscriber queue full for product %s — notification dropped",
                        product_id,
                    )
        except json.JSONDecodeError:
            _logger.error(
                "Invalid JSON in pg_notify payload: %s", payload[:200]
            )
        except Exception:
            _logger.exception("Unexpected error in _on_notify callback")

    # -----------------------------------------------------------------------
    # Connection management — private
    # -----------------------------------------------------------------------

    async def _connect(self) -> None:
        """
        Open the LISTEN connection with exponential backoff retry.

        Protected by _connect_lock so that the keepalive loop and the
        initial connect task cannot both try to open a connection at the
        same time. Double-checks whether a connection already exists after
        acquiring the lock (another task may have connected while we waited).
        """
        async with self._connect_lock:
            # Double-check: another coroutine may have connected first.
            if self._conn is not None and not self._conn.is_closed():
                return

            from app.core.db_listen import create_listen_connection  # noqa: PLC0415

            backoff = 1.0
            while self._running:
                try:
                    conn = await create_listen_connection()
                    await conn.add_listener(CHANNEL, self._on_notify)
                    self._conn = conn
                    _logger.info(
                        "LISTEN connection established (channel=%s)", CHANNEL
                    )
                    return
                except Exception as exc:
                    _logger.warning(
                        "LISTEN connect failed: %s — retry in %.0fs", exc, backoff
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)

    async def _close_connection(self) -> None:
        """Remove the LISTEN registration and close the connection cleanly."""
        conn = self._conn
        self._conn = None
        if conn is None:
            return
        try:
            if not conn.is_closed():
                await conn.remove_listener(CHANNEL, self._on_notify)
                await conn.close()
                _logger.debug("LISTEN connection closed")
        except Exception as exc:
            _logger.debug("Error closing LISTEN connection: %s", exc)

    async def _keepalive_loop(self) -> None:
        """
        Background task — runs every KEEPALIVE_INTERVAL_SECS seconds.

        Two responsibilities:
          1. Send SELECT 1 to keep the Azure PG connection alive.
             Azure LB silently drops idle TCP connections after ~4 minutes.
             At 25s interval we keep the connection alive with headroom.
          2. Detect a dropped connection and reconnect.

        If SELECT 1 fails → connection is dead → close and reconnect.
        If _conn is None or already closed → reconnect directly.
        """
        while self._running:
            await asyncio.sleep(KEEPALIVE_INTERVAL_SECS)
            if not self._running:
                break
            try:
                conn = self._conn
                if conn is None or conn.is_closed():
                    _logger.warning(
                        "LISTEN connection unavailable — reconnecting"
                    )
                    await self._connect()
                else:
                    await conn.fetchval("SELECT 1")
                    _logger.debug("LISTEN keepalive OK")
            except Exception as exc:
                _logger.warning(
                    "LISTEN keepalive failed: %s — reconnecting", exc
                )
                await self._close_connection()
                await self._connect()


# Singleton — one instance for the entire process lifetime.
# Imported by product_status.py (handler) and main.py (lifecycle).
broadcaster = ProductStatusBroadcaster()
