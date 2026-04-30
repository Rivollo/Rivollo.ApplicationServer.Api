"""
WebSocket handler — real-time product status tracking.

One instance of track_product_status() runs per connected browser client.

Exact sequence per connection
──────────────────────────────
  1. broadcaster.subscribe(product_id)     ← MUST be first (race condition)
  2. Short-lived SQLAlchemy session        ← query current status, close session
  3. Send current status to browser        ← "source": "initial_query"
  4. If already terminal → send done, return
  5. Main loop:
       a. wait_for(queue.get(), timeout=30)
       b. Notification arrived:
            - Ignore published / archived (post-pipeline statuses)
            - Forward all others to browser
            - On "ready" → send done, break
       c. Timeout (30s of silence):
            - Send keepalive to browser
            - Recovery poll: re-query DB directly
              (catches any notification missed during LISTEN reconnect)
            - If status is now "ready" → send done, break
  6. finally: unsubscribe — always runs regardless of exit reason

Race condition explanation (why subscribe comes before query)
──────────────────────────────────────────────────────────────
  Without this order:
    T=0  SELECT status → "processing"
    T=1  GPU updates → "ready" — pg_notify fires, nobody listening → LOST
    T=2  add_listener → waiting forever, browser never told it's done

  With this order:
    T=0  subscribe → queue exists in broadcaster._subscribers
    T=1  GPU updates → "ready" — pg_notify fires → goes into our queue
    T=2  SELECT status → "processing" → sent to browser
    T=3  queue.get() returns "ready" notification → sent to browser ✓

Recovery poll explanation (why we re-query on every 30s timeout)
─────────────────────────────────────────────────────────────────
  If the broadcaster's LISTEN connection drops briefly and reconnects,
  any pg_notify fired during that window is lost forever (PostgreSQL
  does not buffer missed notifications). The recovery poll re-queries
  the DB every 30s as a safety net, guaranteeing eventual consistency
  with at most 30s delay even if pg_notify completely fails.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.api.websocket.broadcaster import broadcaster
from app.models.models import Product

_logger = logging.getLogger("rivollo.ws.product_status")

# Only "ready" closes the WebSocket — the processing pipeline is done.
TERMINAL_STATUS = "ready"

# These statuses happen outside the automated pipeline (manual user actions).
# The trigger still fires for them; we silently discard the notification.
IGNORED_STATUSES = frozenset({"published", "archived"})

# Must stay below Azure LB idle timeout (~240s).
KEEPALIVE_SECONDS = 30


async def track_product_status(
    websocket: WebSocket,
    product_id: UUID,
) -> None:
    """
    Core handler — called once per WebSocket connection from main.py.
    Manages the full lifecycle for one browser client watching one product.
    """
    product_id_str = str(product_id)

    # -----------------------------------------------------------------------
    # STEP 1 — Subscribe FIRST (race condition prevention).
    #
    # The queue is registered in broadcaster._subscribers before we touch
    # the database. Any pg_notify that fires from this point forward goes
    # into the queue and will NOT be missed — even during the initial query.
    # -----------------------------------------------------------------------
    queue = await broadcaster.subscribe(product_id_str)

    try:
        # -------------------------------------------------------------------
        # STEP 2 — Query current status with a short-lived SQLAlchemy session.
        #
        # Open session → execute query → close session.
        # The DB connection returns to the pool before the WS loop starts.
        # This prevents pool exhaustion: a WebSocket can live for hours but
        # must not hold a DB connection for its entire lifetime.
        # -------------------------------------------------------------------
        import app.core.db as _db_core  # noqa: PLC0415

        if _db_core._SessionLocal is None:
            await websocket.send_json({
                "type":    "error",
                "message": "Service is initialising — please retry in a moment",
            })
            return

        async with _db_core._SessionLocal() as session:
            result = await session.execute(
                select(Product.status, Product.updated_date)
                .where(Product.id == product_id)
            )
            row = result.fetchone()
        # DB connection returned to pool here — before the loop below.

        # -------------------------------------------------------------------
        # STEP 3 — Product not found.
        # -------------------------------------------------------------------
        if row is None:
            _logger.info("WebSocket: product not found (id=%s)", product_id_str)
            await websocket.send_json({
                "type":    "error",
                "message": f"Product {product_id_str} not found",
            })
            return

        current_status = str(row.status)

        # -------------------------------------------------------------------
        # STEP 4 — Send current status immediately.
        #
        # Browser sees the actual current state without waiting for a change.
        # "source" field helps distinguish initial state from live updates.
        # -------------------------------------------------------------------
        await websocket.send_json({
            "type":         "status_update",
            "product_id":   product_id_str,
            "status":       current_status,
            "updated_date": (
                row.updated_date.isoformat() if row.updated_date else None
            ),
            "source": "initial_query",
        })

        # -------------------------------------------------------------------
        # STEP 5 — Already terminal on connect — no loop needed.
        #
        # Handles the case where the user connects after processing completed.
        # -------------------------------------------------------------------
        if current_status == TERMINAL_STATUS:
            _logger.info(
                "Product %s already in terminal state '%s'",
                product_id_str,
                current_status,
            )
            await websocket.send_json({
                "type":   "done",
                "status": current_status,
            })
            return

        # -------------------------------------------------------------------
        # STEP 6 — Main notification loop.
        # -------------------------------------------------------------------
        _logger.info(
            "Watching product %s — current status: %s",
            product_id_str,
            current_status,
        )

        while True:

            # Wait for a notification from broadcaster (30s timeout).
            notification = None
            try:
                notification = await asyncio.wait_for(
                    queue.get(), timeout=KEEPALIVE_SECONDS
                )
            except asyncio.TimeoutError:
                pass

            # ---------------------------------------------------------------
            # TIMEOUT BRANCH — 30s passed with no notification.
            # ---------------------------------------------------------------
            if notification is None:

                # 1. Keepalive ping — tells browser the connection is alive.
                try:
                    await websocket.send_json({"type": "keepalive"})
                except Exception:
                    # Browser already gone — exit cleanly.
                    break

                # 2. Recovery poll — re-query DB directly.
                #    Guards against notifications missed during LISTEN reconnect.
                recovered_status = await _query_current_status(
                    product_id, product_id_str
                )
                if recovered_status == TERMINAL_STATUS:
                    _logger.info(
                        "Recovery poll: product %s already '%s' — "
                        "notification was missed during reconnect",
                        product_id_str,
                        recovered_status,
                    )
                    await websocket.send_json({
                        "type":       "status_update",
                        "product_id": product_id_str,
                        "status":     recovered_status,
                        "source":     "recovery_poll",
                    })
                    await websocket.send_json({
                        "type":   "done",
                        "status": recovered_status,
                    })
                    break

                continue  # back to wait_for

            # ---------------------------------------------------------------
            # NOTIFICATION BRANCH — pg_notify received via broadcaster queue.
            # ---------------------------------------------------------------
            new_status = str(notification.get("new_status", ""))

            # Skip statuses outside the automated processing pipeline.
            # published and archived are manual user actions that happen
            # after the pipeline ends — they are irrelevant to this WebSocket.
            if new_status in IGNORED_STATUSES:
                _logger.debug(
                    "Ignoring '%s' notification for product %s",
                    new_status,
                    product_id_str,
                )
                continue

            _logger.info(
                "Product %s: %s → %s",
                product_id_str,
                notification.get("old_status"),
                new_status,
            )

            await websocket.send_json({
                "type":         "status_update",
                "product_id":   product_id_str,
                "status":       new_status,
                "old_status":   notification.get("old_status"),
                "updated_date": notification.get("updated_date"),
                "source":       "pg_notify",
            })

            if new_status == TERMINAL_STATUS:
                await websocket.send_json({
                    "type":   "done",
                    "status": new_status,
                })
                _logger.info(
                    "Product %s reached terminal status '%s' — closing WebSocket",
                    product_id_str,
                    new_status,
                )
                break

    except WebSocketDisconnect:
        _logger.info("WebSocket disconnected — product %s", product_id_str)

    except Exception:
        _logger.exception(
            "Unexpected error in WebSocket handler for product %s",
            product_id_str,
        )
        try:
            await websocket.send_json({
                "type":    "error",
                "message": "Internal server error",
            })
        except Exception:
            pass  # browser may already be gone

    finally:
        # Always runs — removes this client's queue from the broadcaster.
        # Runs on: normal close, browser disconnect, any exception.
        await broadcaster.unsubscribe(product_id_str, queue)
        _logger.debug("WebSocket cleanup done — product %s", product_id_str)


async def _query_current_status(
    product_id: UUID,
    product_id_str: str,
) -> str | None:
    """
    Open a short-lived SQLAlchemy session, query status, return it.
    Used only by the recovery poll on 30s timeout. Returns None on error.
    """
    try:
        import app.core.db as _db_core  # noqa: PLC0415

        if _db_core._SessionLocal is None:
            return None

        async with _db_core._SessionLocal() as session:
            result = await session.execute(
                select(Product.status).where(Product.id == product_id)
            )
            row = result.fetchone()

        return str(row.status) if row else None

    except Exception:
        _logger.exception(
            "Recovery poll query failed for product %s", product_id_str
        )
        return None
