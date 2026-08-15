"""Durable inbox for incoming webhook events.

Every webhook is written to tbl_webhook_events and **committed before any
processing happens**, so the payload survives whatever the handler does next.
That ordering is the whole point: previously the row was only flushed, so a
handler failure rolled it back along with the handler's own work, leaving no
record that the event had ever arrived.

`processed` is the state machine, not the row's existence:

    no row              never seen
    row, processed=0    arrived, not yet completed — replayable work
    row, processed=1    completed; a redelivery is a genuine duplicate, skip it

Keying idempotency on `processed` rather than on "did the INSERT conflict" is
what makes a failed event recoverable. Under the old check, a redelivery of an
event that had failed was skipped as a duplicate, so the first failure was final.

Concurrency is handled by claiming the row with SELECT ... FOR UPDATE SKIP
LOCKED. Two simultaneous deliveries of the same event cannot both process it:
the second finds the row locked, gets nothing back, and returns.

This is a database-backed queue, not a message broker. It is deliberately enough
for the current volume — a broker would add an extra delivery hop without
removing the need for this table, since the payload still has to be recorded
somewhere the handler can retry from.
"""

import logging
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.webhook_event import WebhookEvent

_logger = logging.getLogger("rivollo.webhook_inbox")


async def claim_event(
    db: AsyncSession,
    *,
    event_id: str,
    event: str,
    rz_subscription_id: str,
    payload: dict[str, Any],
) -> Optional[WebhookEvent]:
    """Record the event durably and claim it for processing.

    Returns the locked row when this caller owns the work, or None when there is
    nothing to do — either the event is already processed, or another delivery
    of the same event is processing it right now.

    The caller must finish with mark_processed() or record_failure().
    """
    # 1. Persist first, and commit. From here on the payload is recoverable no
    #    matter what the handler does.
    await db.execute(
        pg_insert(WebhookEvent)
        .values(
            event_id=event_id,
            event=event,
            rz_sub_id=rz_subscription_id,
            payload=payload,
            processed=False,
        )
        .on_conflict_do_nothing(index_elements=["event_id"])
    )
    await db.commit()

    # 2. Claim it. skip_locked means a concurrent delivery of the same event
    #    returns no row rather than blocking or double-processing.
    result = await db.execute(
        select(WebhookEvent)
        .where(WebhookEvent.event_id == event_id)
        .with_for_update(skip_locked=True)
    )
    row = result.scalars().first()

    if row is None:
        _logger.info(
            "Webhook event_id=%s is being processed by another delivery — skipping.",
            event_id,
        )
        return None

    if row.processed:
        _logger.info("Webhook event_id=%s already processed — skipping.", event_id)
        return None

    return row


async def mark_processed(db: AsyncSession, row: WebhookEvent) -> None:
    """Commit the handler's work and the completion flag in one transaction.

    Clears any error left by a previous failed attempt, so `error` always
    describes the most recent outcome.
    """
    row.processed = True
    row.error = None
    await db.commit()


async def record_failure(db: AsyncSession, *, event_id: str, error: str) -> None:
    """Roll back the handler's partial work, then record why it failed.

    The row itself is untouched by the rollback because it was committed before
    processing began, so this UPDATE always finds it — and it stays
    processed=false, which is what marks it as outstanding work.
    """
    await db.rollback()
    try:
        await db.execute(
            update(WebhookEvent)
            .where(WebhookEvent.event_id == event_id)
            .values(error=error[:2000])
        )
        await db.commit()
    except Exception:
        # Never let bookkeeping mask the original failure — it is already logged
        # by the caller.
        _logger.exception("Could not record webhook failure for event_id=%s", event_id)
        await db.rollback()


async def unprocessed_events(
    db: AsyncSession, *, limit: int = 50
) -> list[WebhookEvent]:
    """Events that arrived but never completed — outstanding work.

    Nothing consumes this yet. It is the query a reconciliation sweep would run
    (as a scheduled Container Apps Job, claiming rows with the same SKIP LOCKED
    semantics so it is safe alongside live webhook traffic).
    """
    result = await db.execute(
        select(WebhookEvent)
        .where(WebhookEvent.processed.is_(False))
        .order_by(WebhookEvent.created_at.asc())
        .limit(limit)
    )
    return list(result.scalars())
