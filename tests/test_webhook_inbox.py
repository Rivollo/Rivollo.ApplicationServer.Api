"""The webhook inbox must record an event before anything can lose it.

The bug these pin down: the event row used to be flushed but not committed, so a
handler failure rolled it back together with the handler's own work. The webhook
then returned 200, so Razorpay never redelivered it. A single transient failure
silently destroyed the event — no row, no error, no retry, only a log line.

Two properties matter and are asserted here:
  1. the INSERT is committed BEFORE the row is claimed, so it survives a later
     rollback;
  2. `processed` decides whether work is outstanding — a row that exists but was
     never processed is replayable, not a duplicate to skip.
"""

from types import SimpleNamespace

from app.services import webhook_inbox

EVENT = {
    "event_id": "evt_test_1",
    "event": "subscription.charged",
    "rz_subscription_id": "sub_test_1",
    "payload": {"event": "subscription.charged"},
}


class _Scalars:
    """Stands in for SQLAlchemy's ScalarResult."""

    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _Result:
    def __init__(self, row=None):
        self._row = row

    def scalars(self):
        return _Scalars([] if self._row is None else [self._row])


class _RecordingDB:
    """Records the order of execute/commit/rollback calls."""

    def __init__(self, select_row=None):
        self.calls: list[str] = []
        self._select_row = select_row

    async def execute(self, statement):
        verb = type(statement).__name__.lower()
        if "insert" in verb:
            self.calls.append("insert")
            return _Result()
        if "select" in verb:
            self.calls.append("select")
            return _Result(self._select_row)
        self.calls.append("update")
        return _Result()

    async def commit(self):
        self.calls.append("commit")

    async def rollback(self):
        self.calls.append("rollback")


def _row(processed: bool):
    return SimpleNamespace(event_id=EVENT["event_id"], processed=processed, error="old")


async def test_event_is_committed_before_it_is_claimed():
    """The durability guarantee: the row is safe before any handler runs."""
    db = _RecordingDB(select_row=_row(processed=False))

    await webhook_inbox.claim_event(db, **EVENT)

    assert db.calls[:3] == ["insert", "commit", "select"], (
        f"expected insert->commit->select, got {db.calls}. The commit must come "
        "before the claim, or a later rollback destroys the event record."
    )


async def test_unprocessed_event_is_claimable():
    """A row that exists but never completed is outstanding work, not a duplicate."""
    row = _row(processed=False)
    claimed = await webhook_inbox.claim_event(_RecordingDB(select_row=row), **EVENT)
    assert claimed is row


async def test_processed_event_is_skipped():
    """A genuine redelivery of completed work must not run twice."""
    claimed = await webhook_inbox.claim_event(
        _RecordingDB(select_row=_row(processed=True)), **EVENT
    )
    assert claimed is None


async def test_row_locked_by_a_concurrent_delivery_is_skipped():
    """SKIP LOCKED returns no row when another delivery owns it."""
    claimed = await webhook_inbox.claim_event(_RecordingDB(select_row=None), **EVENT)
    assert claimed is None


async def test_marking_processed_clears_a_previous_error():
    """`error` should describe the latest outcome, not a stale earlier one."""
    db = _RecordingDB()
    row = _row(processed=False)

    await webhook_inbox.mark_processed(db, row)

    assert row.processed is True
    assert row.error is None
    assert "commit" in db.calls


async def test_failure_rolls_back_handler_work_then_records_the_error():
    """The rollback must not take the event record with it — hence rollback first,
    then an UPDATE against the already-committed row."""
    db = _RecordingDB()

    await webhook_inbox.record_failure(db, event_id=EVENT["event_id"], error="boom")

    assert db.calls == ["rollback", "update", "commit"], db.calls
