"""You cannot delete your account out from under a live paid subscription.

The rule is a guard rail, not a punishment: cancel first, keep the period you
paid for, delete once it ends. Enforcing it means deletion never has to reach for
the payment gateway at all — no cancellation call, no retry queue, no partial
state to reconcile.

Two things are load-bearing:

  * the block must be in the service, not the route. A frontend that greys out
    the button is a convenience; anyone can POST directly.
  * the block must not catch free users. Every account carries a free-plan
    subscription row from signup, so a check that ignored
    razorpay_subscription_id would lock ~93% of accounts out of deleting.

Razorpay is not stubbed here because nothing should be calling it. The tests that
assert that assert it directly.
"""

import inspect
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models.subscription_enums import SubscriptionStatus
from app.services import account_service as acct
from app.services import subscription_guard as guard

PASSWORD = "correct-horse-battery-staple"


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: self._rows)


class _GuardDB:
    """Returns a fixed set of subscription rows for the guard's single query."""

    def __init__(self, rows):
        self._rows = rows
        self.statements = []

    async def execute(self, statement, *_a, **_k):
        self.statements.append(statement)
        return _ScalarResult(self._rows)


def _sub(
    *,
    status=SubscriptionStatus.ACTIVE,
    rz_id="sub_ABC123",
    period_end_days=12,
):
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        razorpay_subscription_id=rz_id,
        status=status,
        current_period_end=(
            None
            if period_end_days is None
            else datetime.now(timezone.utc) + timedelta(days=period_end_days)
        ),
    )


# ---------------------------------------------------------------------------
# the predicate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_paid_subscription_blocks():
    db = _GuardDB([_sub()])
    assert await guard.has_active_paid_subscription(db, uuid.uuid4()) is True


@pytest.mark.asyncio
async def test_past_due_blocks():
    """The mandate is live and Razorpay is still retrying the charge."""
    db = _GuardDB([_sub(status=SubscriptionStatus.PAST_DUE)])
    assert await guard.has_active_paid_subscription(db, uuid.uuid4()) is True


@pytest.mark.asyncio
async def test_expired_paid_subscription_does_not_block():
    """Once the paid period ends, the account can go."""
    db = _GuardDB([_sub(period_end_days=-1)])
    assert await guard.has_active_paid_subscription(db, uuid.uuid4()) is False


@pytest.mark.asyncio
async def test_no_subscription_does_not_block():
    db = _GuardDB([])
    assert await guard.has_active_paid_subscription(db, uuid.uuid4()) is False


@pytest.mark.asyncio
async def test_null_period_end_still_blocks():
    """An unterminated subscription must not slip through as 'expired'."""
    db = _GuardDB([_sub(period_end_days=None)])
    assert await guard.has_active_paid_subscription(db, uuid.uuid4()) is True


@pytest.mark.asyncio
async def test_blocking_subscription_is_returned_for_reporting():
    sub = _sub()
    db = _GuardDB([sub])
    assert (await guard.get_blocking_subscription(db, uuid.uuid4())) is sub


def test_statuses_come_from_the_existing_cancellation_selector():
    assert SubscriptionStatus.ACTIVE in guard.PAID_ACTIVE_STATUSES
    assert SubscriptionStatus.PAST_DUE in guard.PAID_ACTIVE_STATUSES
    assert SubscriptionStatus.PENDING not in guard.PAID_ACTIVE_STATUSES
    assert SubscriptionStatus.CANCELED not in guard.PAID_ACTIVE_STATUSES


def test_query_filters_out_free_plan_rows():
    """Every account has a free-plan row; catching those would block everyone."""
    source = inspect.getsource(guard.get_blocking_subscription)
    assert "razorpay_subscription_id.is_not(None)" in source


def test_expiry_reuses_the_subscription_services_predicate():
    """Re-implementing it here is how the two quietly disagree later."""
    source = inspect.getsource(guard.get_blocking_subscription)
    assert "SubscriptionService._is_expired" in source


# ---------------------------------------------------------------------------
# deletion is refused
# ---------------------------------------------------------------------------


class _DeleteDB:
    def __init__(self):
        self.commits = 0
        self.statements = []

    async def execute(self, statement, *_a, **_k):
        self.statements.append(statement)
        return SimpleNamespace(first=lambda: None)

    async def commit(self):
        self.commits += 1


def _user(*, password_hash="argon2-hash"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="seller@example.com",
        password_hash=password_hash,
        deleted_at=None,
        purge_after=None,
        is_active=True,
    )


@pytest.fixture(autouse=True)
def _accept_password(monkeypatch):
    monkeypatch.setattr(
        acct, "verify_password", lambda password, hashed: password == PASSWORD
    )


def _block_with(monkeypatch, subscription):
    async def _fake(db, user_id):
        return subscription

    monkeypatch.setattr(acct, "get_blocking_subscription", _fake)


@pytest.mark.asyncio
async def test_paid_user_cannot_delete_account(monkeypatch):
    _block_with(monkeypatch, _sub())
    db = _DeleteDB()

    with pytest.raises(HTTPException) as exc:
        await acct.AccountService.delete_account(
            db=db, user=_user(), password=PASSWORD, confirmation=None
        )

    assert exc.value.status_code == 409
    assert db.commits == 0
    assert db.statements == []


@pytest.mark.asyncio
async def test_paid_user_receives_the_agreed_message(monkeypatch):
    _block_with(monkeypatch, _sub())

    with pytest.raises(HTTPException) as exc:
        await acct.AccountService.delete_account(
            db=_DeleteDB(), user=_user(), password=PASSWORD, confirmation=None
        )

    assert exc.value.detail == (
        "Please cancel your subscription first. You can delete your account "
        "after the current billing period ends."
    )


@pytest.mark.asyncio
async def test_block_is_enforced_in_the_service_not_the_route(monkeypatch):
    """A caller bypassing the frontend hits the same 409."""
    source = inspect.getsource(acct.AccountService.delete_account)
    assert "get_blocking_subscription" in source
    assert "HTTP_409_CONFLICT" in source


@pytest.mark.asyncio
async def test_wrong_password_is_answered_before_billing_state(monkeypatch):
    """Don't disclose subscription state to someone who failed re-auth."""
    monkeypatch.setattr(acct, "verify_password", lambda p, h: False)
    _block_with(monkeypatch, _sub())

    with pytest.raises(HTTPException) as exc:
        await acct.AccountService.delete_account(
            db=_DeleteDB(), user=_user(), password="wrong", confirmation=None
        )

    assert exc.value.status_code == 400
    assert "password" in exc.value.detail.lower()


# ---------------------------------------------------------------------------
# free and expired users can still delete
# ---------------------------------------------------------------------------


class _OkDB:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self._queue = [SimpleNamespace(first=lambda: (now, now + timedelta(days=30)))]
        self.commits = 0

    async def execute(self, *_a, **_k):
        return (
            self._queue.pop(0)
            if self._queue
            else SimpleNamespace(first=lambda: None)
        )

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_free_user_can_delete_account(monkeypatch):
    _block_with(monkeypatch, None)
    db = _OkDB()

    result = await acct.AccountService.delete_account(
        db=db, user=_user(), password=PASSWORD, confirmation=None
    )

    assert result.already_pending is False
    assert db.commits == 1


@pytest.mark.asyncio
async def test_expired_paid_user_can_delete_account(monkeypatch):
    """The guard returns None once the period has ended."""
    _block_with(monkeypatch, None)
    db = _OkDB()

    result = await acct.AccountService.delete_account(
        db=db, user=_user(), password=PASSWORD, confirmation=None
    )

    assert result.purge_after > result.deleted_at
    assert db.commits == 1


# ---------------------------------------------------------------------------
# no Razorpay, no retry queue
# ---------------------------------------------------------------------------


def test_account_service_never_touches_razorpay():
    """Checks the module namespace, not the prose.

    The docstrings mention Razorpay precisely to say we do not call it, so a
    text match would fail on its own explanation. What matters is that nothing
    callable from the gateway module is bound here.
    """
    assert "cancel_subscription" not in vars(acct)
    assert "httpx" not in vars(acct)
    assert not any(
        getattr(v, "__module__", "") == "app.services.razorpay_subscription_service"
        for v in vars(acct).values()
    )


def test_no_cancellation_or_reconciliation_queue_exists():
    """Step 4's wrapper and its derived retry queue are gone for good."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.services.subscription_cancellation")

    source = inspect.getsource(acct)
    for gone in (
        "CancellationOutcome",
        "cancel_for_account_deletion",
        "needs_reconciliation",
        "find_subscriptions_needing_cancellation",
    ):
        assert gone not in source


def test_subscription_guard_makes_no_network_calls():
    """It reads local state only — no gateway client, no cancel function."""
    assert "httpx" not in vars(guard)
    assert "cancel_subscription" not in vars(guard)
    assert not any(
        getattr(v, "__module__", "") == "app.services.razorpay_subscription_service"
        for v in vars(guard).values()
    )


def test_deletion_result_no_longer_reports_a_cancellation():
    assert not hasattr(acct.AccountDeletionResult, "cancellation")
    assert "cancellation" not in acct.AccountDeletionResult.__dataclass_fields__


def test_deletion_never_deletes_payment_records():
    """Payments stay put — they are financial records, retained BY US.

    Retention is local, not delegated: books-of-account retention sits with us
    as merchant of record, so Razorpay holding its own copy does not discharge
    the obligation. Migration d94b62e8c1f5 made tbl_payments.user_id nullable
    with ON DELETE SET NULL, so even the permanent purge keeps the row and only
    strips the link to the person — amount, currency, status, timestamps and
    the razorpay_* ids all survive as an anonymised record.

    Soft delete must therefore not touch payments at all, which is what this
    asserts: the service module never so much as mentions Payment.
    """
    source = inspect.getsource(acct)
    assert "Payment" not in source
