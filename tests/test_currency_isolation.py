"""Every price and promo lookup must name the currency it wants.

tbl_plan_prices and tbl_promo_codes now hold INR and USD rows side by side. That
is what makes a lookup missing its currency filter dangerous rather than merely
untidy:

  * tbl_plan_prices has two rows per (plan, interval) — one per currency — so an
    unfiltered lookup matches both and `scalar_one_or_none()` raises
    MultipleResultsFound. Every rupee checkout for that plan would fail, not
    just charge the wrong number.

  * tbl_promo_codes shares one unique namespace for `code`, so without a
    currency filter an Indian customer could type a USD-only code and have it
    applied to a rupee checkout.

These tests read the SQL each service actually emits, rather than trusting that
the filter is still in the source, so deleting the WHERE clause fails here.
"""

import uuid
from types import SimpleNamespace

import pytest

from app.database.promo_repo import PromoRepository
from app.services import razorpay_subscription_service as inr_svc
from app.services import usd_promo_service
from app.services import usd_subscription_service as usd_svc


def _sql(statement) -> str:
    """Render a statement with its bound values inlined."""
    return str(statement.compile(compile_kwargs={"literal_binds": True}))


class _Result:
    def __init__(self, one=None, rows=None):
        self._one = one
        self._rows = rows or []

    def scalar_one_or_none(self):
        return self._one

    def scalars(self):
        return iter(self._rows)


class _CapturingDB:
    """Returns queued results in order and keeps every statement it was given."""

    def __init__(self, results):
        self._results = list(results)
        self.statements = []

    async def execute(self, statement, *_args, **_kwargs):
        self.statements.append(statement)
        return self._results.pop(0) if self._results else _Result()

    def sql_containing(self, table: str) -> str:
        for statement in self.statements:
            rendered = _sql(statement)
            if table in rendered:
                return rendered
        raise AssertionError(f"no statement queried {table}: {self.statements}")


def _plan():
    # A real UUID: the compiler renders bound values literally here, and the
    # UUID type cannot render a plain string.
    return SimpleNamespace(id=uuid.uuid4(), code="pro", plan_features=[], name="Pro")


def _price():
    return SimpleNamespace(
        price_inr=1999,
        ai_credit_limit=2000,
        razorpay_plan_id="plan_inr",
        total_count=1200,
    )


# ── INR: the paths that must never see a USD row ────────────────────────────


async def test_inr_plan_lookup_filters_to_inr():
    db = _CapturingDB([_Result(one=_plan()), _Result(one=_price())])

    await inr_svc._get_plan_with_features(db, "pro", "monthly")

    sql = db.sql_containing("tbl_plan_prices")
    assert "'INR'" in sql, (
        "the INR price lookup no longer filters on currency. With a USD row "
        f"present this matches two rows and raises MultipleResultsFound:\n{sql}"
    )


async def test_inr_promo_lookup_filters_to_inr():
    db = _CapturingDB([_Result(one=None)])

    await PromoRepository.get_by_code(db, "SOMECODE")

    sql = db.sql_containing("tbl_promo_codes")
    assert "'INR'" in sql, (
        "the INR promo lookup no longer filters on currency — a USD-only code "
        f"could be redeemed against a rupee checkout:\n{sql}"
    )


# ── USD: the mirror of the same risk ────────────────────────────────────────


async def test_usd_plan_lookup_filters_to_usd():
    db = _CapturingDB([_Result(one=_plan()), _Result(one=None)])

    # No USD row seeded, so this raises — the query has still been emitted.
    with pytest.raises(Exception):
        await usd_svc._load_usd_plan(db, "pro", "monthly")

    sql = db.sql_containing("tbl_plan_prices")
    assert "'USD'" in sql, (
        f"the USD price lookup could match the rupee row for this plan:\n{sql}"
    )


async def test_public_promo_lookup_filters_to_usd():
    db = _CapturingDB([_Result(rows=[])])

    await usd_promo_service.get_public_promo(db, plan_code="pro", billing_interval="monthly")

    sql = db.sql_containing("tbl_promo_codes")
    assert "'USD'" in sql, (
        "the advertised-promo lookup could pick up an INR promo and discount a "
        f"dollar checkout by a rupee amount:\n{sql}"
    )
