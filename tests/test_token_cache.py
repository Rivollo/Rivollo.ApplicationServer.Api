"""Rigorous tests for the Managed Identity token caching logic in app.core.db."""
from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.core.db as db_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(value: str = "test-token", expires_in: int = 3600) -> MagicMock:
    """Return a fake azure AccessToken."""
    tok = MagicMock()
    tok.token = value
    tok.expires_on = int(time.time()) + expires_in
    return tok


def _make_credential(token: MagicMock | None = None) -> MagicMock:
    cred = MagicMock()
    cred.get_token.return_value = token or _make_token()
    return cred


def _capture_listener():
    """Return a mock for event.listens_for that captures the decorated fn."""
    captured: dict = {}

    def mock_listens_for(target, event_name):
        def decorator(fn):
            captured["fn"] = fn
            return fn
        return decorator

    return mock_listens_for, captured


# ---------------------------------------------------------------------------
# Fixture: reset ALL module globals after every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_db_globals():
    saved = {
        "_cached_token": db_module._cached_token,
        "_token_expires_on": db_module._token_expires_on,
        "_credential": db_module._credential,
        "_engine": db_module._engine,
        "_SessionLocal": db_module._SessionLocal,
    }
    yield
    for attr, val in saved.items():
        setattr(db_module, attr, val)


# ---------------------------------------------------------------------------
# 1. _attach_managed_identity_token_refresh
# ---------------------------------------------------------------------------

class TestAttachManagedIdentityTokenRefresh:

    def _call_attach(self, fake_cred, mock_listens_for=None):
        fake_engine = MagicMock()
        fake_engine.sync_engine = MagicMock()
        lf, captured = _capture_listener()
        with patch("azure.identity.ManagedIdentityCredential", return_value=fake_cred), \
             patch("app.core.db.event.listens_for", lf):
            db_module._attach_managed_identity_token_refresh(fake_engine)
        return captured

    def test_fetches_token_exactly_once_at_startup(self):
        """Azure AD is called once during attach — not again on do_connect."""
        cred = _make_credential(_make_token("initial"))
        self._call_attach(cred)
        cred.get_token.assert_called_once()

    def test_caches_token_string(self):
        """_cached_token is set to the token string after attach."""
        tok = _make_token("abc-token")
        self._call_attach(_make_credential(tok))
        assert db_module._cached_token == "abc-token"

    def test_stores_expires_on(self):
        """_token_expires_on is set to float(token.expires_on)."""
        tok = _make_token(expires_in=3600)
        self._call_attach(_make_credential(tok))
        assert db_module._token_expires_on == float(tok.expires_on)

    def test_do_connect_injects_cached_password(self):
        """do_connect listener sets cparams['password'] from the cache."""
        tok = _make_token("cached-password")
        captured = self._call_attach(_make_credential(tok))

        cparams: dict = {}
        captured["fn"](None, None, None, cparams)
        assert cparams["password"] == "cached-password"

    def test_do_connect_does_not_call_azure(self):
        """do_connect must NOT call get_token — it only reads the cache."""
        cred = _make_credential()
        captured = self._call_attach(cred)

        cred.get_token.reset_mock()  # clear the startup call
        cparams: dict = {}
        captured["fn"](None, None, None, cparams)
        cred.get_token.assert_not_called()

    def test_do_connect_raises_if_token_is_none(self):
        """If _cached_token is None (startup failed), do_connect raises RuntimeError."""
        cred = _make_credential()
        captured = self._call_attach(cred)

        db_module._cached_token = None  # simulate failed startup
        with pytest.raises(RuntimeError, match="Managed Identity token not available"):
            captured["fn"](None, None, None, {})

    def test_do_connect_picks_up_refreshed_token(self):
        """do_connect always reads the current module-level _cached_token."""
        cred = _make_credential(_make_token("old-token"))
        captured = self._call_attach(cred)

        db_module._cached_token = "refreshed-token"  # simulate background refresh
        cparams: dict = {}
        captured["fn"](None, None, None, cparams)
        assert cparams["password"] == "refreshed-token"


# ---------------------------------------------------------------------------
# 2. token_refresh_loop
# ---------------------------------------------------------------------------

class TestTokenRefreshLoop:

    @pytest.mark.asyncio
    async def test_sleeps_based_on_expires_on(self):
        """Loop sleeps for (expires_on - now - 15min), minimum 60s."""
        db_module._token_expires_on = time.time() + 3600
        db_module._credential = _make_credential()
        expected = 3600 - 15 * 60  # ~2700s

        sleep_calls: list[float] = []

        async def mock_sleep(duration: float):
            sleep_calls.append(duration)
            raise asyncio.CancelledError

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with suppress(asyncio.CancelledError):
                await db_module.token_refresh_loop()

        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - expected) < 5  # within 5 seconds

    @pytest.mark.asyncio
    async def test_minimum_sleep_is_60s_when_token_expired(self):
        """If token is already expired, sleep is at least 60 seconds."""
        db_module._token_expires_on = time.time() - 9999  # long expired
        db_module._credential = _make_credential()

        sleep_calls: list[float] = []

        async def mock_sleep(duration: float):
            sleep_calls.append(duration)
            raise asyncio.CancelledError

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with suppress(asyncio.CancelledError):
                await db_module.token_refresh_loop()

        assert sleep_calls[0] >= 60

    @pytest.mark.asyncio
    async def test_updates_cached_token_after_sleep(self):
        """After sleeping, loop writes the new token to _cached_token."""
        new_tok = _make_token("refreshed", expires_in=3600)
        db_module._token_expires_on = time.time() + 3600
        db_module._credential = _make_credential(new_tok)

        sleep_count = 0

        async def mock_sleep(_: float):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError  # stop after first full iteration

        async def mock_to_thread(fn, *args):
            return fn(*args)  # succeed so assignments run

        with patch("asyncio.sleep", side_effect=mock_sleep), \
             patch("asyncio.to_thread", side_effect=mock_to_thread):
            with suppress(asyncio.CancelledError):
                await db_module.token_refresh_loop()

        assert db_module._cached_token == "refreshed"
        assert db_module._token_expires_on == float(new_tok.expires_on)

    @pytest.mark.asyncio
    async def test_loop_survives_azure_exception(self):
        """If get_token raises, the loop logs and continues — does NOT crash."""
        db_module._token_expires_on = time.time() + 3600
        db_module._credential = MagicMock()

        sleep_count = 0

        async def mock_sleep(_: float):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError  # stop after 2 cycles

        async def mock_to_thread(fn, *args):
            raise Exception("Azure AD unreachable")

        with patch("asyncio.sleep", side_effect=mock_sleep), \
             patch("asyncio.to_thread", side_effect=mock_to_thread):
            with suppress(asyncio.CancelledError):
                await db_module.token_refresh_loop()

        assert sleep_count == 2  # loop ran twice — proved it did not crash

    @pytest.mark.asyncio
    async def test_cached_token_unchanged_on_failure(self):
        """On refresh failure, the old cached token is preserved."""
        db_module._cached_token = "still-valid-token"
        db_module._token_expires_on = time.time() + 3600
        db_module._credential = MagicMock()

        async def mock_sleep(_: float):
            pass

        async def mock_to_thread(fn, *args):
            raise Exception("network error")

        # run one iteration then stop
        call_count = 0

        async def mock_sleep_stop(_: float):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError

        with patch("asyncio.sleep", side_effect=mock_sleep_stop), \
             patch("asyncio.to_thread", side_effect=mock_to_thread):
            with suppress(asyncio.CancelledError):
                await db_module.token_refresh_loop()

        assert db_module._cached_token == "still-valid-token"

    @pytest.mark.asyncio
    async def test_refresh_updates_expires_on(self):
        """After a successful refresh, _token_expires_on is updated."""
        old_expiry = time.time() + 100
        new_tok = _make_token("new", expires_in=3600)
        db_module._token_expires_on = old_expiry
        db_module._credential = _make_credential(new_tok)

        sleep_count = 0

        async def mock_sleep(_: float):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError

        async def mock_to_thread(fn, *args):
            return fn(*args)  # succeed so assignments run

        with patch("asyncio.sleep", side_effect=mock_sleep), \
             patch("asyncio.to_thread", side_effect=mock_to_thread):
            with suppress(asyncio.CancelledError):
                await db_module.token_refresh_loop()

        assert db_module._token_expires_on == float(new_tok.expires_on)
        assert db_module._token_expires_on > old_expiry


# ---------------------------------------------------------------------------
# 3. dispose_engine
# ---------------------------------------------------------------------------

class TestDisposeEngine:

    @pytest.mark.asyncio
    async def test_calls_dispose_when_engine_exists(self):
        """dispose_engine() calls engine.dispose() when engine is set."""
        mock_engine = AsyncMock()
        db_module._engine = mock_engine

        await db_module.dispose_engine()

        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_error_when_engine_is_none(self):
        """dispose_engine() is a no-op when engine is None — must not raise."""
        db_module._engine = None
        await db_module.dispose_engine()  # should not raise


# ---------------------------------------------------------------------------
# 4. _ensure_async_url
# ---------------------------------------------------------------------------

class TestEnsureAsyncUrl:

    @pytest.mark.parametrize("input_url,expected", [
        ("postgresql+asyncpg://host/db", "postgresql+asyncpg://host/db"),
        ("postgres://host/db",           "postgresql+asyncpg://host/db"),
        ("postgresql://host/db",         "postgresql+asyncpg://host/db"),
        ("postgresql+psycopg2://host/db","postgresql+asyncpg://host/db"),
        ("postgresql+psycopg://host/db", "postgresql+asyncpg://host/db"),
        ("sqlite:///./test.db",          "sqlite:///./test.db"),  # unchanged
    ])
    def test_url_conversion(self, input_url: str, expected: str):
        assert db_module._ensure_async_url(input_url) == expected
