"""Test configuration.

The app refuses to start without DATABASE_URL, which is correct: a container
that silently comes up with no database is worse than one that fails loudly.
Tests that drive the app through TestClient therefore need one set before
`app.main` is imported, and conftest runs first.

Nothing connects to it. The URL is never dialled — every test that exercises a
route overrides the `get_db` dependency, so the engine is created and never
used.
"""

import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://test:test@localhost:5432/test_never_connected",
)
