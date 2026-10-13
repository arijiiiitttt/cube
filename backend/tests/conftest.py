import os
import sys

# Make `app` importable and keep config hermetic (no real DB / Redis needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    """Minimal async session stand-in that always returns an empty result set."""

    async def execute(self, *_a, **_kw):
        return _FakeResult([])

    async def commit(self): ...
    async def rollback(self): ...
    async def close(self): ...
    def add(self, *_a): ...
    async def refresh(self, *_a): ...


async def _fake_db():
    yield FakeSession()


@pytest.fixture()
def client():
    app.dependency_overrides[get_db] = _fake_db
    # Not used as a context manager on purpose: skips the lifespan (DB/MinIO init).
    yield TestClient(app)
    app.dependency_overrides.clear()
