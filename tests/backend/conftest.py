"""
API test fixtures. Each test gets a fresh SQLite database file, so the suite needs no
database server. tests/backend/test_migrations.py runs the real Alembic migrations, and
also against PostgreSQL when GESTUREFLOW_TEST_PG_URL is set.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import config
from backend import db as db_module
from backend.deps import login_rate_limit, sentence_limit

GOLDEN = json.loads((Path(__file__).parents[1] / "golden" / "core_golden.json").read_text())


def clear_caches():
    config.get_settings.cache_clear()
    db_module.get_engine.cache_clear()
    db_module.get_sessionmaker.cache_clear()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("GESTUREFLOW_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("GESTUREFLOW_JWT_SECRET", "test-secret-" + "x" * 32)
    clear_caches()
    login_rate_limit.reset()
    sentence_limit.reset()

    from backend import models  # noqa: F401  (registers the tables)
    from backend.db import Base, get_engine
    Base.metadata.create_all(get_engine())

    from backend.main import create_app
    yield create_app()
    get_engine().dispose()
    clear_caches()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def register_and_login(client, email="user@example.com", password="correct-horse-1"):
    r = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def token(client):
    return register_and_login(client)["access_token"]


@pytest.fixture
def auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def confident_case():
    """A golden sample the shipped model classifies with high confidence."""
    return max(GOLDEN["cases"], key=lambda c: max(c["probs"]))
