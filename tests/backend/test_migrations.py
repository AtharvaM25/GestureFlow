"""
The Alembic migrations must build exactly the tables the models describe, and the app must
work on the result. Runs on SQLite always, and on PostgreSQL too when GESTUREFLOW_TEST_PG_URL
points at an empty, disposable database.
"""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient

from backend import models  # noqa: F401
from backend.db import Base, make_engine
from tests.backend.conftest import GOLDEN, clear_caches, register_and_login

ALEMBIC_INI = Path(__file__).parents[2] / "alembic.ini"
PG_URL = os.environ.get("GESTUREFLOW_TEST_PG_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def migrated_url(request, tmp_path, monkeypatch):
    if request.param == "postgres" and not PG_URL:
        pytest.skip("GESTUREFLOW_TEST_PG_URL not set")
    url = PG_URL if request.param == "postgres" else f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    monkeypatch.setenv("GESTUREFLOW_DATABASE_URL", url)
    monkeypatch.setenv("GESTUREFLOW_JWT_SECRET", "test-secret-" + "x" * 32)
    clear_caches()
    cfg = Config(str(ALEMBIC_INI))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield url
    from backend.db import get_engine
    get_engine().dispose()
    command.downgrade(cfg, "base")
    clear_caches()


def test_migrations_match_models(migrated_url):
    engine = make_engine(migrated_url)
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
    engine.dispose()


def test_full_flow_on_migrated_database(migrated_url):
    from backend.main import create_app
    case = max(GOLDEN["cases"], key=lambda c: max(c["probs"]))
    with TestClient(create_app()) as client:
        token = register_and_login(client)["access_token"]
        auth = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/v1/health").json()["database"] == "ok"
        s = client.post("/api/v1/sessions", headers=auth).json()
        with client.websocket_connect("/api/v1/ws/recognition") as conn:
            conn.send_json({"type": "auth", "token": token, "session_id": s["id"]})
            conn.receive_json()
            for _ in range(15):
                conn.send_json({"type": "frame", "landmarks": case["raw"]})
                last = conn.receive_json()
        assert last["committed"] == case["label"]
        detail = client.get(f"/api/v1/sessions/{s['id']}", headers=auth).json()
        assert detail["text"] == case["label"]
        assert detail["letters"][0]["alternatives"][0]["gesture"] == case["label"]
