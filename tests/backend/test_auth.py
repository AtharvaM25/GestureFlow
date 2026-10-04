from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import select

from backend.db import get_sessionmaker
from backend.models import User
from tests.backend.conftest import register_and_login

API = "/api/v1"


def test_register_login_me(client):
    token = register_and_login(client, "Mixed@Example.com")["access_token"]
    me = client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "mixed@example.com"


def test_password_is_hashed(client):
    register_and_login(client)
    with get_sessionmaker()() as s:
        assert s.scalar(select(User.password_hash)).startswith("$argon2id$")


def test_duplicate_email_rejected(client):
    register_and_login(client)
    r = client.post(f"{API}/auth/register",
                    json={"email": "user@example.com", "password": "another-pass-1"})
    assert r.status_code == 409


def test_short_password_and_bad_email_rejected(client):
    assert client.post(f"{API}/auth/register",
                       json={"email": "a@b.co", "password": "short"}).status_code == 422
    r = client.post(f"{API}/auth/register",
                    json={"email": "not-an-email", "password": "long-enough-1"})
    assert r.status_code == 422


def test_wrong_password_and_unknown_email_look_the_same(client):
    register_and_login(client)
    wrong = client.post(f"{API}/auth/login",
                        json={"email": "user@example.com", "password": "wrong-password"})
    unknown = client.post(f"{API}/auth/login",
                          json={"email": "nobody@example.com", "password": "wrong-password"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_protected_routes_need_a_valid_token(client):
    assert client.get(f"{API}/auth/me").status_code == 401
    assert client.get(f"{API}/auth/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get(f"{API}/sessions").status_code == 401


def test_expired_or_forged_token_rejected(client):
    register_and_login(client)
    past = datetime.now(UTC) - timedelta(hours=1)
    expired = jwt.encode({"sub": "1", "exp": past}, "test-secret-" + "x" * 32, algorithm="HS256")
    forged = jwt.encode({"sub": "1", "exp": past + timedelta(days=1)}, "wrong-secret-" + "y" * 32,
                        algorithm="HS256")
    for t in (expired, forged):
        r = client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {t}"})
        assert r.status_code == 401


def test_login_rate_limited(client, monkeypatch):
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "login_attempts_per_minute", 3)
    codes = [client.post(f"{API}/auth/login",
                         json={"email": "x@example.com", "password": "whatever-1"}).status_code
             for _ in range(4)]
    assert codes == [401, 401, 401, 429]
