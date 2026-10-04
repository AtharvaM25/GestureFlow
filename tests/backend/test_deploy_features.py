"""Features added for running online: Postgres URL forms, account deletion, the sentence
provider switch and the per-user sentence limit."""

import pytest
from sqlalchemy import func, select

from backend.config import Settings
from backend.db import get_sessionmaker
from backend.models import CalibrationProfile, CommittedLetter, RecognitionSession, User
from tests.backend.conftest import GOLDEN, register_and_login

API = "/api/v1"
CASES = {c["label"]: c for c in GOLDEN["cases"]}


@pytest.mark.parametrize("given", [
    "postgres://u:p@host.neon.tech/db?sslmode=require",
    "postgresql://u:p@host.neon.tech/db?sslmode=require",
    "postgresql+psycopg://u:p@host.neon.tech/db?sslmode=require",
])
def test_hosted_postgres_urls_use_psycopg(given):
    assert Settings(database_url=given).database_url == \
        "postgresql+psycopg://u:p@host.neon.tech/db?sslmode=require"


def test_sqlite_url_untouched():
    assert Settings(
        database_url="sqlite:///x.db").database_url == "sqlite:///x.db"


# ------------------------------------------------------------ delete account
def count(model):
    with get_sessionmaker()() as s:
        return s.scalar(select(func.count()).select_from(model))


def test_delete_account_removes_everything(client, token, auth):
    # give the account a session with a committed letter, and a calibration
    s = client.post(f"{API}/sessions", headers=auth).json()
    with client.websocket_connect(f"{API}/ws/recognition") as conn:
        conn.send_json({"type": "auth", "token": token, "session_id": s["id"]})
        conn.receive_json()
        for _ in range(15):
            conn.send_json({"type": "frame", "landmarks": CASES["R"]["raw"]})
            conn.receive_json()
    client.post(f"{API}/calibration", headers=auth,
                json={"samples": {k: [c["raw"]] * 3 for k, c in CASES.items()}})
    assert (count(User), count(RecognitionSession), count(CommittedLetter),
            count(CalibrationProfile)) == (1, 1, 1, 1)

    r = client.request("DELETE", f"{API}/auth/me", headers=auth,
                       json={"password": "correct-horse-1"})
    assert r.status_code == 204
    assert (count(User), count(RecognitionSession), count(CommittedLetter),
            count(CalibrationProfile)) == (0, 0, 0, 0)
    assert client.get(f"{API}/auth/me", headers=auth).status_code == 401
    assert client.post(f"{API}/auth/login", json={"email": "user@example.com",
                                                  "password": "correct-horse-1"}).status_code == 401


def test_delete_account_needs_the_password(client, auth):
    r = client.request(
        "DELETE", f"{API}/auth/me", headers=auth, json={"password": "wrong-pass-1"})
    assert r.status_code == 401
    assert count(User) == 1


def test_delete_only_your_own_account(client, auth):
    register_and_login(client, "other@example.com", "other-password-1")
    client.request("DELETE", f"{API}/auth/me",
                   headers=auth, json={"password": "correct-horse-1"})
    with get_sessionmaker()() as s:
        assert [u.email for u in s.scalars(select(User))] == [
            "other@example.com"]


# ---------------------------------------------------------------- sentences
def new_session(client, auth):
    return client.post(f"{API}/sessions", headers=auth).json()["id"]


def test_provider_none_turns_sentences_off(client, auth, monkeypatch):
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "sentence_provider", "none")
    r = client.post(
        f"{API}/sessions/{new_session(client, auth)}/sentence", headers=auth)
    assert r.status_code == 503 and "turned off" in r.json()["detail"]
    assert client.get(f"{API}/health").json()["sentence_provider"] == "none"


def test_groq_without_a_key_is_unavailable_not_a_crash(client, auth, monkeypatch):
    from backend.config import get_settings
    monkeypatch.setattr(get_settings(), "sentence_provider", "groq")
    monkeypatch.setattr(get_settings(), "groq_api_key", "")
    sid = new_session(client, auth)
    # give the session a letter so the LLM would actually be called
    with get_sessionmaker()() as s:
        s.get(RecognitionSession, sid).text = "HI"
        s.commit()
    r = client.post(f"{API}/sessions/{sid}/sentence", headers=auth)
    assert r.status_code == 503 and "RuntimeError" in r.json()["detail"]


def test_provider_settings_reach_the_llm_call(client, auth, monkeypatch):
    import backend.api.sessions as sessions_api
    from backend.config import get_settings
    calls = []
    monkeypatch.setattr(sessions_api, "generate_sentence",
                        lambda *a: calls.append(a) or "Hi.")
    monkeypatch.setattr(get_settings(), "sentence_provider", "groq")
    monkeypatch.setattr(get_settings(), "groq_api_key", "gsk_test")
    client.post(
        f"{API}/sessions/{new_session(client, auth)}/sentence", headers=auth)
    assert calls == [([], "openai/gpt-oss-20b", None, "groq", "gsk_test")]


def test_sentences_are_limited_per_user(client, auth, monkeypatch):
    import backend.api.sessions as sessions_api
    from backend.config import get_settings
    monkeypatch.setattr(sessions_api, "generate_sentence", lambda *a: "Hi.")
    monkeypatch.setattr(get_settings(), "sentences_per_hour", 2)
    sid = new_session(client, auth)
    codes = [client.post(f"{API}/sessions/{sid}/sentence", headers=auth).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]
    # a different user has their own allowance
    other = register_and_login(client, "other@example.com", "other-password-1")
    other_auth = {"Authorization": f"Bearer {other['access_token']}"}
    assert client.post(f"{API}/sessions/{new_session(client, other_auth)}/sentence",
                       headers=other_auth).status_code == 200


def test_make_llm_rejects_unknown_provider_and_missing_key():
    from gestureflow.sentence import make_llm
    with pytest.raises(RuntimeError, match="API key"):
        make_llm("groq", "openai/gpt-oss-20b", api_key="")
    with pytest.raises(ValueError):
        make_llm("openai")
