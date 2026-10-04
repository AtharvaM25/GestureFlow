import pytest
from starlette.websockets import WebSocketDisconnect

from gestureflow.core.temporal import COMMIT_FRAMES
from tests.backend.conftest import register_and_login

API = "/api/v1"
WS = f"{API}/ws/recognition"


def new_session(client, auth):
    r = client.post(f"{API}/sessions", headers=auth, json={})
    assert r.status_code == 201, r.text
    return r.json()


def connect(client, token, session_id):
    ws = client.websocket_connect(WS)
    conn = ws.__enter__()
    conn.send_json({"type": "auth", "token": token, "session_id": session_id})
    return ws, conn


def test_session_lifecycle(client, auth):
    s = new_session(client, auth)
    assert s["text"] == "" and s["ended_at"] is None and len(s["model_version"]) == 12
    assert [x["id"] for x in client.get(f"{API}/sessions", headers=auth).json()] == [s["id"]]
    ended = client.post(f"{API}/sessions/{s['id']}/end", headers=auth).json()
    assert ended["ended_at"] is not None


def test_sessions_are_private(client, auth):
    s = new_session(client, auth)
    other = register_and_login(client, "other@example.com", "other-password-1")
    other_auth = {"Authorization": f"Bearer {other['access_token']}"}
    assert client.get(f"{API}/sessions/{s['id']}", headers=other_auth).status_code == 404
    assert client.get(f"{API}/sessions", headers=other_auth).json() == []


def test_ws_commits_letter_and_persists_it(client, token, auth, confident_case):
    s = new_session(client, auth)
    ws, conn = connect(client, token, s["id"])
    try:
        assert conn.receive_json() == {"type": "ready", "session_id": s["id"], "text": "",
                                       "calibrated": False}
        results = []
        for i in range(COMMIT_FRAMES):
            conn.send_json({"type": "frame", "landmarks": confident_case["raw"], "client_ts": i})
            results.append(conn.receive_json())
        last = results[-1]
        assert last["committed"] == confident_case["label"]
        assert last["text"] == confident_case["label"]
        assert [r["client_ts"] for r in results] == list(range(COMMIT_FRAMES))
        assert all(r["latency_ms"] > 0 for r in results)
        assert results[-2]["stability"] == pytest.approx((COMMIT_FRAMES - 1) / COMMIT_FRAMES)

        conn.send_json({"type": "frame", "landmarks": None})
        assert conn.receive_json()["hand"] is False
    finally:
        ws.__exit__(None, None, None)

    detail = client.get(f"{API}/sessions/{s['id']}", headers=auth).json()
    assert detail["text"] == confident_case["label"]
    assert [x["gesture"] for x in detail["letters"]] == [confident_case["label"]]
    assert detail["frame_count"] == COMMIT_FRAMES + 1      # flushed on disconnect


def test_ws_backspace(client, token, auth, confident_case):
    s = new_session(client, auth)
    ws, conn = connect(client, token, s["id"])
    try:
        conn.receive_json()
        for _ in range(COMMIT_FRAMES):
            conn.send_json({"type": "frame", "landmarks": confident_case["raw"]})
            conn.receive_json()
        conn.send_json({"type": "backspace"})
        assert conn.receive_json() == {"type": "text", "text": ""}
    finally:
        ws.__exit__(None, None, None)
    assert client.get(f"{API}/sessions/{s['id']}", headers=auth).json()["letters"] == []


def test_ws_bad_messages_keep_connection_open(client, token, auth, confident_case):
    s = new_session(client, auth)
    ws, conn = connect(client, token, s["id"])
    try:
        conn.receive_json()
        conn.send_text("{not json")
        assert conn.receive_json()["type"] == "error"
        conn.send_json({"type": "frame", "landmarks": [[0, 0]] * 5})
        assert "21" in conn.receive_json()["message"]
        conn.send_json({"type": "nope"})
        assert conn.receive_json()["type"] == "error"
        conn.send_json({"type": "frame", "landmarks": confident_case["raw"]})
        assert conn.receive_json()["type"] == "result"
    finally:
        ws.__exit__(None, None, None)


@pytest.mark.parametrize("auth_msg, code", [
    ({"type": "auth", "token": "garbage", "session_id": 1}, 4401),
    ({"type": "frame"}, 4401),
])
def test_ws_rejects_bad_auth(client, auth_msg, code):
    with client.websocket_connect(WS) as conn:
        conn.send_json(auth_msg)
        with pytest.raises(WebSocketDisconnect) as e:
            conn.receive_json()
        assert e.value.code == code


def test_ws_rejects_someone_elses_or_ended_session(client, token, auth):
    s = new_session(client, auth)
    other = register_and_login(client, "other@example.com", "other-password-1")
    for t, sid in ((other["access_token"], s["id"]), (token, 9999)):
        with client.websocket_connect(WS) as conn:
            conn.send_json({"type": "auth", "token": t, "session_id": sid})
            with pytest.raises(WebSocketDisconnect) as e:
                conn.receive_json()
            assert e.value.code == 4404
    client.post(f"{API}/sessions/{s['id']}/end", headers=auth)
    with client.websocket_connect(WS) as conn:
        conn.send_json({"type": "auth", "token": token, "session_id": s["id"]})
        with pytest.raises(WebSocketDisconnect) as e:
            conn.receive_json()
        assert e.value.code == 4404


def test_sentence_unavailable_without_ollama(client, auth, monkeypatch):
    import backend.api.sessions as sessions_api

    def boom(*a, **k):
        raise ConnectionError("refused")
    monkeypatch.setattr(sessions_api, "generate_sentence", boom)
    s = new_session(client, auth)
    r = client.post(f"{API}/sessions/{s['id']}/sentence", headers=auth)
    assert r.status_code == 503


def test_sentence_is_stored(client, auth, monkeypatch):
    import backend.api.sessions as sessions_api
    monkeypatch.setattr(sessions_api, "generate_sentence", lambda letters, *a: "Hello.")
    s = new_session(client, auth)
    assert client.post(f"{API}/sessions/{s['id']}/sentence",
                       headers=auth).json() == {"sentence": "Hello."}
    assert client.get(f"{API}/sessions/{s['id']}", headers=auth).json()["sentence"] == "Hello."


def test_timestamps_carry_a_timezone(client, auth):
    """SQLite returns naive datetimes; the API must still say they're UTC, or browsers
    show them in the wrong timezone."""
    s = new_session(client, auth)
    assert s["started_at"].endswith(("Z", "+00:00"))
