from datetime import UTC, datetime, timedelta

from gestureflow.core.temporal import COMMIT_FRAMES
from tests.backend.conftest import GOLDEN, register_and_login

API = "/api/v1"
CASES = {c["label"]: c for c in GOLDEN["cases"]}


def sign(client, token, auth, letters):
    """Open a session and hold each letter long enough to commit it."""
    s = client.post(f"{API}/sessions", headers=auth).json()
    with client.websocket_connect(f"{API}/ws/recognition") as conn:
        conn.send_json({"type": "auth", "token": token, "session_id": s["id"]})
        conn.receive_json()
        for letter in letters:
            for _ in range(COMMIT_FRAMES):
                conn.send_json({"type": "frame", "landmarks": CASES[letter]["raw"]})
                conn.receive_json()
    return s


def test_empty(client, auth):
    body = client.get(f"{API}/analytics?days=7", headers=auth).json()
    assert body["sessions"] == body["letters"] == body["frames"] == 0
    assert body["avg_confidence"] is None and body["per_letter"] == []
    assert len(body["per_day"]) == 7 and all(d["letters"] == 0 for d in body["per_day"])
    assert body["per_day"][-1]["date"] == datetime.now(UTC).date().isoformat()


def test_counts_what_was_signed(client, token, auth):
    sign(client, token, auth, ["R", "R", "B"])     # golden R and B are confident samples
    body = client.get(f"{API}/analytics", headers=auth).json()
    assert body["sessions"] == 1
    assert body["letters"] == 3
    assert body["frames"] == 3 * COMMIT_FRAMES
    assert [(x["gesture"], x["count"]) for x in body["per_letter"]] == [("R", 2), ("B", 1)]
    assert 0.9 < body["avg_confidence"] <= 1 and body["avg_latency_ms"] > 0
    assert body["per_day"][-1]["letters"] == 3 and sum(d["letters"] for d in body["per_day"]) == 3


def test_only_your_own_data(client, token, auth):
    sign(client, token, auth, ["R"])
    other = register_and_login(client, "other@example.com", "other-password-1")
    body = client.get(f"{API}/analytics",
                      headers={"Authorization": f"Bearer {other['access_token']}"}).json()
    assert body["sessions"] == 0 and body["letters"] == 0


def test_window_bounds(client, auth):
    assert client.get(f"{API}/analytics?days=0", headers=auth).status_code == 422
    assert client.get(f"{API}/analytics?days=366", headers=auth).status_code == 422
    body = client.get(f"{API}/analytics?days=90", headers=auth).json()
    first = datetime.fromisoformat(body["per_day"][0]["date"]).date()
    assert first == datetime.now(UTC).date() - timedelta(days=89)
