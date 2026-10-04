from sqlalchemy import update

from backend.db import get_sessionmaker
from backend.models import CalibrationProfile
from tests.backend.conftest import GOLDEN

API = "/api/v1"
CASES = {c["label"]: c for c in GOLDEN["cases"]}


def samples(frames=3, swap=()):
    raw = {label: c["raw"] for label, c in CASES.items()}
    for a, b in swap:
        raw[a], raw[b] = raw[b], raw[a]
    return {"samples": {label: [r] * frames for label, r in raw.items()}}


def test_status_without_calibration(client, auth):
    body = client.get(f"{API}/calibration", headers=auth).json()
    assert body["calibrated"] is False and body["stale"] is False
    assert body["labels"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_save_apply_and_delete(client, auth):
    r = client.post(f"{API}/calibration", headers=auth, json=samples(swap=[("A", "R")]))
    assert r.status_code == 200, r.text
    assert r.json()["calibrated"] is True and r.json()["samples_per_label"] == 3

    # the swapped calibration makes this user's R read as A
    pred = client.post(f"{API}/recognition/landmarks", headers=auth,
                       json={"landmarks": CASES["R"]["raw"]}).json()
    assert pred["calibrated"] is True and pred["gesture"] == "A"

    assert client.delete(f"{API}/calibration", headers=auth).status_code == 204
    pred = client.post(f"{API}/recognition/landmarks", headers=auth,
                       json={"landmarks": CASES["R"]["raw"]}).json()
    assert pred["calibrated"] is False and pred["gesture"] == "R"


def test_calibration_is_per_user(client, auth):
    from tests.backend.conftest import register_and_login
    client.post(f"{API}/calibration", headers=auth, json=samples(swap=[("A", "R")]))
    other = register_and_login(client, "other@example.com", "other-password-1")
    other_auth = {"Authorization": f"Bearer {other['access_token']}"}
    assert client.get(f"{API}/calibration", headers=other_auth).json()["calibrated"] is False
    pred = client.post(f"{API}/recognition/landmarks", headers=other_auth,
                       json={"landmarks": CASES["R"]["raw"]}).json()
    assert pred["gesture"] == "R"


def test_validation(client, auth):
    missing = samples()
    del missing["samples"]["K"]
    r = client.post(f"{API}/calibration", headers=auth, json=missing)
    assert r.status_code == 422 and "K" in r.json()["detail"]

    too_few = samples(frames=2)
    assert client.post(f"{API}/calibration", headers=auth, json=too_few).status_code == 422

    unknown = samples()
    unknown["samples"]["HELLO"] = unknown["samples"]["A"]
    assert client.post(f"{API}/calibration", headers=auth, json=unknown).status_code == 422

    too_many = samples(frames=31)
    assert client.post(f"{API}/calibration", headers=auth, json=too_many).status_code == 422

    bad = samples()
    bad["samples"]["A"] = [[[0, 0]] * 20] * 3
    assert client.post(f"{API}/calibration", headers=auth, json=bad).status_code == 422


def test_stale_after_model_change(client, auth, token):
    client.post(f"{API}/calibration", headers=auth, json=samples(swap=[("A", "R")]))
    with get_sessionmaker()() as s:
        s.execute(update(CalibrationProfile).values(model_version="000000000000"))
        s.commit()
    body = client.get(f"{API}/calibration", headers=auth).json()
    assert body["calibrated"] is False and body["stale"] is True
    pred = client.post(f"{API}/recognition/landmarks", headers=auth,
                       json={"landmarks": CASES["R"]["raw"]}).json()
    assert pred["calibrated"] is False and pred["gesture"] == "R"


def test_websocket_uses_calibration(client, auth, token):
    client.post(f"{API}/calibration", headers=auth, json=samples(swap=[("A", "R")]))
    s = client.post(f"{API}/sessions", headers=auth).json()
    with client.websocket_connect(f"{API}/ws/recognition") as conn:
        conn.send_json({"type": "auth", "token": token, "session_id": s["id"]})
        assert conn.receive_json()["calibrated"] is True
        conn.send_json({"type": "frame", "landmarks": CASES["R"]["raw"]})
        result = conn.receive_json()
    assert result["calibrated"] is True and result["raw"] == "A"
