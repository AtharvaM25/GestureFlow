import pytest

API = "/api/v1"


def test_health(client):
    body = client.get(f"{API}/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_classes": 26, "database": "ok",
                    "sentence_provider": "ollama"}


def test_landmarks_prediction_matches_golden(client, auth, confident_case):
    r = client.post(f"{API}/recognition/landmarks", headers=auth,
                    json={"landmarks": confident_case["raw"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gesture"] == confident_case["label"]
    assert body["confidence"] == pytest.approx(max(confident_case["probs"]), abs=2e-6)
    assert body["alternatives"][0]["gesture"] == confident_case["label"]
    assert len(body["alternatives"]) == 3 and body["latency_ms"] > 0


def test_landmarks_accept_xyz(client, auth, confident_case):
    pts = [p + [0.0] for p in confident_case["raw"]]
    r = client.post(f"{API}/recognition/landmarks", headers=auth, json={"landmarks": pts})
    assert r.status_code == 200 and r.json()["gesture"] == confident_case["label"]


@pytest.mark.parametrize("bad", [
    [[0, 0]] * 20,                    # wrong count
    [[0, 0, 0, 0]] * 21,              # wrong arity
    [[0, "x"]] * 21,                  # not numbers
])
def test_bad_landmarks_rejected(client, auth, bad):
    assert client.post(f"{API}/recognition/landmarks", headers=auth,
                       json={"landmarks": bad}).status_code == 422


def test_recognition_requires_login(client, confident_case):
    assert client.post(f"{API}/recognition/landmarks",
                       json={"landmarks": confident_case["raw"]}).status_code == 401


def test_requests_are_logged_as_json(client, caplog):
    import json
    import logging
    with caplog.at_level(logging.INFO, logger="gestureflow.access"):
        client.get(f"{API}/health")
    line = json.loads(caplog.records[-1].getMessage())
    assert line["path"] == f"{API}/health" and line["status"] == 200 and line["ms"] >= 0


def test_root_opens_the_api_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"
