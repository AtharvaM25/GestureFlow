API = "/api/v1"


def test_lists_the_models_letters_with_f1(client, auth):
    r = client.get(f"{API}/gestures", headers=auth)
    assert r.status_code == 200
    rows = r.json()
    assert [g["label"] for g in rows] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    k = next(g for g in rows if g["label"] == "K")
    assert k["f1_unseen_session"] is not None and k["f1_unseen_session"] < 0.5


def test_requires_login(client):
    assert client.get(f"{API}/gestures").status_code == 401
