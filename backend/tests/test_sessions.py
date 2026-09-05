def test_create_session_defaults_to_all_questions(client):
    r = client.post("/api/sessions", json={"mode": "live"})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "created" and len(body["questions"]) == 5
    assert body["ws_url"].startswith("ws://") and body["id"] in body["ws_url"]


def test_create_session_with_subset(client):
    r = client.post("/api/sessions", json={"question_ids": [3, 1]})
    qs = r.json()["questions"]
    assert [q["question_id"] for q in qs] == [3, 1] and [q["order_index"] for q in qs] == [0, 1]


def test_create_session_unknown_question(client):
    assert client.post("/api/sessions", json={"question_ids": [999]}).status_code == 400


def test_list_and_detail(client):
    ids = [client.post("/api/sessions", json={}).json()["id"] for _ in range(3)]
    r = client.get("/api/sessions?limit=2")
    assert r.json()["total"] == 3 and len(r.json()["items"]) == 2
    assert r.json()["items"][0]["has_report"] is False
    assert client.get(f"/api/sessions/{ids[0]}").status_code == 200


def test_other_user_cannot_see_session(client, other_client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    assert other_client.get(f"/api/sessions/{sid}").status_code == 404
    assert other_client.get("/api/sessions").json()["total"] == 0


def test_finish_creates_report_and_delete(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    assert client.get(f"/api/sessions/{sid}/report").status_code == 409
    r = client.post(f"/api/sessions/{sid}/finish")
    assert r.json() == {"status": "finished", "report_ready": True}
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["overview"]["event_count"] == 0 and len(rep["per_question"]) == 5
    assert client.get("/api/sessions").json()["items"][0]["has_report"] is True
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404
