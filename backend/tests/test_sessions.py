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


def test_advance_question_is_monotonic_and_idempotent(client):
    """중복·역순·건너뛰기 번호가 와도 시각이 역전되지 않고 앞으로만 간다."""
    import uuid

    from app.db import SessionLocal
    from app.models import Session
    from app.services import session_service as svc

    sid = client.post("/api/sessions", json={}).json()["id"]
    with SessionLocal() as db:
        s = db.get(Session, uuid.UUID(sid))
        svc.start_session(db, s)
        assert svc.advance_question(db, s, 2) == 2  # 1 건너뛰고 2
        assert svc.advance_question(db, s, 1) == 2  # 역순 → 무시
        assert svc.advance_question(db, s, 2) == 2  # 중복 → 무시
        assert svc.advance_question(db, s, 4) == 4
        assert svc.current_question_index(s) == 4
        svc.finish_session(db, s)
        for q in s.questions:
            if q.started_at and q.ended_at:
                assert q.ended_at >= q.started_at, f"q{q.order_index} 시각 역전"
        assert s.questions[1].started_at is None and s.questions[3].started_at is None  # 건너뛴 질문
        assert s.questions[0].ended_at == s.questions[2].started_at
