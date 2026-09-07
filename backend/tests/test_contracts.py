"""계약 테스트: 응답 형태가 설계 문서(03-phase1-design.md 5절, 5.6절, 6절)와 일치하는지.
FE 는 이 형태를 보고 개발하므로 필드가 빠지거나 이름이 바뀌면 여기서 잡혀야 한다."""

from app.analysis.types import ATTENTION_LABELS, EMOTION_LABELS

SESSION_KEYS = {"id", "mode", "status", "created_at", "started_at", "finished_at", "questions", "ws_url"}
SESSION_Q_KEYS = {"order_index", "question_id", "text", "started_at", "ended_at"}
LIST_ITEM_KEYS = {"id", "mode", "status", "created_at", "finished_at", "has_report"}
REPORT_KEYS = {
    "session_id",
    "duration_ms",
    "overview",
    "emotion_distribution",
    "attention_distribution",
    "per_question",
    "timeline",
    "feedback",
    # guideline/05 (2026-09-07 추가)
    "status",
    "recording",
    "transcript",
    "llm",
}
OVERVIEW_KEYS = {
    "gaze_hold_rate",
    "stable_emotion_rate",
    "attention_rate",
    "face_found_rate",
    "event_count",
    "frames_analyzed",
}
PER_Q_KEYS = {
    "order_index",
    "question_id",
    "text",
    "duration_ms",
    "gaze_hold_rate",
    "dominant_emotion",
    "attention_rate",
    "event_count",
}
EVENT_KEYS = {"ts_ms", "question_index", "type", "severity", "message"}
WS_RESULT_KEYS = {"type", "ts_ms", "face_found", "gaze", "emotion", "attention"}
WS_EVENT_KEYS = {"type", "ts_ms", "event_type", "severity", "icon", "message"}


def test_question_contract(client):
    q = client.get("/api/questions").json()[0]
    assert set(q) == {"id", "text", "category", "sort_order"}


def test_session_contracts(client):
    s = client.post("/api/sessions", json={"mode": "live"}).json()
    assert set(s) == SESSION_KEYS
    assert set(s["questions"][0]) == SESSION_Q_KEYS
    lst = client.get("/api/sessions").json()
    assert set(lst) == {"items", "total"} and set(lst["items"][0]) == LIST_ITEM_KEYS
    assert set(client.get(f"/api/sessions/{s['id']}").json()) == SESSION_KEYS


def test_error_shape_and_codes(client, other_client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    r = other_client.get(f"/api/sessions/{sid}")
    assert r.status_code == 404 and set(r.json()) == {"detail"}
    r = client.get(f"/api/sessions/{sid}/report")
    assert r.status_code == 409 and "detail" in r.json()
    r = client.post("/api/sessions", json={"question_ids": [999]})
    assert r.status_code == 400 and "detail" in r.json()
    r = client.post("/api/sessions", json={"mode": "nope"})
    assert r.status_code == 422


def test_report_contract_after_live_session(client, auth_ws):
    """실제 WS 흐름으로 로그·이벤트를 만든 뒤 리포트 구조 전체를 대조한다."""
    sid = client.post("/api/sessions", json={}).json()["id"]
    with client.websocket_connect(f"/ws/sessions/{sid}?token=t") as w:
        w.receive_json()
        w.send_json({"type": "start"})
        w.receive_json()
        for _ in range(3):
            w.send_bytes(b"\xff\xd8x")
            msg = w.receive_json()
            assert set(msg) == WS_RESULT_KEYS and msg["type"] == "result"
            assert set(msg["gaze"]) == {"yaw", "pitch", "confidence"}
            assert set(msg["emotion"]) == {"top", "probs"}
        w.send_json({"type": "question", "index": 1})
        w.receive_json()
        w.send_json({"type": "end"})
        assert w.receive_json()["type"] == "report_ready"

    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert set(rep) == REPORT_KEYS
    assert set(rep["status"]) == {"metrics", "recording", "stt", "llm"}
    assert all(v in ("pending", "running", "done", "failed", "skipped") for v in rep["status"].values())
    assert set(rep["overview"]) == OVERVIEW_KEYS
    assert all(
        0.0 <= v <= 1.0 for k, v in rep["overview"].items() if k not in ("event_count", "frames_analyzed")
    )
    assert set(rep["emotion_distribution"]) == set(EMOTION_LABELS)  # v2 4클래스. "기타"는 7클래스 모델일 때만
    assert abs(sum(rep["emotion_distribution"].values()) - 1.0) < 0.01
    assert set(rep["attention_distribution"]) == set(ATTENTION_LABELS)  # 라벨 5개 항상 포함
    assert len(rep["per_question"]) == 5 and set(rep["per_question"][0]) == PER_Q_KEYS
    assert rep["per_question"][0]["duration_ms"] >= 0
    for e in rep["timeline"]:
        assert set(e) == {"ts_ms", "question_index", "type", "message"}
    for f in rep["feedback"]:
        assert set(f) == {"question_index", "note"} and isinstance(f["note"], str)

    ev = client.get(f"/api/sessions/{sid}/events").json()
    assert all(set(e) == EVENT_KEYS for e in ev)


def test_ws_event_contract(client, auth_ws, monkeypatch):
    """Mock 시간을 빠르게 흘려 이벤트를 강제로 발생시키고 event 메시지 필드를 대조한다."""
    from app.services import session_service as svc

    clock = {"ms": 0}
    monkeypatch.setattr(svc, "elapsed_ms", lambda s: clock["ms"])
    sid = client.post("/api/sessions", json={}).json()["id"]
    with client.websocket_connect(f"/ws/sessions/{sid}?token=t") as w:
        w.receive_json()
        w.send_json({"type": "start"})
        w.receive_json()
        got_event = None
        for t in range(
            0, 12_000, 500
        ):  # Mock: yaw 는 20초 주기 사인파, 0~10초 구간이 +쪽 → 2.5~7.5초 |yaw|>20
            clock["ms"] = t
            w.send_bytes(b"\xff\xd8x")
            msgs = [w.receive_json()]
            if msgs[0]["type"] == "result":
                # 이벤트가 있으면 result 바로 뒤에 온다. 없으면 다음 프레임의 result 가 온다 → 살짝 엿보기 위해 ping 사용
                w.send_json({"type": "ping"})
                nxt = w.receive_json()
                if nxt["type"] == "event":
                    got_event = nxt
                    assert w.receive_json()["type"] == "pong"
                    break
        assert got_event is not None, "12초 안에 gaze_off 이벤트가 나와야 함"
        assert set(got_event) == WS_EVENT_KEYS and got_event["event_type"] == "gaze_off"
        w.send_json({"type": "end"})
        w.receive_json()
