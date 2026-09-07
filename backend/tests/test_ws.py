"""WebSocket 시나리오: start → 프레임 → question → end → 리포트. Mock 추론 사용."""

import uuid

import pytest
from starlette.websockets import WebSocketDisconnect


def _connect(client, sid, token="t"):
    return client.websocket_connect(f"/ws/sessions/{sid}?token={token}")


def test_ws_rejects_missing_token(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(f"/ws/sessions/{sid}") as w:
        w.receive_json()


def test_ws_not_found(client, auth_ws):
    with pytest.raises(WebSocketDisconnect), _connect(client, uuid.uuid4()) as w:
        w.receive_json()


def test_full_flow(client, auth_ws):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with _connect(client, sid) as w:
        assert w.receive_json()["type"] == "ready"
        w.send_json({"type": "start"})
        assert w.receive_json() == {"type": "started", "status": "running", "question_index": 0}

        w.send_bytes(b"\xff\xd8fakejpeg")
        msg = w.receive_json()
        assert msg["type"] == "result" and msg["face_found"] is True and msg["gaze"] is not None

        w.send_json({"type": "question", "index": 1})
        assert w.receive_json() == {"type": "question_ack", "index": 1}

        w.send_json({"type": "ping"})
        assert w.receive_json()["type"] == "pong"

        w.send_json({"type": "end"})
        assert w.receive_json() == {"type": "report_ready", "session_id": sid}

    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["status"] == "finished"
    assert detail["questions"][0]["ended_at"] is not None and detail["questions"][1]["started_at"] is not None
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["overview"]["face_found_rate"] == 1.0 and rep["per_question"][0]["gaze_hold_rate"] == 1.0


def test_ws_rejects_second_connection_and_finished_session(client, auth_ws):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with _connect(client, sid) as w:
        w.receive_json()
        with pytest.raises(WebSocketDisconnect), _connect(client, sid) as w2:
            w2.receive_json()
        w.send_json({"type": "end"})
        w.receive_json()
    with pytest.raises(WebSocketDisconnect), _connect(client, sid) as w3:
        w3.receive_json()


def test_bad_question_index_returns_error_not_disconnect(client, auth_ws):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with _connect(client, sid) as w:
        w.receive_json()
        w.send_json({"type": "start"})
        w.receive_json()
        w.send_json({"type": "question", "index": 99})
        assert w.receive_json()["code"] == "bad_question"
        w.send_json({"type": "end"})
        assert w.receive_json()["type"] == "report_ready"


def test_uncertain_frames_excluded_from_report(client, auth_ws, monkeypatch):
    """불확실 프레임은 analysis_logs 에 '불확실' 로 남되 감정 분포·안정 비율 분모에서 빠진다."""
    from app.analysis.types import EMOTION_UNCERTAIN, EmotionResult, FrameResult
    from app.main import app

    class Alternating:
        n = 0

        async def analyze(self, session_id, jpeg, ts_ms):
            self.n += 1
            base = await app.state._mock.analyze(session_id, jpeg, ts_ms)
            if self.n % 2 == 0:
                return FrameResult(
                    ts_ms=ts_ms,
                    face_found=True,
                    face=base.face,
                    gaze=base.gaze,
                    emotion=EmotionResult(
                        base.emotion.probs, EMOTION_UNCERTAIN, accepted=False, confidence=0.5
                    ),
                    attention=base.attention,
                )
            return base

        async def health(self):
            return {}

        async def transcribe(self, audio, language="ko"):
            return {"text": ""}

        async def close(self):
            return None

    app.state._mock = app.state.inference_client
    app.state.inference_client = Alternating()
    sid = client.post("/api/sessions", json={}).json()["id"]
    with client.websocket_connect(f"/ws/sessions/{sid}?token=t") as w:
        w.receive_json()
        w.send_json({"type": "start"})
        w.receive_json()
        for _ in range(6):
            w.send_bytes(b"\xff\xd8x")
            w.receive_json()
        w.send_json({"type": "end"})
        w.receive_json()
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert abs(sum(rep["emotion_distribution"].values()) - 1.0) < 0.01  # 불확실 3장 제외하고 4종 합 1
    assert "불확실" not in rep["emotion_distribution"]
    app.state.inference_client = app.state._mock
