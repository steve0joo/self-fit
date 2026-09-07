"""추론 서버 장애 처리 (03-phase1-design.md 2절 '추론 서버 장애', 5.5절).
가짜 추론 서버(작은 FastAPI 앱)를 httpx ASGITransport 로 붙여 네트워크 없이 검증한다."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI, Response

from app.analysis.http_client import HttpInferenceClient, InferenceError

OK = {
    "face_found": True,
    "face": {"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.9},
    "gaze": {"yaw_deg": -3.2, "pitch_deg": 5.1, "confidence": 0.8},
    "emotion": {
        "probs": {
            "기쁨": 0.1,
            "당황": 0.02,
            "분노": 0.01,
            "불안": 0.05,
            "상처": 0.01,
            "슬픔": 0.01,
            "중립": 0.8,
        },
        "top": "중립",
    },
    "attention": {
        "probs": {"집중": 0.7, "졸림": 0.1, "집중결핍": 0.1, "집중하락": 0.05, "태만": 0.05},
        "top": "집중",
    },
    "timing_ms": {"detect": 4, "gaze": 9, "emotion": 1, "attention": 0},
}


def fake_server(mode: str) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/analyze")
    async def analyze(session_id: str):
        if mode == "ok":
            return OK
        if mode == "no_face":
            return {"face_found": False, "timing_ms": {"detect": 3}}
        if mode == "500":
            return Response(status_code=503, content="model not loaded")
        if mode == "slow":
            await asyncio.sleep(0.3)
            return OK
        if mode == "garbage":
            return {"face_found": True, "face": {}, "gaze": None}
        raise AssertionError(mode)

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "device": "cpu", "models": {}}

    return app


def make_client(mode: str, timeout_ms: int = 1000) -> HttpInferenceClient:
    return HttpInferenceClient(
        "http://inference",
        token="secret",
        timeout_ms=timeout_ms,
        transport=httpx.ASGITransport(app=fake_server(mode)),
    )


@pytest.mark.asyncio
async def test_ok_response_parsed():
    c = make_client("ok")
    r = await c.analyze("sid", b"jpeg", 123)
    assert r.face_found is True and r.ts_ms == 123
    assert r.gaze.yaw_deg == -3.2 and r.emotion.top == "중립" and r.attention.top == "집중"
    assert (await c.health())["status"] == "ok"
    await c.close()


@pytest.mark.asyncio
async def test_no_face_response():
    r = await make_client("no_face").analyze("sid", b"jpeg", 1)
    assert r.face_found is False and r.gaze is None and r.emotion is None


@pytest.mark.asyncio
async def test_5xx_raises():
    with pytest.raises(InferenceError):
        await make_client("500").analyze("sid", b"jpeg", 1)


def test_timeout_is_enforced_by_live_session(client, auth_ws, monkeypatch):
    """ASGITransport 는 httpx 타임아웃을 적용하지 않으므로, 실제 운영 경로인 LiveSession 의 wait_for 로 검증한다.
    추론이 300ms 걸리고 타임아웃이 50ms 면 프레임은 face_found=null 로 기록되어야 한다."""
    from app.config import get_settings
    from app.main import app
    from app.routers import ws as ws_router

    fast = get_settings().model_copy(update={"inference_timeout_ms": 50})
    monkeypatch.setattr(ws_router, "get_settings", lambda: fast)
    app.state.inference_client = make_client("slow")
    sid = client.post("/api/sessions", json={}).json()["id"]
    with client.websocket_connect(f"/ws/sessions/{sid}?token=t") as w:
        w.receive_json()
        w.send_json({"type": "start"})
        w.receive_json()
        w.send_bytes(b"\xff\xd8x")
        msg = w.receive_json()
        assert msg["type"] == "result" and msg["face_found"] is None
        w.send_json({"type": "end"})
        assert w.receive_json()["type"] == "report_ready"


@pytest.mark.asyncio
async def test_garbage_response_raises():
    with pytest.raises(InferenceError):
        await make_client("garbage").analyze("sid", b"jpeg", 1)


def test_session_survives_inference_failure(client, auth_ws):
    """추론이 계속 실패해도 프레임은 face_found=null 로 기록되고, 10회째에 error 메시지, 세션은 끝까지 진행."""
    from app.main import app

    app.state.inference_client = make_client("500")
    sid = client.post("/api/sessions", json={}).json()["id"]
    with client.websocket_connect(f"/ws/sessions/{sid}?token=t") as w:
        w.receive_json()
        w.send_json({"type": "start"})
        w.receive_json()
        error_seen = False
        for i in range(10):
            w.send_bytes(b"\xff\xd8x")
            msg = w.receive_json()
            if msg["type"] == "error":
                error_seen = True
                assert msg["code"] == "inference_unavailable"
                msg = w.receive_json()
            assert msg["type"] == "result" and msg["face_found"] is None
        assert error_seen
        w.send_json({"type": "end"})
        assert w.receive_json()["type"] == "report_ready"
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["overview"]["face_found_rate"] == 0.0 and rep["overview"]["event_count"] == 0


@pytest.mark.asyncio
async def test_uncertain_emotion_parsed():
    """accepted=False 는 top='불확실' 로 바뀌고 확률은 그대로 보존된다."""
    from app.analysis.http_client import parse_analyze
    from app.analysis.types import EMOTION_UNCERTAIN

    d = dict(OK)
    d["emotion"] = {
        "probs": {"기쁨": 0.01, "당황": 0.44, "불안": 0.54, "중립": 0.01},
        "top": "불안",
        "accepted": False,
        "confidence": 0.54,
    }
    r = parse_analyze(d, 1)
    assert (
        r.emotion.top == EMOTION_UNCERTAIN and r.emotion.accepted is False and r.emotion.probs["불안"] == 0.54
    )
