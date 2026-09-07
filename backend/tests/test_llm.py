"""LLM 리포트: OpenAI 호출을 가짜로 바꿔 흐름과 구조를 검증한다."""

import time
import uuid

import pytest

from app.db import SessionLocal
from app.models import Session
from app.services import llm_service
from app.services import session_service as svc

FAKE_OUT = {
    "summary": "전반적으로 차분하게 답변하셨습니다. 시선 유지율이 높았고 집중도도 안정적이었습니다.",
    "per_question": [{"question_index": 0, "feedback": "자기소개가 구조적이었습니다."}],
    "strengths": ["안정적인 시선"],
    "improvements": ["답변 초반에 결론을 먼저 말해 보세요."],
}


class FakeLlm:
    def __init__(self, fail=False):
        self.fail = fail
        self.payloads = []

    @property
    def model_name(self):
        return "fake/test"

    async def generate(self, payload):
        self.payloads.append(payload)
        if self.fail:
            raise RuntimeError("boom")
        return FAKE_OUT


def _wait(client, sid, key="llm", timeout=10):
    for _ in range(int(timeout * 10)):
        rep = client.get(f"/api/sessions/{sid}/report").json()
        if rep["status"][key] in ("done", "failed", "skipped"):
            return rep
        time.sleep(0.1)
    raise AssertionError(f"{key} 가 끝나지 않음")


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLlm()
    monkeypatch.setattr(llm_service, "build_client", lambda: fake)
    monkeypatch.setattr(llm_service.get_settings(), "openai_api_key", "test-key")
    return fake


def test_llm_runs_after_finish(client, fake_llm):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with SessionLocal() as db:
        s = db.get(Session, uuid.UUID(sid))
        svc.start_session(db, s)
        svc.advance_question(db, s, 1)
    r = client.post(f"/api/sessions/{sid}/finish")
    assert r.status_code == 200
    rep = _wait(client, sid)
    assert rep["status"]["llm"] == "done" and rep["status"]["stt"] == "skipped"
    assert set(rep["llm"]) == {"summary", "per_question", "strengths", "improvements", "model"}
    assert rep["llm"]["model"] == "fake/test" and rep["llm"]["summary"].startswith("전반적으로")
    # LLM 입력에 질문 텍스트·지표·이벤트가 들어갔는지
    p = fake_llm.payloads[0]
    assert {q["question_index"] for q in p["questions"]} == {0, 1, 2, 3, 4}
    assert p["questions"][0]["question"].startswith("1분간") and "overview" in p and "events" in p


def test_llm_failed_status(client, monkeypatch):
    fake = FakeLlm(fail=True)
    monkeypatch.setattr(llm_service, "build_client", lambda: fake)
    monkeypatch.setattr(llm_service.get_settings(), "openai_api_key", "test-key")
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/finish")
    rep = _wait(client, sid)
    assert rep["status"]["llm"] == "failed" and rep["llm"] is None
    assert rep["feedback"] is not None  # 규칙 기반 피드백은 그대로 남음


def test_llm_skipped_without_key(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/finish")
    rep = _wait(client, sid)
    assert rep["status"]["llm"] == "skipped" and rep["llm"] is None


def test_build_payload_uses_transcript():
    summary = {
        "duration_ms": 65000,
        "overview": {"gaze_hold_rate": 0.8},
        "emotion_distribution": {},
        "attention_distribution": {},
        "per_question": [{"order_index": 0, "gaze_hold_rate": 0.9}],
        "timeline": [{"ts_ms": 12000, "question_index": 0, "type": "gaze_off"}],
        "transcript": [{"question_index": 0, "text": "안녕하세요 저는", "words": 2, "speech_ms": 30000}],
    }
    p = llm_service.build_payload(summary, [(0, "자기소개"), (1, "지원동기")])
    assert p["questions"][0]["answer_text"] == "안녕하세요 저는" and p["questions"][0]["speech_sec"] == 30
    assert p["questions"][1]["answer_text"] == "" and p["events"][0] == {
        "sec": 12,
        "question_index": 0,
        "type": "gaze_off",
    }
