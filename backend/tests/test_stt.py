"""STT 백그라운드 작업: 실제 webm(무음+톤) 으로 구간 추출 → Mock 추론 → transcript 저장."""

import shutil
import subprocess
import time
import uuid

import pytest

from app.db import SessionLocal
from app.models import Session
from app.services import session_service as svc

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 필요")


@pytest.fixture
def webm(tmp_path):
    out = tmp_path / "rec.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            "-t",
            "4",
            "-c:v",
            "libvpx",
            "-b:v",
            "100k",
            "-c:a",
            "libopus",
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return out.read_bytes()


def _wait_stt(client, sid, timeout=15):
    for _ in range(int(timeout * 10)):
        rep = client.get(f"/api/sessions/{sid}/report").json()
        if rep["status"]["stt"] in ("done", "failed", "skipped"):
            return rep
        time.sleep(0.1)
    raise AssertionError("stt 가 끝나지 않음")


def test_stt_runs_after_finish_and_fills_transcript(client, webm, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "stt_enabled", True)
    sid = client.post("/api/sessions", json={}).json()["id"]
    with SessionLocal() as db:
        s = db.get(Session, uuid.UUID(sid))
        svc.start_session(db, s)
    client.post(
        f"/api/sessions/{sid}/recording/chunks?seq=0", files={"chunk": ("0.webm", webm, "video/webm")}
    )
    time.sleep(1.2)  # 질문 0 이 1초 이상 지속되도록
    with SessionLocal() as db:
        s = db.get(Session, uuid.UUID(sid))
        svc.advance_question(db, s, 1)
    time.sleep(1.2)
    r = client.post(f"/api/sessions/{sid}/finish")
    assert r.status_code == 200
    rep = _wait_stt(client, sid)
    assert rep["status"]["stt"] == "done", rep["status"]
    t = {x["question_index"]: x for x in rep["transcript"]}
    assert 0 in t and 1 in t
    assert t[0]["text"] and t[0]["words"] > 0  # Mock 은 고정 문장
    assert set(t[0]) >= {"question_index", "text", "words", "speech_ms"}


def test_stt_skipped_without_recording(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "stt_enabled", True)
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/finish")
    rep = _wait_stt(client, sid)
    assert rep["status"]["stt"] == "skipped" and rep["transcript"] == []


def test_stt_disabled(client, webm):
    sid = client.post("/api/sessions", json={}).json()["id"]
    with SessionLocal() as db:
        svc.start_session(db, db.get(Session, uuid.UUID(sid)))
    client.post(
        f"/api/sessions/{sid}/recording/chunks?seq=0", files={"chunk": ("0.webm", webm, "video/webm")}
    )
    client.post(f"/api/sessions/{sid}/finish")
    assert client.get(f"/api/sessions/{sid}/report").json()["status"]["stt"] == "skipped"
