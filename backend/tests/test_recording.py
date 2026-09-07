"""녹화 조각 업로드 → 종료 시 합치기 → 재생 (guideline/05 1절)."""

import uuid

from app.db import SessionLocal
from app.models import Session
from app.services import session_service as svc

# 실제 webm 이 아니어도 이어 붙이기·저장 로직은 검증된다. ffmpeg remux 는 실패 시 raw 로 폴백.
CHUNKS = [b"\x1aE\xdf\xa3chunk0", b"chunk1-data", b"chunk2-data"]


def _start(client, sid):
    with SessionLocal() as db:
        svc.start_session(db, db.get(Session, uuid.UUID(sid)))


def _post(client, sid, seq, data):
    return client.post(
        f"/api/sessions/{sid}/recording/chunks?seq={seq}",
        files={"chunk": (f"{seq}.webm", data, "video/webm")},
    )


def test_rejects_before_start(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    assert _post(client, sid, 0, b"x").status_code == 409


def test_upload_assemble_and_stream(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    _start(client, sid)
    for i, c in enumerate(CHUNKS):
        assert _post(client, sid, i, c).status_code == 204
    assert _post(client, sid, 1, CHUNKS[1]).status_code == 204  # 재전송 멱등
    assert client.get(f"/api/sessions/{sid}/recording").status_code == 404  # 아직 합치기 전
    client.post(f"/api/sessions/{sid}/finish")
    r = client.get(f"/api/sessions/{sid}/recording")
    assert r.status_code == 200 and r.headers["content-type"].startswith("video/webm")
    assert r.content == b"".join(CHUNKS)
    # Range 요청 → <video> 탐색
    r = client.get(f"/api/sessions/{sid}/recording", headers={"Range": "bytes=0-4"})
    assert r.status_code == 206 and r.content == b"".join(CHUNKS)[:5]
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"]["recording"] == "done" and rep["recording"]["url"].endswith("/recording")
    assert rep["status"]["metrics"] == "done" and rep["transcript"] == [] and rep["llm"] is None


def test_report_without_recording(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/finish")
    rep = client.get(f"/api/sessions/{sid}/report").json()
    assert rep["status"]["recording"] == "skipped" and rep["recording"] is None


def test_other_user_cannot_upload_or_view(client, other_client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    _start(client, sid)
    assert _post(other_client, sid, 0, b"x").status_code == 404
    assert other_client.get(f"/api/sessions/{sid}/recording").status_code == 404


def test_delete_session_removes_recording(client):
    from app.services import recording_service as rec

    sid = client.post("/api/sessions", json={}).json()["id"]
    _start(client, sid)
    _post(client, sid, 0, b"abc")
    client.post(f"/api/sessions/{sid}/finish")
    assert rec.final_path(uuid.UUID(sid)).exists()
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert not rec.session_dir(uuid.UUID(sid)).exists()


def test_empty_and_oversized_chunk(client, monkeypatch):
    sid = client.post("/api/sessions", json={}).json()["id"]
    _start(client, sid)
    assert _post(client, sid, 0, b"").status_code in (400, 422)
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "recording_max_chunk_bytes", 10)
    assert _post(client, sid, 0, b"x" * 11).status_code == 413
