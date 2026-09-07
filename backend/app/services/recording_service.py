"""면접 녹화 저장 (guideline/05 1절).

media/{session_id}/chunks/000000.webm ... 로 조각을 받고, 세션 종료 시 recording.webm 하나로 합친다.
MediaRecorder 의 timeslice 조각은 같은 스트림의 연속 바이트라 순서대로 이어 붙이면 재생 가능한 webm 이 된다.
이어 붙인 파일은 duration·cue 정보가 없어 탐색이 안 되므로 ffmpeg 로 remux(-c copy) 해 둔다.
"""

import logging
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings

log = logging.getLogger("selffit.recording")
FINAL_NAME = "recording.webm"


def session_dir(session_id: uuid.UUID) -> Path:
    return Path(get_settings().media_dir) / str(session_id)


def chunk_path(session_id: uuid.UUID, seq: int) -> Path:
    return session_dir(session_id) / "chunks" / f"{seq:06d}.webm"


def final_path(session_id: uuid.UUID) -> Path:
    return session_dir(session_id) / FINAL_NAME


def save_chunk(session_id: uuid.UUID, seq: int, data: bytes) -> Path:
    p = chunk_path(session_id, seq)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)  # 같은 seq 재전송은 덮어씀 (멱등)
    return p


def chunk_count(session_id: uuid.UUID) -> int:
    d = session_dir(session_id) / "chunks"
    return len(list(d.glob("*.webm"))) if d.exists() else 0


def within_grace(finished_at: datetime | None) -> bool:
    if finished_at is None:
        return True
    f = finished_at if finished_at.tzinfo else finished_at.replace(tzinfo=UTC)
    return (datetime.now(UTC) - f).total_seconds() <= get_settings().recording_grace_seconds


def assemble(session_id: uuid.UUID) -> Path | None:
    """조각을 seq 순으로 이어 붙이고 remux. 조각이 없으면 None."""
    d = session_dir(session_id) / "chunks"
    chunks = sorted(d.glob("*.webm")) if d.exists() else []
    if not chunks:
        return None
    raw = session_dir(session_id) / "raw.webm"
    with raw.open("wb") as out:
        for c in chunks:
            out.write(c.read_bytes())
    final = final_path(session_id)
    if shutil.which("ffmpeg"):
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(raw), "-c", "copy", str(final)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            log.warning("ffmpeg remux failed (%s), raw 파일 사용: %s", session_id, r.stderr[:200])
            shutil.move(raw, final)
        else:
            raw.unlink(missing_ok=True)
    else:
        log.warning("ffmpeg 없음: remux 생략 (탐색이 안 될 수 있음)")
        shutil.move(raw, final)
    return final


def duration_ms(path: Path) -> int | None:
    if not shutil.which("ffprobe"):
        return None
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    try:
        return int(float(r.stdout.strip()) * 1000)
    except ValueError:
        return None


def info(session_id: uuid.UUID) -> dict | None:
    """리포트용. 합쳐진 영상이 있으면 {url, duration_ms}."""
    f = final_path(session_id)
    if not f.exists():
        return None
    return {"url": f"/api/sessions/{session_id}/recording", "duration_ms": duration_ms(f)}


def delete(session_id: uuid.UUID) -> None:
    shutil.rmtree(session_dir(session_id), ignore_errors=True)
