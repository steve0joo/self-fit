"""STT 백그라운드 작업 (guideline/05 2절).

세션 종료 후 recording.webm 에서 질문 구간별 오디오(16kHz mono wav)를 ffmpeg 로 잘라 추론 서버 /v1/transcribe 에 보내고,
결과를 reports.summary.transcript 에 넣는다. status.stt 는 pending → running → done | failed | skipped.
"""

import asyncio
import logging
import shutil
import subprocess
import uuid
from datetime import UTC

from sqlalchemy.orm.attributes import flag_modified

from app.analysis.client import InferenceClient
from app.config import get_settings
from app.db import SessionLocal
from app.models import Report, Session
from app.services import recording_service as rec

log = logging.getLogger("selffit.stt")


def _ms_between(a, b) -> int:
    a = a if a.tzinfo else a.replace(tzinfo=UTC)
    b = b if b.tzinfo else b.replace(tzinfo=UTC)
    return max(0, int((b - a).total_seconds() * 1000))


def extract_wav(src, start_ms: int, end_ms: int) -> bytes | None:
    """녹화에서 [start_ms, end_ms] 구간을 16kHz mono wav 로. 구간이 0.5초 미만이면 None."""
    if end_ms - start_ms < 500 or not shutil.which("ffmpeg"):
        return None
    r = subprocess.run(  # noqa: PLW1510
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-to",
            f"{end_ms / 1000:.3f}",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "pipe:1",
        ],
        capture_output=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    return r.stdout if r.returncode == 0 and len(r.stdout) > 1000 else None


def set_status(session_id: uuid.UUID, key: str, value: str, **fields) -> None:
    with SessionLocal() as db:
        rep = db.get(Report, session_id)
        if rep is None:
            return
        summary = dict(rep.summary)
        summary["status"] = {**summary.get("status", {}), key: value}
        summary.update(fields)
        rep.summary = summary
        flag_modified(rep, "summary")
        db.commit()


async def run(session_id: uuid.UUID, client: InferenceClient) -> None:
    """질문별 STT 실행 후 LLM 리포트 실행. 예외는 status.failed 로 기록하고 삼킨다 (면접 흐름을 막지 않음)."""
    try:
        await _run_stt(session_id, client)
    finally:
        from app.services import llm_service  # 순환 import 방지

        await llm_service.run(session_id)


async def _run_stt(session_id: uuid.UUID, client: InferenceClient) -> None:
    if not get_settings().stt_enabled:
        set_status(session_id, "stt", "skipped")
        return
    src = rec.final_path(session_id)
    if not src.exists():
        set_status(session_id, "stt", "skipped")
        return
    set_status(session_id, "stt", "running")
    try:
        with SessionLocal() as db:
            s = db.get(Session, session_id)
            if s is None or s.started_at is None:
                set_status(session_id, "stt", "skipped")
                return
            spans = [
                (
                    q.order_index,
                    _ms_between(s.started_at, q.started_at),
                    _ms_between(s.started_at, q.ended_at),
                )
                for q in s.questions
                if q.started_at and q.ended_at
            ]
        transcript = []
        for idx, a, b in spans:
            wav = await asyncio.to_thread(extract_wav, src, a, b)
            if wav is None:
                transcript.append(
                    {
                        "question_index": idx,
                        "text": "",
                        "words": 0,
                        "speech_ms": 0,
                        "note": "오디오 없음 또는 구간이 짧음",
                    }
                )
                continue
            res = await client.transcribe(wav, get_settings().stt_language)
            text = res.get("text", "").strip()
            transcript.append(
                {
                    "question_index": idx,
                    "text": text,
                    "words": len(text.split()),
                    "speech_ms": int(res.get("speech_ms", 0)),
                }
            )
        set_status(session_id, "stt", "done", transcript=transcript)
        log.info("stt done %s: %d questions", session_id, len(transcript))
    except Exception:
        log.exception("stt failed %s", session_id)
        set_status(session_id, "stt", "failed")


def schedule(session_id: uuid.UUID, client: InferenceClient) -> None:
    """이벤트 루프 안에서 호출. STT → LLM 을 한 백그라운드 태스크로 실행."""
    asyncio.get_running_loop().create_task(run(session_id, client))
