"""WebSocket 연결 1개 = LiveSession 1개. 프레임 처리, 판정, 저장, 송신을 묶는다.

백프레셔: 추론 중 새 프레임이 오면 가장 최근 1장만 보관하고 이전 것은 버린다.
"""

import asyncio
import logging
import uuid

from fastapi import WebSocket
from sqlalchemy.orm import Session as DbSession

from app.analysis.client import InferenceClient
from app.analysis.rules import RuleEngine, gaze_state
from app.analysis.types import FrameResult, to_dict
from app.config import Settings
from app.models import AnalysisLog, Event, Session
from app.services import session_service as svc

log = logging.getLogger("selffit.live")


class LiveSession:
    def __init__(
        self, ws: WebSocket, db: DbSession, session_id: uuid.UUID, client: InferenceClient, settings: Settings
    ):
        self.ws, self.db, self.session_id, self.client, self.settings = ws, db, session_id, client, settings
        self.rules = RuleEngine(settings)
        self._latest: bytes | None = None
        self._worker: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._closed = False
        self._fail_streak = 0

    # ---- 수신 측
    def submit_frame(self, jpeg: bytes) -> None:
        self._latest = jpeg  # 이전 프레임은 덮어써서 버림
        self._wake.set()
        if self._worker is None:
            self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._closed = True
        self._wake.set()
        if self._worker:
            try:
                await asyncio.wait_for(self._worker, timeout=2)
            except (TimeoutError, asyncio.CancelledError):
                self._worker.cancel()

    # ---- 처리 루프
    async def _loop(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            jpeg, self._latest = self._latest, None
            if jpeg is None:
                continue
            await self._process(jpeg)

    async def _process(self, jpeg: bytes) -> None:
        session = self.db.get(Session, self.session_id)
        if session is None or session.status != "running":
            return
        ts_ms = svc.elapsed_ms(session)
        q_index = svc.current_question_index(session)
        try:
            result = await asyncio.wait_for(
                self.client.analyze(str(self.session_id), jpeg, ts_ms),
                timeout=self.settings.inference_timeout_ms / 1000,
            )
            self._fail_streak = 0
        except Exception as e:  # noqa: BLE001 - 타임아웃, 5xx, 연결 실패 → 이 프레임만 실패 처리
            self._fail_streak += 1
            log.warning("inference failed (%s): %s", self._fail_streak, e)
            result = FrameResult(ts_ms=ts_ms, face_found=None)
            if self._fail_streak == 10:
                await self._send(
                    {
                        "type": "error",
                        "code": "inference_unavailable",
                        "message": "분석 서버에 연결할 수 없어요. 면접은 계속 진행됩니다.",
                    }
                )

        self._save_log(result, q_index)
        events = self.rules.feed(result) if result.face_found is not None else []
        for ev in events:
            self.db.add(
                Event(
                    session_id=self.session_id,
                    ts_ms=ev.ts_ms,
                    question_index=q_index,
                    type=ev.type,
                    severity=ev.severity,
                    message=ev.message,
                    payload=ev.payload,
                )
            )
        if events:
            self.db.commit()
        await self._send({"type": "result", **to_dict(result)})
        for ev in events:
            await self._send(
                {
                    "type": "event",
                    "ts_ms": ev.ts_ms,
                    "event_type": ev.type,
                    "severity": ev.severity,
                    "icon": ev.icon,
                    "message": ev.message,
                }
            )

    def _save_log(self, r: FrameResult, q_index: int | None) -> None:
        self.db.add(
            AnalysisLog(
                session_id=self.session_id,
                ts_ms=r.ts_ms,
                question_index=q_index,
                face_found=r.face_found,
                gaze_yaw=r.gaze.yaw_deg if r.gaze else None,
                gaze_pitch=r.gaze.pitch_deg if r.gaze else None,
                gaze_state=gaze_state(r, self.settings) if r.face_found is not None else None,
                emotion_probs=r.emotion.probs if r.emotion else None,
                emotion_top=r.emotion.top if r.emotion else None,
                attention_probs=r.attention.probs if r.attention else None,
                attention_top=r.attention.top if r.attention else None,
            )
        )
        self.db.commit()

    async def _send(self, msg: dict) -> None:
        if self._closed:
            return
        try:
            await self.ws.send_json(msg)
        except Exception:  # noqa: BLE001 - 송신 실패 = 연결 끊김
            self._closed = True
