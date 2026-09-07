"""WebSocket 실시간 채널 (03-phase1-design.md 6절).

GET /ws/sessions/{id}?token=<access_token>
- 텍스트: JSON 제어 메시지 (start | resume | question | end | ping)
- 바이너리: JPEG 프레임 1장
"""

import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import get_current_user
from app.config import get_settings
from app.db import SessionLocal
from app.models import Session
from app.services import report_service
from app.services import session_service as svc
from app.services.live_session import LiveSession

router = APIRouter(tags=["ws"])
log = logging.getLogger("selffit.ws")

CLOSE_UNAUTHORIZED, CLOSE_FORBIDDEN, CLOSE_NOT_FOUND, CLOSE_CONFLICT = 4401, 4403, 4404, 4409
_active: set[uuid.UUID] = set()  # 세션당 연결 1개


@router.websocket("/ws/sessions/{session_id}")
async def session_ws(ws: WebSocket, session_id: uuid.UUID, token: str | None = None):
    settings = get_settings()
    await ws.accept()

    # 인증: 토큰은 쿼리로 (브라우저 WebSocket 은 헤더를 못 붙임)
    try:
        if not token:
            raise ValueError("token missing")
        user = get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token), settings)
    except Exception:  # noqa: BLE001 - 어떤 검증 실패든 4401 로 통일
        await ws.close(code=CLOSE_UNAUTHORIZED)
        return

    db = SessionLocal()
    try:
        s = db.get(Session, session_id)
        if s is None:
            await ws.close(code=CLOSE_NOT_FOUND)
            return
        if s.user_id != uuid.UUID(user.id):
            await ws.close(code=CLOSE_FORBIDDEN)
            return
        if s.status == "finished" or session_id in _active:
            await ws.close(code=CLOSE_CONFLICT)
            return
        _active.add(session_id)

        client = ws.app.state.inference_client
        live = LiveSession(ws, db, session_id, client, settings)
        await ws.send_json(
            {
                "type": "ready",
                "session_id": str(session_id),
                "status": s.status,
                "fps_hint": 3,
                "frame_size_hint": 224,
            }
        )

        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if s.status == "running":
                    live.submit_frame(msg["bytes"])
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "code": "bad_json", "message": "JSON 형식이 아닙니다."})
                continue
            mtype = data.get("type")
            if mtype in ("start", "resume"):
                svc.start_session(db, s)
                await ws.send_json(
                    {"type": "started", "status": s.status, "question_index": svc.current_question_index(s)}
                )
            elif mtype == "question":
                try:
                    applied = svc.advance_question(db, s, int(data.get("index", -1)))
                    await ws.send_json({"type": "question_ack", "index": applied})
                except Exception as e:  # noqa: BLE001 - 잘못된 입력은 연결을 끊지 않고 error 로 회신
                    await ws.send_json(
                        {"type": "error", "code": "bad_question", "message": str(getattr(e, "detail", e))}
                    )
            elif mtype == "end":
                await live.stop()
                svc.finish_session(db, s)
                report_service.save_report(db, s)
                await ws.send_json({"type": "report_ready", "session_id": str(session_id)})
                await ws.close(code=1000)
                break
            elif mtype == "ping":
                await ws.send_json({"type": "pong"})
            else:
                await ws.send_json(
                    {"type": "error", "code": "unknown_type", "message": f"알 수 없는 type: {mtype}"}
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws error")
    finally:
        _active.discard(session_id)
        try:
            await live.stop()  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001 - 인증 실패로 live 가 없을 수 있음
            log.debug("live session not started")
        db.close()
