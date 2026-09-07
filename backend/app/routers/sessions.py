import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.auth import CurrentUserDep
from app.db import get_db
from app.models import Event, Session
from app.schemas import EventOut, SessionCreate, SessionList, SessionListItem, SessionOut, SessionQuestionOut
from app.services import recording_service, report_service, stt_service
from app.services import session_service as svc

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
DbDep = Annotated[DbSession, Depends(get_db)]


def _ws_url(request: Request, session_id: uuid.UUID) -> str:
    base = str(request.base_url).rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    return f"{base}/ws/sessions/{session_id}"


def _to_out(s: Session, request: Request) -> SessionOut:
    return SessionOut(
        id=s.id,
        mode=s.mode,
        status=s.status,
        created_at=s.created_at,
        started_at=s.started_at,
        finished_at=s.finished_at,
        questions=[
            SessionQuestionOut(
                order_index=q.order_index,
                question_id=q.question_id,
                text=q.question.text,
                started_at=q.started_at,
                ended_at=q.ended_at,
            )
            for q in s.questions
        ],
        ws_url=_ws_url(request, s.id),
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionOut)
def create(body: SessionCreate, user: CurrentUserDep, db: DbDep, request: Request):
    s = svc.create_session(db, uuid.UUID(user.id), body.mode, body.question_ids)
    return _to_out(s, request)


@router.get("", response_model=SessionList)
def list_(
    user: CurrentUserDep, db: DbDep, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
):
    rows, total = svc.list_sessions(db, uuid.UUID(user.id), limit, offset)
    items = [
        SessionListItem(
            id=s.id,
            mode=s.mode,
            status=s.status,
            created_at=s.created_at,
            finished_at=s.finished_at,
            has_report=has,
        )
        for s, has in rows
    ]
    return SessionList(items=items, total=total)


@router.get("/{session_id}", response_model=SessionOut)
def detail(session_id: uuid.UUID, user: CurrentUserDep, db: DbDep, request: Request):
    return _to_out(svc.get_owned_session(db, session_id, uuid.UUID(user.id)), request)


@router.post("/{session_id}/finish")
async def finish(session_id: uuid.UUID, user: CurrentUserDep, db: DbDep, request: Request):
    """WebSocket 없이 강제 종료 (탭 닫힘 복구용). 리포트를 생성하고 STT 를 백그라운드로 건다."""
    s = svc.get_owned_session(db, session_id, uuid.UUID(user.id))
    svc.finish_session(db, s)
    await run_in_threadpool(recording_service.assemble, session_id)
    report_service.save_report(db, s)
    stt_service.schedule(session_id, request.app.state.inference_client)
    return {"status": "finished", "report_ready": True}


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete(session_id: uuid.UUID, user: CurrentUserDep, db: DbDep):
    s = svc.get_owned_session(db, session_id, uuid.UUID(user.id))
    db.delete(s)
    db.commit()
    recording_service.delete(session_id)  # 세션 삭제 시 영상도 삭제 (개인정보)


@router.get("/{session_id}/events", response_model=list[EventOut])
def events(session_id: uuid.UUID, user: CurrentUserDep, db: DbDep):
    svc.get_owned_session(db, session_id, uuid.UUID(user.id))
    return list(db.scalars(select(Event).where(Event.session_id == session_id).order_by(Event.ts_ms)))
