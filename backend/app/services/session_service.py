"""세션 생명주기. 권한(user_id) 확인은 모두 여기서 한다."""

import uuid
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from app.models import Question, Report, Session, SessionQuestion


def now() -> datetime:
    return datetime.now(UTC)


def list_active_questions(db: DbSession) -> list[Question]:
    return list(
        db.scalars(select(Question).where(Question.is_active.is_(True)).order_by(Question.sort_order))
    )


def create_session(db: DbSession, user_id: uuid.UUID, mode: str, question_ids: list[int] | None) -> Session:
    if question_ids:
        qs = list(db.scalars(select(Question).where(Question.id.in_(question_ids))))
        by_id = {q.id: q for q in qs}
        missing = [i for i in question_ids if i not in by_id]
        if missing:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"없는 질문 id: {missing}")
        ordered = [by_id[i] for i in question_ids]
    else:
        ordered = list_active_questions(db)
    if not ordered:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "질문이 없습니다.")
    s = Session(user_id=user_id, mode=mode, status="created")
    s.questions = [SessionQuestion(question_id=q.id, order_index=i) for i, q in enumerate(ordered)]
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_owned_session(db: DbSession, session_id: uuid.UUID, user_id: uuid.UUID) -> Session:
    s = db.get(Session, session_id)
    if s is None or s.user_id != user_id:
        # 존재하지 않음과 내 것이 아님을 구분하지 않는다 (설계 5절)
        raise HTTPException(status.HTTP_404_NOT_FOUND, "세션을 찾을 수 없습니다.")
    return s


def list_sessions(
    db: DbSession, user_id: uuid.UUID, limit: int, offset: int
) -> tuple[list[tuple[Session, bool]], int]:
    total = db.scalar(select(func.count()).select_from(Session).where(Session.user_id == user_id)) or 0
    rows = db.execute(
        select(Session, Report.session_id.isnot(None))
        .outerjoin(Report, Report.session_id == Session.id)
        .where(Session.user_id == user_id)
        .order_by(Session.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [(r[0], bool(r[1])) for r in rows], total


def start_session(db: DbSession, s: Session) -> None:
    """created → running. 질문 0 시작. 이미 running 이면 무시(resume)."""
    if s.status == "running":
        return
    if s.status != "created":
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 종료된 세션입니다.")
    t = now()
    s.status, s.started_at = "running", t
    if s.questions:
        s.questions[0].started_at = t
    db.commit()


def advance_question(db: DbSession, s: Session, index: int) -> int:
    """질문 전환. 앞으로만 간다(단조 증가). 같은 번호나 이전 번호가 오면 무시하고 현재 번호를 돌려준다.

    FE 가 빠른 클릭·재전송으로 번호를 중복하거나 뒤섞어 보내도 started_at/ended_at 이 역전되지 않는다.
    건너뛴 질문은 시작되지 않은 채(duration 0) 남는다.
    """
    if index < 0 or index >= len(s.questions):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"질문 index 범위 밖: {index}")
    cur = current_question_index(s)
    if cur is not None and index <= cur:
        return cur
    t = now()
    for q in s.questions:
        if q.order_index < index and q.started_at is not None and q.ended_at is None:
            q.ended_at = t
    target = next(q for q in s.questions if q.order_index == index)
    if target.started_at is None:
        target.started_at = t
    db.commit()
    return index


def current_question_index(s: Session) -> int | None:
    """시작됐고 아직 끝나지 않은 질문 중 가장 큰 번호. 없으면 None."""
    open_qs = [q.order_index for q in s.questions if q.started_at is not None and q.ended_at is None]
    return max(open_qs) if open_qs else None


def finish_session(db: DbSession, s: Session) -> None:
    """running/created → finished. 열린 질문을 닫는다. 리포트 생성은 호출자가."""
    if s.status == "finished":
        return
    t = now()
    for q in s.questions:
        if q.started_at is not None and q.ended_at is None:
            q.ended_at = t
    s.status, s.finished_at = "finished", t
    if s.started_at is None:
        s.started_at = t
    db.commit()


def elapsed_ms(s: Session) -> int:
    if s.started_at is None:
        return 0
    started = s.started_at if s.started_at.tzinfo else s.started_at.replace(tzinfo=UTC)
    return int((now() - started).total_seconds() * 1000)
