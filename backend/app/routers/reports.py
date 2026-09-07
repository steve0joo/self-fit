import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DbSession

from app.auth import CurrentUserDep
from app.db import get_db
from app.services import report_service
from app.services import session_service as svc

router = APIRouter(prefix="/api/sessions", tags=["reports"])


@router.get("/{session_id}/report")
def report(session_id: uuid.UUID, user: CurrentUserDep, db: Annotated[DbSession, Depends(get_db)]):
    svc.get_owned_session(db, session_id, uuid.UUID(user.id))
    summary = report_service.get_report(db, session_id)
    if summary is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "세션이 끝나지 않았습니다.")
    return summary
