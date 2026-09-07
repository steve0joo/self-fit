"""녹화 조각 업로드와 재생 (guideline/05 1.2절)."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as DbSession

from app.auth import CurrentUserDep
from app.config import Settings, get_settings
from app.db import get_db
from app.services import recording_service as rec
from app.services import session_service as svc

router = APIRouter(prefix="/api/sessions", tags=["recording"])


@router.post("/{session_id}/recording/chunks", status_code=status.HTTP_204_NO_CONTENT)
async def upload_chunk(
    session_id: uuid.UUID,
    chunk: UploadFile,
    user: CurrentUserDep,
    db: Annotated[DbSession, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    seq: int = Query(..., ge=0),
):
    s = svc.get_owned_session(db, session_id, uuid.UUID(user.id))
    if s.status == "created" or (s.status == "finished" and not rec.within_grace(s.finished_at)):
        raise HTTPException(status.HTTP_409_CONFLICT, "진행 중인 세션에만 녹화 조각을 올릴 수 있습니다.")
    data = await chunk.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "빈 조각입니다.")
    if len(data) > settings.recording_max_chunk_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "조각이 너무 큽니다. timeslice 를 줄이세요."
        )
    rec.save_chunk(session_id, seq, data)
    if s.status == "finished":
        # 종료 후 유예 시간 안에 도착한 마지막 조각: 합쳐진 영상을 다시 만든다
        rec.assemble(session_id)


@router.get("/{session_id}/recording")
def get_recording(session_id: uuid.UUID, user: CurrentUserDep, db: Annotated[DbSession, Depends(get_db)]):
    svc.get_owned_session(db, session_id, uuid.UUID(user.id))
    f = rec.final_path(session_id)
    if not f.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "녹화가 없습니다.")
    # FileResponse 는 Range 요청을 처리하므로 <video> 탐색이 된다
    return FileResponse(f, media_type="video/webm", filename=None)
