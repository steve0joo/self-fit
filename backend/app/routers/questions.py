from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DbSession

from app.db import get_db
from app.schemas import QuestionOut
from app.services.session_service import list_active_questions

router = APIRouter(prefix="/api", tags=["questions"])


@router.get("/questions", response_model=list[QuestionOut])
def list_questions(db: Annotated[DbSession, Depends(get_db)]):
    return list_active_questions(db)
