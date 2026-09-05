"""REST 요청/응답 스키마 (03-phase1-design.md 5절)."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class QuestionOut(BaseModel):
    id: int
    text: str
    category: str | None = None
    sort_order: int


class SessionCreate(BaseModel):
    mode: str = Field(default="live", pattern="^(live|upload)$")
    question_ids: list[int] | None = None


class SessionQuestionOut(BaseModel):
    order_index: int
    question_id: int
    text: str
    started_at: datetime | None = None
    ended_at: datetime | None = None


class SessionOut(BaseModel):
    id: UUID
    mode: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    questions: list[SessionQuestionOut]
    ws_url: str


class SessionListItem(BaseModel):
    id: UUID
    mode: str
    status: str
    created_at: datetime
    finished_at: datetime | None
    has_report: bool


class SessionList(BaseModel):
    items: list[SessionListItem]
    total: int


class EventOut(BaseModel):
    ts_ms: int
    question_index: int | None
    type: str
    severity: str
    message: str
