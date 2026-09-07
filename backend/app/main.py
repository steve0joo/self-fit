import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.analysis.client import build_client
from app.config import get_settings
from app.db import Base, SessionLocal, engine, is_sqlite
from app.models import Question
from app.routers import health, me, questions, recordings, reports, sessions, ws

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SEED_QUESTIONS = [
    ("1분간 자기소개를 해주세요.", "general"),
    ("이 직무에 지원한 동기를 말씀해주세요.", "motivation"),
    ("본인의 강점과 약점은 무엇인가요?", "general"),
    ("협업 중 갈등을 해결했던 경험이 있나요?", "experience"),
    ("마지막으로 하고 싶은 말씀이 있다면 해주세요.", "closing"),
]


def init_sqlite_dev_db() -> None:
    """로컬 SQLite 전용: 테이블 생성 + 질문 seed. Supabase 는 db/migrations/*.sql 로."""
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        if db.scalar(select(Question).limit(1)) is None:
            db.add_all(
                [Question(text=t, category=c, sort_order=i + 1) for i, (t, c) in enumerate(SEED_QUESTIONS)]
            )
            db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if is_sqlite():
        init_sqlite_dev_db()
    app.state.inference_client = build_client(
        settings.inference_backend,
        url=settings.inference_url,
        token=settings.inference_token,
        timeout_ms=settings.inference_timeout_ms,
    )
    logging.getLogger("selffit").info(
        "db=%s inference=%s", engine.url.get_backend_name(), settings.inference_backend
    )
    yield
    await app.state.inference_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="SelfFit Backend", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (health, me, questions, sessions, reports, recordings, ws):
        app.include_router(r.router)
    return app


app = create_app()
