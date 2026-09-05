"""SQLAlchemy 동기 엔진과 세션. 설계 D9: 동기 + SQL 파일 마이그레이션.

- 로컬 개발: DATABASE_URL 미설정 시 SQLite(dev.db). 앱 시작 시 테이블 자동 생성.
- Supabase: db/migrations/*.sql 을 SQL Editor 에서 실행. 앱은 테이블을 만들지 않는다.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    url = get_settings().database_url
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    # Supabase 연결 풀러(transaction 모드)와 호환되도록 앱 쪽 풀은 작게 유지
    return create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def is_sqlite() -> bool:
    return engine.url.get_backend_name() == "sqlite"


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
