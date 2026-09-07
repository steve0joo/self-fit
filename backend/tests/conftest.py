"""테스트 공통: 임시 SQLite, 인증 우회, Mock 추론."""

import os
import uuid

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon")
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["INFERENCE_BACKEND"] = "mock"
os.environ["APP_ENV"] = "test"
os.environ["MEDIA_DIR"] = "./test_media"

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app import db as dbmod
from app.auth import CurrentUser, get_current_user
from app.main import app, init_sqlite_dev_db

TEST_USER = CurrentUser(id=str(uuid.uuid4()), email="tester@example.com", role="authenticated")
OTHER_USER = CurrentUser(id=str(uuid.uuid4()), email="other@example.com", role="authenticated")


@pytest.fixture(autouse=True)
def fresh_db():
    import shutil

    dbmod.Base.metadata.drop_all(dbmod.engine)
    init_sqlite_dev_db()
    shutil.rmtree("./test_media", ignore_errors=True)
    yield
    shutil.rmtree("./test_media", ignore_errors=True)


USERS = {"main": TEST_USER, "other": OTHER_USER}


def _fake_user(request: Request) -> CurrentUser:
    return USERS[request.headers.get("x-test-user", "main")]


@pytest.fixture(autouse=True)
def fake_auth():
    app.dependency_overrides[get_current_user] = _fake_user
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app, headers={"x-test-user": "main"}) as c:
        yield c


@pytest.fixture
def other_client():
    with TestClient(app, headers={"x-test-user": "other"}) as c:
        yield c


@pytest.fixture
def auth_ws(monkeypatch):
    """WebSocket 은 dependency_overrides 를 타지 않으므로 라우터의 검증 함수를 직접 바꾼다."""
    from app.routers import ws as ws_router

    monkeypatch.setattr(ws_router, "get_current_user", lambda creds, settings: TEST_USER)
