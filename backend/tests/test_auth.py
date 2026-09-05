"""인증 의존성을 오버라이드하지 않고 실제 JWT 검증 로직을 탄다."""

import pytest
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.main import app


@pytest.fixture
def real_auth_client():
    app.dependency_overrides.pop(get_current_user, None)
    with TestClient(app) as c:
        yield c


def test_me_requires_token(real_auth_client):
    assert real_auth_client.get("/api/me").status_code == 401


def test_me_rejects_garbage_token(real_auth_client):
    r = real_auth_client.get("/api/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def test_sessions_require_token(real_auth_client):
    assert real_auth_client.post("/api/sessions", json={}).status_code == 401
