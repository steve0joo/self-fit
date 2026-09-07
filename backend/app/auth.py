"""Supabase가 발급한 사용자 JWT를 검증한다.

- 기본: Supabase JWKS(ES256 공개키)로 서명 검증. 시크릿 불필요.
- 예외: 구형 HS256 프로젝트는 SUPABASE_JWT_SECRET을 설정하면 대칭키로 검증.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from app.config import Settings, get_settings

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None
    role: str


@lru_cache
def _jwk_client(jwks_url: str) -> PyJWKClient:
    # 키를 캐시해 요청마다 JWKS를 내려받지 않는다.
    return PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)


def _decode(token: str, settings: Settings) -> dict:
    header = jwt.get_unverified_header(token)
    alg = header.get("alg")
    if alg == "HS256":
        if not settings.supabase_jwt_secret:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "HS256 토큰이지만 SUPABASE_JWT_SECRET이 설정되지 않았습니다.",
            )
        return jwt.decode(token, settings.supabase_jwt_secret, algorithms=["HS256"], audience="authenticated")
    key = _jwk_client(settings.jwks_url).get_signing_key_from_jwt(token).key
    return jwt.decode(token, key, algorithms=[alg], audience="authenticated")


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CurrentUser:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authorization: Bearer <token> 헤더가 필요합니다.")
    try:
        payload = _decode(creds.credentials, settings)
    except HTTPException:
        raise
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "토큰이 만료되었습니다.")
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"토큰 검증 실패: {e}")

    sub = payload.get("sub")
    if not sub:
        # anon 키 같은 서비스 토큰은 sub가 없다. 사용자 토큰만 통과시킨다.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "사용자 토큰이 아닙니다.")
    return CurrentUser(id=sub, email=payload.get("email"), role=payload.get("role", "authenticated"))


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
