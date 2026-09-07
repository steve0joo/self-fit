from fastapi import APIRouter

from app.auth import CurrentUserDep

router = APIRouter(prefix="/api", tags=["auth"])


@router.get("/me")
def me(user: CurrentUserDep) -> dict:
    """Supabase 로그인 토큰이 백엔드에서 검증되는지 확인하는 엔드포인트."""
    return {"id": user.id, "email": user.email, "role": user.role}
