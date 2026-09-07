from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """배포 상태 확인. Render 헬스체크와 FE 연결 확인에 사용."""
    return {"status": "ok", "service": "self-fit-backend"}
