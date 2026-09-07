from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_token(
    x_inference_token: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> None:
    if settings.inference_token and x_inference_token != settings.inference_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "X-Inference-Token 이 없거나 틀립니다.")
