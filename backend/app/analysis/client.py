"""추론 서버 호출 인터페이스. BE 는 이미지를 열어 보지 않고 이 인터페이스만 안다."""

from typing import Protocol

from app.analysis.types import FrameResult


class InferenceClient(Protocol):
    async def analyze(self, session_id: str, jpeg: bytes, ts_ms: int) -> FrameResult: ...

    async def health(self) -> dict: ...

    async def close(self) -> None: ...


def build_client(backend: str, **kwargs) -> InferenceClient:
    if backend == "mock":
        from app.analysis.mock import MockInferenceClient

        return MockInferenceClient()
    if backend == "http":
        from app.analysis.http_client import HttpInferenceClient

        return HttpInferenceClient(**kwargs)
    raise ValueError(f"unknown INFERENCE_BACKEND: {backend}")
