"""추론 서버 HTTP 클라이언트 (03-phase1-design.md 5.5절). Phase 2 에서 INFERENCE_BACKEND=http 로 사용."""

import httpx

from app.analysis.types import (
    AttentionResult,
    EmotionResult,
    FaceBox,
    FrameResult,
    GazeResult,
    emotion_top_from_probs,
)


class InferenceError(RuntimeError):
    pass


class HttpInferenceClient:
    def __init__(
        self,
        url: str,
        token: str = "",
        timeout_ms: int = 1000,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"X-Inference-Token": token} if token else {},
            timeout=httpx.Timeout(timeout_ms / 1000),
            transport=transport,
        )

    async def analyze(self, session_id: str, jpeg: bytes, ts_ms: int) -> FrameResult:
        try:
            r = await self._client.post(
                "/v1/analyze",
                params={"session_id": session_id},
                content=jpeg,
                headers={"Content-Type": "image/jpeg"},
            )
        except httpx.HTTPError as e:  # 타임아웃, 연결 실패
            raise InferenceError(f"inference request failed: {e!r}") from e
        if r.status_code >= 500:
            raise InferenceError(f"inference server {r.status_code}")
        if r.status_code != 200:
            raise InferenceError(f"inference rejected: {r.status_code} {r.text[:120]}")
        return parse_analyze(r.json(), ts_ms)

    async def health(self) -> dict:
        r = await self._client.get("/v1/health")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()


def parse_analyze(d: dict, ts_ms: int) -> FrameResult:
    """5.5절 응답 JSON → FrameResult. 필수 키가 없으면 InferenceError."""
    try:
        if not d["face_found"]:
            return FrameResult(ts_ms=ts_ms, face_found=False, timing_ms=d.get("timing_ms", {}))
        f, g, e = d["face"], d["gaze"], d["emotion"]
        att = d.get("attention")
        return FrameResult(
            ts_ms=ts_ms,
            face_found=True,
            face=FaceBox(int(f["x"]), int(f["y"]), int(f["w"]), int(f["h"]), float(f.get("score", 1.0))),
            gaze=GazeResult(float(g["yaw_deg"]), float(g["pitch_deg"]), float(g.get("confidence", 1.0))),
            emotion=EmotionResult(probs=dict(e["probs"]), top=emotion_top_from_probs(e["probs"])),
            attention=None if att is None else AttentionResult(probs=dict(att["probs"]), top=str(att["top"])),
            timing_ms=d.get("timing_ms", {}),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise InferenceError(f"bad inference response: {e!r}") from e
