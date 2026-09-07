from pydantic import BaseModel


class FaceOut(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float


class GazeOut(BaseModel):
    yaw_deg: float
    pitch_deg: float
    confidence: float


class ProbsOut(BaseModel):
    probs: dict[str, float]
    top: str
    # 감정만 채운다. top-1 확률이 tau 이상인지 — 즉 이 프레임의 판정을 믿을지 여부.
    # 버릴지 말지는 백엔드 판정 규칙이 정한다 (06-backend-handoff.md 4.4절)
    accepted: bool | None = None


class AnalyzeOut(BaseModel):
    face_found: bool
    face: FaceOut | None = None
    gaze: GazeOut | None = None
    emotion: ProbsOut | None = None
    attention: ProbsOut | None = None
    timing_ms: dict[str, int]
