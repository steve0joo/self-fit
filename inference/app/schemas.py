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


class EmotionOut(ProbsOut):
    accepted: bool = True  # 최댓값이 τ 이상이면 True. False 면 판정 보류("불확실")
    confidence: float = 1.0


class AnalyzeOut(BaseModel):
    face_found: bool
    face: FaceOut | None = None
    gaze: GazeOut | None = None
    emotion: EmotionOut | None = None
    attention: ProbsOut | None = None
    timing_ms: dict[str, int]
