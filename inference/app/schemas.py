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


class AnalyzeOut(BaseModel):
    face_found: bool
    face: FaceOut | None = None
    gaze: GazeOut | None = None
    emotion: ProbsOut | None = None
    attention: ProbsOut | None = None
    timing_ms: dict[str, int]
