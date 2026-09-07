"""분석 결과 타입. 추론 서버 응답(5.5절)과 1:1 로 대응한다."""

from dataclasses import dataclass, field
from typing import Any

# 모델 출력 순서 (EmotionNet 원본 7클래스). 파인튜닝 모델 납품 시 확인 후 수정
EMOTION_LABELS = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]
# 서비스가 판정에 쓰는 4종. 1등이 여기 없으면 '기타' (재정규화하지 않는다)
EMOTION_USED = {"중립", "불안", "당황", "기쁨"}
EMOTION_OTHER = "기타"
# Former-DFER 5클래스. 인덱스 순서는 AI-Hub 표기 순 가정 (03-for-ai.md 3절)
ATTENTION_LABELS = ["집중", "졸림", "집중결핍", "집중하락", "태만"]
ATTENTION_FOCUSED = "집중"


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass(frozen=True)
class GazeResult:
    yaw_deg: float
    pitch_deg: float
    confidence: float


@dataclass(frozen=True)
class EmotionResult:
    probs: dict[str, float]  # 모델 출력 그대로
    top: str  # 4종 중 하나 또는 '기타'


@dataclass(frozen=True)
class AttentionResult:
    probs: dict[str, float]
    top: str


@dataclass(frozen=True)
class FrameResult:
    ts_ms: int
    face_found: bool | None  # None = 추론 실패
    face: FaceBox | None = None
    gaze: GazeResult | None = None
    emotion: EmotionResult | None = None
    attention: AttentionResult | None = None
    timing_ms: dict[str, Any] = field(default_factory=dict)


def emotion_top_from_probs(probs: dict[str, float]) -> str:
    """7클래스 확률에서 1등을 뽑고, 서비스 4종이 아니면 '기타'."""
    if not probs:
        return EMOTION_OTHER
    top = max(probs, key=probs.get)
    return top if top in EMOTION_USED else EMOTION_OTHER


def to_dict(r: FrameResult) -> dict:
    """WebSocket result 메시지용."""
    return {
        "ts_ms": r.ts_ms,
        "face_found": r.face_found,
        "gaze": None
        if r.gaze is None
        else {"yaw": r.gaze.yaw_deg, "pitch": r.gaze.pitch_deg, "confidence": r.gaze.confidence},
        "emotion": None if r.emotion is None else {"top": r.emotion.top, "probs": r.emotion.probs},
        "attention": None if r.attention is None else {"top": r.attention.top, "probs": r.attention.probs},
    }
