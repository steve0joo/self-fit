"""Mock 추론 클라이언트 (Phase 1). 이벤트 규칙이 확실히 트리거되도록 시간에 따라 규칙적으로 변한다.

- 시선: yaw 가 20초 주기 사인파(±30°). |yaw|>20° 구간이 주기마다 약 4.6초 지속 → gaze_off 발생
- 감정: 기본 중립. 30초마다 6초 동안 불안 0.7 → emotion_negative 발생. 50초마다 3초 당황
- 집중: 기본 집중. 45초마다 8초 집중하락 0.7 → attention_low 발생. 처음 6초는 None(버퍼 미충족 흉내)
- 얼굴: 70~74초 구간은 얼굴 없음 → face_lost 발생
"""

import math

from app.analysis.types import (
    ATTENTION_LABELS,
    EMOTION_LABELS,
    AttentionResult,
    EmotionResult,
    FaceBox,
    FrameResult,
    GazeResult,
    emotion_top_from_probs,
)


def _dist(labels: list[str], top: str, p: float) -> dict[str, float]:
    rest = (1.0 - p) / (len(labels) - 1)
    return {lab: (p if lab == top else rest) for lab in labels}


class MockInferenceClient:
    async def analyze(self, session_id: str, jpeg: bytes, ts_ms: int) -> FrameResult:
        t = ts_ms / 1000.0
        if 70.0 <= (t % 120.0) < 74.0:
            return FrameResult(ts_ms=ts_ms, face_found=False)

        yaw = 30.0 * math.sin(2 * math.pi * t / 20.0)
        gaze = GazeResult(yaw_deg=round(yaw, 2), pitch_deg=round(3.0 * math.sin(t / 7.0), 2), confidence=0.9)

        cyc30, cyc50 = t % 30.0, t % 50.0
        if 10.0 <= cyc30 < 16.0:
            probs = _dist(EMOTION_LABELS, "불안", 0.7)
        elif 40.0 <= cyc50 < 43.0:
            probs = _dist(EMOTION_LABELS, "당황", 0.75)
        else:
            probs = _dist(EMOTION_LABELS, "중립", 0.8)
        emotion = EmotionResult(probs=probs, top=emotion_top_from_probs(probs))

        attention = None
        if t >= 6.0:
            a_top = "집중하락" if 20.0 <= (t % 45.0) < 28.0 else "집중"
            attention = AttentionResult(probs=_dist(ATTENTION_LABELS, a_top, 0.7), top=a_top)

        return FrameResult(
            ts_ms=ts_ms,
            face_found=True,
            face=FaceBox(x=60, y=40, w=110, h=110, score=0.98),
            gaze=gaze,
            emotion=emotion,
            attention=attention,
            timing_ms={"mock": 0},
        )

    async def transcribe(self, audio: bytes, language: str = "ko") -> dict:
        # 오디오 길이와 무관하게 고정 문장. 리포트 흐름 검증용
        text = "네, 저는 문제를 구조적으로 나누어 해결하는 편이고 팀과 소통하며 일하는 것을 좋아합니다."
        return {
            "text": text,
            "segments": [{"start_ms": 0, "end_ms": 4000, "text": text}],
            "language": language,
            "duration_ms": 5000,
            "speech_ms": 4000,
        }

    async def health(self) -> dict:
        return {"status": "ok", "device": "mock", "models": {}}

    async def close(self) -> None:
        return None
