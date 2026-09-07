"""판정 엔진: 시각 기반 지속·쿨다운 경계."""

from app.analysis.rules import RuleEngine
from app.analysis.types import EmotionResult, FaceBox, FrameResult, GazeResult, emotion_top_from_probs
from app.config import get_settings

S = get_settings()


def frame(ts_ms, yaw=0.0, emotion_top="중립", prob=0.8, face=True):
    if not face:
        return FrameResult(ts_ms=ts_ms, face_found=False)
    probs = {"기쁨": 0, "당황": 0, "불안": 0, "중립": 0}
    probs[emotion_top] = prob
    return FrameResult(
        ts_ms=ts_ms,
        face_found=True,
        face=FaceBox(0, 0, 10, 10, 1.0),
        gaze=GazeResult(yaw, 0.0, 0.9),
        emotion=EmotionResult(probs, emotion_top_from_probs(probs)),
    )


def test_gaze_off_fires_after_hold_and_respects_cooldown():
    e = RuleEngine(S)
    hold = int(S.gaze_off_seconds * 1000)
    assert e.feed(frame(0, yaw=25)) == []
    assert e.feed(frame(hold - 1, yaw=25)) == []
    fired = e.feed(frame(hold, yaw=25))
    assert [f.type for f in fired] == ["gaze_off"]
    # 발화 직후 조건이 계속 참이어도 지속 시간을 다시 채워야 하고, 쿨다운(10초) 안에서는 재발행 없음
    assert e.feed(frame(hold + 100, yaw=25)) == []  # since 재설정
    assert e.feed(frame(hold * 2 + 200, yaw=25)) == []  # 지속 충족이지만 쿨다운 중
    # 쿨다운이 끝난 시점(발화 + 10초)에 지속도 충족 → 재발행
    assert [f.type for f in e.feed(frame(hold + 10_000, yaw=25))] == ["gaze_off"]


def test_gaze_off_resets_when_back_to_center():
    e = RuleEngine(S)
    e.feed(frame(0, yaw=25))
    e.feed(frame(2000, yaw=0))  # 리셋
    assert e.feed(frame(4000, yaw=25)) == []  # 다시 0부터


def test_emotion_negative_needs_prob_threshold():
    e = RuleEngine(S)
    hold = int(S.emotion_negative_seconds * 1000)
    e.feed(frame(0, emotion_top="불안", prob=0.4))
    assert e.feed(frame(hold, emotion_top="불안", prob=0.4)) == []  # 확률 미달
    e.feed(frame(0, emotion_top="불안", prob=0.7))
    assert [f.type for f in e.feed(frame(hold, emotion_top="불안", prob=0.7))] == ["emotion_negative"]


def test_other_emotion_never_fires():
    e = RuleEngine(S)
    e.feed(frame(0, emotion_top="분노", prob=0.9))
    fired = e.feed(frame(60_000, emotion_top="분노", prob=0.9))
    assert not [f for f in fired if f.type.startswith("emotion")]
    assert frame(0, emotion_top="분노", prob=0.9).emotion.top == "기타"


def test_face_lost():
    e = RuleEngine(S)
    e.feed(frame(0, face=False))
    assert [f.type for f in e.feed(frame(int(S.face_lost_seconds * 1000), face=False))] == ["face_lost"]


def test_uncertain_emotion_frames_are_ignored_by_rules():
    """추론 서버가 accepted=False(τ 미달)로 준 프레임은 감정 이벤트를 만들지 않는다."""
    from app.analysis.types import EMOTION_UNCERTAIN

    e = RuleEngine(S)
    probs = {"기쁨": 0.0, "당황": 0.0, "불안": 0.9, "중립": 0.1}
    mk = lambda ts: FrameResult(
        ts_ms=ts,
        face_found=True,
        face=FaceBox(0, 0, 10, 10, 1.0),
        gaze=GazeResult(0.0, 0.0, 0.9),
        emotion=EmotionResult(probs, EMOTION_UNCERTAIN, accepted=False, confidence=0.9),
    )
    e.feed(mk(0))
    assert not [f for f in e.feed(mk(20_000)) if f.type.startswith("emotion")]
