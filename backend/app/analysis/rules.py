"""이벤트 판정 엔진 (03-phase1-design.md 8절).

세션마다 1개. FrameResult 를 시간순으로 feed 하면 조건이 지속 시간을 넘길 때 Event 를 돌려준다.
프레임 수가 아니라 ts_ms 차이로 판단하므로 프레임 누락에 견딘다.
"""

from dataclasses import dataclass, field

from app.analysis.types import FrameResult
from app.config import Settings


@dataclass(frozen=True)
class Event:
    ts_ms: int
    type: str
    severity: str
    icon: str
    message: str
    payload: dict = field(default_factory=dict)


# (type, severity, icon, message)
EVENT_SPECS = {
    "gaze_off": ("warn", "👁️", "시선이 화면 밖으로 벗어났어요"),
    "face_lost": ("warn", "🙈", "얼굴이 화면에서 보이지 않아요"),
    "emotion_negative": ("warn", "🙂", "표정에서 긴장이 감지됐어요"),
    "emotion_surprised": ("info", "😮", "당황한 표정이 감지됐어요"),
    "attention_low": ("warn", "😴", "집중이 흐트러진 것 같아요"),
    "gaze_stable": ("info", "✅", "좋아요, 안정적인 시선이에요"),
}
COOLDOWN_MS = {
    "gaze_off": 10_000,
    "face_lost": 10_000,
    "emotion_negative": 15_000,
    "emotion_surprised": 15_000,
    "attention_low": 20_000,
    "gaze_stable": 60_000,
}
ATTENTION_LOW_LABELS = {"졸림", "집중결핍", "집중하락", "태만"}


def gaze_state(r: FrameResult, s: Settings) -> str:
    if not r.face_found or r.gaze is None or r.gaze.confidence < 0.3:
        return "unknown"
    off = abs(r.gaze.yaw_deg) > s.gaze_yaw_threshold_deg or abs(r.gaze.pitch_deg) > s.gaze_pitch_threshold_deg
    return "off" if off else "center"


class RuleEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        self._since: dict[str, int | None] = {k: None for k in EVENT_SPECS}  # 조건이 참이 된 시각
        self._last_fired: dict[str, int] = {}

    def _conditions(self, r: FrameResult) -> dict[str, tuple[bool, float, dict]]:
        s = self.s
        gs = gaze_state(r, s)
        face_lost = r.face_found is False
        neg = sur = low = False
        payload_e: dict = {}
        payload_a: dict = {}
        if r.face_found and r.emotion and r.emotion.accepted:  # τ 미달(불확실) 프레임은 감정 판정에 쓰지 않음
            p = r.emotion.probs.get(r.emotion.top, 0.0)
            neg = r.emotion.top == "불안" and p >= s.emotion_negative_min_prob
            sur = r.emotion.top == "당황" and p >= s.emotion_surprised_min_prob
            payload_e = {"top": r.emotion.top, "prob": round(p, 3)}
        if r.face_found and r.attention:
            pa = r.attention.probs.get(r.attention.top, 0.0)
            low = r.attention.top in ATTENTION_LOW_LABELS and pa >= s.attention_low_min_prob
            payload_a = {"top": r.attention.top, "prob": round(pa, 3)}
        return {
            "gaze_off": (gs == "off", s.gaze_off_seconds, {"yaw": r.gaze.yaw_deg if r.gaze else None}),
            "face_lost": (face_lost, s.face_lost_seconds, {}),
            "emotion_negative": (neg, s.emotion_negative_seconds, payload_e),
            "emotion_surprised": (sur, s.emotion_surprised_seconds, payload_e),
            "attention_low": (low, s.attention_low_seconds, payload_a),
            "gaze_stable": (gs == "center", s.gaze_stable_seconds, {}),
        }

    def feed(self, r: FrameResult) -> list[Event]:
        fired: list[Event] = []
        for etype, (cond, hold_s, payload) in self._conditions(r).items():
            if not cond:
                self._since[etype] = None
                continue
            if self._since[etype] is None:
                self._since[etype] = r.ts_ms
            held = r.ts_ms - self._since[etype]
            last = self._last_fired.get(etype)
            in_cooldown = last is not None and (r.ts_ms - last) < COOLDOWN_MS[etype]
            if held >= hold_s * 1000 and not in_cooldown:
                sev, icon, msg = EVENT_SPECS[etype]
                fired.append(Event(r.ts_ms, etype, sev, icon, msg, {**payload, "held_ms": held}))
                self._last_fired[etype] = r.ts_ms
                self._since[etype] = None  # 다시 지속 시간을 채워야 재발행
        return fired
