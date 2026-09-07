"""세션별 얼굴 크롭 16장 버퍼와 집중 상태 추론 주기 관리 (03-phase1-design.md 2절, 7절)."""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _State:
    frames: deque = field(default_factory=lambda: deque(maxlen=16))
    last_face_at: float = 0.0
    last_infer_at: float = 0.0
    last_result: dict | None = None
    touched_at: float = 0.0


class AttentionBuffer:
    def __init__(self, window: int, interval_s: float, reset_s: float, ttl_s: float):
        self.window, self.interval_s, self.reset_s, self.ttl_s = window, interval_s, reset_s, ttl_s
        self._states: dict[str, _State] = {}

    def _get(self, sid: str) -> _State:
        st = self._states.get(sid)
        if st is None:
            st = self._states[sid] = _State(frames=deque(maxlen=self.window))
        st.touched_at = time.monotonic()
        return st

    def push(self, sid: str, frame112: np.ndarray | None) -> tuple[bool, list[np.ndarray] | None]:
        """프레임을 넣고 (추론해야 하는지, 추론에 쓸 16장) 을 돌려준다. frame112=None 은 얼굴 없음."""
        st = self._get(sid)
        now = time.monotonic()
        if frame112 is None:
            if st.last_face_at and now - st.last_face_at > self.reset_s:
                st.frames.clear()
                st.last_result = None
            return False, None
        st.last_face_at = now
        st.frames.append(frame112)
        ready = len(st.frames) == self.window and (now - st.last_infer_at) >= self.interval_s
        return ready, (list(st.frames) if ready else None)

    def set_result(self, sid: str, result: dict) -> None:
        st = self._get(sid)
        st.last_result, st.last_infer_at = result, time.monotonic()

    def last(self, sid: str) -> dict | None:
        return self._get(sid).last_result

    def sweep(self) -> int:
        cutoff = time.monotonic() - self.ttl_s
        stale = [k for k, v in self._states.items() if v.touched_at < cutoff]
        for k in stale:
            del self._states[k]
        return len(stale)
