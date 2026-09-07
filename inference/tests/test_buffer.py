import time

import numpy as np

from app.attention_buffer import AttentionBuffer


def frame():
    return np.zeros((112, 112, 3), dtype=np.uint8)


def test_ready_only_after_window_and_interval():
    b = AttentionBuffer(window=4, interval_s=0.0, reset_s=3.0, ttl_s=600)
    assert all(b.push("s", frame())[0] is False for _ in range(3))
    ready, frames = b.push("s", frame())
    assert ready and len(frames) == 4
    b.set_result("s", {"집중": 1.0})
    assert b.last("s") == {"집중": 1.0}


def test_interval_gate():
    b = AttentionBuffer(window=2, interval_s=10.0, reset_s=3.0, ttl_s=600)
    b.push("s", frame()); assert b.push("s", frame())[0] is True
    b.set_result("s", {"집중": 1.0})
    assert b.push("s", frame())[0] is False  # 10초 안 지남


def test_reset_when_face_lost_long():
    b = AttentionBuffer(window=2, interval_s=0.0, reset_s=0.01, ttl_s=600)
    b.push("s", frame()); b.push("s", frame()); b.set_result("s", {"집중": 1.0})
    time.sleep(0.02)
    b.push("s", None)
    assert b.last("s") is None and b.push("s", frame())[0] is False  # 버퍼 비워짐


def test_sessions_isolated_and_sweep():
    b = AttentionBuffer(window=2, interval_s=0.0, reset_s=3.0, ttl_s=0.0)
    b.push("a", frame()); b.push("b", frame())
    assert b.push("a", frame())[0] is True and b.push("b", frame())[0] is True
    time.sleep(0.01)
    assert b.sweep() == 2
