import cv2
import numpy as np

from app import preprocess as pp
from app.detector import Face


def img(h=100, w=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_crop_no_margin_exact():
    roi = pp.crop(img(), Face(10, 20, 30, 30, 1.0), margin=0.0)
    assert roi.shape == (30, 30, 3)


def test_crop_margin_and_padding_at_border():
    roi = pp.crop(img(), Face(0, 0, 40, 40, 1.0), margin=0.25)  # 여백 10px, 좌상단은 이미지 밖 → 패딩
    assert roi.shape == (60, 60, 3)
    assert roi[0, 0].tolist() == [0, 0, 0] and roi[30, 30].tolist() == [128, 128, 128]


def test_crop_without_padding_clips_to_frame():
    """감정 경로(pad=False)는 학습과 같이 클리핑만 한다 — 검은 띠가 붙으면 학습 분포 밖이다."""
    roi = pp.crop(img(h=100, w=200), Face(-10, -10, 40, 40, 1.0), margin=0.0, pad=False)
    assert roi.shape == (30, 30, 3)
    assert (roi == 128).all()


def test_squared_keeps_centre_and_takes_long_side():
    """MediaPipe 박스는 세로로 긴 편(종횡비 중앙값 0.73). 시선·집중 경로만 이 정사각 박스를 쓴다."""
    sq = Face(10, 20, 30, 50, 0.9).squared()
    assert (sq.w, sq.h) == (50, 50)
    assert (sq.x + sq.w / 2, sq.y + sq.h / 2) == (25.0, 45.0)  # 중심 유지
    assert sq.score == 0.9


def test_emotion_resize_is_bilinear_stretch():
    """학습의 Resize((48,48)) 이 bilinear 이라 서빙도 같아야 한다 (핸드오프 3.3절).
    INTER_AREA 로 돌아가면 이 테스트가 깨진다."""
    grad = np.tile(np.arange(84, dtype=np.uint8)[None, :], (84, 1))  # 가로 그라데이션
    face = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
    want = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_LINEAR)
    got = (pp.emotion_tensor(face)[0, 0].numpy() * 255.0).round().astype(np.uint8)
    assert (got == want).all()
    assert not (got == cv2.resize(grad, (48, 48), interpolation=cv2.INTER_AREA)).all()


def test_tensor_shapes_and_ranges():
    face = img(80, 80)
    g = pp.gaze_tensor(face, 448)
    assert g.shape == (1, 3, 448, 448)
    e = pp.emotion_tensor(face)
    assert e.shape == (1, 1, 48, 48) and 0.0 <= float(e.min()) and float(e.max()) <= 1.0
    a = pp.attention_tensor([pp.attention_frame(face)] * 16)
    assert a.shape == (1, 16, 3, 112, 112) and float(a.max()) <= 1.0
