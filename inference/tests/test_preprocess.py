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


def test_tensor_shapes_and_ranges():
    face = img(80, 80)
    g = pp.gaze_tensor(face, 448)
    assert g.shape == (1, 3, 448, 448)
    e = pp.emotion_tensor(face)
    assert e.shape == (1, 1, 48, 48) and 0.0 <= float(e.min()) and float(e.max()) <= 1.0
    a = pp.attention_tensor([pp.attention_frame(face)] * 16)
    assert a.shape == (1, 16, 3, 112, 112) and float(a.max()) <= 1.0
