"""모델별 전처리 (03-phase1-design.md 7.1절). 학습 조건과 같아야 하므로 수치는 여기서만 관리."""

import cv2
import numpy as np
import torch

from app.detector import Face

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop(bgr: np.ndarray, face: Face, margin: float) -> np.ndarray:
    """얼굴 박스에 margin 비율만큼 여백을 주고 이미지 경계 안에서 자른다. 부족한 부분은 검정 패딩."""
    h, w = bgr.shape[:2]
    m = int(round(face.w * margin))
    x0, y0, x1, y1 = face.x - m, face.y - m, face.x + face.w + m, face.y + face.h + m
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    roi = bgr[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    if pad_l or pad_t or pad_r or pad_b:
        roi = cv2.copyMakeBorder(roi, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return roi


def gaze_tensor(bgr_face: np.ndarray, size: int) -> torch.Tensor:
    """L2CS: RGB, Resize(size), ImageNet 정규화 → (1,3,size,size)"""
    rgb = cv2.cvtColor(cv2.resize(bgr_face, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def emotion_tensor(bgr_face: np.ndarray) -> torch.Tensor:
    """EmotionNet: 흑백, 48×48 INTER_AREA, /255 → (1,1,48,48). 원본 video.py 와 동일."""
    gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(roi.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def attention_frame(bgr_face: np.ndarray) -> np.ndarray:
    """Former-DFER 버퍼용 프레임: RGB 112×112 uint8. 텐서 변환은 16장 모였을 때."""
    return cv2.cvtColor(cv2.resize(bgr_face, (112, 112), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def attention_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """(16,112,112,3) uint8 → /255 → (1,16,3,112,112). 원본 ToTorchFormatTensor(div=True) 와 동일."""
    x = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0)
