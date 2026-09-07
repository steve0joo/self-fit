"""모델별 전처리 (03-phase1-design.md 7.1절). 학습 조건과 같아야 하므로 수치는 여기서만 관리."""

import cv2
import numpy as np
import torch

from app.detector import Face

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop(bgr: np.ndarray, face: Face, margin: float, pad: bool = True) -> np.ndarray:
    """얼굴 박스에 margin 비율만큼 여백을 주고 자른다.

    pad=True  이미지 밖으로 나간 만큼 검정 패딩을 채운다 (요청한 크기가 그대로 나온다).
    pad=False 이미지 경계로 클리핑만 한다 — 학습 전처리와 같은 동작. 감정 경로가 쓴다.
              학습 데이터에 검은 띠가 붙은 얼굴은 없다 (핸드오프 3.2절).
    """
    h, w = bgr.shape[:2]
    m = int(round(face.w * margin))
    x0, y0, x1, y1 = face.x - m, face.y - m, face.x + face.w + m, face.y + face.h + m
    roi = bgr[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    if not pad:
        return roi
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        roi = cv2.copyMakeBorder(roi, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return roi


def gaze_tensor(bgr_face: np.ndarray, size: int) -> torch.Tensor:
    """L2CS: RGB, Resize(size), ImageNet 정규화 → (1,3,size,size)"""
    rgb = cv2.cvtColor(cv2.resize(bgr_face, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def emotion_tensor(bgr_face: np.ndarray) -> torch.Tensor:
    """EmotionNet: 흑백, 48×48 bilinear 스트레치, /255 → (1,1,48,48).

    보간은 INTER_LINEAR 고정 — 학습이 torchvision Resize((48,48))(bilinear)로 맞춰져 있다
    (meta.json 의 input.resize, 핸드오프 3.3절). 160px 중간 단계를 끼우지 말 것: 그건 학습 캐시
    사정이지 계약이 아니고, 서빙 크롭은 이미 ~84px 라 리샘플만 하나 더 붙는다.

    BGR 배열에 COLOR_BGR2GRAY 는 학습의 RGB 배열 + COLOR_RGB2GRAY 와 같은 연산이다. 통일한다고
    한쪽만 바꾸면 그때 채널이 뒤집힌다 (스펙 2.4절).
    """
    gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(roi.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


def attention_frame(bgr_face: np.ndarray) -> np.ndarray:
    """Former-DFER 버퍼용 프레임: RGB 112×112 uint8. 텐서 변환은 16장 모였을 때."""
    return cv2.cvtColor(cv2.resize(bgr_face, (112, 112), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


def attention_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """(16,112,112,3) uint8 → /255 → (1,16,3,112,112). 원본 ToTorchFormatTensor(div=True) 와 동일."""
    x = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0)
