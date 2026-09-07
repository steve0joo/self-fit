"""MediaPipe 얼굴 검출. 검출 점수가 가장 높은 얼굴 1개를 원본 박스 그대로 돌려준다."""

from dataclasses import dataclass

import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float

    def squared(self) -> "Face":
        """긴 변 기준 정사각형, 중심 유지.

        정사각 크롭을 전제로 검증된 시선(L2CS)·집중(Former-DFER) 경로 전용이다.
        감정 경로는 학습과 같은 원본 박스를 써야 하므로 이걸 거치지 않는다 —
        ai/docs/06-backend-handoff.md 3.2절.
        """
        side = max(self.w, self.h)
        cx, cy = self.x + self.w / 2, self.y + self.h / 2
        return Face(x=int(round(cx - side / 2)), y=int(round(cy - side / 2)), w=side, h=side, score=self.score)


class FaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        # model_selection=0 (2m 이내 근거리용) + min_detection_confidence=0.5 는 학습 전처리와 같은 값이다
        # (ai/scripts/preprocess.py 의 MP_MODEL_SELECTION / MP_MIN_CONFIDENCE). 바꾸면 크롭이 학습과 갈라진다.
        self._det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_confidence)

    def detect(self, bgr: np.ndarray) -> Face | None:
        h, w = bgr.shape[:2]
        rgb = bgr[:, :, ::-1]
        res = self._det.process(np.ascontiguousarray(rgb))
        if not res.detections:
            return None
        # 다중 얼굴 규칙은 면적이 아니라 검출 점수 최고 — 학습 전처리와 동일(스펙 2.3절, 핸드오프 3.1절).
        # 면접자가 화면에서 가장 크다는 보장이 없다(뒤에 선 사람이 카메라에 더 가까울 수 있다).
        best = max(res.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box
        x0, y0 = int(round(bb.xmin * w)), int(round(bb.ymin * h))
        x1, y1 = int(round((bb.xmin + bb.width) * w)), int(round((bb.ymin + bb.height) * h))
        # 화면 안에 남는 부분이 2px 미만이면 쓸 수 없는 검출이다 (학습의 degenerate_box 규칙과 같음).
        if min(x1, w) - max(x0, 0) < 2 or min(y1, h) - max(y0, 0) < 2:
            return None
        return Face(x=x0, y=y0, w=x1 - x0, h=y1 - y0, score=float(best.score[0]))

    def close(self) -> None:
        self._det.close()
