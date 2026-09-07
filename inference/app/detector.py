"""MediaPipe 얼굴 검출. 가장 큰 얼굴 1개를 정사각형 박스로 돌려준다."""

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


class FaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        # model_selection=0: 2m 이내 근거리용 (웹캠 면접 환경)
        self._det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_confidence)

    def detect(self, bgr: np.ndarray) -> Face | None:
        h, w = bgr.shape[:2]
        rgb = bgr[:, :, ::-1]
        res = self._det.process(np.ascontiguousarray(rgb))
        if not res.detections:
            return None
        best, best_area = None, 0
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            bx, by, bw, bh = bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h
            if bw * bh > best_area:
                best_area, best = bw * bh, (bx, by, bw, bh, float(d.score[0]))
        bx, by, bw, bh, score = best
        # 정사각형으로 보정 (긴 변 기준, 중심 유지)
        side = max(bw, bh)
        cx, cy = bx + bw / 2, by + bh / 2
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        return Face(x=x0, y=y0, w=int(round(side)), h=int(round(side)), score=score)

    def close(self) -> None:
        self._det.close()
