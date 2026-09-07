"""모델 3개 래퍼. 원본 검증 스크립트의 로드·후처리 방식을 그대로 옮겼다."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from app.emotion_contract import load_emotion_contract
from app.models.emotionnet import EmotionNet
from app.models.former import GenerateModel
from app.models.l2cs_net import L2CS

log = logging.getLogger("inference.models")

# 감정 클래스는 납품 meta.json 이 정한다 (app/emotion_contract.py).
# 집중 5클래스. 인덱스 순서는 AI-Hub 표기 순 가정 (03-for-ai.md 3절)
ATTENTION_LABELS = ["집중", "졸림", "집중결핍", "집중하락", "태만"]


class _AnyObject:
    """체크포인트에 함께 pickle 된 학습용 보조 객체(RecorderMeter 등)를 받아 버리기 위한 대체 클래스."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


def _stub_training_modules() -> None:
    """Former-DFER 체크포인트는 학습 스크립트의 runner_helper.RecorderMeter 를 포함한다.
    가중치만 필요하므로 해당 모듈을 더미로 등록해 unpickle 이 통과되게 한다."""
    import sys
    import types

    if "runner_helper" not in sys.modules:
        m = types.ModuleType("runner_helper")
        for name in ("RecorderMeter", "AverageMeter", "ProgressMeter"):
            setattr(m, name, type(name, (_AnyObject,), {}))
        sys.modules["runner_helper"] = m


def _load(path: Path):
    _stub_training_modules()
    return torch.load(path, map_location="cpu", weights_only=False)


class GazeModel:
    """L2CS-Net ResNet50, 90-bin. run_valid_l2cs.py 후처리: softmax 기대값 ×4 − 180."""

    name = "l2cs_v1"

    def __init__(self, weights: Path, device: str):
        self.device = device
        self.model = L2CS(Bottleneck, [3, 4, 6, 3], 90)
        self.model.load_state_dict(_load(weights))
        self.model.eval().to(device)
        self.idx = torch.arange(90, dtype=torch.float32, device=device)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> tuple[float, float, float]:
        yaw_logits, pitch_logits = self.model(x.to(self.device))
        yaw_p, pitch_p = F.softmax(yaw_logits, dim=1), F.softmax(pitch_logits, dim=1)
        yaw = (torch.sum(yaw_p * self.idx, 1) * 4 - 180).item()
        pitch = (torch.sum(pitch_p * self.idx, 1) * 4 - 180).item()
        # 확신도: 90-bin(4°) 분포에서 상위 3개 bin(±4°) 에 실린 확률 질량의 평균.
        # 단일 bin 최댓값은 정상 입력에서도 0.2~0.3 이라 임계값으로 쓰기 어렵다.
        conf = float((yaw_p.topk(3).values.sum() + pitch_p.topk(3).values.sum()) / 2)
        return yaw, pitch, conf


class EmotionModel:
    """EmotionNet 파인튜닝본. 판정 계약은 ai/docs/06-backend-handoff.md 4절:

        prob = softmax(model(x) + bias);  top = argmax(prob);  accepted = prob.max() >= tau

    bias 는 argmax 이전, tau 비교 이전에 더한다 — softmax 이후에 더하면 틀린다. 학습이 클래스당
    6,800개로 균형을 맞춘 탓에 모델은 균등 prior 를 학습했고, 중립이 지배적인 면접 스트림에서는
    bias 없이 중립 프레임의 16.8%를 불안으로 부른다(bias 적용 시 7.65%).
    """

    def __init__(self, weights: Path, meta: Path, device: str):
        self.name = weights.stem
        self.device = device
        contract = load_emotion_contract(meta)
        self.labels, self.tau = contract.labels, contract.tau
        self.model = EmotionNet(num_classes=len(self.labels))
        self.model.load_state_dict(_load(weights)["model"])
        self.model.eval().to(device)
        self.bias = torch.tensor(contract.bias, dtype=torch.float32, device=device)
        log.info("emotion contract: %s bias=%s tau=%s", self.labels, contract.bias, self.tau)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> tuple[dict[str, float], bool]:
        """(클래스별 확률, tau 통과 여부). 모델 출력은 LogSoftmax 라 bias 를 그대로 더할 수 있다."""
        logprob = self.model(x.to(self.device))
        prob = F.softmax(logprob + self.bias, dim=1)[0].cpu().numpy()
        accepted = bool(prob.max() >= self.tau)
        return {lab: round(float(p), 4) for lab, p in zip(self.labels, prob, strict=True)}, accepted


class AttentionModel:
    """Former-DFER 5클래스. DataParallel 로 저장돼 'module.' 접두사 제거. logit → softmax."""

    name = "former_dfer_v1"

    def __init__(self, weights: Path, device: str):
        self.device = device
        self.model = GenerateModel(num_classes=5)
        sd = _load(weights)["state_dict"]
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.eval().to(device)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> dict[str, float]:
        out = F.softmax(self.model(x.to(self.device)), dim=1)[0].cpu().numpy()
        return {lab: round(float(p), 4) for lab, p in zip(ATTENTION_LABELS, out, strict=True)}


def top_of(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def warmup(gaze: GazeModel, emotion: EmotionModel, attention: AttentionModel, gaze_size: int) -> None:
    """첫 요청 지연을 없애기 위해 더미 입력으로 1회 실행."""
    gaze.predict(torch.zeros(1, 3, gaze_size, gaze_size))
    emotion.predict(torch.zeros(1, 1, 48, 48))
    attention.predict(torch.zeros(1, 16, 3, 112, 112))
    log.info("warmup done")


def cuda_available() -> bool:
    return torch.cuda.is_available()
