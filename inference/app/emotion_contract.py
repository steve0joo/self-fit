"""감정 모델의 배포 계약 (ai/docs/06-backend-handoff.md 4절).

bias·tau·클래스 순서는 체크포인트마다 다시 튜닝되는 값이라 가중치와 함께 배포된 meta.json 에서
읽는다. 코드에 박아 두면 가중치만 교체했을 때 잘못된 조합이 조용히 돈다.

torch 를 쓰지 않는다 — 계약 파싱은 모델 로딩과 분리해 두어야 검증이 싸다.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# 납품 meta.json 은 영문 클래스명을 쓴다. 서비스 표기는 한글.
EMOTION_KOREAN = {"happy": "기쁨", "embarrassed": "당황", "anxious": "불안", "neutral": "중립"}


@dataclass(frozen=True)
class EmotionContract:
    labels: list[str]  # 한글 표기, 모델 출력 순서
    bias: list[float]  # 로그확률에 가산
    tau: float  # bias 적용 후 확률 기준


def load_emotion_contract(path: Path) -> EmotionContract:
    """납품 meta.json 에서 배포 계약 필드만 읽는다. 어긋나면 기동 시점에 실패시킨다."""
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
        order = [str(c) for c in m["class_order"]]
        bias_field = m["bias"]
        bias = [float(v) for v in bias_field["value"]]
        # tau 가 아니라 tau_biased — bias 없는 확률 척도에서 고른 tau 와는 다른 값이다 (4.2절)
        tau = float(m["tau_biased"]["value"])
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"감정 모델 계약을 읽을 수 없습니다 ({path}): {e!r}") from e

    if bias_field.get("class_order", order) != order:
        raise RuntimeError(f"meta.json 의 bias.class_order 가 class_order 와 다릅니다 ({path}).")
    if len(bias) != len(order):
        raise RuntimeError(f"bias 길이({len(bias)})가 클래스 수({len(order)})와 다릅니다 ({path}).")
    unknown = [c for c in order if c not in EMOTION_KOREAN]
    if unknown:
        raise RuntimeError(f"표기를 모르는 감정 클래스 {unknown} ({path}).")
    return EmotionContract(labels=[EMOTION_KOREAN[c] for c in order], bias=bias, tau=tau)
