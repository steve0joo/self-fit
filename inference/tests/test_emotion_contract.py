"""감정 모델 배포 계약 파싱 (ai/docs/06-backend-handoff.md 4절)."""

import json
from pathlib import Path

import pytest

from app.emotion_contract import load_emotion_contract

# 납품 원본. 저장소에 있으면 실제 값으로 계약을 검증한다.
DELIVERABLE_META = Path(__file__).resolve().parents[2] / "ai" / "models" / "deliverable" / "meta.json"

VALID = {
    "class_order": ["happy", "embarrassed", "anxious", "neutral"],
    "bias": {"value": [0.0, 1.4, 0.0, 1.9], "class_order": ["happy", "embarrassed", "anxious", "neutral"]},
    "tau_biased": {"value": 0.98},
    "tau": {"value": 0.94},  # bias 없는 척도의 값 — 이걸 읽으면 안 된다
}


def write(tmp_path: Path, meta: dict) -> Path:
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return p


def test_reads_labels_bias_and_tau(tmp_path):
    c = load_emotion_contract(write(tmp_path, VALID))
    assert c.labels == ["기쁨", "당황", "불안", "중립"]
    assert c.bias == [0.0, 1.4, 0.0, 1.9]
    assert c.tau == 0.98  # tau 가 아니라 tau_biased


def test_bias_order_mismatch_is_rejected(tmp_path):
    meta = json.loads(json.dumps(VALID))
    meta["bias"]["class_order"] = ["neutral", "anxious", "embarrassed", "happy"]
    with pytest.raises(RuntimeError, match="class_order"):
        load_emotion_contract(write(tmp_path, meta))


def test_bias_length_mismatch_is_rejected(tmp_path):
    meta = json.loads(json.dumps(VALID))
    meta["bias"]["value"] = [0.0, 1.4, 0.0]
    meta["bias"].pop("class_order")
    with pytest.raises(RuntimeError, match="bias 길이"):
        load_emotion_contract(write(tmp_path, meta))


def test_unknown_class_is_rejected(tmp_path):
    meta = json.loads(json.dumps(VALID))
    meta["class_order"] = ["happy", "embarrassed", "anxious", "sad"]
    meta["bias"]["class_order"] = meta["class_order"]
    with pytest.raises(RuntimeError, match="sad"):
        load_emotion_contract(write(tmp_path, meta))


def test_missing_tau_biased_is_rejected(tmp_path):
    meta = json.loads(json.dumps(VALID))
    meta.pop("tau_biased")
    with pytest.raises(RuntimeError, match="계약을 읽을 수 없습니다"):
        load_emotion_contract(write(tmp_path, meta))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="계약을 읽을 수 없습니다"):
        load_emotion_contract(tmp_path / "없는파일.json")


@pytest.mark.skipif(not DELIVERABLE_META.exists(), reason="납품 meta.json 이 없는 체크아웃")
def test_delivered_contract_matches_handoff_doc():
    """납품물이 문서(06-backend-handoff.md 4절)와 어긋나면 여기서 잡는다."""
    c = load_emotion_contract(DELIVERABLE_META)
    assert c.labels == ["기쁨", "당황", "불안", "중립"]
    assert c.bias == [0.0, 1.4, 0.0, 1.9]
    assert c.tau == 0.98
