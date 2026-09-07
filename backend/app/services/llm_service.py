"""LLM 리포트 (guideline/05 2절). 수치 리포트 + 질문·답변 텍스트 → 총평·질문별 피드백·강점·개선점.

- OpenAI Chat Completions, JSON Schema 로 출력 형식 강제.
- STT 가 끝난 뒤 실행 (stt_service.run 마지막에 호출). STT 가 없으면 수치만으로 생성.
- status.llm: pending → running → done | failed | skipped(키 없음).
"""

import json
import logging
import uuid
from typing import Any, Protocol

from openai import AsyncOpenAI

from app.config import get_settings
from app.db import SessionLocal
from app.models import Report, Session
from app.services.stt_service import set_status

log = logging.getLogger("selffit.llm")

LLM_SCHEMA = {
    "name": "interview_feedback",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string", "description": "전체 면접 총평 3~4문장, 존댓말"},
            "per_question": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"question_index": {"type": "integer"}, "feedback": {"type": "string"}},
                    "required": ["question_index", "feedback"],
                },
            },
            "strengths": {"type": "array", "items": {"type": "string"}},
            "improvements": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "per_question", "strengths", "improvements"],
    },
}

SYSTEM_PROMPT = """당신은 취업 준비생의 모의 면접을 코칭하는 면접관입니다.
아래 데이터로 피드백을 작성하세요. 데이터에 없는 사실을 지어내지 마세요.

데이터 설명:
- 행동 지표는 AI 가 웹캠으로 측정한 값입니다. gaze_hold_rate(시선 유지율), stable_emotion_rate(안정 표정 비율), attention_rate(집중 유지율)은 0~1 비율, 이벤트는 실시간으로 감지된 순간입니다.
- transcript 는 음성 인식으로 얻은 답변 텍스트입니다. 비어 있으면 답변 내용은 평가하지 말고 행동만 평가하세요. 인식 오류가 섞여 있을 수 있으니 단어 하나에 집착하지 마세요.
- 이벤트 종류: gaze_off(시선 이탈), face_lost(얼굴 사라짐), emotion_negative(긴장 표정), emotion_surprised(당황 표정), attention_low(집중 저하).
- overview.frames_analyzed 가 0 이거나 face_found_rate 가 0.3 미만이면 행동 지표는 측정되지 않았거나 신뢰할 수 없는 것입니다. 그 경우 비율을 인용하지 말고 "웹캠 분석 데이터가 부족했다"고만 언급하세요.

작성 원칙:
- 존댓말, 구체적으로, 격려와 개선점을 균형 있게. 수치를 언급할 때는 퍼센트로.
- per_question 은 질문마다 하나씩, 답변 내용(있으면)과 그 질문 구간의 행동을 함께 봅니다. 2~3문장.
- strengths, improvements 는 각각 2~4개, 한 문장씩. improvements 는 실천 가능한 행동으로."""


class LlmClient(Protocol):
    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    @property
    def model_name(self) -> str: ...


class OpenAiLlmClient:
    def __init__(self, api_key: str, model: str, timeout_s: float):
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=1)
        self._model = model

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "면접 데이터(JSON):\n" + json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_schema", "json_schema": LLM_SCHEMA},
        )
        content = r.choices[0].message.content or "{}"
        return json.loads(content)


def build_client() -> LlmClient | None:
    s = get_settings()
    if not s.openai_api_key:
        return None
    return OpenAiLlmClient(s.openai_api_key, s.openai_model, s.openai_timeout_s)


def build_payload(summary: dict, questions: list[tuple[int, str]]) -> dict[str, Any]:
    """LLM 에 줄 입력. 토큰을 아끼기 위해 필요한 것만 추린다."""
    transcript = {t["question_index"]: t for t in summary.get("transcript", [])}
    return {
        "duration_sec": round(summary.get("duration_ms", 0) / 1000),
        "overview": summary.get("overview", {}),
        "emotion_distribution": summary.get("emotion_distribution", {}),
        "attention_distribution": summary.get("attention_distribution", {}),
        "questions": [
            {
                "question_index": idx,
                "question": text,
                "answer_text": transcript.get(idx, {}).get("text", ""),
                "speech_sec": round(transcript.get(idx, {}).get("speech_ms", 0) / 1000),
                "metrics": next((q for q in summary.get("per_question", []) if q["order_index"] == idx), {}),
            }
            for idx, text in questions
        ],
        "events": [
            {"sec": round(e["ts_ms"] / 1000), "question_index": e["question_index"], "type": e["type"]}
            for e in summary.get("timeline", [])
        ][:60],
    }


async def run(session_id: uuid.UUID, client: LlmClient | None = None) -> None:
    client = client or build_client()
    if client is None:
        set_status(session_id, "llm", "skipped")
        return
    set_status(session_id, "llm", "running")
    try:
        with SessionLocal() as db:
            rep = db.get(Report, session_id)
            s = db.get(Session, session_id)
            if rep is None or s is None:
                set_status(session_id, "llm", "skipped")
                return
            payload = build_payload(rep.summary, [(q.order_index, q.question.text) for q in s.questions])
        out = await client.generate(payload)
        llm = {
            "summary": out.get("summary", ""),
            "per_question": out.get("per_question", []),
            "strengths": out.get("strengths", []),
            "improvements": out.get("improvements", []),
            "model": client.model_name,
        }
        set_status(session_id, "llm", "done", llm=llm)
        log.info("llm done %s (%s)", session_id, client.model_name)
    except Exception:
        log.exception("llm failed %s", session_id)
        set_status(session_id, "llm", "failed")
