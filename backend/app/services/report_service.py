"""세션 종료 시 analysis_logs + events → reports.summary (03-phase1-design.md 5.6절, 9절)."""

import uuid
from collections import Counter
from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.analysis.types import ATTENTION_FOCUSED, ATTENTION_LABELS, EMOTION_OTHER, EMOTION_USED
from app.models import AnalysisLog, Event, Report, Session

STABLE_EMOTIONS = {"중립", "기쁨"}


def _rate(n: int, d: int) -> float:
    return round(n / d, 3) if d else 0.0


def _ms(a, b) -> int:
    if a is None or b is None:
        return 0
    a = a if a.tzinfo else a.replace(tzinfo=UTC)
    b = b if b.tzinfo else b.replace(tzinfo=UTC)
    return int((b - a).total_seconds() * 1000)


def _metrics(logs: list[AnalysisLog], events: list[Event]) -> dict:
    valid = [l for l in logs if l.face_found]
    with_att = [l for l in valid if l.attention_top]
    emo = Counter(l.emotion_top or EMOTION_OTHER for l in valid)
    att = Counter(l.attention_top for l in with_att)
    return {
        "gaze_hold_rate": _rate(sum(1 for l in valid if l.gaze_state == "center"), len(valid)),
        "stable_emotion_rate": _rate(sum(1 for l in valid if l.emotion_top in STABLE_EMOTIONS), len(valid)),
        "attention_rate": _rate(att.get(ATTENTION_FOCUSED, 0), len(with_att)),
        "face_found_rate": _rate(len(valid), len([l for l in logs if l.face_found is not None])),
        "event_count": len(events),
        "_emo": emo,
        "_att": att,
        "_valid": len(valid),
    }


def build_report(db: DbSession, s: Session) -> dict:
    logs = list(
        db.scalars(select(AnalysisLog).where(AnalysisLog.session_id == s.id).order_by(AnalysisLog.ts_ms))
    )
    events = list(db.scalars(select(Event).where(Event.session_id == s.id).order_by(Event.ts_ms)))
    m = _metrics(logs, events)

    emo_keys = sorted(EMOTION_USED) + [EMOTION_OTHER]
    emotion_distribution = {k: _rate(m["_emo"].get(k, 0), m["_valid"]) for k in emo_keys}
    att_total = sum(m["_att"].values())
    attention_distribution = {k: _rate(m["_att"].get(k, 0), att_total) for k in ATTENTION_LABELS}

    per_question = []
    for q in s.questions:
        q_logs = [l for l in logs if l.question_index == q.order_index]
        q_events = [e for e in events if e.question_index == q.order_index]
        qm = _metrics(q_logs, q_events)
        per_question.append(
            {
                "order_index": q.order_index,
                "question_id": q.question_id,
                "text": q.question.text,
                "duration_ms": _ms(q.started_at, q.ended_at),
                "gaze_hold_rate": qm["gaze_hold_rate"],
                "dominant_emotion": (qm["_emo"].most_common(1)[0][0] if qm["_emo"] else None),
                "attention_rate": qm["attention_rate"],
                "event_count": qm["event_count"],
                "_neg_events": sum(1 for e in q_events if e.type == "emotion_negative"),
                "_gaze_off": sum(1 for e in q_events if e.type == "gaze_off"),
            }
        )

    summary = {
        "session_id": str(s.id),
        "duration_ms": _ms(s.started_at, s.finished_at),
        "overview": {
            k: m[k]
            for k in (
                "gaze_hold_rate",
                "stable_emotion_rate",
                "attention_rate",
                "face_found_rate",
                "event_count",
            )
        },
        "emotion_distribution": emotion_distribution,
        "attention_distribution": attention_distribution,
        "per_question": [{k: v for k, v in pq.items() if not k.startswith("_")} for pq in per_question],
        "timeline": [
            {"ts_ms": e.ts_ms, "question_index": e.question_index, "type": e.type, "message": e.message}
            for e in events
        ],
        "feedback": _feedback(per_question, m),
    }
    return summary


def _feedback(pq: list[dict], m: dict) -> list[dict]:
    fb: list[dict] = []
    if m["face_found_rate"] < 0.3:
        fb.append(
            {
                "question_index": None,
                "note": "얼굴이 잘 보이지 않아 분석 신뢰도가 낮아요. 조명과 카메라 위치를 확인해보세요.",
            }
        )
    if not pq:
        return fb
    for q in pq:
        if q["_gaze_off"] >= 2:
            fb.append(
                {
                    "question_index": q["order_index"],
                    "note": f"시선 이탈이 {q['_gaze_off']}회 감지됐어요. 답변 시작 전 카메라를 먼저 응시해보세요.",
                }
            )
    most_neg = max(pq, key=lambda q: q["_neg_events"])
    if most_neg["_neg_events"] > 0:
        fb.append(
            {
                "question_index": most_neg["order_index"],
                "note": "답변 중 긴장 표정이 가장 오래 지속됐어요. 호흡을 고르고 천천히 답해보세요.",
            }
        )
    best = max(pq, key=lambda q: (q["gaze_hold_rate"], q["attention_rate"]))
    if best["gaze_hold_rate"] > 0:
        fb.append(
            {
                "question_index": best["order_index"],
                "note": "시선 유지율과 집중도가 가장 높았어요. 이 리듬을 기억해두세요.",
            }
        )
    for q in pq:
        if 0 < q["attention_rate"] < 0.6:
            fb.append(
                {
                    "question_index": q["order_index"],
                    "note": "이 질문에서 집중이 흐트러진 시간이 길었어요. 질문을 끝까지 듣고 잠시 생각한 뒤 답해보세요.",
                }
            )
    return fb


def save_report(db: DbSession, s: Session) -> dict:
    summary = build_report(db, s)
    existing = db.get(Report, s.id)
    if existing:
        existing.summary = summary
    else:
        db.add(Report(session_id=s.id, summary=summary))
    db.commit()
    return summary


def get_report(db: DbSession, session_id: uuid.UUID) -> dict | None:
    r = db.get(Report, session_id)
    return r.summary if r else None
