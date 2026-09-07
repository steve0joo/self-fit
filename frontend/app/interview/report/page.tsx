'use client';

import React, { Suspense, useEffect, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { apiFetch } from '@/lib/api';

const DEMO_OVERVIEW = {
  gaze_hold_rate: 0.82,
  stable_emotion_rate: 0.74,
  attention_rate: 0.81,
  event_count: 6,
};

const QUESTION_SCORES = [
  { label: '질문 1', value: 78 },
  { label: '질문 2', value: 84 },
  { label: '질문 3', value: 71 },
  { label: '질문 4', value: 88 },
  { label: '질문 5', value: 90 },
];

const FEEDBACK = [
  { q: '질문 1', note: '초반 시선 이탈이 2회 감지됐어요. 답변 시작 전 카메라를 먼저 응시해보세요.' },
  { q: '질문 3', note: '답변 중 긴장 표정이 가장 오래 지속됐어요. 호흡을 고르고 천천히 답해보세요.' },
  { q: '질문 5', note: '시선 유지율과 표정 안정도 모두 가장 높았어요. 이 리듬을 기억해두세요.' },
];

type Overview = {
  gaze_hold_rate: number;
  stable_emotion_rate: number;
  attention_rate: number;
  event_count: number;
};

type PerQuestion = {
  order_index: number;
  text: string;
  gaze_hold_rate: number;
  dominant_emotion: string;
  attention_rate: number;
  event_count: number;
};

type Report = {
  overview: Overview;
  per_question: PerQuestion[];
  feedback: { question_index: number; note: string }[];
  emotion_distribution?: Record<string, number>;
  timeline?: { ts_ms: number; question_index: number; type: string; message: string }[];
};

type Status = 'demo' | 'loading' | 'ready' | 'unfinished';

const pct = (v: number) => `${Math.round((v ?? 0) * 100)}%`;

function ReportContent() {
  const sessionId = useSearchParams().get('session');
  const [status, setStatus] = useState<Status>(sessionId ? 'loading' : 'demo');
  const [report, setReport] = useState<Report | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    setStatus('loading');

    (async () => {
      try {
        const res = await apiFetch(`/api/sessions/${sessionId}/report`);
        if (res.status === 409) {
          if (!cancelled) setStatus('unfinished');
          return;
        }
        if (!res.ok) throw new Error(`GET /api/sessions/${sessionId}/report → ${res.status}`);
        const data: Report = await res.json();
        if (cancelled) return;
        setReport(data);
        setStatus('ready');
      } catch (e) {
        console.warn('[report] 리포트 조회 실패 — 데모 데이터로 표시합니다.', e);
        if (!cancelled) setStatus('demo');
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const overview = report?.overview ?? DEMO_OVERVIEW;
  const bars = report
    ? (report.per_question ?? []).map((q) => ({
        label: `질문 ${q.order_index + 1}`,
        value: Math.round((q.gaze_hold_rate ?? 0) * 100),
      }))
    : QUESTION_SCORES;
  const feedback = report
    ? (report.feedback ?? []).map((f) => ({ q: `질문 ${f.question_index + 1}`, note: f.note }))
    : FEEDBACK;

  if (status === 'loading') {
    return (
      <div className="question-card">
        <div className="question-index">행동 리포트</div>
        <p className="question-text">리포트를 불러오는 중...</p>
      </div>
    );
  }

  if (status === 'unfinished') {
    return (
      <div className="question-card">
        <div className="question-index">행동 리포트</div>
        <p className="question-text">세션이 아직 끝나지 않았습니다.</p>
        <div className="action-row">
          <Link className="btn-solid" href="/interview">면접으로 돌아가기 →</Link>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="report-stat-grid">
        <div className="report-stat"><div className="k">시선 유지율</div><div className="v num">{pct(overview.gaze_hold_rate)}</div></div>
        <div className="report-stat"><div className="k">안정 표정 비율</div><div className="v num">{pct(overview.stable_emotion_rate)}</div></div>
        <div className="report-stat"><div className="k">집중 유지율</div><div className="v num">{pct(overview.attention_rate)}</div></div>
        <div className="report-stat"><div className="k">알림 횟수</div><div className="v num">{overview.event_count ?? 0}회</div></div>
      </div>

      <div className="question-card">
        <div className="question-index">질문별 시선 유지율</div>
        {bars.map((s, i) => (
          <div className="bar-row" key={`${s.label}-${i}`}>
            <span className="bar-label">{s.label}</span>
            <div className="bar-track"><div className="bar-fill" style={{ width: `${s.value}%` }} /></div>
            <span className="bar-value num">{s.value}%</span>
          </div>
        ))}
      </div>

      <div className="question-card">
        <div className="question-index">피드백</div>
        <ul className="feedback-list">
          {feedback.map((f, i) => (
            <li key={`${f.q}-${i}`}><b>{f.q}</b>{f.note}</li>
          ))}
        </ul>
      </div>

      <div className="action-row">
        <Link className="btn-ghost" href="/upload">영상으로 다시 분석</Link>
        <Link className="btn-solid" href="/interview">다시 도전하기 →</Link>
      </div>
    </>
  );
}

export default function ReportPage() {
  return (
    <div className="page-shell">
      <Link className="page-back" href="/">← 홈으로</Link>
      <h1 className="page-title">행동 리포트</h1>
      <p className="page-sub" style={{ color: 'var(--ink)' }}>방금 진행한 모의면접의 시선, 표정 분석 결과입니다.</p>

      <Suspense
        fallback={
          <div className="question-card">
            <div className="question-index">행동 리포트</div>
            <p className="question-text">리포트를 불러오는 중...</p>
          </div>
        }
      >
        <ReportContent />
      </Suspense>
    </div>
  );
}