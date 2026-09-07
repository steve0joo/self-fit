'use client';

import React, { Suspense, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { apiFetch, apiFetchRaw } from '@/lib/api';

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
  frames_analyzed?: number;
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
  status?: { metrics: string; stt: string; llm: string; recording: string };
  recording?: { url: string; duration_ms: number } | null;
  transcript?: { question_index: number; text: string; words: number; speech_ms: number }[];
  llm?: {
    summary: string;
    per_question: { question_index: number; feedback: string }[];
    strengths: string[];
    improvements: string[];
    model: string;
  } | null;
};

type Status = 'demo' | 'loading' | 'ready' | 'unfinished';

const POLL_MS = 3000;

const pct = (v: number) => `${Math.round((v ?? 0) * 100)}%`;

const isRunning = (s?: string) => s === 'pending' || s === 'running';

const needsPoll = (s?: Report['status']) =>
  !!s && [s.metrics, s.stt, s.llm, s.recording].some(isRunning);

function formatTime(ms: number): string {
  const total = Math.max(0, Math.floor((ms ?? 0) / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function ReportContent() {
  const sessionId = useSearchParams().get('session');
  const [status, setStatus] = useState<Status>(sessionId ? 'loading' : 'demo');
  const [report, setReport] = useState<Report | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoFailed, setVideoFailed] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    setStatus('loading');

    const load = async () => {
      try {
        const res = await apiFetch(`/api/sessions/${sessionId}/report`);
        if (cancelled) return;
        if (res.status === 409) {
          setStatus('unfinished');
          return;
        }
        if (!res.ok) throw new Error(`GET /api/sessions/${sessionId}/report → ${res.status}`);
        const data: Report = await res.json();
        if (cancelled) return;
        setReport(data);
        setStatus('ready');
        if (needsPoll(data.status)) timer = setTimeout(load, POLL_MS);
      } catch (e) {
        if (cancelled) return;
        console.warn('[report] 리포트 조회 실패', e);
        setStatus((prev) => (prev === 'ready' ? 'ready' : 'demo'));
      }
    };

    load();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
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
  const noMetrics = report?.overview?.frames_analyzed === 0;
  const statValue = (value: React.ReactNode) =>
    noMetrics ? (
      <div className="v" style={{ fontSize: 15 }}>데이터 없음</div>
    ) : (
      <div className="v num">{value}</div>
    );
  const jobs = report?.status;
  const transcript = report?.transcript ?? [];
  const llm = report?.llm ?? null;
  const recording = report?.recording ?? null;
  const timeline = report?.timeline ?? [];
  const canPlay = !!recording && jobs?.recording === 'done';
  const recordingUrl = canPlay ? recording.url : null;

  useEffect(() => {
    if (!recordingUrl) return;
    let cancelled = false;
    let objectUrl: string | null = null;
    setVideoFailed(false);

    (async () => {
      try {
        const res = await apiFetchRaw(recordingUrl);
        if (!res.ok) throw new Error(`GET ${recordingUrl} → ${res.status}`);
        const blob = await res.blob();
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setVideoSrc(objectUrl);
      } catch (e) {
        if (cancelled) return;
        console.warn('[report] 녹화 영상 로드 실패', e);
        setVideoFailed(true);
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setVideoSrc(null);
    };
  }, [recordingUrl]);

  const seekTo = (ms: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = ms / 1000;
    void v.play().catch(() => {});
  };

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
      {isRunning(jobs?.llm) && (
        <div className="question-card">
          <div className="question-index">AI 분석</div>
          <p className="question-text">답변 분석 중… 완료되면 총평이 자동으로 나타납니다.</p>
        </div>
      )}

      {jobs?.stt === 'failed' && (
        <div className="question-card">
          <div className="question-index">음성 분석</div>
          <p className="question-text">음성 분석에 실패했어요. 행동 지표만 표시합니다.</p>
        </div>
      )}

      {jobs?.llm === 'failed' && (
        <div className="question-card">
          <div className="question-index">AI 분석</div>
          <p className="question-text">AI 총평 생성에 실패했어요. 기본 피드백만 표시합니다.</p>
        </div>
      )}

      <div className="report-stat-grid">
        <div className="report-stat"><div className="k">시선 유지율</div>{statValue(pct(overview.gaze_hold_rate))}</div>
        <div className="report-stat"><div className="k">안정 표정 비율</div>{statValue(pct(overview.stable_emotion_rate))}</div>
        <div className="report-stat"><div className="k">집중 유지율</div>{statValue(pct(overview.attention_rate))}</div>
        <div className="report-stat"><div className="k">알림 횟수</div>{statValue(`${overview.event_count ?? 0}회`)}</div>
      </div>

      <div className="question-card">
        <div className="question-index">질문별 시선 유지율</div>
        {noMetrics ? (
          <p className="question-text">행동 지표가 측정되지 않았어요. 카메라에 얼굴이 잡히지 않았을 수 있어요.</p>
        ) : (
          bars.map((s, i) => (
            <div className="bar-row" key={`${s.label}-${i}`}>
              <span className="bar-label">{s.label}</span>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${s.value}%` }} /></div>
              <span className="bar-value num">{s.value}%</span>
            </div>
          ))
        )}
      </div>

      {(canPlay || isRunning(jobs?.recording)) && (
        <div className="question-card">
          <div className="question-index">
            면접 영상
            {canPlay && recording?.duration_ms ? ` · ${formatTime(recording.duration_ms)}` : ''}
          </div>
          {!canPlay && <p className="question-text">영상 처리 중…</p>}
          {canPlay && videoFailed && <p className="question-text">영상을 불러오지 못했어요.</p>}
          {canPlay && !videoFailed && !videoSrc && <p className="question-text">영상 불러오는 중…</p>}
          {canPlay && videoSrc && (
            <video ref={videoRef} className="recording-player" src={videoSrc} controls playsInline />
          )}
        </div>
      )}

      {timeline.length > 0 && (
        <div className="question-card">
          <div className="question-index">타임라인</div>
          <ul className="feedback-list">
            {timeline.map((ev, i) => (
              <li
                key={`timeline-${ev.ts_ms}-${i}`}
                style={{ cursor: videoSrc ? 'pointer' : 'default' }}
                onClick={() => seekTo(ev.ts_ms)}
              >
                <b>{formatTime(ev.ts_ms)}</b>질문 {ev.question_index + 1} · {ev.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {transcript.length > 0 && (
        <div className="question-card">
          <div className="question-index">답변 내용</div>
          <ul className="feedback-list">
            {transcript.map((t, i) => (
              <li key={`transcript-${t.question_index}-${i}`}>
                <b>질문 {t.question_index + 1}</b>{t.text}
              </li>
            ))}
          </ul>
        </div>
      )}

      {llm ? (
        <>
          <div className="question-card">
            <div className="question-index">총평</div>
            <p className="question-text">{llm.summary}</p>

            {(llm.strengths ?? []).length > 0 && (
              <>
                <div className="question-index" style={{ marginTop: 18 }}>강점</div>
                <ul className="feedback-list">
                  {llm.strengths.map((s, i) => (
                    <li key={`strength-${i}`}>{s}</li>
                  ))}
                </ul>
              </>
            )}

            {(llm.improvements ?? []).length > 0 && (
              <>
                <div className="question-index" style={{ marginTop: 18 }}>개선점</div>
                <ul className="feedback-list">
                  {llm.improvements.map((s, i) => (
                    <li key={`improvement-${i}`}>{s}</li>
                  ))}
                </ul>
              </>
            )}
          </div>

          {(llm.per_question ?? []).length > 0 && (
            <div className="question-card">
              <div className="question-index">질문별 피드백</div>
              <ul className="feedback-list">
                {llm.per_question.map((f, i) => (
                  <li key={`llm-feedback-${f.question_index}-${i}`}>
                    <b>질문 {f.question_index + 1}</b>{f.feedback}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <div className="question-card">
          <div className="question-index">피드백</div>
          <ul className="feedback-list">
            {feedback.map((f, i) => (
              <li key={`${f.q}-${i}`}><b>{f.q}</b>{f.note}</li>
            ))}
          </ul>
        </div>
      )}

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