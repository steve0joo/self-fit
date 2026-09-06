'use client';

// 개발용 면접 테스트 페이지. BE 의 세션·WebSocket·토스트·리포트를 실제 웹캠으로 확인한다.
// 서비스 화면이 아니며, 오가는 JSON 을 그대로 보여 주는 것이 목적. FE 통합 후 지워도 된다.
// 규격: backend/guideline/02-for-frontend.md

import React, { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import type { Session } from '@supabase/supabase-js';
import ToastStack, { ToastItem } from '../../components/ToastStack';
import { getSupabase } from '@/lib/supabase';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const FPS = 3;
const FRAME_PX = 224;

type Q = { order_index: number; question_id: number; text: string };
type Phase = 'idle' | 'created' | 'running' | 'finished';

let toastSeq = 0;

export default function DevInterviewPage() {
  const supabase = getSupabase();
  const [auth, setAuth] = useState<Session | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [questions, setQuestions] = useState<Q[]>([]);
  const [qIndex, setQIndex] = useState(0);
  const [wsState, setWsState] = useState('닫힘');
  const [sent, setSent] = useState(0);
  const [toastOn, setToastOn] = useState(true);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [report, setReport] = useState<unknown>(null);
  const [err, setErr] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const toastOnRef = useRef(true);
  toastOnRef.current = toastOn;

  const push = (line: string) => setLog((l) => [line, ...l].slice(0, 30));

  // 로그인 세션
  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setAuth(data.session));
    const { data: sub } = supabase.auth.onAuthStateChange((_e, s) => setAuth(s));
    return () => sub.subscription.unsubscribe();
  }, [supabase]);

  // 웹캠
  useEffect(() => {
    let stream: MediaStream | null = null;
    navigator.mediaDevices?.getUserMedia({ video: { width: 640, height: 480 } })
      .then((s) => { stream = s; if (videoRef.current) videoRef.current.srcObject = s; })
      .catch((e) => setErr(`웹캠 오류: ${e.message}`));
    return () => { stream?.getTracks().forEach((t) => t.stop()); stopFrames(); wsRef.current?.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const api = async (path: string, init: RequestInit = {}) => {
    const token = auth?.access_token;
    const r = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { ...(init.headers || {}), Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    });
    if (!r.ok) throw new Error(`${init.method ?? 'GET'} ${path} → ${r.status} ${await r.text()}`);
    return r.status === 204 ? null : r.json();
  };

  // 1) 세션 생성
  const createSession = async () => {
    setErr(null); setReport(null); setLog([]); setSent(0);
    try {
      const s = await api('/api/sessions', { method: 'POST', body: JSON.stringify({ mode: 'live' }) });
      setSessionId(s.id); setQuestions(s.questions); setQIndex(0); setPhase('created');
      push(`세션 생성 ${s.id}`);
    } catch (e) { setErr(String(e)); }
  };

  // 2) WebSocket 연결 + start
  const start = () => {
    if (!sessionId || !auth) return;
    const url = `${API_URL.replace(/^http/, 'ws')}/ws/sessions/${sessionId}?token=${auth.access_token}`;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    setWsState('연결 중');
    ws.onopen = () => setWsState('열림');
    ws.onclose = (e) => { setWsState(`닫힘 (code ${e.code})`); stopFrames(); };
    ws.onerror = () => setErr('WebSocket 오류. 백엔드가 떠 있는지, 토큰이 유효한지 확인');
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      switch (msg.type) {
        case 'ready':
          ws.send(JSON.stringify({ type: msg.status === 'running' ? 'resume' : 'start' }));
          break;
        case 'started':
          setPhase('running'); setQIndex(msg.question_index ?? 0); startFrames();
          push(`started q=${msg.question_index}`);
          break;
        case 'result':
          push(`result ts=${msg.ts_ms} face=${msg.face_found} yaw=${msg.gaze?.yaw ?? '-'} emo=${msg.emotion?.top ?? '-'} att=${msg.attention?.top ?? '-'}`);
          break;
        case 'event':
          push(`EVENT ${msg.event_type} @${msg.ts_ms} "${msg.message}"`);
          if (toastOnRef.current) showToast(msg.icon, msg.message);
          break;
        case 'question_ack':
          setQIndex(msg.index); push(`question_ack ${msg.index}`);
          break;
        case 'report_ready':
          push('report_ready'); setPhase('finished'); stopFrames();
          api(`/api/sessions/${sessionId}/report`).then(setReport).catch((er) => setErr(String(er)));
          break;
        case 'error':
          push(`error ${msg.code}: ${msg.message}`);
          break;
        default:
          push(`(${msg.type}) ${JSON.stringify(msg)}`);
      }
    };
  };

  // 3) 프레임 전송: 캔버스에 224px 로 그려 JPEG 로 보냄
  const startFrames = () => {
    stopFrames();
    timerRef.current = window.setInterval(() => {
      const v = videoRef.current, c = canvasRef.current, ws = wsRef.current;
      if (!v || !c || !ws || ws.readyState !== WebSocket.OPEN) return;
      c.getContext('2d')!.drawImage(v, 0, 0, FRAME_PX, FRAME_PX);
      c.toBlob((blob) => { if (blob) { ws.send(blob); setSent((n) => n + 1); } }, 'image/jpeg', 0.7);
    }, 1000 / FPS);
  };
  const stopFrames = () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };

  const nextQuestion = () => {
    const next = qIndex + 1;
    if (next >= questions.length) return;
    wsRef.current?.send(JSON.stringify({ type: 'question', index: next }));
  };
  const end = () => { stopFrames(); wsRef.current?.send(JSON.stringify({ type: 'end' })); };
  const forceFinish = async () => {
    if (!sessionId) return;
    try { await api(`/api/sessions/${sessionId}/finish`, { method: 'POST' }); setPhase('finished');
      setReport(await api(`/api/sessions/${sessionId}/report`)); } catch (e) { setErr(String(e)); }
  };

  const showToast = (icon: string, message: string) => {
    const id = ++toastSeq;
    setToasts((t) => [...t, { id, icon, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  };

  if (!auth) {
    return (
      <div className="page-shell">
        <h1 className="page-title">개발용 면접 테스트</h1>
        <p className="page-sub">로그인이 필요합니다. <Link href="/login">로그인</Link> 또는 <Link href="/dev/connect">연결 확인 페이지</Link>에서 로그인 후 돌아오세요.</p>
      </div>
    );
  }

  const isLast = qIndex >= questions.length - 1;
  return (
    <div className="page-shell">
      {toastOn && <ToastStack toasts={toasts} />}
      <Link className="page-back" href="/">← 홈으로</Link>
      <h1 className="page-title">개발용 면접 테스트</h1>
      <p className="page-sub">API {API_URL} · 사용자 {auth.user.email} · 세션 {sessionId ?? '-'} · 상태 {phase} · WS {wsState} · 보낸 프레임 {sent}</p>
      {err && <p style={{ color: 'crimson' }}>{err}</p>}

      <div className="video-panel" style={{ maxWidth: 480 }}>
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} width={FRAME_PX} height={FRAME_PX} style={{ display: 'none' }} />
      </div>

      <div className="question-card">
        <div className="question-index">{questions.length ? `질문 ${qIndex + 1} / ${questions.length}` : '세션 없음'}</div>
        <p className="question-text">{questions[qIndex]?.text ?? '세션을 만들면 질문이 표시됩니다.'}</p>
      </div>

      <div className="action-row" style={{ flexWrap: 'wrap', gap: 8 }}>
        <button className="btn-solid" onClick={createSession} disabled={phase === 'running'}>1. 세션 만들기</button>
        <button className="btn-solid" onClick={start} disabled={phase !== 'created'}>2. 시작 (WS 연결)</button>
        <button className="btn-ghost" onClick={nextQuestion} disabled={phase !== 'running' || isLast}>다음 질문 →</button>
        <button className="btn-ghost" onClick={end} disabled={phase !== 'running'}>종료 → 리포트</button>
        <button className="btn-ghost" onClick={forceFinish} disabled={!sessionId || phase === 'finished'}>강제 종료 (/finish)</button>
        <label style={{ alignSelf: 'center' }}>
          <input type="checkbox" checked={toastOn} onChange={(e) => setToastOn(e.target.checked)} /> 토스트 표시
        </label>
      </div>

      <div className="question-card">
        <div className="question-index">메시지 로그 (최근 30개)</div>
        <pre style={{ fontSize: 12, maxHeight: 260, overflow: 'auto', margin: 0 }}>{log.join('\n')}</pre>
      </div>

      {report != null && (
        <div className="question-card">
          <div className="question-index">리포트 JSON (GET /api/sessions/{'{id}'}/report)</div>
          <pre style={{ fontSize: 12, maxHeight: 400, overflow: 'auto', margin: 0 }}>{JSON.stringify(report, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
