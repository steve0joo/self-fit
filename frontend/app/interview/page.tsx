'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import ToastStack, { ToastItem } from '../components/ToastStack';
import { apiFetch, apiFetchRaw, getWsUrl } from '@/lib/api';
import { getSupabase } from '@/lib/supabase';

const QUESTIONS = [
  '1분간 자기소개를 해주세요.',
  '이 직무에 지원한 동기를 말씀해주세요.',
  '본인의 강점과 약점은 무엇인가요?',
  '협업 중 갈등을 해결했던 경험이 있나요?',
  '마지막으로 하고 싶은 말씀이 있다면 해주세요.',
];

const TOAST_POOL: { icon: string; message: string }[] = [
  { icon: '👁️', message: '시선이 화면 밖으로 벗어났어요' },
  { icon: '🙂', message: '표정에서 긴장이 감지됐어요' },
  { icon: '✅', message: '좋아요, 안정적인 시선이에요' },
  { icon: '😮', message: '당황한 표정이 감지됐어요' },
];

const FRAME_SHORT_PX = 224;
const FRAME_INTERVAL_MS = 333;
const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT = 3;
const REPORT_FALLBACK_MS = 3000;
const TOAST_TTL_MS = 4000;
const MOCK_TOAST_INTERVAL_MS = 5000;
const TOAST_PREF_KEY = 'selffit:toastEnabled';
const RECORD_TIMESLICE_MS = 3000;
const RECORD_MIME = 'video/webm;codecs=vp8,opus';
const RECORD_MIME_FALLBACK = 'video/webm';
const RECORD_VIDEO_BPS = 800_000;

function readToastPref(): boolean {
  if (typeof window === 'undefined') return true;
  try {
    return window.localStorage.getItem(TOAST_PREF_KEY) !== 'false';
  } catch (e) {
    console.warn('[interview] 토스트 설정을 읽지 못했습니다.', e);
    return true;
  }
}

function writeToastPref(value: boolean) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(TOAST_PREF_KEY, String(value));
  } catch (e) {
    console.warn('[interview] 토스트 설정을 저장하지 못했습니다.', e);
  }
}

function syncCanvasToVideo(video: HTMLVideoElement, canvas: HTMLCanvasElement): boolean {
  const { videoWidth, videoHeight } = video;
  if (!videoWidth || !videoHeight) return false;
  const scale = FRAME_SHORT_PX / Math.min(videoWidth, videoHeight);
  const width = Math.max(1, Math.round(videoWidth * scale));
  const height = Math.max(1, Math.round(videoHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    console.log('[interview] frame canvas', canvas.width, '×', canvas.height, `(video ${videoWidth}×${videoHeight})`);
  }
  return true;
}

type ApiQuestion = { id: number; text: string; sort_order: number };
type SessionQuestion = { order_index: number; question_id: number; text: string };
type CreatedSession = { id: string; questions?: SessionQuestion[]; ws_url?: string };

let toastId = 0;

function InterviewSession() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraError, setCameraError] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [questions, setQuestions] = useState<string[]>(QUESTIONS);
  const [live, setLive] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const tokenRef = useRef<string | null>(null);
  const frameTimerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reportTimerRef = useRef<number | null>(null);
  const retryRef = useRef(0);
  const teardownRef = useRef(false);
  const navigatedRef = useRef(false);

  // 녹화(MediaRecorder) — 3fps 프레임 전송과는 완전히 독립적으로 동작한다.
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const seqRef = useRef(0);
  const [streamReady, setStreamReady] = useState(false);
  const [recordingSessionId, setRecordingSessionId] = useState<string | null>(null);
  const [toastEnabled, setToastEnabled] = useState(true);
  const toastEnabledRef = useRef(true);
  toastEnabledRef.current = toastEnabled;

  useEffect(() => {
    setToastEnabled(readToastPref());
  }, []);

  const toggleToast = () => {
    const next = !toastEnabledRef.current;
    toastEnabledRef.current = next;
    setToastEnabled(next);
    writeToastPref(next);
    if (!next) setToasts([]);
  };

  const showToast = useCallback((icon: string, message: string) => {
    if (!toastEnabledRef.current) return;
    const id = ++toastId;
    setToasts((prev) => [...prev, { id, icon, message }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), TOAST_TTL_MS);
  }, []);

  const handleVideoMetadata = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) syncCanvasToVideo(video, canvas);
  }, []);

  const stopFrames = useCallback(() => {
    if (frameTimerRef.current !== null) {
      clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
  }, []);

  const startFrames = useCallback(() => {
    stopFrames();
    frameTimerRef.current = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ws = wsRef.current;
      if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
      if (video.paused || video.ended || video.readyState < 2) return;
      if (!syncCanvasToVideo(video, canvas)) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
        },
        'image/jpeg',
        0.7
      );
    }, FRAME_INTERVAL_MS);
  }, [stopFrames]);

  const uploadChunk = useCallback(async (sessionId: string, seq: number, blob: Blob) => {
    const form = new FormData();
    form.append('chunk', blob, `chunk-${seq}.webm`);
    try {
      const res = await apiFetchRaw(`/api/sessions/${sessionId}/recording/chunks?seq=${seq}`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) throw new Error(`POST recording/chunks?seq=${seq} → ${res.status}`);
      console.log(`[interview] 녹화 청크 업로드 완료 seq=${seq} (${blob.size} bytes)`);
    } catch (e) {
      console.warn(`[interview] 녹화 청크 업로드 실패 seq=${seq} — 면접은 계속 진행합니다.`, e);
    }
  }, []);

  const stopRecording = useCallback(() => {
    const rec = mediaRecorderRef.current;
    mediaRecorderRef.current = null;
    if (!rec || rec.state === 'inactive') return;
    try {
      rec.stop(); // 남은 버퍼가 ondataavailable로 한 번 더 떨어져 마지막 청크까지 업로드된다.
    } catch (e) {
      console.warn('[interview] 녹화 중지 실패', e);
    }
  }, []);

  const goReport = useCallback(() => {
    if (navigatedRef.current) return;
    navigatedRef.current = true;
    teardownRef.current = true;
    if (reportTimerRef.current !== null) {
      clearTimeout(reportTimerRef.current);
      reportTimerRef.current = null;
    }
    const sessionId = sessionIdRef.current;
    router.push(sessionId ? `/interview/report?session=${sessionId}` : '/interview/report');
  }, [router]);

  const connect = useCallback(
    (resume: boolean) => {
      const sessionId = sessionIdRef.current;
      const token = tokenRef.current;
      if (!sessionId || !token || teardownRef.current) return;

      let ws: WebSocket;
      try {
        ws = new WebSocket(getWsUrl(sessionId, token));
      } catch (e) {
        console.warn('[interview] WebSocket 생성 실패 — 로컬 진행으로 폴백합니다.', e);
        return;
      }
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setLive(true);
        ws.send(JSON.stringify({ type: resume ? 'resume' : 'start' }));
        startFrames();
      };

      ws.onmessage = (e) => {
        if (typeof e.data !== 'string') return;
        let msg: { type?: string; icon?: string; message?: string; code?: string };
        try {
          msg = JSON.parse(e.data);
        } catch {
          return;
        }
        switch (msg.type) {
          case 'event':
            showToast(msg.icon ?? '👁️', msg.message ?? '');
            break;
          case 'result':
            break;
          case 'report_ready':
            goReport();
            break;
          case 'error':
            console.warn('[interview] ws error', msg.code, msg.message);
            break;
          default:
            break;
        }
      };

      ws.onerror = () => {
        console.warn('[interview] WebSocket 오류 — 백엔드가 떠 있는지 확인하세요.');
      };

      ws.onclose = (e) => {
        stopFrames();
        if (wsRef.current === ws) wsRef.current = null;
        setLive(false);
        if (teardownRef.current) return;
        const fatal = e.code === 1000 || (e.code >= 4000 && e.code < 5000);
        if (fatal) {
          if (e.code !== 1000) console.warn(`[interview] WebSocket 종료 (code ${e.code}) — 로컬 진행으로 폴백합니다.`);
          return;
        }
        if (retryRef.current >= MAX_RECONNECT) {
          console.warn('[interview] 재연결 3회 실패 — 로컬 진행(목업 알림)으로 폴백합니다.');
          return;
        }
        retryRef.current += 1;
        reconnectTimerRef.current = window.setTimeout(() => connect(true), RECONNECT_DELAY_MS);
      };
    },
    [goReport, showToast, startFrames, stopFrames]
  );

  // 스트림과 세션 id가 모두 준비됐을 때만 녹화를 시작한다.
  // 세션이 없는 로컬 폴백 모드에서는 업로드할 곳이 없으므로 녹화 자체를 하지 않는다.
  useEffect(() => {
    if (!streamReady || !recordingSessionId) return;
    const stream = streamRef.current;
    if (!stream) return;
    if (typeof MediaRecorder === 'undefined') {
      console.warn('[interview] 이 브라우저는 MediaRecorder를 지원하지 않습니다 — 녹화를 건너뜁니다.');
      return;
    }

    const mimeType = MediaRecorder.isTypeSupported(RECORD_MIME) ? RECORD_MIME : RECORD_MIME_FALLBACK;
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: RECORD_VIDEO_BPS });
    } catch (e) {
      console.warn('[interview] MediaRecorder 생성 실패 — 녹화 없이 진행합니다.', e);
      return;
    }

    rec.ondataavailable = (e) => {
      if (!e.data || e.data.size === 0) return;
      const seq = seqRef.current; // 재연결(resume) 후에도 초기화하지 않고 이어서 증가
      seqRef.current += 1;
      void uploadChunk(recordingSessionId, seq, e.data);
    };
    rec.onerror = (e) => console.warn('[interview] 녹화 오류', e);

    mediaRecorderRef.current = rec;
    try {
      rec.start(RECORD_TIMESLICE_MS);
      console.log(`[interview] 녹화 시작 (${mimeType}, ${RECORD_TIMESLICE_MS}ms 단위)`);
    } catch (e) {
      console.warn('[interview] 녹화 시작 실패 — 녹화 없이 진행합니다.', e);
      mediaRecorderRef.current = null;
    }

    return () => stopRecording();
  }, [recordingSessionId, stopRecording, streamReady, uploadChunk]);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    const attach = (s: MediaStream) => {
      if (cancelled) {
        s.getTracks().forEach((t) => t.stop());
        return;
      }
      stream = s;
      streamRef.current = s;
      if (videoRef.current) videoRef.current.srcObject = s;
      setStreamReady(true);
    };

    navigator.mediaDevices
      ?.getUserMedia({ video: true, audio: true })
      .then(attach)
      .catch((e) => {
        // 마이크가 거부/부재여도 영상만으로 계속 진행한다.
        console.warn('[interview] 오디오 포함 getUserMedia 실패 — 영상만으로 재시도합니다.', e);
        return navigator.mediaDevices
          ?.getUserMedia({ video: true, audio: false })
          .then(attach)
          .catch(() => setCameraError(true));
      });

    return () => {
      cancelled = true;
      streamRef.current = null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    teardownRef.current = false;

    (async () => {
      try {
        const res = await apiFetch('/api/questions');
        if (!res.ok) throw new Error(`GET /api/questions → ${res.status}`);
        const list: ApiQuestion[] = await res.json();
        const texts = [...list].sort((a, b) => a.sort_order - b.sort_order).map((q) => q.text);
        if (!cancelled && texts.length) setQuestions(texts);
      } catch (e) {
        console.warn('[interview] 질문 목록 로드 실패 — 기본 질문으로 진행합니다.', e);
      }

      if (cancelled) return;

      try {
        const res = await apiFetch('/api/sessions', {
          method: 'POST',
          body: JSON.stringify({ mode: 'live' }),
        });
        if (!res.ok) throw new Error(`POST /api/sessions → ${res.status}`);
        const session: CreatedSession = await res.json();
        const { data } = await getSupabase().auth.getSession();
        const token = data.session?.access_token;
        if (cancelled) return;
        if (!session?.id) throw new Error('세션 id 없음');
        if (!token) throw new Error('access_token 없음');
        sessionIdRef.current = session.id;
        tokenRef.current = token;
        setRecordingSessionId(session.id);
        if (session.questions?.length) {
          setQuestions(
            [...session.questions].sort((a, b) => a.order_index - b.order_index).map((q) => q.text)
          );
        }
        connect(false);
      } catch (e) {
        console.warn('[interview] 세션 생성 실패 — 로컬 진행(목업 알림)으로 폴백합니다.', e);
      }
    })();

    return () => {
      cancelled = true;
      teardownRef.current = true;
      stopFrames();
      if (reconnectTimerRef.current !== null) clearTimeout(reconnectTimerRef.current);
      if (reportTimerRef.current !== null) clearTimeout(reportTimerRef.current);
      wsRef.current?.close(1000, 'unmount');
      wsRef.current = null;
    };
  }, [connect, stopFrames]);

  useEffect(() => {
    if (live) return;
    const interval = setInterval(() => {
      const pick = TOAST_POOL[Math.floor(Math.random() * TOAST_POOL.length)];
      showToast(pick.icon, pick.message);
    }, MOCK_TOAST_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [live, showToast]);

  const isLast = currentIndex >= questions.length - 1;
  const progress = ((currentIndex + 1) / questions.length) * 100;

  const handleNext = () => {
    const ws = wsRef.current;
    const open = ws?.readyState === WebSocket.OPEN;

    if (isLast) {
      stopRecording(); // end 전에 멈춰서 마지막 청크까지 업로드되도록 한다.
      if (open) {
        stopFrames();
        ws!.send(JSON.stringify({ type: 'end' }));
        teardownRef.current = true;
        reportTimerRef.current = window.setTimeout(goReport, REPORT_FALLBACK_MS);
      } else {
        goReport();
      }
      return;
    }

    const next = currentIndex + 1;
    setCurrentIndex(next);
    if (open) ws!.send(JSON.stringify({ type: 'question', index: next }));
  };

  return (
    <div className="page-shell">
      <ToastStack toasts={toasts} />
      <Link className="page-back" href="/">← 홈으로</Link>
      <h1 className="page-title">모의면접 진행 중</h1>
      <p className="page-sub" style={{ color: 'var(--ink)' }}>웹캠으로 시선과 표정을 실시간으로 관찰하고 있습니다.</p>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="video-panel">
        {cameraError ? (
          <div className="video-empty">웹캠 권한이 필요합니다. 브라우저 설정에서 카메라 접근을 허용해주세요.</div>
        ) : (
          <video ref={videoRef} autoPlay playsInline muted onLoadedMetadata={handleVideoMetadata} />
        )}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        <div className="video-rec"><span className="dot" />분석 중</div>
        <button className="toast-toggle" type="button" onClick={toggleToast} aria-pressed={toastEnabled}>
          {toastEnabled ? '🔔 알림 켜짐' : '🔕 알림 꺼짐'}
        </button>
      </div>

      <div className="question-card">
        <div className="question-index">질문 {currentIndex + 1} / {questions.length}</div>
        <p className="question-text">{questions[currentIndex]}</p>
      </div>

      <div className="action-row">
        <button className="btn-solid" onClick={handleNext}>
          {isLast ? '결과 보기 →' : '다음 질문 →'}
        </button>
      </div>
    </div>
  );
}

export default function InterviewPage() {
  const router = useRouter();
  const [authed, setAuthed] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const { data } = await getSupabase().auth.getSession();
        if (cancelled) return;
        if (data.session) {
          setAuthed(true);
          return;
        }
      } catch (e) {
        console.warn('[interview] 세션 확인 실패 — 로그인 화면으로 보냅니다.', e);
        if (cancelled) return;
      }
      setAuthed(false);
      router.replace('/login?redirect=/interview');
    })();

    return () => {
      cancelled = true;
    };
  }, [router]);

  if (authed !== true) {
    return (
      <div className="page-shell">
        <div className="question-index">확인 중...</div>
      </div>
    );
  }

  return <InterviewSession />;
}