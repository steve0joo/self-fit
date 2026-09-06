'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import type { Session } from '@supabase/supabase-js';
import { getSupabase } from '@/lib/supabase';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const supabase = getSupabase();

type Check = { label: string; state: 'idle' | 'ok' | 'fail'; detail: string };

export default function ConnectCheckPage() {
  const [health, setHealth] = useState<Check>({ label: 'FastAPI /health', state: 'idle', detail: '' });
  const [me, setMe] = useState<Check>({ label: 'FastAPI /api/me (토큰 검증)', state: 'idle', detail: '로그인 후 확인' });
  const [session, setSession] = useState<Session | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [authMsg, setAuthMsg] = useState('');

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(async (r) => setHealth({ label: 'FastAPI /health', state: r.ok ? 'ok' : 'fail', detail: `${r.status} ${await r.text()}` }))
      .catch((e) => setHealth({ label: 'FastAPI /health', state: 'fail', detail: `${API_URL} 에 연결 실패: ${e.message}` }));
  }, []);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data: sub } = supabase.auth.onAuthStateChange((_e, s) => setSession(s));
    return () => sub.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (!session) return;
    fetch(`${API_URL}/api/me`, { headers: { Authorization: `Bearer ${session.access_token}` } })
      .then(async (r) => setMe({ label: 'FastAPI /api/me (토큰 검증)', state: r.ok ? 'ok' : 'fail', detail: `${r.status} ${await r.text()}` }))
      .catch((e) => setMe({ label: 'FastAPI /api/me (토큰 검증)', state: 'fail', detail: e.message }));
  }, [session]);

  const signUp = async () => {
    const { error } = await supabase.auth.signUp({ email, password });
    setAuthMsg(error ? `가입 실패: ${error.message}` : '가입 요청 완료. 이메일의 확인 링크를 누른 뒤 로그인하세요.');
  };
  const signIn = async () => {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setAuthMsg(error ? `로그인 실패: ${error.message}` : '로그인 성공');
  };
  const signOut = async () => {
    await supabase.auth.signOut();
    setMe({ label: 'FastAPI /api/me (토큰 검증)', state: 'idle', detail: '로그인 후 확인' });
    setAuthMsg('로그아웃');
  };

  const mark = (s: Check['state']) => (s === 'ok' ? '✅' : s === 'fail' ? '❌' : '⏳');

  return (
    <div className="page-shell">
      <Link className="page-back" href="/">← 홈으로</Link>
      <h1 className="page-title">연결 확인</h1>
      <p className="page-sub">FE → FastAPI → Supabase 인증이 이어지는지 확인하는 개발용 페이지입니다.</p>

      <div className="question-card">
        <div className="question-index">API: {API_URL}</div>
        {[health, me].map((c) => (
          <p key={c.label} style={{ margin: '8px 0' }}>
            {mark(c.state)} <b>{c.label}</b>
            <br />
            <code style={{ fontSize: 13, wordBreak: 'break-all' }}>{c.detail}</code>
          </p>
        ))}
      </div>

      <div className="question-card">
        <div className="question-index">Supabase 인증</div>
        {session ? (
          <>
            <p>로그인됨: <b>{session.user.email}</b></p>
            <p><code style={{ fontSize: 12 }}>user id: {session.user.id}</code></p>
            <div className="action-row"><button className="btn-ghost" onClick={signOut}>로그아웃</button></div>
          </>
        ) : (
          <>
            <input placeholder="email" value={email} onChange={(e) => setEmail(e.target.value)} style={{ display: 'block', width: '100%', margin: '6px 0', padding: 8 }} />
            <input placeholder="password (6자 이상)" type="password" value={password} onChange={(e) => setPassword(e.target.value)} style={{ display: 'block', width: '100%', margin: '6px 0', padding: 8 }} />
            <div className="action-row">
              <button className="btn-ghost" onClick={signUp}>가입</button>
              <button className="btn-solid" onClick={signIn}>로그인</button>
            </div>
          </>
        )}
        {authMsg && <p style={{ marginTop: 8 }}>{authMsg}</p>}
      </div>
    </div>
  );
}