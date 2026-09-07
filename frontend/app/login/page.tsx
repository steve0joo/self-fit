'use client';

import React, { Suspense, useState } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { getSupabase } from '@/lib/supabase';

const safeRedirect = (value: string | null) =>
  value && value.startsWith('/') && !value.startsWith('//') ? value : '/interview';

function LoginForm() {
  const router = useRouter();
  const redirectTo = safeRedirect(useSearchParams().get('redirect'));
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    const { error: signInError } = await getSupabase().auth.signInWithPassword({ email, password });

    if (signInError) {
      setLoading(false);
      setError(signInError.message);
      return;
    }
    router.push(redirectTo);
  };

  return (
    <>
      <div className="blob" style={{ width: 420, height: 420, background: '#8B7CFF', top: -140, left: -120 }} />
      <div className="blob" style={{ width: 360, height: 360, background: '#7C8BFF', bottom: -140, right: -110 }} />
      <div className="blob" style={{ width: 280, height: 280, background: '#6FE3B4', bottom: '10%', left: '4%' }} />

      <div className="page-shell auth-shell">
        <Link className="page-back" href="/">← 홈으로</Link>
        <h1 className="page-title">로그인</h1>
        <p className="page-sub">가입한 이메일로 로그인하고 모의면접을 이어서 진행하세요.</p>

        <div className="question-card">
          <form onSubmit={handleSubmit}>
            <div className="form-field">
              <label className="form-label" htmlFor="email">이메일</label>
              <input
                id="email"
                className="form-input"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </div>

            <div className="form-field">
              <label className="form-label" htmlFor="password">비밀번호</label>
              <input
                id="password"
                className="form-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="비밀번호를 입력하세요"
                autoComplete="current-password"
                required
              />
            </div>

            {error && <p className="form-error">{error}</p>}

            <button className="btn-solid form-submit" type="submit" disabled={loading}>
              {loading ? '로그인 중…' : '로그인'}
            </button>

            <div className="form-foot">
              계정이 없으신가요? <Link className="form-link" href="/signup">회원가입</Link>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="page-shell auth-shell">
          <div className="question-index">확인 중...</div>
        </div>
      }
    >
      <LoginForm />
    </Suspense>
  );
}