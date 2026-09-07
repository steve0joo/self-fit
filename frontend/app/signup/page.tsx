'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { getSupabase } from '@/lib/supabase';

export default function SignupPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== passwordConfirm) {
      setError('비밀번호가 일치하지 않습니다.');
      return;
    }

    setLoading(true);
    const { error: signUpError } = await getSupabase().auth.signUp({ email, password });
    setLoading(false);

    if (signUpError) {
      setError(signUpError.message);
      return;
    }
    setSent(true);
  };

  return (
    <>
      <div className="blob" style={{ width: 420, height: 420, background: '#8B7CFF', top: -140, left: -120 }} />
      <div className="blob" style={{ width: 360, height: 360, background: '#7C8BFF', bottom: -140, right: -110 }} />
      <div className="blob" style={{ width: 280, height: 280, background: '#6FE3B4', bottom: '10%', left: '4%' }} />

      <div className="page-shell auth-shell">
        <Link className="page-back" href="/">← 홈으로</Link>
        <h1 className="page-title">회원가입</h1>
        <p className="page-sub">이메일로 계정을 만들고 모의면접 기록을 저장하세요.</p>

        <div className="question-card">
          {sent ? (
            <>
              <p className="question-text">이메일을 확인해주세요</p>
              <p className="form-note">
                {email} 주소로 인증 메일을 보냈습니다. 메일의 링크를 눌러 인증을 완료한 뒤 로그인해주세요.
              </p>
              <div className="form-foot">
                <Link className="form-link" href="/login">로그인으로 이동</Link>
              </div>
            </>
          ) : (
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
                  placeholder="6자 이상"
                  autoComplete="new-password"
                  minLength={6}
                  required
                />
              </div>

              <div className="form-field">
                <label className="form-label" htmlFor="passwordConfirm">비밀번호 확인</label>
                <input
                  id="passwordConfirm"
                  className="form-input"
                  type="password"
                  value={passwordConfirm}
                  onChange={(e) => setPasswordConfirm(e.target.value)}
                  placeholder="비밀번호를 다시 입력하세요"
                  autoComplete="new-password"
                  required
                />
              </div>

              {error && <p className="form-error">{error}</p>}

              <button className="btn-solid form-submit" type="submit" disabled={loading}>
                {loading ? '가입 중…' : '회원가입'}
              </button>

              <div className="form-foot">
                이미 계정이 있으신가요? <Link className="form-link" href="/login">로그인</Link>
              </div>
            </form>
          )}
        </div>
      </div>
    </>
  );
}