'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getSupabase } from '@/lib/supabase';

interface StatCard {
  figure: string;
  unit: string;
  desc: React.ReactNode;
  dotColor: string;
  tag: string;
}

interface ExpCard {
  className: string;
  title: string;
  desc: React.ReactNode;
  href: string;
}

const WHY_STATS: StatCard[] = [
  { figure: '21.7', unit: '%', desc: <>현재 채용 과정에서<br />AI를 활용 중인 기업 비율</>, dotColor: 'var(--blue)', tag: '고용노동부, 2025 기업 채용동향조사' },
  { figure: '74.5', unit: '%', desc: <>향후 AI 도구 도입<br />확대를 계획 중인 기업 비율</>, dotColor: 'var(--violet)', tag: '고용노동부, 2025 기업 채용동향조사' },
  { figure: '50', unit: '만+', desc: <>표정, 감정 분석 모델 학습에<br />활용된 얼굴 이미지 건수</>, dotColor: 'var(--good)', tag: 'AI-Hub, 한국인 감정인식 위한 복합 영상' },
  { figure: '4', unit: '개', desc: <>중립, 당황, 불안, 기쁨<br />4개의 감정 라벨</>, dotColor: '#E69622', tag: 'AI-Hub' },
];

const DATASETS = [
  { name: '디스플레이 중심 안구 움직임 영상 데이터', href: 'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71421' },
  { name: '한국인 감정인식을 위한 복합 영상', href: 'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=82' },
  { name: 'AI-Hub', href: 'https://aihub.or.kr' },
];

const EXP_CARDS: ExpCard[] = [
  { className: 'exp-a', title: '질문 리스트로 바로 시작', desc: <>회원가입 없이 고정 질문 세트로<br />첫 모의면접을 시작할 수 있습니다.</>, href: '/interview' },
  { className: 'exp-b', title: '실시간 분석 체험', desc: <>웹캠 앞에서 시선, 표정 분석과<br />토스트 알림을 바로 확인합니다.</>, href: '/interview' },
  { className: 'exp-c', title: '영상 업로드 분석', desc: <>이미 녹화해둔 면접 영상을 올리면<br />같은 방식으로 분석합니다.</>, href: '/upload' },
  { className: 'exp-d', title: '행동 리포트 확인', desc: <>면접이 끝나면<br />시선 유지율과 감정 변화를 리포트로 받습니다.</>, href: '/interview/report' },
];

const BLOBS = [
  { width: 420, height: 420, background: '#8B7CFF', top: -120, left: -140 },
  { width: 360, height: 360, background: '#FFB98A', top: 60, right: -140 },
  { width: 320, height: 320, background: '#6FE3B4', bottom: -100, left: '8%' as const },
  { width: 380, height: 380, background: '#7C8BFF', bottom: -140, right: -100 },
  { width: 260, height: 260, background: '#C9A6FF', top: '40%' as const, left: '50%' as const },
];

const LandingPage: React.FC = () => {
  const router = useRouter();
  const [signedIn, setSignedIn] = useState(false);

  useEffect(() => {
    let supabase;
    try {
      supabase = getSupabase();
    } catch (e) {
      console.warn('[landing] Supabase 클라이언트를 만들 수 없습니다.', e);
      return;
    }
    supabase.auth.getSession().then(({ data }) => setSignedIn(!!data.session));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => setSignedIn(!!session));
    return () => sub.subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    try {
      await getSupabase().auth.signOut();
    } catch (e) {
      console.warn('[landing] 로그아웃 실패', e);
    }
    router.refresh();
  };

  return (
    <>
      {BLOBS.map((b, i) => (
        <div
          key={i}
          className="blob"
          style={{
            width: b.width,
            height: b.height,
            background: b.background,
            top: b.top,
            left: b.left,
            right: b.right,
            bottom: b.bottom,
          }}
        />
      ))}

      <div className="wrap">
        <nav>
          <div className="logo"><img className="logo-mark" src="/logo.png" alt="" />SelfFit</div>
          <div className="nav-links">
            <a href="#why">도입 배경</a>
            <a href="#feature">핵심 기능</a>
            <a href="#track">서비스 흐름</a>
            <a href="#experience">체험하기</a>
          </div>
          <div className="nav-right">
            {signedIn ? (
              <button
                className="nav-signin"
                type="button"
                onClick={handleSignOut}
                style={{ background: 'none', padding: 0 }}
              >
                로그아웃
              </button>
            ) : (
              <Link className="nav-signin" href="/login">로그인</Link>
            )}
            <Link className="nav-cta" href="/interview">모의면접 시작</Link>
          </div>
        </nav>

        <section className="hero">
          <h1>보이지 않던 습관을<br />숫자로 만나보세요</h1>
          <p className="hero-sub">웹캠으로 시선과 표정을 실시간 분석하고,<br />면접이 끝나면 행동 리포트로 확인합니다.</p>
          <div className="hero-ctas">
            <a className="btn-ghost" href="#feature">기능 살펴보기</a>
            <Link className="btn-solid" href="/interview">모의면접 시작하기</Link>
          </div>

          <div className="laptop-wrap">
            <div className="floating-badge" style={{ top: -6, left: -30 }}>
              <span className="fb-icon" style={{ background: '#EAF1FF' }}>👁️</span>시선 이탈 감지
            </div>
            <div className="floating-badge" style={{ top: 60, right: -46 }}>
              <span className="fb-icon" style={{ background: '#F1EAFF' }}>🙂</span>표정: 긴장 감지
            </div>
            <div className="floating-badge" style={{ bottom: 30, left: -56 }}>
              <span className="fb-icon" style={{ background: '#E6FBF1' }}>✅</span>질문 3/5 진행 중
            </div>
            <div className="floating-badge" style={{ bottom: -10, right: -24 }}>
              <span className="fb-icon" style={{ background: '#FFF1E6' }}>📈</span>리포트 준비 완료
            </div>

            <div className="laptop-screen">
              <div className="laptop-bar"><span /><span /><span /></div>
              <div className="laptop-inner">
                <div className="l-side">
                  <div className="item active">면접 진행</div>
                  <div className="item">행동 리포트</div>
                  <div className="item">영상 업로드</div>
                </div>
                <div className="l-main">
                  <div className="l-title">3번째 질문 · 시선·표정 분석 중</div>
                  <div className="l-row">
                    <div className="l-card"><div className="k">시선 유지율</div><div className="v num">82%</div></div>
                    <div className="l-card"><div className="k">안정 표정 비율</div><div className="v num">74%</div></div>
                    <div className="l-card"><div className="k">알림 횟수</div><div className="v num">6<span style={{ fontSize: 11, color: 'var(--ink-muted)' }}>회</span></div></div>
                  </div>
                  <div className="l-chart">
                    <svg viewBox="0 0 500 70" preserveAspectRatio="none">
                      <polyline points="10,55 110,44 210,36 310,22 410,14" fill="none" stroke="var(--blue)" strokeWidth={3} strokeLinecap="round" />
                      <circle cx={410} cy={14} r={4} fill="var(--blue)" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>
            <div className="laptop-base" />
            <div className="laptop-stand" />
          </div>
        </section>

        <section className="why-section" id="why">
          <h2>취업 준비생을 위한 면접 자가진단</h2>
          <p className="sub">AI 채용이 빠르게 확대되는 만큼 카메라 앞 행동도 스스로 점검할 수 있어야 합니다.</p>
          <div className="why-grid">
            {WHY_STATS.map((s) => (
              <div className="why-card" key={`${s.figure}${s.unit}`}>
                <div className="why-figure num">
                  {s.figure}
                  <span style={{ fontSize: 16, color: 'var(--ink-muted)' }}>{s.unit}</span>
                </div>
                <p>{s.desc}</p>
                <div className="why-tag">
                  <span className="dot" style={{ background: s.dotColor }} />
                  {s.tag}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="feature-section" id="feature">
          <div className="feature-head">
            <h2>안구, 표정 <span className="grad">2개 모델</span>로 실시간 관찰</h2>
            <p>일정 시간 이상 관찰되는 습관은 토스트 알림으로 바로 알려드립니다.</p>
          </div>
          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-visual">
                <img
                  src="/feature-gaze.png"
                  alt="디스플레이 중심의 시선 추적을 나타낸 일러스트"
                  style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 8 }}
                />
              </div>
              <h3>실시간 시선 추적</h3>
              <p>디스플레이 중심 안구 움직임 데이터로 학습한 모델이<br />시선 방향과 이탈 여부를 프레임 단위로 판정합니다.</p>
            </div>
            <div className="feature-card">
              <div className="feature-visual">
                <div className="mini-chip-row">
                  <div className="mini-chip"><span className="mini-icon" style={{ background: 'var(--good)' }} />중립</div>
                  <div className="mini-chip"><span className="mini-icon" style={{ background: '#E69622' }} />당황</div>
                  <div className="mini-chip"><span className="mini-icon" style={{ background: 'var(--violet)' }} />불안</div>
                  <div className="mini-chip"><span className="mini-icon" style={{ background: 'var(--blue)' }} />기쁨</div>
                </div>
              </div>
              <h3>얼굴 표정, 감정 분석</h3>
              <p>중립, 당황, 불안, 기쁨 4개 감정 라벨로<br />답변 중 표정 변화를 인식합니다.</p>
            </div>
            <div className="feature-card">
              <div className="feature-visual">
                <div className="mini-chip-row" style={{ flexDirection: 'column', width: '100%' }}>
                  <div className="mini-chip" style={{ justifyContent: 'space-between', width: '100%' }}>
                    시선 이탈 <b style={{ color: '#B5790E' }}>3초 지속</b>
                  </div>
                  <div className="mini-chip" style={{ justifyContent: 'space-between', width: '100%' }}>
                    긴장 표정 <b style={{ color: '#B5790E' }}>토스트 알림</b>
                  </div>
                </div>
              </div>
              <h3>즉각적인 토스트 알림</h3>
              <p>일정 시간 이상 같은 행동이 관찰되면 화면에 바로 알려<br />스스로 인지할 수 있게 합니다.</p>
            </div>
            <div className="feature-card">
              <div className="feature-visual">
                <svg className="mini-chart" viewBox="0 0 280 90" height={90}>
                  <polyline points="10,70 70,55 130,50 190,30 260,18" fill="none" stroke="var(--violet)" strokeWidth={3} strokeLinecap="round" />
                  <g fill="var(--violet)">
                    <circle cx={10} cy={70} r={3.5} />
                    <circle cx={70} cy={55} r={3.5} />
                    <circle cx={130} cy={50} r={3.5} />
                    <circle cx={190} cy={30} r={3.5} />
                    <circle cx={260} cy={18} r={3.5} />
                  </g>
                </svg>
              </div>
              <h3>질문 진행 → 행동 리포트</h3>
              <p>정해진 질문 리스트를 따라 면접을 진행하고<br />종료 후 시선 및 표정 변화를 리포트로 확인합니다.</p>
            </div>
          </div>
        </section>

        <section className="track-section" id="track">
          <div className="feature-head">
            <h2><span className="grad">실시간 탐지</span>, 영상 분석</h2>
            <p>지금 웹캠으로 진행하거나 이미 녹화한 영상을 올려도 같은 방식으로 분석합니다.</p>
          </div>

          <div className="track-block">
            <div>
              <span className="track-tag">MAIN</span>
              <h3>질문에 답하는 동안 실시간으로 관찰합니다</h3>
              <p>질문 리스트를 따라 면접을 진행하는 동안<br />안구, 표정 모델이 계속 관찰하고<br />일정 시간 이상 지속되면 토스트로 즉시 알려줍니다.</p>
              <Link className="track-link" href="/interview">면접 시작하기 →</Link>
            </div>
            <div className="track-visual">
              <img
                src="/track-live.png"
                alt="면접 화면에서 시선과 표정을 실시간으로 분석하고 토스트로 알려주는 모습"
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
              />
            </div>
          </div>

          <div className="track-block">
            <div>
              <span className="track-tag">SUB</span>
              <h3>이미 녹화한 영상도 분석할 수 있어요</h3>
              <p>이미 녹화한 면접 영상을 올리면<br />동일한 안구, 표정 모델로 분석합니다.</p>
              <Link className="track-link" href="/upload">영상 업로드하기 →</Link>
            </div>
            <div className="track-visual">
              <svg viewBox="0 0 200 150" width="70%">
                <circle cx={100} cy={75} r={46} fill="none" stroke="var(--line)" strokeWidth={10} />
                <circle
                  cx={100} cy={75} r={46} fill="none" stroke="var(--violet)" strokeWidth={10}
                  strokeDasharray="220 289" strokeLinecap="round" transform="rotate(-90 100 75)"
                />
                <text x={100} y={82} textAnchor="middle" fontFamily="Space Grotesk" fontSize={26} fontWeight={700} fill="var(--ink)">76</text>
              </svg>
            </div>
          </div>
        </section>

        <section className="exp-section" id="experience">
          <div className="feature-head">
            <h2>직접 진행해보면 다릅니다</h2>
            <p>노트북, 모니터 웹캠 환경에서 바로 시작할 수 있습니다.</p>
          </div>
          <div className="exp-grid">
            {EXP_CARDS.map((c) => (
              <Link className={`exp-card ${c.className}`} key={c.title} href={c.href}>
                <div>
                  <h3>{c.title}</h3>
                  <p>{c.desc}</p>
                </div>
              </Link>
            ))}
          </div>
        </section>

        <section className="cta-band">
          <h2>SelfFit</h2>
          <p>웹캠 앞에 앉는 순간부터 스스로 확인이 시작됩니다</p>
          <Link className="btn-solid" href="/interview">모의면접 시작하기</Link>
        </section>

        <footer>
          <div>SelfFit: 면접 행동 자가진단 서비스</div>
          <div className="footer-datasets">
            <span className="footer-datasets-label">모델 학습에 활용한 AI-Hub 공개 데이터셋</span>
            {DATASETS.map((d) => (
              <a key={d.href} href={d.href} target="_blank" rel="noreferrer noopener">{d.name}</a>
            ))}
          </div>
        </footer>
      </div>
    </>
  );
};

export default LandingPage;