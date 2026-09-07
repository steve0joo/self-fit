# Phase 0: 기반 구축 및 FE-BE-Supabase 연결 확인

- 작성일: 2026-09-04
- 작성자: 박기호 (BE / AI 서비스 통합)
- 브랜치: `dev`
- 상태: **연결 검증 완료** (2026-09-04, 브라우저 수동 확인까지 통과)

## 1. 목표

`01-dev-plan.md`의 Phase 0. AI 모델 없이 **FE(Next.js) → BE(FastAPI) → Supabase 인증**이 이어지는 뼈대를 만들고 연결을 확인한다.

## 2. Supabase 프로젝트

| 항목 | 값 |
|---|---|
| Project URL | `https://rreihvyukcbcjrhcntnq.supabase.co` |
| anon key | `.env` / `.env.local`에 있음. 공개 가능 키 |
| JWT 서명 | **ES256 (JWKS)**. `/auth/v1/.well-known/jwks.json`으로 공개키 제공 → 백엔드에 JWT 시크릿 불필요 |
| 인증 방식 | 이메일 + 비밀번호. 소셜 로그인 미설정 |
| 이메일 확인 | **필수** (`mailer_autoconfirm: false`). 가입 후 확인 메일 링크를 눌러야 로그인 가능 |
| DB | 아직 테이블 없음. 세션/리포트 스키마는 설계 단계에서 |

> service_role 키와 DB 비밀번호는 문서와 채팅에 남기지 않는다. 필요 시 `.env`에만 넣는다.

## 3. 백엔드 구조

```
backend/
├── app/
│   ├── main.py            # FastAPI 앱 생성, CORS, 라우터 등록
│   ├── config.py          # pydantic-settings. .env 로드
│   ├── auth.py            # Supabase JWT 검증 (JWKS ES256, HS256 폴백)
│   └── routers/
│       ├── health.py      # GET /health
│       ├── me.py          # GET /api/me  (토큰 필수)
│       └── questions.py   # GET /api/questions (seed 5개, FE 하드코딩과 동일)
├── tests/test_api.py      # 5개 테스트
├── docs/                  # 기획·설계 문서
├── .env.example           # 환경변수 목록
├── pyproject.toml         # uv, Python 3.12
└── README.md
```

| 의존성 | 용도 |
|---|---|
| fastapi, uvicorn[standard] | 서버. websockets 포함(Phase 1 실시간 채널용) |
| pydantic-settings | `.env` 로드 |
| pyjwt[crypto] | JWKS 기반 JWT 검증 |
| httpx | 향후 Supabase REST 호출 |
| pytest, ruff (dev) | 테스트, 린트 |

### 인증 흐름
1. FE가 Supabase JS로 로그인 → `access_token`(ES256 JWT) 획득
2. FE가 `Authorization: Bearer <token>`으로 BE 호출
3. BE `auth.py`가 JWKS에서 `kid`에 맞는 공개키를 받아 서명·만료·`aud=authenticated` 검증
4. `sub`(사용자 UUID)가 없는 토큰(anon 키 등)은 거부
5. 통과하면 `CurrentUser(id, email, role)`를 라우터에 주입

## 4. 프론트엔드 추가분

FE 담당자 코드는 수정하지 않았다. 아래 파일만 추가했다.

| 파일 | 내용 |
|---|---|
| `lib/supabase.ts` | Supabase 클라이언트 싱글턴, `API_URL` 상수 |
| `app/dev/connect/page.tsx` | 연결 확인 페이지. 헬스체크, 가입/로그인, 토큰으로 `/api/me` 호출 결과 표시. 통합 후 삭제 가능 |
| `.env.local.example` | `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL` |
| `package.json` | `@supabase/supabase-js` 추가 |
| `.gitignore` | `.env.local` 추가 |

## 5. 검증 결과

### 자동 검증 (완료)
| 확인 | 결과 |
|---|---|
| `uv run pytest` | 5 passed |
| `uv run ruff check .` | 통과 |
| `GET /health` | 200 |
| `GET /api/questions` | 200, 5개 |
| `GET /api/me` 토큰 없음 | 401 |
| `GET /api/me` anon 키 | 401 (사용자 토큰 아님) |
| CORS preflight, Origin `localhost:3000` | 허용 |
| `npm run build` | 성공, `/dev/connect` 포함 |
| `GET localhost:3000/dev/connect` | 200 렌더링 |

### 브라우저 검증 (완료, 2026-09-04)
| 확인 | 결과 |
|---|---|
| `/dev/connect` 페이지에서 `FastAPI /health` | ✅ (브라우저 fetch → 백엔드 로그 `GET /health 200`) |
| 이메일 가입 → 확인 메일 → 로그인 | 성공 |
| 로그인 토큰으로 `FastAPI /api/me` | ✅ (백엔드 로그 `GET /api/me 200`, JWKS ES256 검증 통과) |

### 재현 절차
1. 두 서버 실행
   ```bash
   cd backend && uv run uvicorn app.main:app --reload --port 8000
   cd frontend && npm run dev
   ```
2. `http://localhost:3000/dev/connect` 접속 → `FastAPI /health` 항목이 ✅
3. 이메일·비밀번호 입력 후 **가입** → 메일의 확인 링크 클릭
4. 같은 정보로 **로그인** → `FastAPI /api/me` 항목이 ✅ 이고 user id가 표시되면 **연결 완료**

> 개발 중 확인 메일이 번거로우면 Supabase 대시보드 Authentication → Providers → Email에서 "Confirm email"을 끌 수 있다. 배포 전에는 다시 켠다.

## 6. 다음 단계 (Phase 1)

- 세션 API와 WebSocket 실시간 채널을 **Mock Predictor**로 구현
- DB 스키마(sessions, analysis_logs, events, reports) 설계 후 Supabase에 생성
- FE 면접 화면이 하드코딩 질문 대신 `/api/questions`를 쓰도록 FE 담당자와 협의
- Supabase MCP 연결 시 테이블 생성과 확인을 Claude Code에서 직접 수행 가능
