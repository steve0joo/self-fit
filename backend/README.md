# SelfFit Backend (FastAPI)

팀원용 가이드는 [`guideline/`](guideline/README.md), 기획·설계 문서는 [`docs/`](docs/)에 있습니다.

## 실행

```bash
cd backend
cp .env.example .env   # 값 채우기
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

- 헬스체크: http://localhost:8000/health
- API 문서: http://localhost:8000/docs

## 테스트

```bash
uv run pytest
uv run ruff check .
```

## DB
- 기본은 로컬 SQLite(`dev.db`, git 제외). 앱 시작 시 테이블과 질문 seed 를 자동 생성하므로 바로 실행됩니다.
- Supabase 를 쓰려면 `db/migrations/001_init.sql` 을 Supabase SQL Editor 에서 실행하고 `.env` 에 `DATABASE_URL` 을 넣습니다.

## 추론
- `INFERENCE_BACKEND=mock`(기본)이면 모델 없이 규칙적인 가짜 결과로 전체 흐름이 돕니다. Phase 2 에서 `http` 로 바꿉니다.

## 인증

FE가 Supabase로 로그인해 받은 access token을 `Authorization: Bearer <token>` 으로 보내면,
백엔드는 Supabase JWKS(공개키)로 서명을 검증한다. 시크릿 불필요.
