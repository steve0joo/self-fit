# Phase 1 개발 설계: 세션·실시간 분석·리포트

- 작성일: 2026-09-04
- 작성자: 박기호 (BE / AI 서비스 통합)
- 선행 문서: `01-dev-plan.md`(기획), `02-phase0-setup.md`(기반 구축)
- 상태: 초안. 6절(FE 협의 항목)은 FE 담당자 확인 필요

이 문서는 "어떻게 만들 것인가"를 정한다. 구현은 이 문서를 기준으로 하고, 구현 중 바뀐 결정은 이 문서에 다시 반영한다.

> **현재 목표 환경 (2026-09-04 결정): 실제 배포는 하지 않는다.** FE(3000), BE(8000), 추론 서버(9000)를 모두 노트북 한 대에서 띄우고, **같은 네트워크(Wi-Fi)의 팀원 PC가 노트북 IP로 접속**할 수 있게 한다 (10.2절). Render·Vercel·Cloudflare Tunnel 관련 내용(10.3절)은 나중에 배포하게 될 때를 위한 참고이며 지금은 적용하지 않는다. 같은 네트워크에서는 프레임 왕복이 50ms 안팎이다.

> **단순화 원칙 (2026-09-04 리뷰):** 7인 팀에서 FE·AI 팀이 읽고 바로 붙일 수 있어야 한다. 그래서 (1) DB는 동기 SQLAlchemy + SQL 파일 마이그레이션, (2) 집중 상태 버퍼는 추론 서버가 관리해 BE에서 이미지 처리 제거, (3) 페이지네이션은 limit/offset, (4) 세션 자동 종료 타이머·upload_jobs 테이블은 Phase 3으로 미룸. 팀 공유용 요약은 `backend/guideline/`에 있다.

---

## 1. 범위와 목표

Phase 1은 **AI 모델 없이 면접 전체 흐름이 돌아가는 상태**를 만든다. 모델 자리에는 Mock Predictor를 둔다. Phase 2에서 Mock을 실제 모델로 갈아끼운다.

서빙 모델은 3개다: **시선**(L2CS-Net), **감정**(EmotionNet), **집중 상태**(Former-DFER). 시선과 감정은 프레임마다, 집중 상태는 16프레임 묶음으로 주기 추론한다.

**2026-09-04 결정: 감정은 4종만 쓴다. 중립, 불안, 당황, 기쁨.** 분노·상처·슬픔은 면접 상황에서 드물고 구분 신뢰도가 낮아 제외했다. 처리 방식은 **모델 출력(7클래스 확률)은 손대지 않고, 1등이 4종 중 하나일 때만 그 라벨을 쓰고 아니면 "기타"로 둔다.** 확률을 4종으로 재정규화하지 않는다 — 버린 3종의 확률을 4종에 몰아주면 모델이 말하지 않은 확신을 만들어 과대해석이 되기 때문이다. AI 팀이 4클래스로 파인튜닝하면 "기타"가 사라질 뿐 로직은 같다.

**2026-09-04 결정: 추론은 별도 추론 서버(GPU, Docker)에서 하고 BE는 HTTP로 호출한다.** 세 모델은 추론 서버 이미지 1개에 함께 올린다. AI-Hub 제공 Docker 이미지는 쓰지 않고 가중치와 구조 코드만 가져온다. Phase 1은 BE 쪽 Mock으로 진행하고, 추론 서버는 Phase 2에서 만든다.

| 포함 | 제외 (Phase 2~3) |
|---|---|
| 세션 생성·시작·질문 전환·종료 REST API | 추론 서버 구현과 실제 모델 추론 |
| WebSocket 실시간 채널 (프레임 수신, 결과·이벤트 송신) | 얼굴 검출기 |
| 이벤트 판정 엔진 (시계열 규칙) | 영상 업로드 분석 |
| 분석 로그·이벤트·리포트 DB 저장 | ONNX 변환, 성능 튜닝 |
| 리포트 집계와 조회 API | 외부 배포 |
| Predictor 인터페이스 + Mock 구현 | |

완료 기준: FE가 실제 웹캠으로 면접을 진행하면 Mock 결과로 토스트가 뜨고, 종료 후 DB에 저장된 로그로 만든 리포트가 조회된다.

---

## 2. 아키텍처

```
┌──────────────┐  REST (JWT)   ┌───────────────────────────────┐  내부 HTTP   ┌─────────────────────────┐
│  Next.js FE  │──────────────▶│  BE: FastAPI (노트북 :8000)   │────────────▶│ 추론 서버: FastAPI+Torch │
│              │  WebSocket    │  ┌──────┐ ┌────────┐ ┌──────┐ │  JPEG 전송   │ (노트북 GPU, Docker)     │
│  웹캠 캡처   │◀─────────────▶│  │세션  │ │판정엔진│ │리포트│ │◀────────────│  얼굴 검출 → L2CS-Net    │
│  3fps 224px │  frame/event  │  └──────┘ └────────┘ └──────┘ │  결과 JSON   │  → EmotionNet            │
└──────────────┘               │  InferenceClient (HTTP)       │             │  → Former-DFER (16장)    │
        │ 로그인               └──────────────┬────────────────┘             └─────────────────────────┘
        ▼                                     │ asyncpg
┌──────────────┐  JWKS 검증           ┌───────▼────────┐
│ Supabase Auth│◀─────────────────────│ Supabase        │
└──────────────┘                      │ PostgreSQL      │
                                      └────────────────┘
```

역할 분담: **BE**는 세션·인증·판정·저장·리포트를 맡고 이미지를 해석하지 않는다. **추론 서버**는 이미지를 받아 세 모델의 결과 JSON만 돌려주는 무상태(stateless) 서비스다. 세션이나 DB를 모른다.

### 핵심 결정
| 결정 | 선택 | 이유 |
|---|---|---|
| DB 접근 | SQLAlchemy 2.0 **동기** + psycopg. 마이그레이션은 번호 붙인 SQL 파일(`backend/db/migrations/*.sql`)을 Supabase SQL Editor에서 실행 | 누구나 SQL 파일을 읽으면 스키마를 안다. Alembic·asyncpg 학습 비용 제거. FastAPI가 동기 DB 호출을 스레드풀로 처리하므로 성능 문제 없음 |
| 접속 경로 | Supabase 연결 풀러 transaction 모드, 포트 6543 | 서버 재시작·`--reload` 시 커넥션 폭주 방지 |
| 권한 모델 | 백엔드가 service 연결로 접근, `user_id` 필터를 코드에서 강제. RLS 미사용 | FE는 DB에 직접 접근하지 않음 |
| 세션 상태 | DB가 정본. 진행 중 세션의 판정 상태는 프로세스 메모리 | 단일 인스턴스 전제. 다중 인스턴스는 범위 밖 |
| 집중 상태 추론 주기 | 추론 서버가 버퍼 16장이 찬 뒤 **2초마다 1회** Former-DFER 실행. 사이 프레임은 직전 값 반환 | 16프레임 입력 모델. 매 프레임 실행은 낭비 |
| 프레임 전송 | WebSocket **바이너리 메시지**로 JPEG 원본 전송, 제어 메시지는 JSON 텍스트 | base64 오버헤드 33% 제거. 한 연결로 양방향 |
| 추론 위치 | **별도 추론 서버** (GPU, 모델 3개). BE는 `InferenceClient`로 HTTP 호출. 지금은 같은 노트북의 localhost:9000 | GPU 활용. BE에 torch를 싣지 않아 가볍고, 모델 교체가 BE와 분리됨 |
| 추론 서버 상태 | `/v1/analyze`에 `session_id`를 함께 보내면 추론 서버가 세션별 16장 얼굴 버퍼를 메모리에 유지하고 집중 상태를 계산 | BE에서 이미지 크롭·opencv를 없앰. 추론 서버가 이미 이미지를 들고 있으니 버퍼는 거기가 자연스러움. 재시작하면 버퍼만 비고 세션은 영향 없음 |
| BE ↔ 추론 서버 보안 | 공유 시크릿 헤더 `X-Inference-Token`. 로컬에서는 형식만 유지 | 나중에 외부 노출 시 그대로 사용 |
| 추론 서버 장애 | 요청 타임아웃 1초. 실패 시 해당 프레임은 `face_found=null`로 기록하고 세션은 계속. 10회 연속 실패 시 FE에 `error` 메시지 | 노트북 GPU 서버가 꺼져도 면접 흐름은 유지 |
| 질문 전환 | WebSocket 제어 메시지로 전달, 서버가 DB에 기록 | REST 왕복 없이 프레임과 같은 채널에서 순서 보장 |

---

## 3. 폴더 구조

Phase 0 구조를 확장한다. 계층은 `routers → services → repositories/models`이고, 분석 파이프라인은 `analysis/`로 분리한다.

```
backend/
├── app/
│   ├── main.py                 # 앱 생성, 라이프사이클(모델 로드, DB 엔진)
│   ├── config.py               # Settings (+ DB_URL, 판정 임계값)
│   ├── auth.py                 # Supabase JWT 검증 (Phase 0)
│   ├── db.py                   # 엔진, 세션 팩토리, get_db 의존성
│   ├── models/                 # SQLAlchemy ORM
│   │   ├── question.py  session.py  analysis_log.py  event.py  report.py  upload_job.py
│   ├── schemas/                # Pydantic 요청/응답
│   │   ├── session.py  report.py  ws.py
│   ├── routers/
│   │   ├── health.py  me.py  questions.py           # Phase 0
│   │   ├── sessions.py                              # REST 세션 생명주기
│   │   ├── reports.py                               # 리포트 조회
│   │   └── ws.py                                    # WebSocket 엔드포인트
│   ├── services/
│   │   ├── session_service.py   # 세션 상태 전이, 권한 확인
│   │   ├── report_service.py    # 로그 → 리포트 집계
│   │   └── live_session.py      # 연결당 상태: 프레임 버퍼, 판정기, 현재 질문
│   └── analysis/
│       ├── types.py             # FaceBox, GazeResult, EmotionResult, AttentionResult, FrameResult
│       ├── client.py            # InferenceClient 인터페이스 + HttpInferenceClient (추론 서버 호출)
│       ├── mock.py              # MockInferenceClient (Phase 1)
│       └── rules.py             # 이벤트 판정 엔진
├── db/migrations/              # 001_init.sql, 002_....sql (Supabase SQL Editor에서 순서대로 실행)
├── guideline/                  # 팀 공유 가이드 (FE·AI 팀용)
├── tests/
│   ├── test_api.py             # Phase 0
│   ├── test_sessions.py  test_ws.py  test_rules.py  test_report.py
│   └── conftest.py             # 테스트 DB, 인증 우회 fixture
└── docs/
```

BE 추가 의존성: `sqlalchemy`, `psycopg[binary]`. **BE에는 torch·opencv를 넣지 않는다.** 이미지는 받아서 그대로 추론 서버에 넘길 뿐이다.

추론 서버는 저장소 루트의 별도 프로젝트로 둔다 (담당: BE).

```
inference/
├── app/
│   ├── main.py                 # FastAPI, lifespan에서 모델 3개 로드
│   ├── config.py               # MODEL_DIR, DEVICE(cuda|cpu), INFERENCE_TOKEN
│   ├── auth.py                 # X-Inference-Token 검증
│   ├── routers/infer.py        # /v1/analyze, /v1/health
│   ├── detector.py             # MediaPipe 얼굴 검출
│   ├── preprocess.py           # 모델별 크롭·리사이즈·정규화 (7.1절)
│   ├── attention_buffer.py     # session_id별 얼굴 크롭 16장 버퍼 (dict + deque, 10분 미사용 시 삭제)
│   └── models/
│       ├── l2cs.py             # zip의 l2cs/model.py 구조 코드 이식
│       ├── emotionnet.py       # zip의 models/emotionnet.py 이식
│       └── former_dfer.py      # zip의 models/ST_Former.py + S/T_Former 이식
├── weights/                    # git 제외. l2cs_trained.pkl, model.pth, former_trained.pth
├── tests/                      # 샘플 이미지 → 원본 스크립트 결과와 일치 검증
├── scripts/bench.py            # 모델별 추론 시간 측정
├── Dockerfile                  # pytorch/pytorch:2.x-cuda 베이스. 의존성만 굽고 소스·가중치는 볼륨
├── compose.yaml                # docker compose up 한 줄로 실행. GPU 예약, 볼륨, --reload
├── pyproject.toml              # torch, torchvision, einops, mediapipe, fastapi, uvicorn, opencv-python-headless
└── README.md
```

---

## 4. DB 스키마

Supabase PostgreSQL. `auth.users`는 Supabase가 관리하고, 서비스 테이블은 `public` 스키마에 둔다.

```sql
-- 질문 (seed 데이터, 관리자만 변경)
questions (
  id            serial PRIMARY KEY,
  text          text NOT NULL,
  category      text,                     -- 'general' | 'motivation' | ... (P2)
  sort_order    int NOT NULL,
  is_active     bool NOT NULL DEFAULT true
)

-- 면접 세션
sessions (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  mode          text NOT NULL,            -- 'live' | 'upload'
  status        text NOT NULL,            -- 'created' | 'running' | 'finished' | 'failed'
  created_at    timestamptz NOT NULL DEFAULT now(),
  started_at    timestamptz,
  finished_at   timestamptz
)
INDEX (user_id, created_at DESC)

-- 세션에 배정된 질문과 진행 시각
session_questions (
  id            serial PRIMARY KEY,
  session_id    uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  question_id   int  NOT NULL REFERENCES questions(id),
  order_index   int  NOT NULL,            -- 0부터
  started_at    timestamptz,
  ended_at      timestamptz,
  UNIQUE (session_id, order_index)
)

-- 프레임 단위 분석 결과 (원본 이미지는 저장하지 않음)
analysis_logs (
  id            bigserial PRIMARY KEY,
  session_id    uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  ts_ms         int  NOT NULL,            -- 세션 시작 기준 경과 ms
  question_index int,                     -- 그 시점의 질문. null = 시작 전
  face_found    bool NOT NULL,
  gaze_yaw      real,  gaze_pitch real,   -- 도(deg). face_found=false면 null
  gaze_state    text,                     -- 'center' | 'off' | 'unknown'
  emotion_probs jsonb,                    -- 모델 출력 그대로 (7개). 재정규화 없음
  emotion_top   text,                     -- 4종 중 1등이면 그 라벨, 아니면 '기타'
  attention_probs jsonb,                  -- {"집중":0.7, ...} 5개. 2초마다 갱신되므로 사이 프레임은 직전 값 복사
  attention_top text                      -- 'focused' | 'drowsy' | 'deficit' | 'declining' | 'negligent' | null
)
INDEX (session_id, ts_ms)

-- 판정 엔진이 발행한 이벤트 (= 토스트)
events (
  id            bigserial PRIMARY KEY,
  session_id    uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  ts_ms         int  NOT NULL,
  question_index int,
  type          text NOT NULL,            -- 'gaze_off' | 'face_lost' | 'emotion_negative' | 'emotion_surprised' | 'attention_low' | 'gaze_stable'
  severity      text NOT NULL,            -- 'info' | 'warn'
  message       text NOT NULL,            -- FE에 그대로 표시할 문구
  payload       jsonb                     -- 지속 시간, 확률 등
)
INDEX (session_id, ts_ms)

-- 세션 종료 시 1회 생성되는 집계 결과
reports (
  session_id    uuid PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  summary       jsonb NOT NULL,           -- 5.6절 구조
  created_at    timestamptz NOT NULL DEFAULT now()
)
```

설계 메모
- 영상 업로드용 `upload_jobs` 테이블은 Phase 3에서 추가한다.
- `analysis_logs`는 3fps × 5분 ≈ 900행/세션. 100세션이어도 9만 행이라 프레임 단위 저장으로 충분하다. 몰리면 1초 집계로 바꾼다.
- 얼굴 이미지는 어디에도 저장하지 않는다 (기획서 비기능 요구).
- `emotion_probs` 키는 한국어 라벨 그대로 둔다. 클래스 순서는 `analysis/types.py`의 `EMOTION_LABELS` 상수 하나로 관리한다. `emotion_top`은 BE가 `EMOTION_USED`로 판정해 넣는다. 재정규화는 하지 않는다.
- `attention_probs`도 같은 방식. Former-DFER 클래스 순서는 AI-Hub 표기 순(집중 F, 졸림 S, 집중결핍 D, 집중하락 A, 태만 N)으로 가정하며, **학습 라벨 파일로 인덱스 순서 확인 필요.**

---

## 5. REST API 명세

공통: 모든 `/api/*`는 `Authorization: Bearer <Supabase access_token>` 필수. 에러 응답은 `{"detail": "<메시지>"}`. 404는 "존재하지 않거나 내 것이 아님"을 구분하지 않는다.

### 5.1 질문
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/questions` | 활성 질문 목록. `[{id, text, category, sort_order}]` |

### 5.2 세션 생명주기
| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| POST | `/api/sessions` | `{mode: "live", question_ids?: [int]}` 생략 시 활성 질문 전체 | `201 {id, status:"created", questions:[{order_index, question_id, text}], ws_url}` |
| GET | `/api/sessions` | `?limit=20&offset=0` | 내 세션 목록 (최신순) `{items:[{id, mode, status, created_at, finished_at, has_report}], total}` |
| GET | `/api/sessions/{id}` | | 세션 상세 + 질문 목록 + 상태 |
| POST | `/api/sessions/{id}/finish` | | WebSocket 없이 강제 종료 (탭 닫힘 복구용). 리포트 생성 후 `{status:"finished", report_ready:true}` |
| DELETE | `/api/sessions/{id}` | | 세션과 하위 데이터 삭제 |

상태 전이: `created --(WS 연결 + start)--> running --(WS end 또는 /finish)--> finished`. WS가 끊긴 `running` 세션은 그대로 두고, FE가 다시 들어오면 `/finish`로 정리한다. 자동 정리는 Phase 3.

### 5.3 리포트
| 메서드 | 경로 | 응답 |
|---|---|---|
| GET | `/api/sessions/{id}/report` | 5.6절 구조. 미생성이면 `409 {detail:"세션이 끝나지 않았습니다"}` |
| GET | `/api/sessions/{id}/events` | 이벤트 타임라인 `[{ts_ms, question_index, type, severity, message}]` |

### 5.4 WebSocket 진입점
`GET /ws/sessions/{id}?token=<access_token>` → 101 Switching Protocols. 브라우저 WebSocket은 헤더를 못 붙이므로 토큰은 쿼리로 받고, 연결 직후 검증 실패 시 close code `4401`.

### 5.5 추론 서버 HTTP 규격 (BE → 추론 서버)

인증: 모든 요청에 `X-Inference-Token: <공유 시크릿>`. 없거나 틀리면 `401`.

| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| GET | `/v1/health` | | `{status:"ok", device:"cuda", models:{gaze:"l2cs_v1", emotion:"emotionnet_v1", attention:"former_dfer_v1"}}` |
| POST | `/v1/analyze?session_id=<uuid>` | body: JPEG bytes, `Content-Type: image/jpeg` | 아래 |

`/v1/analyze` 응답:
```json
{
  "face_found": true,
  "face": {"x": 120, "y": 80, "w": 200, "h": 200, "score": 0.98},
  "gaze":      {"yaw_deg": -3.2, "pitch_deg": 5.1, "confidence": 0.81},
  "emotion":   {"probs": {"기쁨":0.1, "당황":0.02, "분노":0.01, "불안":0.05, "상처":0.01, "슬픔":0.01, "중립":0.8}, "top": "중립"},
  "attention": {"probs": {"집중":0.7, "졸림":0.05, "집중결핍":0.1, "집중하락":0.1, "태만":0.05}, "top": "집중"},
  "timing_ms": {"detect": 4, "gaze": 9, "emotion": 1, "attention": 0}
}
```
- 얼굴이 없으면 `face_found=false`이고 나머지는 `null`. 얼굴이 여러 개면 가장 큰 것 하나.
- `attention`은 `session_id`별 버퍼에 얼굴 16장이 찰 때까지 `null`. 이후 2초마다 갱신되고 사이 프레임은 직전 값.
- `confidence`는 yaw/pitch softmax 최댓값의 평균.
- 에러: `400` 디코딩 실패, `413` 이미지 1MB 초과, `503` 모델 미로드. BE는 `5xx`와 타임아웃을 같은 방식으로 처리한다 (2절 "추론 서버 장애").

### 5.6 리포트 응답 구조
```json
{
  "session_id": "uuid",
  "duration_ms": 312000,
  "overview": {
    "gaze_hold_rate": 0.82,          // gaze_state == center 인 유효 프레임 비율
    "stable_emotion_rate": 0.74,     // emotion_top ∈ {중립, 기쁨} 비율
    "attention_rate": 0.81,          // attention_top == 집중 비율
    "face_found_rate": 0.97,
    "event_count": 6
  },
  "emotion_distribution": {"중립":0.58, "불안":0.12, "당황":0.05, "기쁨":0.19, "기타":0.06},
  "attention_distribution": {"집중":0.81, "졸림":0.02, "집중결핍":0.07, "집중하락":0.08, "태만":0.02},
  "per_question": [
    {"order_index":0, "question_id":1, "text":"...", "duration_ms":61000,
     "gaze_hold_rate":0.78, "dominant_emotion":"중립", "attention_rate":0.85, "event_count":2}
  ],
  "timeline": [ {"ts_ms":12000, "question_index":0, "type":"gaze_off", "message":"..."} ],
  "feedback": [
    {"question_index":0, "note":"초반 시선 이탈이 2회 감지됐어요. 답변 시작 전 카메라를 먼저 응시해보세요."}
  ]
}
```
FE 리포트 화면(시선 유지율, 안정 표정 비율, 알림 횟수, 질문별 막대, 피드백 문장)과 1:1로 대응한다. `attention_rate`는 FE에 아직 없는 지표라 12절에서 협의한다.

---

## 6. WebSocket 프로토콜

한 세션 = 한 연결. 텍스트 프레임은 JSON 제어 메시지, 바이너리 프레임은 JPEG 이미지 1장.

### 6.1 FE → BE
| 형식 | 내용 | 설명 |
|---|---|---|
| text | `{"type":"start"}` | 세션 시작. 서버가 `started_at` 기록, 질문 0 시작. 프레임은 `start` 이후에만 처리됨 |
| binary | JPEG bytes | 웹캠 프레임. **권장 224×224 이하, 품질 70, 3fps.** 서버가 수신 시각을 `ts_ms`로 기록 |
| text | `{"type":"resume"}` | 재연결 시. 세션이 `running`이면 그대로 이어 감. `created`면 `start`와 동일 처리 |
| text | `{"type":"question","index":1}` | 질문 전환. 이전 질문 `ended_at`, 새 질문 `started_at` 기록. **앞으로만 간다**: 현재 번호 이하가 오면 무시하고 `question_ack`에 현재 번호를 돌려줌 (2026-09-06 QA 후 추가) |
| text | `{"type":"end"}` | 면접 종료 요청. 서버가 리포트 생성 후 `report_ready` 송신하고 연결 종료 |
| text | `{"type":"ping"}` | 15초 간격 keepalive (선택) |

### 6.2 BE → FE
| 내용 | 설명 |
|---|---|
| `{"type":"ready","session_id":"...","status":"created","fps_hint":3,"frame_size_hint":224}` | 연결·인증 성공 직후. `status`가 `running`이면 재연결 상황 |
| `{"type":"started","status":"running","question_index":0}` | `start`/`resume` 처리 후 |
| `{"type":"pong"}` | `ping` 응답 |
| `{"type":"result","ts_ms":12345,"face_found":true,"gaze":{"yaw":-3.2,"pitch":5.1,"state":"center"},"emotion":{"top":"중립","probs":{...}},"attention":{"top":"집중","probs":{...}}}` | 프레임마다. `attention`은 버퍼가 16장 차기 전까지 `null`. FE는 표시하지 않아도 됨(디버그 오버레이용) |
| `{"type":"event","ts_ms":15000,"event_type":"gaze_off","severity":"warn","icon":"👁️","message":"시선이 화면 밖으로 벗어났어요"}` | 판정 이벤트. FE `ToastStack` 형식과 동일한 `icon`, `message`. **표시 여부는 FE의 사용자 설정이 결정**한다. BE는 설정과 무관하게 항상 보내고 DB에 기록한다 |
| `{"type":"question_ack","index":1}` | 질문 전환 반영 확인. `index`는 **실제 적용된 번호** |
| `{"type":"report_ready","session_id":"..."}` | 종료 처리 완료. FE는 리포트 화면으로 이동 |
| `{"type":"error","code":"frame_decode_failed","message":"..."}` | 복구 가능한 오류. 연결 유지 |

### 6.3 close code
| code | 의미 |
|---|---|
| 1000 | 정상 종료 (`end` 처리 후) |
| 4401 | 토큰 검증 실패 |
| 4403 | 내 세션이 아님 |
| 4404 | 세션 없음 |
| 4409 | 이미 종료된 세션 또는 다른 연결이 점유 중 |

### 6.4 서버 처리 흐름 (프레임 1장)
```
JPEG bytes ─▶ InferenceClient.analyze(session_id, jpeg) ──HTTP──▶ 추론 서버 (/v1/analyze)
                        │                                             얼굴 검출 → L2CS → EmotionNet
                        ▼ 결과 JSON                                   → 버퍼에 크롭 추가 → (2초마다) Former-DFER
                  FrameResult(face, gaze, emotion, attention)
                        │
        ┌───────────────┼──────────────────┐
        ▼               ▼                  ▼
 analysis_logs 저장  RuleEngine.feed()  result 메시지 송신
                        │
                        ▼ Event 발생 시
                events 저장 + event 메시지 송신
```
추론 호출은 `httpx.AsyncClient`로 비동기 처리한다. 처리 중 새 프레임이 오면 **가장 최근 1장만 유지**하고 나머지는 버린다. 타임아웃 1초.

## 7. 분석 파이프라인과 인터페이스

### 7.0 BE 쪽: InferenceClient
BE는 모델을 모른다. 추론 서버를 호출하는 클라이언트 인터페이스와 그 Mock만 가진다.

```python
# analysis/types.py
EMOTION_LABELS = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]  # 모델 출력 순서 (EmotionNet 원본)
EMOTION_USED = {"중립", "불안", "당황", "기쁨"}  # 서비스에서 판정에 쓰는 4종. top이 여기 없으면 "기타"
ATTENTION_LABELS = ["집중", "졸림", "집중결핍", "집중하락", "태만"]  # Former-DFER. 인덱스 순서 확인 필요


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass(frozen=True)
class GazeResult:
    yaw_deg: float
    pitch_deg: float
    confidence: float


@dataclass(frozen=True)
class EmotionResult:
    probs: dict[str, float]
    top: str


@dataclass(frozen=True)
class AttentionResult:
    probs: dict[str, float]
    top: str
    window_ts_ms: tuple[int, int]


@dataclass(frozen=True)
class FrameResult:
    ts_ms: int
    face_found: bool | None  # None = 추론 실패
    face: FaceBox | None
    gaze: GazeResult | None
    emotion: EmotionResult | None
    attention: AttentionResult | None  # 버퍼 미충족 시 None, 이후엔 최근 추론값


# analysis/client.py
class InferenceClient(Protocol):
    async def analyze(self, session_id: str, jpeg: bytes) -> FrameResult: ...
    async def health(self) -> dict: ...
```

| 구현 | Phase | 동작 |
|---|---|---|
| `MockInferenceClient` | 1 | 항상 중앙 얼굴. yaw는 20초 주기 사인파, 감정은 중립 위주로 30초마다 불안 5초, 집중은 45초마다 집중하락 6초. 이벤트 규칙을 확실히 트리거 |
| `HttpInferenceClient` | 2 | `httpx.AsyncClient`로 5.5절 규격 호출. 타임아웃·재시도 없음(다음 프레임이 곧 옴) |

선택은 설정값 `INFERENCE_BACKEND=mock|http`. BE는 이미지를 열어 보지 않는다.

### 7.0.1 추론 서버 쪽: 모델 파이프라인
추론 서버 내부는 `FaceDetector → 모델 3개` 구조이고, 각 모델은 zip의 구조 코드를 이식해 가중치를 로드한다.

| 구성요소 | 전처리 (모델 스펙 기준, `01-dev-plan.md` D-5) |
|---|---|
| `MediaPipeFaceDetector` | short-range 모델. 가장 큰 얼굴 1개. 박스는 정사각형으로 보정 |
| `L2CSGaze` | 얼굴 박스에 **20% 여백**(7.1절) → RGB → 448×448 → ImageNet 정규화 → 90-bin softmax 기대값 ×4−180. GPU라 224 축소 불필요 |
| `EmotionNet` | 얼굴 박스 **여백 없음** → 흑백 → 48×48 INTER_AREA → /255 → log-softmax → exp. `torch.load(...)['model']`. 확률 7개를 **그대로** 반환. 4종 판정은 BE 몫 |
| `FormerDFER` | `session_id`별 버퍼의 112×112 크롭 16장 각각 RGB → /255 → `(1,16,3,112,112)` → fc(512,5) logit → softmax. `state_dict`의 `module.` 접두사 제거 후 로드. 얼굴이 3초 이상 없으면 버퍼 초기화 |

모델 로드는 lifespan에서 1회, `DEVICE=cuda|cpu` 설정. 세 모델 합계 GPU 메모리 약 1GB.

### 7.1 얼굴 크롭 규칙 (2026-09-04 확정)

AI-Hub 제공물(소스, Docker 데모 이미지 3종, README)을 모두 확인한 결과 **세 모델 모두 얼굴 검출·크롭 코드가 없다.** 검증 스크립트는 이미 잘린 얼굴 이미지를 `/data`에서 읽는다. 안구 데이터셋 샘플(`Sample (1).zip`)을 확인하니 원본은 1920×1080 전체 프레임이고 **라벨에 얼굴 박스가 없어** NIA 쪽도 검출기로 잘랐음이 확실하다. 따라서 크롭 규칙은 각 모델 원저자의 관례를 따르고, Phase 2에서 샘플의 `pose.head` 정답으로 여백 0/10/20/30%를 비교 측정해 확정한다 (`guideline/03-for-ai.md` 3.3절).

| 모델 | 크롭 규칙 | 근거 |
|---|---|---|
| L2CS-Net | 검출 박스에 상하좌우 **20% 여백** 후 정사각형, 448 리사이즈 | 원저자 `demo.py` 관례. NIA 데모 이미지가 같은 계열 검출기(RetinaFace, `face-detection` 패키지)를 설치함 |
| EmotionNet | 검출 박스 **여백 없음**, 48 흑백 | 학습 데이터가 어노테이터 박스를 그대로 자름. 제공 `video.py`도 Haar 박스를 그대로 사용 |
| Former-DFER | 검출 박스 **여백 없음**, 112 리사이즈 | 원저자는 DFEW의 정렬된 얼굴 클립 사용. NIA용 requirements에 `mediapipe`가 있어 MediaPipe 검출 박스로 잘랐을 가능성이 높음 |

검출기는 **MediaPipe Face Detection 1개**를 추론 서버가 공유하고, 박스 하나에서 여백만 달리해 세 입력을 만든다. 검출은 프레임당 1회다.

### 7.2 Former-DFER 클래스 인덱스
학습 라벨 파일(`validation.txt`)은 Docker 이미지 밖 `/data`에 있어 제공물에 없다. AI-Hub 표기 순서 **0 집중(F), 1 졸림(S), 2 집중결핍(D), 3 집중하락(A), 4 태만(N)** 을 가정한다. 샘플 데이터에 `condition=A`, `D` 프레임이 있으므로 Phase 2에서 모델에 넣어 켜지는 인덱스를 보고 두 클래스는 실측으로 확정한다. 순서가 다르면 `ATTENTION_LABELS` 상수만 바꾼다.

---

## 8. 이벤트 판정 규칙

`analysis/rules.py`의 `RuleEngine`이 세션마다 1개 생성되어 `FrameResult`를 순서대로 받는다. 모든 임계값은 `Settings`에서 읽는다.

| 이벤트 | 조건 | 지속 | 쿨다운 | severity | 토스트 (FE 문구 그대로) |
|---|---|---|---|---|---|
| `gaze_off` | `|yaw| > 20°` 또는 `|pitch| > 15°` | ≥ 3초 | 10초 | warn | 👁️ 시선이 화면 밖으로 벗어났어요 |
| `face_lost` | `face_found == false` | ≥ 3초 | 10초 | warn | 🙈 얼굴이 화면에서 보이지 않아요 |
| `emotion_negative` | top == 불안 이고 prob ≥ 0.5 (7클래스 원본 확률 기준) | ≥ 5초 | 15초 | warn | 🙂 표정에서 긴장이 감지됐어요 |
| `emotion_surprised` | top == 당황 이고 prob ≥ 0.6 | ≥ 2초 | 15초 | info | 😮 당황한 표정이 감지됐어요 |
| `attention_low` | attention top ∈ {졸림, 집중결핍, 집중하락, 태만} 이고 prob ≥ 0.5 | ≥ 6초 (추론 3회 연속) | 20초 | warn | 😴 집중이 흐트러진 것 같아요 |
| `gaze_stable` | gaze_state == center 연속 | ≥ 30초 | 60초 | info | ✅ 좋아요, 안정적인 시선이에요 |

토스트 표시는 사용자 선택이다(2026-09-04 결정). BE는 이 설정을 모르고 항상 이벤트를 발행·저장한다. 이유: 실시간 알림을 꺼도 종료 후 리포트에는 같은 이벤트가 있어야 하고, 설정을 BE에 두면 세션·API가 늘어난다. FE가 `localStorage` 등에 on/off를 저장하고 `event` 메시지를 받았을 때 표시만 건너뛴다.

판정 방식: 조건이 처음 참이 된 시각을 기억하고, 거짓이 되면 리셋. 지속 시간을 넘기면 이벤트 발행 후 쿨다운 동안 같은 타입을 억제. 프레임 누락(3fps 미만)에 견디도록 **프레임 수가 아니라 시각(ts_ms) 차이**로 계산한다.

`gaze_state` 산출: `center` if 임계값 이내, `off` if 초과, `unknown` if 얼굴 없음 또는 confidence < 0.3.

---

## 9. 리포트 집계 규칙

세션 종료 시 `report_service.build(session_id)`가 `analysis_logs`와 `events`를 읽어 5.6절 구조를 만들고 `reports`에 저장한다.

- 유효 프레임 = `face_found == true`. 비율 지표의 분모.
- `gaze_hold_rate` = center 프레임 / 유효 프레임.
- `stable_emotion_rate` = top ∈ {중립, 기쁨} 프레임 / 유효 프레임. top이 "기타"인 프레임은 분모에 포함(안정으로 치지 않음).
- `emotion_distribution` = 유효 프레임의 `emotion_top` 빈도 비율. 4종 + "기타" 5개 키. "기타"를 숨기지 않는 이유는 판정 밖 표정의 빈도 자체가 정보이기 때문.
- `attention_rate` = attention_top == 집중 프레임 / attention이 있는 프레임. `attention_distribution`도 같은 분모.
- 질문별 지표는 `session_questions`의 `started_at~ended_at` 구간으로 나눈다.
- `feedback`은 규칙 템플릿으로 생성한다 (LLM 미사용):
  - 질문별 `gaze_off` 2회 이상 → "초반 시선 이탈이 N회 감지됐어요. 답변 시작 전 카메라를 먼저 응시해보세요."
  - 질문별 부정 감정 지속 최댓값이 세션 최댓값 → "답변 중 긴장 표정이 가장 오래 지속됐어요. 호흡을 고르고 천천히 답해보세요."
  - 질문별 gaze_hold_rate와 stable_emotion_rate 모두 최고 → "시선 유지율과 표정 안정도 모두 가장 높았어요. 이 리듬을 기억해두세요."
  - 유효 프레임이 30% 미만 → "얼굴이 잘 보이지 않아 분석 신뢰도가 낮아요. 조명과 카메라 위치를 확인해보세요."
  - 질문별 attention_rate가 0.6 미만 → "이 질문에서 집중이 흐트러진 시간이 길었어요. 질문을 끝까지 듣고 잠시 생각한 뒤 답해보세요."

---

## 10. 설정값과 환경변수

`.env.example`에 추가:
```
DATABASE_URL=postgresql+psycopg://postgres.<ref>:<pw>@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres
# 비우면 로컬 SQLite(backend/dev.db)를 쓰고 앱 시작 시 테이블·seed 자동 생성 (2026-09-06 구현 시 추가: Supabase 비밀번호 없이도 FE 연동 가능)
INFERENCE_BACKEND=mock            # mock | http
INFERENCE_URL=http://localhost:9000   # BE와 추론 서버가 같은 노트북이면 localhost
INFERENCE_TOKEN=<공유 시크릿>
INFERENCE_TIMEOUT_MS=1000
GAZE_YAW_THRESHOLD_DEG=20
GAZE_PITCH_THRESHOLD_DEG=15
GAZE_OFF_SECONDS=3
FACE_LOST_SECONDS=3
EMOTION_NEGATIVE_SECONDS=5
EMOTION_NEGATIVE_MIN_PROB=0.5
ATTENTION_WINDOW_FRAMES=16
ATTENTION_INTERVAL_SECONDS=2
ATTENTION_LOW_SECONDS=6
ATTENTION_LOW_MIN_PROB=0.5
SESSION_IDLE_TIMEOUT_SECONDS=300
```

추론 서버 `inference/.env.example`:
```
DEVICE=cuda                       # cuda | cpu
MODEL_DIR=./weights
INFERENCE_TOKEN=<BE와 동일한 공유 시크릿>
MAX_IMAGE_BYTES=1048576
```

### 10.1 추론 서버 실행: Docker 필수 (2026-09-04 결정)
추론 서버는 개발 중에도 항상 Docker 컨테이너로 띄운다. BE와 FE는 네이티브로 실행한다.

| 항목 | 내용 |
|---|---|
| 사전 준비 (1회) | Windows에 **Docker Desktop** 설치 → Settings → Resources → WSL Integration에서 사용 중인 배포판 켜기. GPU는 Windows NVIDIA 드라이버만 있으면 `--gpus all`로 바로 사용됨 (WSL 안에 CUDA 툴킷·Container Toolkit 별도 설치 불필요) |
| 이미지 | `inference/Dockerfile`. 베이스 `pytorch/pytorch:2.x-cuda12.x-cudnn-runtime`. 의존성(`einops`, `mediapipe`, `fastapi`, `uvicorn`, `opencv-python-headless`)만 이미지에 굽고 **소스와 가중치는 굽지 않는다** |
| 소스·가중치 | `compose.yaml`에서 볼륨 마운트: `./app → /app/app`, `./weights → /app/weights`. 코드 수정은 컨테이너 안 `uvicorn --reload`가 즉시 반영. 가중치는 git 밖에 두고 폴더째 마운트 |
| 실행 | `cd inference && docker compose up` (처음 한 번 `--build`). 포트 9000, `0.0.0.0` 바인딩 |
| 확인 | `curl localhost:9000/v1/health` → `device: cuda` |
| CPU 폴백 | GPU 없는 팀원 PC: `.env`에 `DEVICE=cpu`, compose의 gpu 예약 블록만 주석 처리 |

`inference/compose.yaml` 골격:
```yaml
services:
  inference:
    build: .
    ports: ["9000:9000"]
    env_file: .env
    volumes:
      - ./app:/app/app
      - ./weights:/app/weights
    command: uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Docker를 필수로 한 이유: 팀원 누구의 PC에서든 같은 PyTorch·CUDA 조합으로 같은 결과가 나와야 모델 검증 결과를 믿을 수 있다. 나중에 배포하게 되면 이 이미지를 그대로 쓴다.

### 10.1.1 Phase 2 구축 중 확인된 것 (2026-09-07)
| 항목 | 내용 |
|---|---|
| 가중치 호환 | 세 모델 모두 PyTorch 2.4 에서 로드됨. Former-DFER 체크포인트는 학습 스크립트의 `runner_helper.RecorderMeter` 객체를 함께 pickle 해 두어, 로드 시 더미 모듈을 등록해 통과시킴 (`predictors._stub_training_modules`) |
| 시선 확신도 | 90-bin softmax 최댓값은 정상 입력에서도 0.2~0.3 이라 BE 임계값(0.3)에 걸림 → **상위 3개 bin 확률 합의 평균**으로 정의 변경. 정상 입력에서 0.65~0.82 |
| 지연 | GPU(RTX 4060) 기준 640px JPEG 1장: 검출 5~9ms + 시선 9~12ms + 감정 3~5ms, 왕복 17~30ms. 3fps 에 여유 큼 |
| compose env | `env_file` 은 빈 값 줄의 뒤 주석을 값으로 읽음 → `.env` 주석은 별도 줄에 |
| WSL 포트 | Docker 가 게시한 포트는 WSL `localhost` 로도 열린다. 단, 컨테이너 기동 시 WSL 안에서 그 포트를 다른 프로세스가 쓰고 있으면 WSL 쪽 게시가 빠지므로 `ss -ltn` 으로 확인하고 비운 뒤 `docker compose up -d --force-recreate`. (2026-09-07 다른 프로젝트 Jupyter 커널이 9000 점유 → 종료 후 `localhost:9000` 정상) |
| 미검증 | 얼굴 크롭 여백 실측(안구 샘플 라벨 `pose.head` 단위 미확인), Former-DFER 인덱스 순서, 감정 모델 정확도 |

### 10.2 같은 네트워크(LAN)에서 팀원 접속
목표: 노트북에서 세 서버를 띄우고, 같은 Wi-Fi의 팀원 PC 브라우저가 노트북 IP로 접속해 면접 흐름 전체를 쓴다. 인터넷 공개·TLS·터널 없음.

| 단계 | 내용 |
|---|---|
| 1. WSL 네트워크 | 노트북은 Windows 11 24H2라 **mirrored 모드** 사용 가능. `C:\Users\<사용자>\.wslconfig`에 `[wsl2]` 아래 `networkingMode=mirrored`를 넣고 PowerShell에서 `wsl --shutdown` 후 재시작. WSL이 Windows와 같은 IP를 쓰게 되어 LAN에서 WSL 포트에 직접 접근됨 |
| 1-대안 | mirrored가 안 되면 관리자 PowerShell에서 포트 전달: `netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL IP>` (3000, 9000도 동일). WSL IP는 WSL에서 `hostname -I` |
| 2. Windows 방화벽 | 3000, 8000, 9000 인바운드 허용: `netsh advfirewall firewall add rule name="selffit" dir=in action=allow protocol=TCP localport=3000,8000,9000` |
| 3. 서버 바인딩 | BE `uvicorn app.main:app --host 0.0.0.0 --port 8000`, FE `npm run dev -- -H 0.0.0.0`. 추론 서버는 compose가 이미 `0.0.0.0:9000` |
| 4. 주소 설정 | 노트북 IP를 `ipconfig`(Wi-Fi IPv4)로 확인. 팀원 FE `.env.local`의 `NEXT_PUBLIC_API_URL=http://<노트북IP>:8000`. BE `.env`의 `CORS_ORIGINS`에 팀원 FE 주소 추가. `INFERENCE_URL`은 같은 노트북이므로 `localhost:9000` 유지 |
| 5. 확인 | 팀원 PC에서 `http://<노트북IP>:8000/health` → 200, 자기 FE의 `/dev/connect` → ✅ |

주의 두 가지:
- 노트북 IP는 Wi-Fi 재접속 때 바뀔 수 있다. 시연 전 확인하고 `.env.local`을 맞춘다.
- **브라우저는 `localhost`가 아닌 http 주소에서 웹캠을 막는다.** 그래서 팀원은 FE를 **자기 PC에서 localhost로 띄우고 BE 주소만 노트북 IP로** 잡는 방식을 권장한다. 노트북 FE에 직접 접속해 웹캠까지 쓰려면 Chrome `chrome://flags/#unsafely-treat-insecure-origin-as-secure`에 `http://<노트북IP>:3000`을 등록해야 한다.

### 10.3 (참고) 나중에 배포할 때 WebSocket 설계가 받는 제약

지금은 로컬이라 해당 없음. 배포 결정 시 다시 본다.
| 환경 | 제약 | 대응 |
|---|---|---|
| Render 무료 (BE) | CPU 0.1코어, RAM 512MB, 단일 인스턴스, 15분 유휴 시 슬립 | BE는 이미지를 열지 않고 바이트만 전달. 판정 상태는 메모리, 로그는 DB. 시연 전 `/health`로 깨움 |
| Render 리전 | 한국 사용자·한국 노트북 기준 **싱가포르**가 가장 가까움 | 싱가포르 선택. 미국 리전은 왕복이 2배 |
| Cloudflare Tunnel (BE → 추론 서버) | 프레임당 인터넷 왕복 100~150ms 추가 | 3fps에서는 허용. 최근 1장만 유지하는 백프레셔로 밀림 방지 |
| Vercel (FE) https | 혼합 콘텐츠 차단 | `wss://` 필수. Render TLS 기본 제공 |
| 슬립·재시작 시 WS 끊김 | 면접 중 연결 유실 | FE가 같은 세션 ID로 **재연결**. BE는 DB의 세션 상태로 이어 감 (FE 가이드 4절) |

프레임 1장 왕복 추정: 브라우저↔Render 70~100ms + 터널 100~150ms + GPU 20~40ms ≈ **250~350ms**. 목표 500ms 이내지만 실측 필요. 지연이 문제면 시연 한정으로 BE도 노트북에서 띄워 터널 구간을 없앤다 (코드 변경 없음).

## 11. 테스트 전략

| 대상 | 방법 |
|---|---|
| `RuleEngine` | 순수 함수 단위 테스트. 시각을 직접 주입해 지속·쿨다운 경계 검증 (가장 촘촘히) |
| `report_service` | 고정 로그 fixture → 기대 리포트 JSON 비교 |
| REST | `TestClient` + Supabase 테스트용 스키마 + 인증 의존성 오버라이드 |
| WebSocket | `TestClient.websocket_connect`로 start → 바이너리 프레임 N장 → question → end 시나리오. Mock Predictor 사용 |
| `HttpInferenceClient` | 가짜 추론 서버(작은 FastAPI 앱, ASGITransport)로 정상·얼굴없음·5xx·깨진 응답 파싱 검증. 타임아웃은 LiveSession 경로로 검증 (`tests/test_inference_failure.py`) |
| 계약 테스트 | REST 응답·리포트 JSON·WS 메시지의 필드 집합을 이 문서와 대조 (`tests/test_contracts.py`). FE 가 의존하는 형태가 바뀌면 여기서 실패 |
| CI | `.github/workflows/backend.yml`. `backend/` 변경이 있는 push·PR 에서 ruff + pytest |
| 추론 서버 (Phase 2) | 샘플 이미지에 대해 원본 검증 스크립트와 출력 일치 확인. `scripts/bench.py`로 GPU/CPU 추론 시간 측정. `/v1/analyze` 계약 테스트 |

---

## 12. FE 협의 항목

FE 담당자 확인이 필요한 결정. 답이 없으면 괄호 값으로 진행.

1. 프레임 전송: 캔버스에서 224px로 축소 후 JPEG 품질 70, 3fps로 바이너리 전송 (가정: 가능)
2. 질문 전환을 WebSocket 제어 메시지로 보내는 것 (가정: 가능)
3. 토스트 메시지의 `icon`, `message`를 서버가 내려주는 것. 현재 FE 하드코딩 4종과 문구 동일 (가정: 서버 값 사용)
3-1. 토스트 on/off 토글은 FE가 구현하고 설정은 FE에만 저장. 꺼진 동안 받은 `event`는 화면에만 안 띄우고 버림 (가정: 단일 on/off, 종류별 토글은 P2)
4. 리포트 화면이 5.6절 JSON을 그대로 렌더링 (가정: 가능. `feedback`은 배열 길이 가변)
5. 면접 시작 전 로그인 필수 여부. 세션은 `user_id`가 있어야 생성됨 (가정: 로그인 필수, 로그인 UI는 FE)
6. `result` 메시지를 화면에 쓸지 (가정: MVP는 무시, 디버그 오버레이는 P2)
7. 집중 상태 토스트 "😴 집중이 흐트러진 것 같아요" 추가와 리포트의 집중도 지표(`attention_rate`) 표시 (가정: 토스트는 서버 값 그대로 표시, 리포트에는 "집중 유지율" 타일 추가)

---

## 13. 구현 순서

| # | 작업 | 산출물 | 완료 기준 |
|---|---|---|---|
| 1 | DB 기반 | `db.py`, `models.py`, `db/migrations/001_init.sql`, questions seed | **완료 (2026-09-06).** SQLite 로컬 검증. Supabase 적용은 대기 |
| 2 | 분석 타입·클라이언트·Mock | `analysis/types.py, client.py, mock.py` | **완료.** Mock 은 20초 주기 시선 이탈, 30초마다 불안, 45초마다 집중하락, 70초에 얼굴 사라짐 |
| 3 | 판정 엔진 | `analysis/rules.py` | **완료.** 지속·쿨다운·리셋 경계 테스트 5개 통과 |
| 4 | 세션 REST | `routers/sessions.py`, `services/session_service.py` | **완료.** 생성·목록·상세·종료·삭제·이벤트, 소유권 검사 테스트 통과 |
| 5 | WebSocket | `routers/ws.py`, `services/live_session.py` | **완료.** start→프레임→question→end 시나리오, 중복 연결·종료 세션 거부, 잘못된 입력 시 연결 유지 테스트 통과 |
| 6 | 리포트 | `services/report_service.py`, `routers/reports.py` | **완료 (기본).** 5.6절 구조 전부 생성. 집계값 정밀 검증 테스트는 추가 예정 |
| 7 | FE 통합 (Mock) | FE와 함께 실제 웹캠 시연 | 토스트·리포트 확인 |
| 8 | 추론 서버 (Phase 2) | Docker Desktop 설치(선행), `inference/` 프로젝트, Dockerfile + compose, 모델 3개 이식, `/v1/*` | **완료 (2026-09-07).** `/v1/health`가 `cuda`, 샘플 추론 성공, 640px 프레임 기준 왕복 17~27ms. 단위 테스트 7개 |
| 9 | 실제 연결 (Phase 2) | `HttpInferenceClient`, `INFERENCE_BACKEND=http` | 진행 중. BE `.env` 전환 완료, 실제 웹캠 확인 대기 |

각 작업은 `dev`에서 브랜치를 따 PR로 합친다. 작업 1~3은 서로 독립이라 병렬 가능.
