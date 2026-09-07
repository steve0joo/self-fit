# SelfFit

웹캠 기반 모의 면접 자가진단 서비스. 면접 중 **시선·표정·집중 상태**를 AI가 실시간으로 분석해 토스트로 알려 주고, 종료 후 행동 리포트를 제공합니다.

경기 AI 멤버십 채용연계형 교육 1차 프로젝트 (2026-09).

## 사용자 흐름
```
로그인 → 질문 목록 확인 → 면접 시작(웹캠) → 질문별 답변 → 종료 → 리포트
                                   │
                          시선 이탈·긴장 표정·집중 저하 시 토스트 알림 (사용자가 켜고 끌 수 있음)
```

## 시스템 흐름
```
 브라우저                    노트북 (로컬)                  노트북 GPU
┌────────────┐  로그인   ┌────────────┐
│ FE Next.js │─────────▶│ Supabase   │  인증 + PostgreSQL
│            │◀─────────│            │
│ 웹캠 3fps  │  토큰     └─────┬──────┘
│            │  REST/WS ┌─────┴──────┐  JPEG 1장  ┌─────────────┐
│            │─────────▶│ BE FastAPI │──────────▶│ 추론 서버    │ 얼굴 검출
│            │◀─────────│ 세션·판정  │◀──────────│ 모델 3개    │ L2CS-Net (시선)
└────────────┘ 토스트    │ 리포트     │  결과 JSON └─────────────┘ EmotionNet (감정)
               리포트    └────────────┘                            Former-DFER (집중)
```

## 폴더
| 폴더 | 담당 | 내용 |
|---|---|---|
| `frontend/` | FE | Next.js 14. 랜딩, 면접, 리포트, 업로드 화면 |
| `backend/` | BE | FastAPI. 인증, 세션, WebSocket, 판정, 리포트 |
| `inference/` | BE | 추론 서버 (Phase 2에서 생성). 모델 3개 + 얼굴 검출. **Docker로 실행** |
| `ai/` | AI | 모델 학습·검증 코드 |

## 시작하기
| 파트 | 명령 |
|---|---|
| FE | `cd frontend && cp .env.local.example .env.local && npm install && npm run dev` → http://localhost:3000 |
| BE | `cd backend && cp .env.example .env && uv sync && uv run uvicorn app.main:app --reload --port 8000` → http://localhost:8000/docs |
| 추론 서버 (Phase 2 이후) | Docker Desktop 설치 후 `cd inference && docker compose up --build` → http://localhost:9000/v1/health |
| 연결 확인 | 둘 다 띄운 뒤 http://localhost:3000/dev/connect |
| 같은 Wi-Fi 팀원 | 각자 PC에서 FE만 띄우고 `.env.local`의 `NEXT_PUBLIC_API_URL`을 `http://<백엔드 노트북IP>:8000`으로. 절차는 `backend/docs/03-phase1-design.md` 10.2절 |

`.env` 값(Supabase URL, 키)은 팀 채널에서 받습니다. 저장소에 올리지 않습니다.

## 협업 규칙 요약
- 모든 작업은 `dev` 브랜치 기준. 브랜치는 `파트/작업` 이름으로 따고 PR은 `dev`로.
- 자기 파트 폴더만 커밋. 루트 파일과 다른 파트 폴더는 담당자와 먼저 상의.
- 모델 파일, 데이터셋, `.env`는 커밋 금지 (`.gitignore`가 막고 있음).

## 문서
| 읽을 사람 | 문서 |
|---|---|
| 전원 | [backend/guideline/00-summary.md](backend/guideline/00-summary.md) 백엔드 설계 요약 보고: 결정 사항, 현재 상태, 남은 일 |
| 전원 | [backend/guideline/01-architecture.md](backend/guideline/01-architecture.md) 설계 개요 |
| FE | [backend/guideline/02-for-frontend.md](backend/guideline/02-for-frontend.md) API·WebSocket 사용법 |
| AI | [backend/guideline/03-for-ai.md](backend/guideline/03-for-ai.md) 모델 실행 위치, 납품 규약 |
| 전원 | [backend/guideline/04-dev-workflow.md](backend/guideline/04-dev-workflow.md) 깃·로컬 실행 |
| BE | [backend/docs/](backend/docs/) 기획·설계 상세 |

## 진행 단계
| 단계 | 내용 | 상태 |
|---|---|---|
| Phase 0 | BE 골격, 로그인 연동, FE-BE-Supabase 연결 | 완료 |
| Phase 1 | 세션·WebSocket·판정·리포트를 Mock으로 완성 | 설계 완료, 구현 예정 |
| Phase 2 | 추론 서버, 실제 모델 연결 | 예정 |
| Phase 3 | 영상 업로드 분석 | 예정 |
| 배포 | 실제 배포는 하지 않음 (2026-09-04 결정). 모든 서비스를 노트북 한 대에서 로컬 실행 | 보류 |

## 참고
- 데이터·모델: AI-Hub [디스플레이 중심 안구 움직임 영상 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71421), [한국인 감정인식을 위한 복합 영상](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=82)
- 실행 환경: 노트북 한 대에서 FE(3000), BE(8000), 추론 서버(9000) 로컬 실행. 배포는 보류
