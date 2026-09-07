# 04. 운영 규칙: 깃, 로컬 실행, 문서

이 작업을 이어받거나 함께 하는 사람이 지켜야 할 규칙입니다. 팀 합의 사항(깃 전략)과 백엔드가 정한 실행 방법을 담았습니다.

## 깃
- **`dev`에서 모든 기능을 구현하고, 마지막에 `main`으로 합칩니다.** `main`에는 직접 커밋하지 않습니다.
- 브랜치 이름: `파트/작업` 예) `backend/session-api`, `frontend/login`, `ai/gaze-eval`
- 흐름:
  ```bash
  git switch dev && git pull
  git switch -c backend/session-api
  # 작업 후
  git add backend/            # 자기 파트 폴더만
  git commit -m "feat(backend): 세션 생성 API"
  git push -u origin backend/session-api
  # GitHub에서 dev로 PR
  ```
- 커밋 메시지: `feat|fix|docs|chore(파트): 내용`. 한 커밋에 한 가지.
- 다른 파트 폴더를 건드려야 하면 그 담당자에게 먼저 말합니다. 루트 파일(`.gitignore`, `README.md`)도 마찬가지.
- `git push --force`는 공유 브랜치에 쓰지 않습니다.

## 절대 커밋하면 안 되는 것
`.env`, `.env.local`, 모델 가중치, 데이터셋, zip. 루트 `.gitignore`가 막고 있지만 `git status`로 한 번 확인하는 습관.

## 백엔드 로컬 실행
```bash
cd backend
cp .env.example .env          # 값은 팀 채널에서
uv sync
uv run uvicorn app.main:app --reload --port 8000
```
- API 문서: http://localhost:8000/docs
- 테스트: `uv run pytest` / 린트: `uv run ruff check .`
- `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`는 도구가 만드는 캐시입니다. 지워도 되고 커밋되지 않습니다.

## 추론 서버 실행 (Phase 2 이후, Docker 필수)
Docker Desktop을 설치하고 WSL Integration을 켠 뒤:
```bash
cd inference
cp .env.example .env
docker compose up --build     # 처음 한 번. 이후엔 docker compose up
curl localhost:9000/v1/health # device: cuda 면 정상
```
소스와 가중치는 볼륨으로 붙어 있어 코드를 고치면 컨테이너 안에서 바로 반영됩니다. 이미지를 다시 빌드하는 건 의존성이 바뀔 때뿐입니다. GPU가 없는 PC는 `.env`에 `DEVICE=cpu`로 두고 `compose.yaml`의 gpu 블록을 주석 처리하면 느리지만 동작합니다.

## 문서 위치
| 위치 | 내용 |
|---|---|
| `README.md` (루트) | 전체 플로우, 시작하기 |
| `backend/guideline/` | 인수인계·보고 문서 (이 폴더). `00-summary.md`부터 |
| `backend/docs/01-dev-plan.md` | 백엔드 기획: 범위, 결정 사항, 리스크 |
| `backend/docs/02-phase0-setup.md` | 기반 구축과 연결 검증 기록 |
| `backend/docs/03-phase1-design.md` | 상세 설계: API, WebSocket, DB, 판정 규칙 |

## 백엔드 작업 방식
기획 → 설계 → 리뷰 → QA → 문서 확인 → 커밋 순서로 갑니다. 코드와 문서가 다르면 코드가 맞고, 문서는 바로 고칩니다.
