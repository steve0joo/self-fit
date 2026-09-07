# 아키텍처 (AI 파트)

## 모델이 실행되는 곳
AI 팀은 **학습만** 한다. 추론은 백엔드가 만든 **추론 서버(Docker, 노트북 GPU)**에서 돈다.

```
BE :8000 ──HTTP: JPEG 1장──▶ 추론 서버 :9000 ──▶ 결과 JSON
                              ├─ MediaPipe 얼굴 검출 (1회)
                              ├─ L2CS-Net    → yaw, pitch
                              ├─ EmotionNet  → 감정 7클래스
                              └─ Former-DFER → 집중 5클래스 (16장 모아 2초마다)
```
세 모델은 이미지 1개·프로세스 1개에 함께 올라가고, 얼굴 검출은 프레임당 1회만 해서 세 모델이 나눠 쓴다.

## 전처리 규약 (모델별)
| | L2CS-Net | EmotionNet | Former-DFER |
|---|---|---|---|
| 입력 | 448×448 RGB, ImageNet 정규화 | 48×48 흑백, 0~1 | 112×112 RGB 16장, 0~1 |
| 크롭 여백 | 박스 + 20% (가정, Phase 2 실측) | 없음 | 없음 (가정) |
| 출력 | yaw, pitch 각도 | 7클래스 확률 | 5클래스 확률 |
| 가중치 | `l2cs_trained.pkl` 96MB | `model.pth` 57MB | `former_trained.pth` 144MB |

가중치가 작아 보이는 건 정상이다(파라미터 수 × 4바이트). AI-Hub Docker 이미지가 11GB였던 건 OS·conda·PyTorch 때문이다. 상세: [`../../backend/guideline/03-for-ai.md`](../../backend/guideline/03-for-ai.md) 2절.

## ai/ 폴더 구조
```
ai/
├── README.md          # 파트 홈, 할 일
├── CLAUDE.md          # 코딩 규칙
├── pyproject.toml     # uv, Python 3.12
├── docs/              # PRD, ARCHITECTURE, ADR
├── scripts/           # 보조 스크립트
├── data/       (git 제외) 데이터셋 원본
├── notebooks/  (git 제외) 실험 노트북
├── runs/       (git 제외) 학습 로그·체크포인트
└── weights/    (git 제외) 산출 가중치 → inference/weights/ 로 납품
```
`data/ notebooks/ runs/ weights/` 는 커밋하지 않는다(`ai/.gitignore`).
아래 넷은 아직 없는 **권장 구조**이며 작업 시작할 때 만든다.

## 납품 흐름
```
파인튜닝 → 가중치 파일(버전 파일명) + 전처리/출력 명세 + 성능표
   → GitHub Release나 Drive로 전달 (git 커밋 금지)
   → 백엔드가 inference/weights/ 에 넣고 .env 로 교체
```
추론 서버는 파일·명세만 맞으면 코드 수정 없이 모델을 갈아끼운다. 상세 규약: [`../../backend/guideline/03-for-ai.md`](../../backend/guideline/03-for-ai.md) 5절.
