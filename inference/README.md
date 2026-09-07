# SelfFit Inference Server

모델 3개(시선 L2CS-Net, 감정 EmotionNet, 집중 Former-DFER)와 얼굴 검출(MediaPipe)을 한 컨테이너에 올린 추론 서버.
백엔드(`backend/`)가 `POST /v1/analyze` 로 JPEG 1장을 보내면 결과 JSON 을 돌려준다. 규격은 `backend/docs/03-phase1-design.md` 5.5절.

## 실행 (Docker 필수)
```bash
cd inference
cp .env.example .env
# weights/ 에 가중치 3개 (l2cs_v1.pkl, emotionnet_v1.pth, former_dfer_v1.pth) — 팀 채널에서 받기
docker compose up --build        # 처음 한 번. 이후 docker compose up
curl localhost:9000/v1/health    # {"status":"ok","device":"cuda",...}
python scripts/smoke.py          # 샘플 이미지로 추론·지연 확인
```
소스(`app/`)는 볼륨으로 붙어 있어 수정하면 컨테이너 안 `--reload` 가 바로 반영한다. 이미지를 다시 빌드하는 건 `requirements.txt` 가 바뀔 때뿐.

GPU 가 없는 PC: `.env` 의 `DEVICE=cpu`, `compose.yaml` 의 `deploy:` 블록 주석 처리.

## 구조
```
app/main.py            모델 로드(lifespan), 워밍업
app/routers/infer.py   GET /v1/health, POST /v1/analyze
app/detector.py        MediaPipe 얼굴 검출 → 정사각형 박스
app/preprocess.py      모델별 크롭·리사이즈·정규화 (학습 조건과 동일)
app/predictors.py      모델 3개 래퍼, 후처리, 클래스 라벨
app/attention_buffer.py 세션별 16장 버퍼, 2초 주기
app/models/            AI-Hub 베이스라인 구조 코드 원본 이식
weights/               가중치 (git 제외)
```

## 백엔드와 연결
`backend/.env`: `INFERENCE_BACKEND=http`, `INFERENCE_URL=http://localhost:9000`, `INFERENCE_TOKEN` 은 양쪽 동일.
