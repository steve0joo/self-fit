"""SelfFit 추론 서버. 모델 3개(시선·감정·집중)를 한 프로세스에 올리고 /v1/analyze 로 제공한다.
규격: backend/docs/03-phase1-design.md 5.5절."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app import predictors
from app.attention_buffer import AttentionBuffer
from app.config import get_settings
from app.detector import FaceDetector
from app.routers import infer
from app.stt import SttModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inference")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    device = s.device
    if device == "cuda" and not predictors.cuda_available():
        log.warning("CUDA 를 쓸 수 없어 CPU 로 전환합니다.")
        device = "cpu"
    wd = Path(s.model_dir)
    app.state.gaze = predictors.GazeModel(wd / s.gaze_weights, device)
    labels = [x.strip() for x in s.emotion_labels.split(",") if x.strip()]
    bias = [float(x) for x in s.emotion_bias.split(",") if x.strip()] if s.emotion_bias.strip() else None
    app.state.emotion = predictors.EmotionModel(wd / s.emotion_weights, device, labels, bias, s.emotion_tau)
    app.state.attention = predictors.AttentionModel(wd / s.attention_weights, device)
    app.state.detector = FaceDetector()
    app.state.attention_buffer = AttentionBuffer(s.attention_window, s.attention_interval_s, s.attention_reset_s, s.session_ttl_s)
    predictors.warmup(app.state.gaze, app.state.emotion, app.state.attention, s.gaze_input_size)
    app.state.stt = SttModel(s.stt_model, device, s.stt_language)
    app.state.models_loaded = True
    log.info("models loaded on %s: gaze=%s emotion=%s attention=%s stt=%s", device, app.state.gaze.name, app.state.emotion.name, app.state.attention.name, app.state.stt.name)
    yield
    app.state.detector.close()


app = FastAPI(title="SelfFit Inference", version="0.1.0", lifespan=lifespan)
app.include_router(infer.router)
