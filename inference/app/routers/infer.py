import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool

from app import preprocess as pp
from app.auth import require_token
from app.config import Settings, get_settings
from app.predictors import top_of
from app.schemas import AnalyzeOut, EmotionOut, FaceOut, GazeOut, ProbsOut

router = APIRouter(prefix="/v1", tags=["inference"])


@router.get("/health")
def health(request: Request, settings: Annotated[Settings, Depends(get_settings)]):
    st = request.app.state
    loaded = getattr(st, "models_loaded", False)
    if not loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "모델이 아직 로드되지 않았습니다.")
    return {
        "status": "ok",
        "device": settings.device,
        "models": {"gaze": st.gaze.name, "emotion": st.emotion.name, "attention": st.attention.name, "stt": getattr(getattr(st, "stt", None), "name", None)},
    }


@router.post("/analyze", response_model=AnalyzeOut, dependencies=[Depends(require_token)])
async def analyze(request: Request, session_id: str = Query(..., min_length=1), settings: Settings = Depends(get_settings)):
    st = request.app.state
    if not getattr(st, "models_loaded", False):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "모델이 아직 로드되지 않았습니다.")
    body = await request.body()
    if len(body) > settings.max_image_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"이미지가 {settings.max_image_bytes} bytes 를 넘습니다.")
    bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "이미지를 디코딩할 수 없습니다 (JPEG/PNG).")

    timing: dict[str, int] = {}
    t = time.perf_counter()
    face = st.detector.detect(bgr)
    timing["detect"] = int((time.perf_counter() - t) * 1000)
    if face is None:
        st.attention_buffer.push(session_id, None)
        return AnalyzeOut(face_found=False, timing_ms=timing)

    t = time.perf_counter()
    yaw, pitch, conf = st.gaze.predict(pp.gaze_tensor(pp.crop(bgr, face, settings.gaze_crop_margin), settings.gaze_input_size))
    timing["gaze"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    face_tight = pp.crop(bgr, face, 0.0)
    emo, emo_top, emo_ok, emo_conf = st.emotion.predict(pp.emotion_tensor(face_tight, settings.emotion_resize))
    timing["emotion"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    ready, frames = st.attention_buffer.push(session_id, pp.attention_frame(face_tight))
    if ready:
        att = st.attention.predict(pp.attention_tensor(frames))
        st.attention_buffer.set_result(session_id, att)
    att_last = st.attention_buffer.last(session_id)
    timing["attention"] = int((time.perf_counter() - t) * 1000)

    return AnalyzeOut(
        face_found=True,
        face=FaceOut(x=face.x, y=face.y, w=face.w, h=face.h, score=round(face.score, 3)),
        gaze=GazeOut(yaw_deg=round(yaw, 2), pitch_deg=round(pitch, 2), confidence=round(conf, 3)),
        emotion=EmotionOut(probs=emo, top=emo_top, accepted=emo_ok, confidence=round(emo_conf, 4)),
        attention=None if att_last is None else ProbsOut(probs=att_last, top=top_of(att_last)),
        timing_ms=timing,
    )


@router.post("/transcribe", dependencies=[Depends(require_token)])
async def transcribe(request: Request, language: str | None = Query(None), settings: Settings = Depends(get_settings)):
    """오디오(webm/wav/mp3, ffmpeg 가 읽는 형식) → {text, segments[], duration_ms, speech_ms}. guideline/05 STT."""
    st = request.app.state
    if not getattr(st, "stt", None):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "STT 모델이 로드되지 않았습니다.")
    body = await request.body()
    if not body:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "오디오가 비어 있습니다.")
    if len(body) > settings.stt_max_audio_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "오디오가 너무 큽니다. 질문 구간별로 나눠 보내세요.")
    t = time.perf_counter()
    try:
        result = await run_in_threadpool(st.stt.transcribe, body, language)
    except Exception as e:  # noqa: BLE001 - 디코딩 실패 등은 400 으로
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"오디오를 처리할 수 없습니다: {e!s}"[:200]) from e
    result["timing_ms"] = int((time.perf_counter() - t) * 1000)
    return result
