from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    device: str = Field(default="cuda", description="cuda | cpu")
    model_dir: str = Field(default="./weights")
    inference_token: str = Field(default="", description="비우면 인증 생략 (로컬 개발)")
    max_image_bytes: int = Field(default=1_048_576)

    gaze_weights: str = "l2cs_v1.pkl"
    # 감정 모델 (AI 팀 납품 v2, ai/models/deliverable/meta.json 기준). v1(원본 7클래스)로 되돌리려면
    # EMOTION_WEIGHTS=emotionnet_v1.pth EMOTION_LABELS=기쁨,당황,분노,불안,상처,슬픔,중립 EMOTION_BIAS= EMOTION_TAU=0
    emotion_weights: str = "emotionnet_v2.pth"
    emotion_labels: str = "기쁨,당황,불안,중립"  # 모델 출력 인덱스 순서
    emotion_bias: str = "0.0,1.4,0.0,1.9"  # 로그확률에 더함 (meta.json bias). 비우면 0
    emotion_tau: float = 0.98  # softmax 최댓값이 이 값 이상일 때만 채택. 0 이면 항상 채택
    emotion_resize: str = "linear"  # linear(v2 학습과 동일) | area(원본 video.py)
    attention_weights: str = "former_dfer_v1.pth"

    gaze_input_size: int = 448  # L2CS 검증 스크립트 기준. 224 로 줄이면 빠르지만 정확도 측정 필요
    gaze_crop_margin: float = 0.20  # 원저자 데모 관례 (03-phase1-design.md 7.1)
    attention_window: int = 16
    attention_interval_s: float = 2.0
    attention_reset_s: float = 3.0  # 얼굴이 이 시간 이상 없으면 버퍼 초기화
    session_ttl_s: float = 600.0  # 미사용 세션 버퍼 정리

    # STT (faster-whisper). 팀 결정: small, 한국어
    stt_model: str = Field(default="small", description="tiny | base | small | medium")
    stt_language: str = Field(default="ko")
    stt_max_audio_bytes: int = Field(default=50_000_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
