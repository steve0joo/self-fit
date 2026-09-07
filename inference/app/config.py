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
    emotion_weights: str = "emotionnet_v2.pth"
    # 가중치와 같은 디렉터리에 함께 배포한다. 클래스 순서·bias·tau 가 여기서 온다 (06-backend-handoff.md 2절)
    emotion_meta: str = "emotionnet_v2.meta.json"
    attention_weights: str = "former_dfer_v1.pth"

    gaze_input_size: int = 448  # L2CS 검증 스크립트 기준. 224 로 줄이면 빠르지만 정확도 측정 필요
    gaze_crop_margin: float = 0.20  # 원저자 데모 관례 (03-phase1-design.md 7.1)
    attention_window: int = 16
    attention_interval_s: float = 2.0
    attention_reset_s: float = 3.0  # 얼굴이 이 시간 이상 없으면 버퍼 초기화
    session_ttl_s: float = 600.0  # 미사용 세션 버퍼 정리


@lru_cache
def get_settings() -> Settings:
    return Settings()
