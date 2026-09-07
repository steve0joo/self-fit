from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수로 주입되는 설정. 값 목록은 .env.example 참고."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="development", description="development | production | test")
    supabase_url: AnyHttpUrl = Field(description="Supabase Project URL")
    supabase_anon_key: str = Field(description="Supabase anon(public) key")
    supabase_jwt_secret: str | None = Field(default=None, description="구형 HS256 프로젝트에서만 필요")
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # DB. 비워 두면 로컬 SQLite(dev.db). Supabase 사용 시 postgresql+psycopg://... 로 교체
    database_url: str = Field(default="sqlite:///./dev.db")

    # 추론 백엔드. Phase 1은 mock, Phase 2부터 http
    inference_backend: str = Field(default="mock", description="mock | http")
    inference_url: str = Field(default="http://localhost:9000")
    inference_token: str = Field(default="")
    inference_timeout_ms: int = Field(default=1000)

    # 녹화 (guideline/05). 조각·합친 영상 저장 위치. git 제외
    media_dir: str = Field(default="./media")
    recording_grace_seconds: float = Field(
        default=5.0, description="세션 종료 후 이 시간까지 도착한 조각은 허용"
    )
    recording_max_chunk_bytes: int = Field(default=5_000_000)

    # STT / LLM (guideline/05). 구현 전까지 리포트 status 는 skipped
    stt_enabled: bool = Field(default=False)
    stt_language: str = Field(default="ko")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_timeout_s: float = Field(default=60.0)

    # 이벤트 판정 임계값 (03-phase1-design.md 8절)
    gaze_yaw_threshold_deg: float = 20.0
    gaze_pitch_threshold_deg: float = 15.0
    gaze_off_seconds: float = 3.0
    face_lost_seconds: float = 3.0
    emotion_negative_seconds: float = 5.0
    emotion_negative_min_prob: float = 0.5
    emotion_surprised_seconds: float = 2.0
    emotion_surprised_min_prob: float = 0.6
    attention_low_seconds: float = 6.0
    attention_low_min_prob: float = 0.5
    gaze_stable_seconds: float = 30.0

    @property
    def jwks_url(self) -> str:
        return f"{str(self.supabase_url).rstrip('/')}/auth/v1/.well-known/jwks.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
