"""Application settings loaded from environment variables and dotenv files."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "development", "test", "staging", "production"]


class Settings(BaseSettings):
    """Validated application settings for the foundation layer."""

    app_name: str = Field(default="Face Attendance API", description="Public application name.")
    environment: Environment = Field(default="local", description="Runtime environment name.")
    debug: bool = Field(default=False, description="Enable development diagnostics when true.")
    api_v1_prefix: str = Field(default="/api/v1", description="Prefix for versioned HTTP APIs.")

    log_level: str = Field(default="INFO", description="Minimum Loguru level for emitted logs.")
    log_json: bool = Field(default=False, description="Emit structured JSON logs when true.")
    log_to_file: bool = Field(default=False, description="Write logs to a file in addition to stderr.")
    log_file_path: Path = Field(default=Path("data/logs/app.log"), description="Log file path.")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FACE_ATTENDANCE_",
        extra="ignore",
        case_sensitive=False,
    )


def load_settings() -> Settings:
    """Create a new settings object from the current process environment."""

    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings for dependency injection."""

    return load_settings()
