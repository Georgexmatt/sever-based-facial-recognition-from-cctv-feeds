"""Unit tests for Loguru configuration."""

from config.settings import Settings
from face_attendance.infrastructure.logging import configure_logging


def test_configure_logging_accepts_default_settings() -> None:
    """Logging setup should complete without requiring file sinks."""

    configure_logging(Settings(environment="test"))
