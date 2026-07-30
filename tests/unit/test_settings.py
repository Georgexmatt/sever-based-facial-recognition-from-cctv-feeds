"""Unit tests for configuration loading."""

from config.settings import Settings


def test_settings_use_prefixed_environment(monkeypatch) -> None:
    """Settings should read FACE_ATTENDANCE-prefixed environment variables."""

    monkeypatch.setenv("FACE_ATTENDANCE_APP_NAME", "Test Attendance")
    monkeypatch.setenv("FACE_ATTENDANCE_ENVIRONMENT", "test")

    settings = Settings()

    assert settings.app_name == "Test Attendance"
    assert settings.environment == "test"
