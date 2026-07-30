"""Unit tests for FastAPI application factory wiring."""

from fastapi.routing import APIRoute

from config.settings import Settings
from face_attendance.api.app import create_app


def test_create_app_registers_settings_and_health_route() -> None:
    """The app factory should register foundation state and versioned routes."""

    settings = Settings(app_name="Unit Test API", environment="test")
    app = create_app(settings)
    routes = {route.path for route in app.routes if isinstance(route, APIRoute)}

    assert app.title == "Unit Test API"
    assert app.state.settings is settings
    assert "/api/v1/health" in routes
