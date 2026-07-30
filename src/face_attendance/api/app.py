"""FastAPI application factory."""

from fastapi import FastAPI

from config.settings import Settings, load_settings
from face_attendance.api.v1.router import router as v1_router
from face_attendance.core.lifecycle import lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""

    resolved_settings = settings or load_settings()
    app = FastAPI(
        title=resolved_settings.app_name,
        debug=resolved_settings.debug,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.include_router(v1_router, prefix=resolved_settings.api_v1_prefix)
    return app
