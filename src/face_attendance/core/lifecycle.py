"""FastAPI startup and shutdown lifecycle hooks."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from face_attendance.core.container import Container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run application startup and shutdown foundation hooks."""

    existing_settings = getattr(app.state, "settings", None)
    container = Container.build(existing_settings)
    app.state.container = container
    app.state.settings = container.settings
    logger.info("Application startup complete", app_name=container.settings.app_name)
    try:
        yield
    finally:
        logger.info("Application shutdown complete", app_name=container.settings.app_name)
