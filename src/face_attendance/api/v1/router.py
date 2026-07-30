"""Version 1 API router containing foundation endpoints only."""

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check(request: Request) -> dict[str, Any]:
    """Return basic process health without invoking recognition components."""

    settings = request.app.state.settings
    return {"status": "ok", "app_name": settings.app_name, "environment": settings.environment}
