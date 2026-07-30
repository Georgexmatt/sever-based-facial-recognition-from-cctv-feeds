"""ASGI entry point for local development and production servers."""

from face_attendance.api.app import create_app

app = create_app()

__all__ = ["app", "create_app"]
