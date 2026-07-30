"""Small dependency injection container for foundation services."""

from dataclasses import dataclass

from config.settings import Settings, load_settings
from face_attendance.edge.detection import FaceDetector, ScrfdFaceDetector
from face_attendance.infrastructure.logging import configure_logging


@dataclass(slots=True)
class Container:
    """Application dependency container.

    The container intentionally exposes only foundation dependencies until the
    recognition, tracking, persistence, and vector-search layers are reviewed.
    """

    settings: Settings
    face_detector: FaceDetector

    @classmethod
    def build(cls, settings: Settings | None = None) -> "Container":
        """Build a container from explicit or environment-loaded settings."""

        resolved_settings = settings or load_settings()
        configure_logging(resolved_settings)
        face_detector = ScrfdFaceDetector.get_instance(resolved_settings)
        return cls(settings=resolved_settings, face_detector=face_detector)
