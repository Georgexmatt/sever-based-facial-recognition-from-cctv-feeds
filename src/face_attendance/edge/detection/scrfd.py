"""InsightFace SCRFD detector service."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from config.settings import Settings
from face_attendance.edge.detection.interfaces import (
    BoundingBox,
    FaceDetection,
    FacialLandmarks,
    ImageArray,
)


class DetectorConfigurationError(RuntimeError):
    """Raised when SCRFD detector configuration or model loading fails."""


class FaceDetectionError(RuntimeError):
    """Raised when SCRFD inference fails."""


class ScrfdFaceDetector:
    """Production SCRFD face detector backed by InsightFace.

    The service lazily loads a single SCRFD model instance and protects loading
    with a process-local lock so dependency injection can safely share one
    runtime across API requests or stream workers.
    """

    _instance: "ScrfdFaceDetector | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        """Initialize detector configuration without loading model weights."""

        self._settings = settings
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._providers = self._select_providers(settings.scrfd_execution_provider)
        self._ctx_id = 0 if self._providers[0] == "CUDAExecutionProvider" else -1

    @classmethod
    def get_instance(cls, settings: Settings) -> "ScrfdFaceDetector":
        """Return the singleton SCRFD detector for the current process."""

        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(settings)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance for isolated tests."""

        with cls._instance_lock:
            cls._instance = None

    def detect_image(self, image: ImageArray) -> list[FaceDetection]:
        """Detect faces in an image and return confidence-filtered results.

        Args:
            image: Three-channel ``numpy.uint8`` image array accepted by
                InsightFace/OpenCV.

        Returns:
            List of face detections containing bounding boxes and five-point
            landmarks.

        Raises:
            ValueError: If the input array is not an image-like array.
            FaceDetectionError: If model inference fails.
        """

        self._validate_image(image)
        try:
            model = self._load_model()
            bboxes, landmarks = model.detect(
                image,
                max_num=self._settings.scrfd_max_faces,
                metric=self._settings.scrfd_nms_metric,
            )
        except Exception as exc:  # noqa: BLE001 - wrap vendor/runtime failures
            logger.exception("SCRFD inference failed")
            raise FaceDetectionError("SCRFD inference failed") from exc

        detections = self._to_detections(bboxes, landmarks)
        logger.debug("SCRFD detected faces", count=len(detections))
        return detections

    def detect_video_frame(self, frame: ImageArray) -> list[FaceDetection]:
        """Detect faces in a decoded video frame.

        This method intentionally delegates to :meth:`detect_image` so image and
        video-frame inference share validation, logging, filtering, and error
        handling behavior.
        """

        return self.detect_image(frame)

    def _load_model(self) -> Any:
        """Lazily load and prepare the InsightFace SCRFD model."""

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from insightface.model_zoo import get_model

                model_path = str(Path(self._settings.scrfd_model_path))
                logger.info(
                    "Loading SCRFD model",
                    model_path=model_path,
                    providers=self._providers,
                    input_size=self._settings.scrfd_input_size,
                    confidence_threshold=self._settings.scrfd_confidence_threshold,
                )
                model = get_model(model_path, providers=self._providers)
                model.prepare(
                    ctx_id=self._ctx_id,
                    det_thresh=self._settings.scrfd_confidence_threshold,
                    input_size=self._settings.scrfd_input_size,
                )
                self._model = model
                logger.info("SCRFD model loaded", provider=self._providers[0])
                return model
            except Exception as exc:  # noqa: BLE001 - include import/config/vendor errors
                logger.exception("Failed to load SCRFD model")
                raise DetectorConfigurationError("Failed to load SCRFD model") from exc

    @staticmethod
    def _select_providers(provider: str) -> list[str]:
        """Translate configured provider policy to ONNX Runtime providers."""

        if provider == "cpu":
            return ["CPUExecutionProvider"]
        if provider == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        raise DetectorConfigurationError(f"Unsupported SCRFD provider: {provider}")

    @staticmethod
    def _validate_image(image: ImageArray) -> None:
        """Validate an image/frame before passing it to InsightFace."""

        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise ValueError("image must use dtype uint8")

    def _to_detections(self, bboxes: Any, landmarks: Any) -> list[FaceDetection]:
        """Convert raw SCRFD arrays into typed detection value objects."""

        if bboxes is None or landmarks is None:
            return []

        detections: list[FaceDetection] = []
        for bbox, landmark in zip(np.asarray(bboxes), np.asarray(landmarks), strict=False):
            confidence = float(bbox[4])
            if confidence < self._settings.scrfd_confidence_threshold:
                continue
            detections.append(
                FaceDetection(
                    bounding_box=BoundingBox(
                        x1=float(bbox[0]),
                        y1=float(bbox[1]),
                        x2=float(bbox[2]),
                        y2=float(bbox[3]),
                        confidence=confidence,
                    ),
                    landmarks=self._parse_landmarks(landmark),
                )
            )
        return detections

    @staticmethod
    def _parse_landmarks(landmark: Sequence[Sequence[float]]) -> FacialLandmarks:
        """Convert a five-point landmark sequence to a value object."""

        points = [(float(point[0]), float(point[1])) for point in landmark]
        if len(points) != 5:
            raise FaceDetectionError("SCRFD must return five facial landmarks")
        return FacialLandmarks(
            left_eye=points[0],
            right_eye=points[1],
            nose=points[2],
            left_mouth=points[3],
            right_mouth=points[4],
        )
