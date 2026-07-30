"""Face detection interfaces and value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space face bounding box with detector confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True, slots=True)
class FacialLandmarks:
    """Five-point facial landmarks returned by SCRFD."""

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    left_mouth: tuple[float, float]
    right_mouth: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """Single detected face with bounding box and landmarks."""

    bounding_box: BoundingBox
    landmarks: FacialLandmarks


class FaceDetector(Protocol):
    """Protocol implemented by production face detector services."""

    def detect_image(self, image: ImageArray) -> list[FaceDetection]:
        """Detect faces in a single BGR/RGB image array."""

    def detect_video_frame(self, frame: ImageArray) -> list[FaceDetection]:
        """Detect faces in a single decoded video frame."""
