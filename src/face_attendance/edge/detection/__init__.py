"""Face detection services."""

from face_attendance.edge.detection.interfaces import BoundingBox, FaceDetection, FaceDetector, FacialLandmarks
from face_attendance.edge.detection.scrfd import ScrfdFaceDetector

__all__ = ["BoundingBox", "FaceDetection", "FaceDetector", "FacialLandmarks", "ScrfdFaceDetector"]
