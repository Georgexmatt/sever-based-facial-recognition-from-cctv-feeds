"""Unit tests for the SCRFD detector adapter."""

import sys
import types

import numpy as np
import pytest

from config.settings import Settings
from face_attendance.edge.detection.scrfd import FaceDetectionError, ScrfdFaceDetector


class FakeScrfdModel:
    """Small fake InsightFace model used to avoid loading ONNX weights."""

    def __init__(self) -> None:
        """Track prepare arguments for assertions."""

        self.prepare_kwargs: dict[str, object] = {}

    def prepare(self, **kwargs: object) -> None:
        """Capture SCRFD preparation options."""

        self.prepare_kwargs = kwargs

    def detect(self, image: np.ndarray, max_num: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
        """Return one accepted and one rejected detection."""

        assert image.shape == (10, 10, 3)
        assert max_num == 4
        assert metric == "default"
        return (
            np.array([[1, 2, 8, 9, 0.9], [3, 4, 5, 6, 0.2]], dtype=np.float32),
            np.array(
                [
                    [[1, 1], [2, 1], [1.5, 2], [1, 3], [2, 3]],
                    [[3, 3], [4, 3], [3.5, 4], [3, 5], [4, 5]],
                ],
                dtype=np.float32,
            ),
        )


def install_fake_insightface(monkeypatch: pytest.MonkeyPatch, model: FakeScrfdModel) -> None:
    """Install a fake insightface.model_zoo module into sys.modules."""

    module = types.ModuleType("insightface.model_zoo")
    module.get_model = lambda model_path, providers: model
    package = types.ModuleType("insightface")
    package.model_zoo = module
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.setitem(sys.modules, "insightface.model_zoo", module)


def test_scrfd_detector_loads_model_and_filters_by_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detector should load SCRFD once and return typed, thresholded detections."""

    ScrfdFaceDetector.reset_instance()
    fake_model = FakeScrfdModel()
    install_fake_insightface(monkeypatch, fake_model)
    settings = Settings(
        scrfd_confidence_threshold=0.5,
        scrfd_input_size=(320, 320),
        scrfd_max_faces=4,
    )

    detector = ScrfdFaceDetector.get_instance(settings)
    detections = detector.detect_image(np.zeros((10, 10, 3), dtype=np.uint8))

    assert fake_model.prepare_kwargs == {"ctx_id": -1, "det_thresh": 0.5, "input_size": (320, 320)}
    assert len(detections) == 1
    assert detections[0].bounding_box.confidence == pytest.approx(0.9)
    assert detections[0].landmarks.left_eye == (1.0, 1.0)


def test_detect_video_frame_delegates_to_image_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Video-frame inference should expose the same result as image inference."""

    ScrfdFaceDetector.reset_instance()
    fake_model = FakeScrfdModel()
    install_fake_insightface(monkeypatch, fake_model)
    detector = ScrfdFaceDetector.get_instance(Settings(scrfd_max_faces=4))

    assert detector.detect_video_frame(np.zeros((10, 10, 3), dtype=np.uint8))


def test_scrfd_detector_rejects_invalid_images() -> None:
    """Detector should validate image inputs before invoking SCRFD."""

    ScrfdFaceDetector.reset_instance()
    detector = ScrfdFaceDetector.get_instance(Settings())

    with pytest.raises(ValueError):
        detector.detect_image(np.zeros((10, 10), dtype=np.uint8))


def test_scrfd_detector_wraps_inference_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vendor inference exceptions should be wrapped in FaceDetectionError."""

    class BrokenModel(FakeScrfdModel):
        def detect(self, image: np.ndarray, max_num: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
            raise RuntimeError("boom")

    ScrfdFaceDetector.reset_instance()
    install_fake_insightface(monkeypatch, BrokenModel())
    detector = ScrfdFaceDetector.get_instance(Settings())

    with pytest.raises(FaceDetectionError):
        detector.detect_image(np.zeros((10, 10, 3), dtype=np.uint8))
