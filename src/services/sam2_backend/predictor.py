"""SAM2 predictor backend with lazy model loading.

Provides a unified interface for segmentation inference that supports
both mock mode (deterministic, no GPU) and real SAM2 mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ModelError
from .masks import (
    bbox_from_mask,
    confidence_from_mask,
    create_bbox_mask,
    create_points_mask,
    mask_array_to_polygon,
    save_mask,
    validate_image_path,
)
from .settings import get_settings

logger = logging.getLogger(__name__)


class SAM2PredictorBackend:
    """Lazy-loaded SAM2 predictor with mock fallback.

    Loads the model once on first inference call. In mock mode,
    returns deterministic masks without loading any model.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None
        self._predictor: Any = None
        self._loaded = False
        self._load_error: str | None = None
        self._device: str = "unknown"
        logger.info(
            "SAM2PredictorBackend initialised (mode=%s, device=%s)",
            self.settings.mode,
            self.settings.resolved_device,
        )

    @property
    def device(self) -> str:
        """Return the resolved device string."""
        if self._device != "unknown":
            return self._device
        return self.settings.resolved_device

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is loaded and ready."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load the model if not already loaded (no-op in mock mode)."""
        if self.settings.mode == "mock":
            self._loaded = True
            self._device = "mock"
            logger.info("Mock mode active, no model to load")
            return

        if self._loaded:
            return

        self._load_sam2()

    def _load_sam2(self) -> None:
        """Load the SAM2 model and checkpoint.

        Raises:
            ModelError: If the model cannot be loaded.
        """
        try:
            import torch
        except ImportError as exc:
            msg = "torch is not installed. Install with: pip install torch"
            logger.error(msg)
            self._load_error = msg
            raise ModelError(msg) from exc

        device_str = self.settings.resolved_device
        if device_str == "cuda":
            if not torch.cuda.is_available():
                msg = "CUDA requested but not available"
                logger.error(msg)
                self._load_error = msg
                raise ModelError(msg)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self._device = device_str

        try:
            from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
            from sam2.sam2_image_predictor import (
                SAM2ImagePredictor,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            msg = (
                "SAM2 package not installed. "
                "Install with: pip install sam2 "
                "or clone from https://github.com/facebookresearch/segment-anything-2"
            )
            logger.error(msg)
            self._load_error = msg
            raise ModelError(msg) from exc

        model_cfg = self.settings.model_cfg
        checkpoint = self.settings.checkpoint

        if not checkpoint:
            msg = "SAM2_BACKEND_CHECKPOINT is not set"
            logger.error(msg)
            self._load_error = msg
            raise ModelError(msg)

        cp_path = Path(checkpoint)
        if not cp_path.exists():
            msg = f"Checkpoint not found: {checkpoint}"
            logger.error(msg)
            self._load_error = msg
            raise ModelError(msg)

        if not model_cfg:
            model_cfg = "sam2_hiera_l.yaml"
            logger.info("No model config specified, using default: %s", model_cfg)

        logger.info(
            "Loading SAM2 model (cfg=%s, checkpoint=%s, device=%s)",
            model_cfg,
            checkpoint,
            device_str,
        )
        try:
            self._model = build_sam2(model_cfg, checkpoint, device=device)
            self._predictor = SAM2ImagePredictor(self._model)
        except Exception as exc:
            msg = f"Failed to load SAM2 model: {exc}"
            logger.error(msg)
            self._load_error = msg
            raise ModelError(msg) from exc

        self._loaded = True
        logger.info("SAM2 model loaded successfully on %s", device_str)

    def segment_bbox(
        self,
        image_path: str,
        bbox: list[float],
        output_dir: Path,
        label: str = "",
    ) -> dict[str, Any]:
        """Run segmentation with a bounding-box prompt.

        Args:
            image_path: Path to the input image.
            bbox: [x_min, y_min, x_max, y_max].
            output_dir: Directory to save mask.
            label: Optional label.

        Returns:
            Segment response dict.
        """
        self.ensure_loaded()

        if self.settings.mode == "mock":
            return self._mock_segment_bbox(image_path, bbox, output_dir, label)

        return self._sam2_segment_bbox(image_path, bbox, output_dir, label)

    def segment_points(
        self,
        image_path: str,
        points: list[list[float]],
        point_labels: list[int],
        output_dir: Path,
        label: str = "",
    ) -> dict[str, Any]:
        """Run segmentation with point prompts.

        Args:
            image_path: Path to the input image.
            points: [[x, y], ...].
            point_labels: 1 = foreground, 0 = background.
            output_dir: Directory to save mask.
            label: Optional label.

        Returns:
            Segment response dict.
        """
        self.ensure_loaded()

        if self.settings.mode == "mock":
            return self._mock_segment_points(image_path, points, point_labels, output_dir, label)

        return self._sam2_segment_points(image_path, points, point_labels, output_dir, label)

    # ── SAM2 real inference ──────────────────────────────────────

    def _sam2_segment_bbox(
        self,
        image_path: str,
        bbox: list[float],
        output_dir: Path,
        label: str,
    ) -> dict[str, Any]:
        """Run real SAM2 inference with a bbox prompt."""
        image = np.array(_image_open(image_path))
        self._predictor.set_image(image)

        box = np.array(bbox, dtype=np.float32)
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8) * 255
        confidence = float(scores[best_idx])

        return _build_segment_result(mask, image_path, bbox, output_dir, label, confidence)

    def _sam2_segment_points(
        self,
        image_path: str,
        points: list[list[float]],
        point_labels: list[int],
        output_dir: Path,
        label: str,
    ) -> dict[str, Any]:
        """Run real SAM2 inference with point prompts."""
        image = np.array(_image_open(image_path))
        self._predictor.set_image(image)

        coords = np.array(points, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=None,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8) * 255
        confidence = float(scores[best_idx])

        result_bbox = bbox_from_mask(mask)
        return _build_segment_result(mask, image_path, result_bbox, output_dir, label, confidence)

    # ── Mock implementations ─────────────────────────────────────

    def _mock_segment_bbox(
        self,
        image_path: str,
        bbox: list[float],
        output_dir: Path,
        label: str,
    ) -> dict[str, Any]:
        """Return a deterministic mock segmentation for a bbox prompt."""
        w, h = validate_image_path(image_path, self.settings.max_image_pixels)
        mask = create_bbox_mask(w, h, bbox)
        return _build_segment_result(mask, image_path, bbox, output_dir, label, confidence=0.85)

    def _mock_segment_points(
        self,
        image_path: str,
        points: list[list[float]],
        point_labels: list[int],
        output_dir: Path,
        label: str,
    ) -> dict[str, Any]:
        """Return a deterministic mock segmentation for point prompts."""
        w, h = validate_image_path(image_path, self.settings.max_image_pixels)
        fg_points = [p for p, lbl in zip(points, point_labels, strict=True) if lbl == 1]
        if not fg_points:
            mask = np.zeros((h, w), dtype=np.uint8)
            return _build_segment_result(
                mask, image_path, [0.0, 0.0, 0.0, 0.0], output_dir, label, confidence=0.0
            )
        mask = create_points_mask(w, h, points, point_labels, radius=8)
        return _build_segment_result(
            mask, image_path, bbox_from_mask(mask), output_dir, label, confidence=0.75
        )


def _image_open(path: str) -> Any:
    """Thin wrapper around PIL.Image.open for deferred import."""
    from PIL import Image

    return Image.open(path)


def _build_segment_result(
    mask: np.ndarray,
    image_path: str,
    bbox: list[float],
    output_dir: Path,
    label: str,
    confidence: float,
) -> dict[str, Any]:
    """Build a segment response dict from a mask array.

    Saves the mask, extracts polygon, and assembles the response.

    Args:
        mask: Binary mask array.
        image_path: Source image path.
        bbox: Bounding box [x_min, y_min, x_max, y_max].
        output_dir: Output directory.
        label: Label string.
        confidence: Confidence score.

    Returns:
        Response dict matching SegmentResponse schema.
    """
    ph = prompt_hash_mock(bbox)
    fname = make_mask_filename_mock(image_path, ph)
    mask_path = save_mask(mask, fname, output_dir)

    polygon = mask_array_to_polygon(mask)
    conf = confidence_from_mask(mask) if confidence == 0.0 else confidence

    metadata: dict[str, Any] = {
        "polygon_quality": "contour" if len(polygon) > 4 else "bbox_fallback",
        "mask_shape": list(mask.shape),
    }

    return {
        "image_path": image_path,
        "label": label,
        "bbox": [round(v, 4) for v in bbox],
        "polygon": [[round(v, 4) for v in pt] for pt in polygon],
        "mask_path": mask_path,
        "confidence": round(min(max(conf, 0.0), 1.0), 4),
        "metadata": metadata,
    }


def prompt_hash_mock(data: Any) -> str:
    """Hash prompt data for deterministic naming."""
    from .masks import prompt_hash

    return prompt_hash(data)


def make_mask_filename_mock(image_path: str, prompt_h: str) -> str:
    """Build a deterministic mask filename."""
    from .masks import make_mask_filename

    return make_mask_filename(image_path, prompt_h)
