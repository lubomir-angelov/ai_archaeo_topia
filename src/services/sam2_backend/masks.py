"""Mask and polygon utilities.

Handles binary mask creation, saving, and polygon extraction.
Uses OpenCV for contour extraction when available, falls back to
simple bbox-based rectangle polygon.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import OutputError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def mask_array_to_polygon(
    mask: np.ndarray,
    min_area: int = 10,
) -> list[list[float]]:
    """Extract polygon vertices from a binary mask array.

    Uses OpenCV contour approximation when available. Falls back to
    a simple bounding-box rectangle when contour extraction fails.

    Args:
        mask: Binary mask array (0 and 255 values).
        min_area: Minimum pixel area to consider as valid.

    Returns:
        List of [x, y] vertex pairs.
    """
    binary = (mask > 127).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < min_area:
        return []

    xmin = int(np.min(coords[:, 1]))
    xmax = int(np.max(coords[:, 1]))
    ymin = int(np.min(coords[:, 0]))
    ymax = int(np.max(coords[:, 0]))

    try:
        import cv2  # type: ignore[import-not-found]

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _bbox_to_polygon(xmin, ymin, xmax, ymax)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < min_area:
            return _bbox_to_polygon(xmin, ymin, xmax, ymax)

        perimeter = cv2.arcLength(largest, True)
        if perimeter < 1e-3:
            return _bbox_to_polygon(xmin, ymin, xmax, ymax)

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) < 3:
            return _bbox_to_polygon(xmin, ymin, xmax, ymax)

        polygon = [[float(pt[0][0]), float(pt[0][1])] for pt in approx.reshape(-1, 1, 2)]
        return polygon

    except ImportError:
        logger.debug("OpenCV not available, using bbox fallback for polygon")
        return _bbox_to_polygon(xmin, ymin, xmax, ymax)
    except Exception as exc:
        logger.warning("Contour extraction failed (%s), using bbox fallback", exc)
        return _bbox_to_polygon(xmin, ymin, xmax, ymax)


def _bbox_to_polygon(xmin: int, ymin: int, xmax: int, ymax: int) -> list[list[float]]:
    """Create a rectangular polygon from bounding-box coordinates."""
    return [
        [float(xmin), float(ymin)],
        [float(xmax), float(ymin)],
        [float(xmax), float(ymax)],
        [float(xmin), float(ymax)],
    ]


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    """Derive a bounding box from a binary mask array.

    Returns:
        [x_min, y_min, x_max, y_max] or [0, 0, 0, 0] if empty.
    """
    binary = (mask > 127).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    xmin = float(np.min(coords[:, 1]))
    xmax = float(np.max(coords[:, 1]))
    ymin = float(np.min(coords[:, 0]))
    ymax = float(np.max(coords[:, 0]))
    return [xmin, ymin, xmax + 1.0, ymax + 1.0]


def confidence_from_mask(mask: np.ndarray) -> float:
    """Compute a simple confidence score from mask fill ratio.

    Higher fill ratio relative to bbox area yields higher confidence.

    Returns:
        Float in [0.0, 1.0].
    """
    binary = (mask > 127).astype(np.uint8)
    total = binary.size
    if total == 0:
        return 0.0
    filled = np.count_nonzero(binary)
    ratio = filled / total
    return round(min(ratio * 2.0, 1.0), 4)


def make_mask_filename(image_path: str, prompt_hash: str) -> str:
    """Build a deterministic mask filename.

    Args:
        image_path: Source image path.
        prompt_hash: Hash of the prompt data.

    Returns:
        Filename string.
    """
    stem = Path(image_path).stem
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_hash = prompt_hash[:12]
    return f"mask_{stem}_{ts}_{short_hash}.png"


def prompt_hash(data: Any) -> str:
    """Create a hash from prompt data for deterministic naming.

    Args:
        data: Any serializable data.

    Returns:
        Hex digest string.
    """
    raw = str(data).encode()
    return hashlib.sha256(raw).hexdigest()


def save_mask(
    mask_array: np.ndarray,
    filename: str,
    output_dir: Path,
) -> str:
    """Save a binary mask as PNG under the output directory.

    Args:
        mask_array: Binary mask (0 and 255).
        filename: Target filename.
        output_dir: Output directory path.

    Returns:
        Absolute path to the saved mask.

    Raises:
        OutputError: If the mask cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    try:
        img = Image.fromarray(mask_array.astype(np.uint8), mode="L")
        img.save(str(path), "PNG")
    except Exception as exc:
        raise OutputError(f"Failed to write mask {path}: {exc}") from exc
    logger.info("Wrote mask to %s", path)
    return str(path.resolve())


def create_bbox_mask(
    image_width: int,
    image_height: int,
    bbox: list[float],
) -> np.ndarray:
    """Create a binary mask filled inside a bounding box.

    Args:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        bbox: [x_min, y_min, x_max, y_max].

    Returns:
        Binary mask array (0 and 255).
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    xmin = max(0, int(bbox[0]))
    ymin = max(0, int(bbox[1]))
    xmax = min(image_width, int(bbox[2]))
    ymax = min(image_height, int(bbox[3]))
    mask[ymin:ymax, xmin:xmax] = 255
    return mask


def create_points_mask(
    image_width: int,
    image_height: int,
    points: list[list[float]],
    point_labels: list[int],
    radius: int = 5,
) -> np.ndarray:
    """Create a binary mask around foreground points.

    Args:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        points: [[x, y], ...].
        point_labels: 1 = foreground, 0 = background.
        radius: Radius of the circle to draw around each foreground point.

    Returns:
        Binary mask array (0 and 255).
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for pt, lbl in zip(points, point_labels, strict=True):
        if lbl != 1:
            continue
        cx, cy = int(pt[0]), int(pt[1])
        if not (0 <= cx < image_width and 0 <= cy < image_height):
            continue
        y_indices, x_indices = np.ogrid[:image_height, :image_width]
        dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        mask[dist <= radius] = 255
    return mask


def validate_image_path(image_path: str, max_pixels: int = 10_000) -> tuple[int, int]:
    """Validate that an image file exists and is readable.

    Args:
        image_path: Path to the image file.
        max_pixels: Maximum allowed dimension.

    Returns:
        (width, height) tuple.

    Raises:
        ImageError: If the image is invalid or too large.
    """
    from .errors import ImageError

    p = Path(image_path)
    if not p.exists():
        raise ImageError(f"Image not found: {image_path}")
    if not p.is_file():
        raise ImageError(f"Not a file: {image_path}")
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ImageError(f"Unsupported image format: {p.suffix}")

    try:
        img = Image.open(p)
        img.verify()
    except Exception as exc:
        raise ImageError(f"Cannot read image {image_path}: {exc}") from exc

    img = Image.open(p)
    w, h = img.size
    if w > max_pixels or h > max_pixels:
        raise ImageError(f"Image {w}x{h} exceeds max dimension {max_pixels}px")
    return w, h


def validate_output_dir(
    requested_dir: str,
    base_output_dir: Path,
) -> Path:
    """Validate that a requested output directory is under the base output dir.

    Args:
        requested_dir: The output directory requested by the client.
        base_output_dir: The configured base output directory.

    Returns:
        Resolved output directory path.

    Raises:
        ValidationError: If the path escapes the base output directory.
    """
    from .errors import ValidationError

    if not requested_dir:
        return base_output_dir.resolve()

    requested = Path(requested_dir)
    if not requested.is_absolute():
        requested = base_output_dir / requested

    base = base_output_dir.resolve()
    try:
        requested.resolve().relative_to(base)
    except ValueError:
        raise ValidationError(
            f"output_dir {requested_dir!r} escapes base output_dir {base}"
        ) from None
    return requested.resolve()
