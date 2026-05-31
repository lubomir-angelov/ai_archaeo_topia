"""Validation logic for annotation run outputs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from PIL import Image

from .errors import ValidationError

logger = logging.getLogger(__name__)

VALID_LABELS = {"mound", "hard_negative_symbol", "uncertain_ignore"}
VALID_RELIEF = {
    "plain",
    "hilly",
    "mountain",
    "slope",
    "ridge",
    "valley",
    "urban",
    "mixed",
    "unknown",
    "",
}
VALID_DIFFICULTY = {"easy", "medium", "hard", "very_hard", ""}
VALID_STATUS = {
    "not_started",
    "in_progress",
    "annotated",
    "needs_review",
    "reviewed",
    "rejected",
}


def validate_run(
    csv_path: Path,
    run_dir: Path,
) -> dict:
    """Validate a complete annotation run.

    Checks:
    - CSV exists and has required columns
    - No duplicate sample IDs
    - Image files exist
    - Bbox coordinates are numeric and valid
    - Labels conform to taxonomy
    - Review status is valid
    - Paths stay under run directory

    Args:
        csv_path: Path to annotation CSV.
        run_dir: Run output directory.

    Returns:
        Validation summary dict.

    Raises:
        ValidationError: If CSV cannot be read.
    """
    errors: list[str] = []
    warnings: list[str] = []
    sample_ids: set[str] = set()
    rows = 0

    if not csv_path.exists():
        raise ValidationError(f"CSV not found: {csv_path}")

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Check required columns
            required = {
                "sample_id",
                "image_path",
                "label",
                "bbox_x_min",
                "bbox_y_min",
                "bbox_x_max",
                "bbox_y_max",
                "review_status",
            }
            if reader.fieldnames:
                missing = required - set(reader.fieldnames)
                if missing:
                    errors.append(f"Missing required columns: {missing}")
            else:
                errors.append("CSV has no header row")

            for row in reader:
                rows += 1
                sid = row.get("sample_id", "")

                # Duplicate check
                if sid in sample_ids:
                    errors.append(f"Duplicate sample_id: {sid}")
                sample_ids.add(sid)

                # Image path exists
                img_path = row.get("image_path", "") or row.get("clip_image_path", "")
                if img_path and not Path(img_path).exists():
                    errors.append(f"Image not found: {img_path}")

                # Bbox validation
                _validate_bbox(row, sid, errors)

                # Label validation
                label = row.get("label", "")
                if label and label not in VALID_LABELS:
                    warnings.append(f"Unknown label '{label}' for {sid}")

                # Review status
                review = row.get("review_status", "")
                if review and review != "pending_review":
                    warnings.append(f"Unexpected review_status '{review}' for {sid}")

                # Annotation status
                status = row.get("annotation_status", "")
                if status and status not in VALID_STATUS:
                    warnings.append(f"Invalid annotation_status '{status}' for {sid}")

                # Relief
                relief = row.get("relief_normalized", "")
                if relief and relief not in VALID_RELIEF:
                    warnings.append(f"Invalid relief_normalized '{relief}' for {sid}")

                # Difficulty
                diff = row.get("difficulty", "")
                if diff and diff not in VALID_DIFFICULTY:
                    warnings.append(f"Invalid difficulty '{diff}' for {sid}")

                # Path containment
                for path_field in [
                    "image_path",
                    "clip_image_path",
                    "review_image_path",
                    "mask_path",
                ]:
                    p = row.get(path_field, "")
                    if p:
                        try:
                            Path(p).resolve().relative_to(run_dir.resolve())
                        except ValueError:
                            errors.append(f"Path {path_field}={p} escapes run directory {run_dir}")

    except Exception as e:
        raise ValidationError(f"Failed to validate CSV: {e}") from e

    # Sample image dimension check
    _check_image_dimensions(csv_path, run_dir, warnings)

    return {
        "total_rows": rows,
        "unique_sample_ids": len(sample_ids),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors,
        "warning_messages": warnings,
    }


def _validate_bbox(row: dict, sid: str, errors: list[str]) -> None:
    """Validate bbox coordinates in a CSV row."""
    try:
        xmin = float(row.get("bbox_x_min", ""))
        ymin = float(row.get("bbox_y_min", ""))
        xmax = float(row.get("bbox_x_max", ""))
        ymax = float(row.get("bbox_y_max", ""))
    except (ValueError, TypeError):
        errors.append(f"Non-numeric bbox for {sid}")
        return

    if xmin >= xmax:
        errors.append(f"bbox x_min ({xmin}) >= x_max ({xmax}) for {sid}")
    if ymin >= ymax:
        errors.append(f"bbox y_min ({ymin}) >= y_max ({ymax}) for {sid}")

    # Check against image dimensions if available
    img_path = row.get("image_path", "") or row.get("clip_image_path", "")
    if img_path and Path(img_path).exists():
        try:
            img = Image.open(img_path)
            w, h = img.size
            if xmax > w or ymax > h:
                errors.append(
                    f"bbox [{xmin},{ymin},{xmax},{ymax}] outside image {w}x{h} for {sid}"
                )
        except Exception:
            pass


def _check_image_dimensions(
    csv_path: Path,
    run_dir: Path,
    warnings: list[str],
) -> None:
    """Check a sample of images for minimum dimensions."""
    if not csv_path.exists():
        return

    checked = 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if checked >= 10:
                    break
                img_path = row.get("image_path", "") or row.get("clip_image_path", "")
                if img_path and Path(img_path).exists():
                    try:
                        img = Image.open(img_path)
                        w, h = img.size
                        if w < 50 or h < 50:
                            warnings.append(
                                f"Very small image {row.get('sample_id', '')}: {w}x{h}"
                            )
                        checked += 1
                    except Exception as e:
                        warnings.append(f"Cannot open image {row.get('sample_id', '')}: {e}")
    except Exception:
        pass


def validate_bbox_in_image(
    bbox: list[float],
    image_path: str,
) -> bool:
    """Check if bbox coordinates are inside image dimensions.

    Args:
        bbox: [x_min, y_min, x_max, y_max].
        image_path: Path to image file.

    Returns:
        True if bbox is valid and inside image.
    """
    if len(bbox) != 4:
        return False
    try:
        img = Image.open(image_path)
        w, h = img.size
        xmin, ymin, xmax, ymax = bbox
        return 0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h
    except Exception:
        return False
