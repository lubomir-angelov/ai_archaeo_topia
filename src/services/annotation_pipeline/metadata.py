"""Metadata extraction and sidecar handling."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from .errors import PdfError
from .schemas import ClipMetadata

logger = logging.getLogger(__name__)

# Columns from the protocol's master_samples.csv
PROTOCOL_COLUMNS = [
    "sample_id",
    "annotator",
    "image_path",
    "sheet_25k",
    "sheet_5k",
    "province",
    "position_on_25k_sheet",
    "original_description",
    "target_present",
    "target_count_claimed",
    "contains_necropolis",
    "contains_single_mound",
    "contains_hard_negative",
    "relief_original",
    "relief_normalized",
    "difficulty",
    "uncertainty",
    "annotation_status",
    "review_status",
    "notes",
]

# Extended columns for machine-generated annotations
ANNOTATION_COLUMNS = [
    "sample_id",
    "annotator",
    "image_path",
    "source_pdf",
    "pdf_page",
    "clip_index",
    "clip_image_path",
    "review_image_path",
    "annotation_source",
    "model_backend",
    "label",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "polygon",
    "mask_path",
    "confidence",
    "review_status",
    "sheet_25k",
    "sheet_5k",
    "province",
    "position_on_25k_sheet",
    "original_description",
    "target_present",
    "target_count_claimed",
    "contains_necropolis",
    "contains_single_mound",
    "contains_hard_negative",
    "relief_original",
    "relief_normalized",
    "difficulty",
    "uncertainty",
    "annotation_status",
    "notes",
]


def load_sidecar_metadata(sidecar_path: str) -> dict[str, dict]:
    """Load optional sidecar metadata file (CSV or JSON).

    Args:
        sidecar_path: Path to sidecar file.

    Returns:
        Dict keyed by sample_id or clip identifier.

    Raises:
        PdfError: If file cannot be read.
    """
    path = Path(sidecar_path)
    if not path.exists():
        logger.warning("Sidecar metadata not found: %s", sidecar_path)
        return {}

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_sidecar(path)
    elif suffix == ".json":
        return _load_json_sidecar(path)
    else:
        raise PdfError(f"Unsupported sidecar format: {suffix}")


def _load_csv_sidecar(path: Path) -> dict[str, dict]:
    """Load CSV sidecar metadata."""
    result: dict[str, dict] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("sample_id", row.get("clip_index", ""))
                if key:
                    result[str(key)] = dict(row)
    except Exception as e:
        raise PdfError(f"Failed to read sidecar CSV: {e}") from e
    logger.info("Loaded %d rows from sidecar CSV: %s", len(result), path)
    return result


def _load_json_sidecar(path: Path) -> dict[str, dict]:
    """Load JSON sidecar metadata."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            result: dict[str, dict] = {}
            for item in data:
                key = item.get("sample_id", item.get("clip_index", ""))
                if key:
                    result[str(key)] = item
            return result
    except Exception as e:
        raise PdfError(f"Failed to read sidecar JSON: {e}") from e
    logger.info("Loaded %d entries from sidecar JSON: %s", len(data), path)
    return data if isinstance(data, dict) else {}


def merge_sidecar(
    clip: ClipMetadata,
    sidecar: dict[str, dict],
) -> list[str]:
    """Merge sidecar metadata into clip metadata.

    Args:
        clip: Clip metadata to enrich.
        sidecar: Sidecar metadata dict.

    Returns:
        List of missing metadata field names.
    """
    key = clip.sample_id or f"{clip.pdf_page}_{clip.clip_index}"
    extra = sidecar.get(key, {})
    if not extra:
        return []

    missing = []
    for field in [
        "sheet_25k",
        "sheet_5k",
        "province",
        "original_description",
        "relief_original",
        "relief_normalized",
        "difficulty",
        "notes",
    ]:
        val = extra.get(field)
        if val and not getattr(clip, field, ""):
            setattr(clip, field, str(val))
        elif not getattr(clip, field, "") and not val:
            missing.append(field)
    return missing


def report_missing_metadata(
    clips: list[ClipMetadata],
) -> list[str]:
    """Check clips for missing metadata fields.

    Returns:
        List of field names that are missing across clips.
    """
    optional_fields = [
        "sheet_25k",
        "sheet_5k",
        "province",
        "position_on_25k_sheet",
        "original_description",
        "relief_original",
        "relief_normalized",
        "difficulty",
        "notes",
    ]
    missing_all: dict[str, int] = {}
    for clip in clips:
        for field in optional_fields:
            val = getattr(clip, field, "")
            if not val:
                missing_all[field] = missing_all.get(field, 0) + 1

    missing_fields = []
    for field, count in sorted(missing_all.items()):
        if count == len(clips):
            missing_fields.append(f"{field} (all {count} clips)")
        elif count > 0:
            missing_fields.append(f"{field} ({count}/{len(clips)} clips)")
    return missing_fields
