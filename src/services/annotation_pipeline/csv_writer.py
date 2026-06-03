"""CSV writing for annotation results."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from .errors import CsvError
from .metadata import ANNOTATION_COLUMNS
from .schemas import AnnotationRow

logger = logging.getLogger(__name__)


def write_annotation_csv(
    rows: list[AnnotationRow],
    csv_path: Path,
) -> int:
    """Write annotation CSV with protocol-compliant columns.

    Args:
        rows: Annotation rows to write.
        csv_path: Output CSV path.

    Returns:
        Number of rows written.

    Raises:
        CsvError: If writing fails.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump())
    except Exception as e:
        raise CsvError(f"Failed to write CSV to {csv_path}: {e}") from e

    logger.info("Wrote %d rows to %s", len(rows), csv_path)
    return len(rows)


def append_annotation_csv(
    rows: list[AnnotationRow],
    csv_path: Path,
) -> int:
    """Append annotation rows to existing CSV.

    Args:
        rows: Annotation rows to append.
        csv_path: Existing CSV path.

    Returns:
        Number of rows appended.

    Raises:
        CsvError: If appending fails.
    """
    if not csv_path.exists():
        return write_annotation_csv(rows, csv_path)

    existing_ids = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row.get("sample_id", ""))
    except Exception as e:
        raise CsvError(f"Failed to read existing CSV {csv_path}: {e}") from e

    new_rows = [r for r in rows if r.sample_id not in existing_ids]
    if not new_rows:
        logger.info("No new rows to append (all %d already exist)", len(rows))
        return 0

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
            for row in new_rows:
                writer.writerow(row.model_dump())
    except Exception as e:
        raise CsvError(f"Failed to append CSV to {csv_path}: {e}") from e

    logger.info("Appended %d new rows to %s", len(new_rows), csv_path)
    return len(new_rows)


def polygon_to_json(polygon: list[list[float]]) -> str:
    """Convert polygon to JSON string for CSV storage."""
    if not polygon:
        return ""
    return json.dumps(polygon)
