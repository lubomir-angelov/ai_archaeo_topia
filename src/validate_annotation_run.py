#!/usr/bin/env python3
"""Validate the annotation run outputs against the protocol."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = Path("/home/naim/repos/ai_archaeo_topia/artifacts/annotation_runs/annotations/master_samples.csv")
IMAGE_ROOT = Path("/home/naim/repos/ai_archaeo_topia/artifacts/annotation_runs/raw/images")
REVIEW_ROOT = Path("/home/naim/repos/ai_archaeo_topia/artifacts/annotation_runs/review/samples_for_review")

VALID_LABELS = {"mound", "hard_negative_symbol", "uncertain_ignore"}
VALID_RELIEF = {"plain", "hilly", "mountain", "slope", "ridge", "valley", "urban", "mixed", "unknown"}
VALID_DIFFICULTY = {"easy", "medium", "hard", "very_hard"}
VALID_STATUS = {"not_started", "in_progress", "annotated", "needs_review", "reviewed", "rejected"}
REQUIRED_COLUMNS = [
    "sample_id", "annotator", "image_path", "sheet_25k", "sheet_5k", "province",
    "position_on_25k_sheet", "original_description", "target_present",
    "target_count_claimed", "contains_necropolis", "contains_single_mound",
    "contains_hard_negative", "relief_original", "relief_normalized", "difficulty",
    "uncertainty", "annotation_status", "review_status", "notes",
]

def validate() -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    sample_ids: set[str] = set()
    rows = 0

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check columns
        if reader.fieldnames:
            missing_cols = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing_cols:
                errors.append(f"Missing CSV columns: {missing_cols}")

        for row in reader:
            rows += 1
            sid = row.get("sample_id", "")

            # Duplicate check
            if sid in sample_ids:
                errors.append(f"Duplicate sample_id: {sid}")
            sample_ids.add(sid)

            # Image path exists
            img_path = row.get("image_path", "")
            if img_path and not Path(img_path).exists():
                errors.append(f"Image not found: {img_path}")

            # Relief normalized
            relief = row.get("relief_normalized", "")
            if relief and relief not in VALID_RELIEF:
                warnings.append(f"Invalid relief_normalized '{relief}' for {sid}")

            # Difficulty
            diff = row.get("difficulty", "")
            if diff and diff not in VALID_DIFFICULTY:
                warnings.append(f"Invalid difficulty '{diff}' for {sid}")

            # Annotation status
            status = row.get("annotation_status", "")
            if status and status not in VALID_STATUS:
                warnings.append(f"Invalid annotation_status '{status}' for {sid}")

            # Review status
            review = row.get("review_status", "")
            if review and review != "pending_review":
                warnings.append(f"Unexpected review_status '{review}' for {sid}")

    # Check image dimensions consistency
    for sid in list(sample_ids)[:10]:  # Sample check
        csv_row = None
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["sample_id"] == sid:
                    csv_row = row
                    break
        if csv_row:
            img_path = csv_row.get("image_path", "")
            if img_path and Path(img_path).exists():
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    if w < 50 or h < 50:
                        warnings.append(f"Very small image {sid}: {w}x{h}")
                except Exception as e:
                    errors.append(f"Cannot open image {sid}: {e}")

    summary = {
        "total_rows": rows,
        "unique_sample_ids": len(sample_ids),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors[:10],
        "warning_messages": warnings[:10],
    }
    return summary


if __name__ == "__main__":
    result = validate()
    print("\n=== Validation Summary ===")
    print(f"  Total rows: {result['total_rows']}")
    print(f"  Unique sample IDs: {result['unique_sample_ids']}")
    print(f"  Errors: {result['errors']}")
    print(f"  Warnings: {result['warnings']}")
    if result["error_messages"]:
        print("\n  ERRORS:")
        for e in result["error_messages"]:
            print(f"    - {e}")
    if result["warning_messages"]:
        print("\n  WARNINGS:")
        for w in result["warning_messages"]:
            print(f"    - {w}")
