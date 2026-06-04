"""Write extracted data to output artifacts.

Load stage: saves images, CSV, and report to the output directory.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from .models import DocxSample, ExtractionReport

logger = logging.getLogger(__name__)

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


class ArtifactWriter:
    """Writes extraction outputs to disk."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def write_csv(self, samples: list[DocxSample], csv_path: Path) -> None:
        """Write master_samples.csv with protocol-compliant columns."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        existing_ids: set[str] = set()
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_ids.add(row.get("sample_id", ""))

        new_count = 0
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PROTOCOL_COLUMNS)
            writer.writeheader()

            for s in samples:
                if s.sample_id in existing_ids:
                    continue
                writer.writerow(
                    {
                        "sample_id": s.sample_id or "",
                        "annotator": s.annotator or "",
                        "image_path": s.image_path or "",
                        "sheet_25k": s.sheet_25k or "",
                        "sheet_5k": s.sheet_5k or "",
                        "province": s.province or "",
                        "position_on_25k_sheet": s.position_on_25k_sheet or "",
                        "original_description": s.original_description or "",
                        "target_present": str(s.target_present).lower(),
                        "target_count_claimed": s.target_count_claimed or 0,
                        "contains_necropolis": str(s.contains_necropolis).lower(),
                        "contains_single_mound": str(s.contains_single_mound).lower(),
                        "contains_hard_negative": str(s.contains_hard_negative).lower(),
                        "relief_original": s.relief_original or "",
                        "relief_normalized": s.relief_normalized or "",
                        "difficulty": s.difficulty or "medium",
                        "uncertainty": str(s.uncertainty).lower(),
                        "annotation_status": s.annotation_status or "not_started",
                        "review_status": s.review_status or "pending_review",
                        "notes": s.notes or "",
                    }
                )
                new_count += 1

        logger.info("Wrote %d new rows to %s", new_count, csv_path)

    def write_report(self, report: ExtractionReport, report_path: Path) -> None:
        """Write JSON extraction report."""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_dir": report.source_dir,
            "output_dir": report.output_dir,
            "files_processed": report.files_processed,
            "total_samples": report.total_samples,
            "total_images": report.total_images,
            "file_stats": [
                {
                    "filename": fs.filename,
                    "annotator": fs.annotator,
                    "total_images": fs.total_images,
                    "total_samples": fs.total_samples,
                    "samples_with_multiple_images": fs.samples_with_multiple_images,
                    "provinces": fs.provinces,
                }
                for fs in report.file_stats
            ],
            "errors": report.errors,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Wrote report to %s", report_path)
