#!/usr/bin/env python3
"""ETL job: extract mound samples from docx files.

Usage:
    python -m data_lake.docx_mound_extractor.main \
        --source ~/repos/ai_archaeo_topia/data/cleansed/labeling/mounds \
        --output ~/repos/ai_archaeo_topia/data/cleansed/annotation_protocol
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .loader import DocxLoader
from .models import DocxSample, ExtractionReport
from .writer import ArtifactWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def run_etl(source_dir: Path, output_dir: Path) -> ExtractionReport:
    """Execute the full extraction pipeline.

    1. Discover docx files in source_dir
    2. Extract images + text metadata from each
    3. Parse metadata into protocol-compliant fields
    4. Write images to output_dir/raw/images/<ANNOTATOR>/
    5. Write master_samples.csv to output_dir/annotations/
    6. Write JSON report to output_dir/report.json
    """
    docx_files = sorted(source_dir.glob("*.docx"))
    if not docx_files:
        raise RuntimeError(f"No .docx files found in {source_dir}")

    logger.info("Found %d docx files in %s", len(docx_files), source_dir)

    images_dir = output_dir / "raw" / "images"
    csv_path = output_dir / "annotations" / "master_samples.csv"
    report_path = output_dir / "report.json"

    all_samples: list[DocxSample] = []
    report = ExtractionReport(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        files_processed=0,
        total_samples=0,
        total_images=0,
    )

    for docx_path in docx_files:
        logger.info("Processing %s", docx_path.name)
        try:
            loader = DocxLoader(docx_path, images_dir)
            samples, stats = loader.load()
            all_samples.extend(samples)
            report.file_stats.append(stats)
            report.files_processed += 1
            report.total_samples += stats.total_samples
            report.total_images += stats.total_images
            logger.info(
                "  -> %d samples, %d images, %d multi-image",
                stats.total_samples,
                stats.total_images,
                stats.samples_with_multiple_images,
            )
        except Exception as exc:
            msg = f"{docx_path.name}: {exc}"
            logger.error(msg)
            report.errors.append(msg)

    writer = ArtifactWriter(output_dir)
    writer.write_csv(all_samples, csv_path)
    writer.write_report(report, report_path)

    logger.info(
        "Done. %d files, %d samples, %d images, %d errors",
        report.files_processed,
        report.total_samples,
        report.total_images,
        len(report.errors),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract mound samples from docx files")
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing .docx files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root directory for annotation protocol artifacts",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.is_dir():
        logger.error("Source directory does not exist: %s", source)
        sys.exit(1)

    output.mkdir(parents=True, exist_ok=True)

    try:
        report = run_etl(source, output)
        if report.errors:
            sys.exit(1)
    except Exception as exc:
        logger.error("ETL failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
