#!/usr/bin/env python3
"""PDF → SAM2 seed annotation pipeline runner.

End-to-end pipeline with clear boundaries:

  1. PDF PREPROCESSING (this module):
     PDF → pdf_clip_extractor → clip images + metadata
     This step extracts images from the PDF.  SAM 2 MCP never sees
     the raw PDF; it only receives the extracted map images.

  2. SAM 2 MAP SEGMENTATION (sam2_mcp / sam2_backend):
     clip images → sam2_mcp (generate_proposals) → masks + polygons
     SAM 2 MCP operates only on the extracted map images.

  3. ANNOTATION OUTPUT (this module):
     masks + polygons + metadata → annotation CSV
     → validate_annotation_run

Architecture boundary:
  - PDF handling: this module (OCR belongs to DeepSeek OCR MCP)
  - Map segmentation: SAM 2 MCP (map-only, no PDF/document support)
  - Reasoning/decisions: Qwen geospatial agent

Usage:
  python -m services.annotation_pipeline.pdf_sam2_runner \
    --pdf "${PDF_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --sam2-mcp-url "${SAM2_MCP_URL}" \
    --sam2-mode mock \
    --dry-run false
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from .artifacts import (
    create_run_directory,
    render_review_image,
    save_clip_image,
)
from .csv_writer import polygon_to_json, write_annotation_csv
from .errors import (
    ArtifactError,
    CsvError,
    PdfError,
    PipelineError,
    Sam2McpError,
    ValidationError,
)
from .metadata import (
    report_missing_metadata,
)
from .schemas import (
    AnnotationRow,
    ClipMetadata,
    RunReport,
    Sam2Proposal,
)
from .validation import validate_run

logger = logging.getLogger(__name__)


def extract_clips_from_pdf(
    pdf_path: Path,
    dry_run: bool = False,
    max_clips: int = 5,
) -> tuple[list[ClipMetadata], dict[int, Image.Image]]:
    """Extract image clips and metadata from PDF.

    Args:
        pdf_path: Path to input PDF.
        dry_run: Limit extraction for preview.
        max_clips: Max clips in dry-run mode.

    Returns:
        Tuple of (clip metadata list, PIL image dict keyed by xref).
    """
    if not pdf_path.exists():
        raise PdfError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise PdfError(f"Cannot open PDF {pdf_path}: {e}") from e
    logger.info("Opened PDF: %d pages", doc.page_count)

    all_clips: list[ClipMetadata] = []
    all_images: dict[int, Image.Image] = {}
    clip_counter = 0

    for page in doc:
        images = page.get_images(full=True)
        if not images:
            continue

        text = page.get_text()
        page_num = page.number + 1

        for idx, img in enumerate(images):
            if dry_run and clip_counter >= max_clips:
                break

            xref = img[0]

            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                pil_img = Image.open(img_bytes)
            except Exception as e:
                first_error = e
                try:
                    img_base = doc.extract_image(xref)
                    raw = img_base["image"]
                    pil_img = Image.open(io.BytesIO(raw))
                except Exception:
                    logger.warning(
                        "Failed to extract image xref=%d: %s", xref, first_error
                    )
                    continue

            # Parse metadata from page text
            parsed = _parse_page_text(text)

            annotator = parsed.get("annotator", "UNK")
            sample_id = (
                f"{annotator}_{clip_counter + 1:03d}_"
                f"{parsed.get('sheet_25k', 'UNKNOWN').replace(' ', '_')}_"
                f"{parsed.get('sheet_5k', 'UNKNOWN').replace(' ', '_')}.png"
            )

            clip = ClipMetadata(
                source_pdf=str(pdf_path),
                pdf_page=page_num,
                clip_index=idx,
                clip_image_path="",
                annotator=annotator,
                sample_id=sample_id,
                sheet_25k=parsed.get("sheet_25k", ""),
                sheet_5k=parsed.get("sheet_5k", ""),
                province=parsed.get("province", ""),
                position_on_25k_sheet=parsed.get("position", ""),
                original_description=parsed.get("description", ""),
                target_present=parsed.get("target_present", True),
                target_count_claimed=parsed.get("target_count", ""),
                contains_necropolis=parsed.get("contains_necropolis", False),
                contains_single_mound=parsed.get("contains_single_mound", False),
                contains_hard_negative=parsed.get("contains_hard_negative", False),
                relief_original=parsed.get("relief_original", ""),
                relief_normalized=parsed.get("relief_normalized", ""),
                difficulty=parsed.get("difficulty", "medium"),
                uncertainty=parsed.get("is_uncertain", False),
                notes=parsed.get("notes", ""),
                width=pil_img.width,
                height=pil_img.height,
            )

            all_clips.append(clip)
            all_images[xref] = pil_img
            clip_counter += 1

        if dry_run and clip_counter >= max_clips:
            break

    doc.close()
    logger.info("Extracted %d clips from PDF", len(all_clips))
    return all_clips, all_images


def _parse_page_text(text: str) -> dict[str, Any]:
    """Parse structured metadata from page text.

    Reuses logic from pdf_clip_extractor.py.
    """
    import re

    result: dict[str, Any] = {
        "annotator": "",
        "sheet_25k": "",
        "sheet_5k": "",
        "province": "",
        "position": "",
        "description": "",
        "relief_original": "",
        "relief_normalized": "",
        "target_count": "",
        "target_present": True,
        "is_uncertain": False,
        "notes": "",
        "contains_necropolis": False,
        "contains_single_mound": False,
        "contains_hard_negative": False,
        "difficulty": "medium",
    }

    prefixes = {"AG", "VG", "JTZ", "IK"}
    for prefix in sorted(prefixes, key=len, reverse=True):
        if re.search(rf"^\s*{prefix}\s*$", text, re.MULTILINE):
            result["annotator"] = prefix
            break

    m25 = re.search(
        r"Картен лист\s+1:\s*(?:25\s*000|25000)\s*:\s*([^\n,]+)",
        text,
        re.IGNORECASE,
    )
    if m25:
        result["sheet_25k"] = m25.group(1).strip()

    m5 = re.search(
        r"картен лист\s+1:\s*(?:5\s*000|5000)\s*:\s*([^\n,]+)",
        text,
        re.IGNORECASE,
    )
    if m5:
        raw = m5.group(1).strip()
        nums = re.findall(r"\(([^)]+)\)", raw)
        result["sheet_5k"] = " ".join(nums) if nums else raw

    prov = re.search(r"обл\.\s*([^\n,]+)", text)
    if prov:
        result["province"] = prov.group(1).strip()

    relief = re.search(r"Релеф:\s*(.+)", text)
    if relief:
        result["relief_original"] = relief.group(1).strip()
        result["relief_normalized"] = _normalize_relief(relief.group(1).strip())

    desc_lower = text.lower()
    uncertain_kws = [
        "не е могила",
        "няма могили",
        "няма знак за могила",
        "не е ясен",
        "не знам",
        "наподобява",
        "може и да е",
    ]
    if any(kw in desc_lower for kw in uncertain_kws):
        result["is_uncertain"] = True

    neg_kws = [
        "воденица",
        "мелница",
        "резервоар",
        "полезащитен",
        "изкоп",
        "каптаж",
        "кладенец",
        "рибарник",
    ]
    if any(kw in text.lower() for kw in neg_kws):
        result["contains_hard_negative"] = True

    if "некропол" in text.lower():
        result["contains_necropolis"] = True

    if re.search(r"Една\s+могила", text):
        result["contains_single_mound"] = True

    return result


_RELIEF_MAP = {
    "равнинен": "plain",
    "равниен": "plain",
    "хълмист": "hilly",
    "слабо хълмист": "hilly",
    "полупланински": "mixed",
    "планински": "mountain",
    "склон": "slope",
    "склон на хълм": "slope",
    "било": "ridge",
    "било на възвишение": "ridge",
    "речна долина": "valley",
    "дере": "valley",
    "градска среда": "urban",
}


def _normalize_relief(original: str) -> str:
    """Normalize relief description."""
    cleaned = original.strip().lower()
    if cleaned in _RELIEF_MAP:
        return _RELIEF_MAP[cleaned]
    for key, val in _RELIEF_MAP.items():
        if key in cleaned or cleaned in key:
            return val
    return "unknown"


def call_sam2_proposals(
    clip: ClipMetadata,
    backend_url: str,
    output_dir: str,
    max_proposals: int = 50,
    label_hint: str = "mound",
    run_id: str = "",
) -> list[Sam2Proposal]:
    """Call SAM2 MCP client to generate proposals.

    Uses Sam2BackendClient directly (same client used by sam2_mcp service).
    SAM 2 MCP receives only the extracted map image, never the raw PDF.

    Args:
        clip: Clip metadata with image path.
        backend_url: Backend URL or 'mock'.
        output_dir: Output directory for masks.
        max_proposals: Max proposals per image.
        label_hint: Label hint for proposals.
        run_id: Run identifier for provenance tracking.

    Returns:
        List of SAM2 proposals.
    """
    from services.sam2_mcp.client import Sam2BackendClient

    client = Sam2BackendClient(
        backend_url=backend_url,
        output_dir=output_dir,
    )

    image_path = clip.clip_image_path
    if not image_path or not Path(image_path).exists():
        raise Sam2McpError(f"Clip image not found: {image_path}")

    try:
        items = client.generate_proposals(
            image_path=image_path,
            max_proposals=max_proposals,
            label_hint=label_hint,
        )
    except Exception as e:
        raise Sam2McpError(f"Failed to generate proposals for {clip.sample_id}: {e}") from e

    proposals = []
    for item in items:
        proposals.append(
            Sam2Proposal(
                bbox=item.bbox,
                polygon=item.polygon,
                mask_path=item.mask_path,
                confidence=item.confidence,
                label=item.label or label_hint,
            )
        )

    logger.debug("Got %d proposals for %s", len(proposals), clip.sample_id)
    return proposals


def build_annotation_rows(
    clips: list[ClipMetadata],
    proposals_map: dict[str, list[Sam2Proposal]],
    sam2_mode: str,
    review_status: str,
) -> list[AnnotationRow]:
    """Build annotation CSV rows from clips and proposals.

    Args:
        clips: Extracted clip metadata.
        proposals_map: Proposals keyed by sample_id.
        sam2_mode: Backend mode (mock/sam2).
        review_status: Review status for all rows.

    Returns:
        List of annotation rows.
    """
    rows: list[AnnotationRow] = []

    for clip in clips:
        props = proposals_map.get(clip.sample_id, [])
        if not props:
            # Create a row with no proposals
            row = AnnotationRow(
                sample_id=clip.sample_id,
                annotator=clip.annotator,
                image_path=clip.clip_image_path,
                source_pdf=clip.source_pdf,
                pdf_page=clip.pdf_page,
                clip_index=clip.clip_index,
                clip_image_path=clip.clip_image_path,
                annotation_source="sam2_mcp",
                model_backend=sam2_mode,
                label="",
                review_status=review_status,
                sheet_25k=clip.sheet_25k,
                sheet_5k=clip.sheet_5k,
                province=clip.province,
                position_on_25k_sheet=clip.position_on_25k_sheet,
                original_description=clip.original_description,
                target_present=str(clip.target_present).lower(),
                target_count_claimed=clip.target_count_claimed,
                contains_necropolis=str(clip.contains_necropolis).lower(),
                contains_single_mound=str(clip.contains_single_mound).lower(),
                contains_hard_negative=str(clip.contains_hard_negative).lower(),
                relief_original=clip.relief_original,
                relief_normalized=clip.relief_normalized,
                difficulty=clip.difficulty,
                uncertainty=str(clip.uncertainty).lower(),
                annotation_status="annotated",
                notes=clip.notes,
            )
            rows.append(row)
            continue

        for i, prop in enumerate(props):
            bbox = prop.bbox
            row = AnnotationRow(
                sample_id=f"{clip.sample_id}_obj{i}",
                annotator=clip.annotator,
                image_path=clip.clip_image_path,
                source_pdf=clip.source_pdf,
                pdf_page=clip.pdf_page,
                clip_index=clip.clip_index,
                clip_image_path=clip.clip_image_path,
                annotation_source="sam2_mcp",
                model_backend=sam2_mode,
                label=prop.label or "mound",
                bbox_x_min=bbox[0] if len(bbox) >= 4 else 0.0,
                bbox_y_min=bbox[1] if len(bbox) >= 4 else 0.0,
                bbox_x_max=bbox[2] if len(bbox) >= 4 else 0.0,
                bbox_y_max=bbox[3] if len(bbox) >= 4 else 0.0,
                polygon=polygon_to_json(prop.polygon),
                mask_path=prop.mask_path,
                confidence=prop.confidence,
                review_status=review_status,
                sheet_25k=clip.sheet_25k,
                sheet_5k=clip.sheet_5k,
                province=clip.province,
                position_on_25k_sheet=clip.position_on_25k_sheet,
                original_description=clip.original_description,
                target_present=str(clip.target_present).lower(),
                target_count_claimed=clip.target_count_claimed,
                contains_necropolis=str(clip.contains_necropolis).lower(),
                contains_single_mound=str(clip.contains_single_mound).lower(),
                contains_hard_negative=str(clip.contains_hard_negative).lower(),
                relief_original=clip.relief_original,
                relief_normalized=clip.relief_normalized,
                difficulty=clip.difficulty,
                uncertainty=str(clip.uncertainty).lower(),
                annotation_status="annotated",
                notes=clip.notes,
            )
            rows.append(row)

    logger.info("Built %d annotation rows from %d clips", len(rows), len(clips))
    return rows


def run_pipeline(
    pdf_path: str,
    output_root: str,
    run_id: str,
    sam2_mcp_url: str = "mock",
    sam2_mode: str = "mock",
    review_status: str = "pending_review",
    dry_run: bool = False,
    max_proposals: int = 50,
    protocol_dir: str = "docs/annotation",
) -> RunReport:
    """Execute the full PDF → SAM2 annotation pipeline.

    Args:
        pdf_path: Path to input PDF.
        output_root: Base output directory.
        run_id: Unique run identifier.
        sam2_mcp_url: SAM2 MCP backend URL or 'mock'.
        sam2_mode: Backend mode.
        review_status: Review status for annotations.
        dry_run: If True, limit processing.
        max_proposals: Max proposals per image.
        protocol_dir: Protocol directory path.

    Returns:
        Run report with summary and diagnostics.
    """
    pdf = Path(pdf_path)
    out_root = Path(output_root)
    proto_dir = Path(protocol_dir)

    errors: list[str] = []
    warnings: list[str] = []

    # Pre-flight checks
    if not pdf.exists():
        raise PdfError(f"PDF not found: {pdf_path}")

    if not proto_dir.exists():
        raise ValidationError(f"Protocol directory not found: {protocol_dir}")

    if not out_root.exists():
        try:
            out_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ArtifactError(f"Cannot create output root: {e}") from e

    # Create run directory
    run_dir = create_run_directory(out_root, run_id, dry_run)
    logger.info("Run directory: %s", run_dir)

    # Extract clips
    try:
        clips, images = extract_clips_from_pdf(
            pdf, dry_run=dry_run, max_clips=5 if dry_run else 9999
        )
    except PdfError as e:
        errors.append(str(e))
        return RunReport(
            run_id=run_id,
            pdf_path=str(pdf),
            output_root=str(run_dir),
            dry_run=dry_run,
            sam2_mode=sam2_mode,
            sam2_mcp_url=sam2_mcp_url,
            errors=errors,
        )

    if not clips:
        warnings.append("No clips extracted from PDF")
        return RunReport(
            run_id=run_id,
            pdf_path=str(pdf),
            output_root=str(run_dir),
            dry_run=dry_run,
            sam2_mode=sam2_mode,
            sam2_mcp_url=sam2_mcp_url,
            pages_processed=len(set(c.pdf_page for c in clips)),
            warnings=warnings,
        )

    # Save clip images
    for clip in clips:
        pil_img = images.get(clip.clip_index)
        if pil_img is None:
            # Try to find by page
            for xref, img in images.items():
                if xref not in [c.clip_index for c in clips]:
                    pil_img = img
                    break
        if pil_img is None:
            # Fallback: re-extract from PDF
            try:
                doc = fitz.open(str(pdf))
                page = doc[clip.pdf_page - 1]
                for img_xref in page.get_images(full=True):
                    pix = fitz.Pixmap(doc, img_xref[0])
                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pil_img = Image.open(pix.tobytes("png"))
                    break
                doc.close()
            except Exception as e:
                logger.warning("Cannot re-extract image for %s: %s", clip.sample_id, e)
                continue

        try:
            save_clip_image(clip, pil_img, run_dir)
        except ArtifactError as e:
            errors.append(str(e))
            continue

    # Generate proposals via SAM2
    proposals_map: dict[str, list[Sam2Proposal]] = {}
    total_proposals = 0

    for clip in clips:
        if not clip.clip_image_path or not Path(clip.clip_image_path).exists():
            warnings.append(f"Skipping proposals for {clip.sample_id}: no image")
            continue

        try:
            props = call_sam2_proposals(
                clip=clip,
                backend_url=sam2_mcp_url,
                output_dir=str(run_dir / "masks"),
                max_proposals=max_proposals,
                run_id=run_id,
            )
            proposals_map[clip.sample_id] = props
            total_proposals += len(props)
        except Sam2McpError as e:
            errors.append(f"Proposal error for {clip.sample_id}: {e}")

    # Render review images
    for clip in clips:
        props = proposals_map.get(clip.sample_id, [])
        pil_img = None
        for _xref, img in images.items():
            if img.width == clip.width and img.height == clip.height:
                pil_img = img
                break
        if pil_img is None and clips:
            pil_img = list(images.values())[0] if images else None

        if pil_img:
            try:
                review_path = render_review_image(clip, pil_img, props, run_dir)
                # Store for CSV
                for c in clips:
                    if c.sample_id == clip.sample_id:
                        c.review_image_path = str(review_path)
                        break
            except ArtifactError as e:
                warnings.append(f"Review image error for {clip.sample_id}: {e}")

    # Build annotation rows
    rows = build_annotation_rows(clips, proposals_map, sam2_mode, review_status)

    # Write CSV
    csv_path = run_dir / "annotations" / "annotation_results.csv"
    try:
        written = write_annotation_csv(rows, csv_path)
    except CsvError as e:
        errors.append(str(e))
        written = 0

    # Check missing metadata
    missing = report_missing_metadata(clips)

    # Validate run
    validation_result: dict[str, Any] = {}
    if written > 0:
        try:
            validation_result = validate_run(csv_path, run_dir)
            if validation_result["errors"] > 0:
                errors.extend(validation_result["error_messages"])
            if validation_result["warnings"] > 0:
                warnings.extend(validation_result["warning_messages"])
        except ValidationError as e:
            errors.append(f"Validation error: {e}")

    # Build report
    report = RunReport(
        run_id=run_id,
        pdf_path=str(pdf),
        output_root=str(run_dir),
        dry_run=dry_run,
        sam2_mode=sam2_mode,
        sam2_mcp_url=sam2_mcp_url,
        pages_processed=len(set(c.pdf_page for c in clips)),
        clips_extracted=len(clips),
        proposals_generated=total_proposals,
        csv_rows_written=written,
        csv_path=str(csv_path),
        errors=errors,
        warnings=warnings,
        missing_metadata=missing,
    )

    logger.info(
        "Pipeline complete: %d clips, %d proposals, %d CSV rows, %d errors, %d warnings",
        report.clips_extracted,
        report.proposals_generated,
        report.csv_rows_written,
        len(report.errors),
        len(report.warnings),
    )

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="PDF → SAM2 seed annotation pipeline")
    parser.add_argument("--pdf", required=True, help="Path to input PDF file")
    parser.add_argument(
        "--output-root",
        default="./data/annotations/runs",
        help="Base output directory for runs",
    )
    parser.add_argument(
        "--run-id",
        default=f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        help="Unique run identifier",
    )
    parser.add_argument(
        "--sam2-mcp-url",
        default="mock",
        help="SAM2 MCP backend URL or 'mock'",
    )
    parser.add_argument(
        "--sam2-mode",
        default="mock",
        choices=["mock", "sam2"],
        help="SAM2 backend mode",
    )
    parser.add_argument(
        "--review-status",
        default="pending_review",
        help="Review status for generated annotations",
    )
    parser.add_argument(
        "--dry-run",
        type=str,
        default="false",
        help="Run in dry-run mode (process first 5 clips only)",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=50,
        help="Maximum proposals per image",
    )
    parser.add_argument(
        "--protocol-dir",
        default="docs/annotation",
        help="Path to annotation protocol directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    is_dry_run = args.dry_run.lower() in ("true", "1", "yes")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        report = run_pipeline(
            pdf_path=args.pdf,
            output_root=args.output_root,
            run_id=args.run_id,
            sam2_mcp_url=args.sam2_mcp_url,
            sam2_mode=args.sam2_mode,
            review_status=args.review_status,
            dry_run=is_dry_run,
            max_proposals=args.max_proposals,
            protocol_dir=args.protocol_dir,
        )

        print("\n=== Pipeline Run Report ===")
        print(f"  run_id: {report.run_id}")
        print(f"  pdf_path: {report.pdf_path}")
        print(f"  output_root: {report.output_root}")
        print(f"  dry_run: {report.dry_run}")
        print(f"  sam2_mode: {report.sam2_mode}")
        print(f"  sam2_mcp_url: {report.sam2_mcp_url}")
        print(f"  pages_processed: {report.pages_processed}")
        print(f"  clips_extracted: {report.clips_extracted}")
        print(f"  proposals_generated: {report.proposals_generated}")
        print(f"  csv_rows_written: {report.csv_rows_written}")
        print(f"  csv_path: {report.csv_path}")

        if report.missing_metadata:
            print("\n  Missing metadata:")
            for m in report.missing_metadata:
                print(f"    - {m}")

        if report.warnings:
            print("\n  Warnings:")
            for w in report.warnings[:20]:
                print(f"    - {w}")

        if report.errors:
            print("\n  Errors:")
            for e in report.errors[:20]:
                print(f"    - {e}")

        sys.exit(1 if report.errors else 0)

    except PipelineError as e:
        logger.error("Pipeline failed: %s", e)
        print(f"\nPipeline error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
