#!/usr/bin/env python3
"""Generate box/point proposals from extracted clips and run SAM2 segmentation.

Reads the CSV metadata from the PDF extraction step, generates context-aware
box/point proposals based on clip metadata (necropolis, single mound, hard
negative, uncertain), sends each proposal through the SAM2 backend for
segmentation, and saves masks + annotation CSV pinned to original metadata.

Usage:
    python src/sam2_mcp_proposal_run.py \
        --input-dir data/sam2_test \
        --output-dir data/sam2_test/automated_mcp \
        --backend-url mock \
        --max-proposals 10
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Ensure src/ is on path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.sam2_mcp.client import Sam2BackendClient
from services.sam2_mcp.schemas import SegmentationItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_clip_metadata(csv_path: Path) -> list[dict[str, str]]:
    """Load clip metadata from master_samples.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info("Loaded %d clip metadata rows from %s", len(rows), csv_path)
    return rows


def generate_proposal_boxes(
    clip: dict[str, str],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Generate targeted box/point proposals based on clip metadata.

    Returns list of proposal dicts with:
      - bbox: [x1, y1, x2, y2] in pixel coords
      - points: [[x, y], ...] for point prompts (optional)
      - point_labels: [1, 0, ...] for point prompts (optional)
      - label: str
      - prompt_type: "box" or "points"
    """
    contains_necropolis = clip.get("contains_necropolis", "false").lower() == "true"
    contains_single_mound = clip.get("contains_single_mound", "false").lower() == "true"
    contains_hard_negative = clip.get("contains_hard_negative", "false").lower() == "true"
    uncertainty = clip.get("uncertainty", "false").lower() == "true"

    proposals: list[dict[str, Any]] = []
    cx, cy = width / 2, height / 2

    if contains_necropolis:
        # Multiple mounds: grid of boxes across image
        count_str = clip.get("target_count_claimed", "3")
        nums = re.findall(r"\d+", count_str)
        n_boxes = min(5, max(2, int(nums[0]) if nums else 3))
        rows = int(np.ceil(np.sqrt(n_boxes)))
        cols = int(np.ceil(n_boxes / rows))
        step_x = width / (cols + 1)
        step_y = height / (rows + 1)
        box_w = width * 0.12
        box_h = height * 0.12
        for i in range(rows):
            for j in range(cols):
                if len(proposals) >= n_boxes:
                    break
                bx = cx * (j + 1) / (cols + 1)
                by = cy * (i + 1) / (rows + 1)
                proposals.append({
                    "bbox": [
                        max(0, int(bx - box_w / 2)),
                        max(0, int(by - box_h / 2)),
                        min(width, int(bx + box_w / 2)),
                        min(height, int(by + box_h / 2)),
                    ],
                    "points": [[int(bx), int(by)]],
                    "point_labels": [1],
                    "label": "mound",
                    "prompt_type": "box",
                })
            if len(proposals) >= n_boxes:
                break

    if contains_single_mound:
        # Single mound: center box with point prompt
        box_w = width * 0.18
        box_h = height * 0.18
        proposals.append({
            "bbox": [
                int(cx - box_w / 2),
                int(cy - box_h / 2),
                int(cx + box_w / 2),
                int(cy + box_h / 2),
            ],
            "points": [[int(cx), int(cy)]],
            "point_labels": [1],
            "label": "mound",
            "prompt_type": "box",
        })

    if contains_hard_negative:
        # Hard negative: center box to identify false positive
        box_w = width * 0.15
        box_h = height * 0.15
        proposals.append({
            "bbox": [
                int(cx - box_w / 2),
                int(cy - box_h / 2),
                int(cx + box_w / 2),
                int(cy + box_h / 2),
            ],
            "points": [[int(cx), int(cy)]],
            "point_labels": [1],
            "label": "hard_negative_symbol",
            "prompt_type": "box",
        })

    if uncertainty:
        # Uncertain: smaller center box
        box_w = width * 0.12
        box_h = height * 0.12
        proposals.append({
            "bbox": [
                int(cx - box_w / 2),
                int(cy - box_h / 2),
                int(cx + box_w / 2),
                int(cy + box_h / 2),
            ],
            "points": [[int(cx), int(cy)]],
            "point_labels": [1],
            "label": "uncertain_ignore",
            "prompt_type": "points",
        })

    if not proposals:
        # Default: single center box
        box_w = width * 0.2
        box_h = height * 0.2
        proposals.append({
            "bbox": [
                int(cx - box_w / 2),
                int(cy - box_h / 2),
                int(cx + box_w / 2),
                int(cy + box_h / 2),
            ],
            "points": [[int(cx), int(cy)]],
            "point_labels": [1],
            "label": "mound",
            "prompt_type": "box",
        })

    return proposals


def make_mask_filename(sample_id: str, idx: int) -> str:
    """Build deterministic mask filename."""
    stem = Path(sample_id).stem
    short_hash = hashlib.sha256(f"{stem}_{idx}".encode()).hexdigest()[:10]
    return f"mask_{stem}_{short_hash}.png"


def run_proposals(
    input_dir: str,
    output_dir: str,
    backend_url: str,
    max_proposals: int,
) -> dict[str, Any]:
    """Run the full proposal + SAM2 segmentation pipeline.

    Args:
        input_dir: Path to extraction output (with raw/images/ and annotations/).
        output_dir: Path to write SAM2 results.
        backend_url: SAM2 backend URL or 'mock'.
        max_proposals: Max proposals per image.

    Returns:
        Summary dict.
    """
    in_root = Path(input_dir)
    out_root = Path(output_dir)

    csv_path = in_root / "annotations" / "master_samples.csv"
    clips = load_clip_metadata(csv_path)

    # Create output structure
    masks_dir = out_root / "masks"
    review_dir = out_root / "review"
    masks_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SAM2 client
    client = Sam2BackendClient(
        backend_url=backend_url,
        output_dir=str(masks_dir),
    )
    health = client.check_health()
    logger.info("SAM2 backend health: %s", health)

    # Process each clip
    all_results: list[dict[str, Any]] = []
    total_masks = 0
    success_count = 0
    fail_count = 0

    for clip in clips:
        sample_id = clip.get("sample_id", "")
        image_path = clip.get("image_path", "")

        if not image_path or not Path(image_path).exists():
            logger.warning("Image not found for %s: %s", sample_id, image_path)
            fail_count += 1
            continue

        # Get image dimensions
        try:
            img = Image.open(image_path)
            w, h = img.size
        except Exception as e:
            logger.warning("Cannot open %s: %s", image_path, e)
            fail_count += 1
            continue

        # Generate proposals from metadata
        proposals = generate_proposal_boxes(clip, w, h)
        proposals = proposals[:max_proposals]

        clip_results: list[dict[str, Any]] = []

        for idx, prop in enumerate(proposals):
            label = prop["label"]
            prompt_type = prop["prompt_type"]
            bbox = prop["bbox"]

            try:
                if prompt_type == "box":
                    seg = client.segment_box(
                        image_path=image_path,
                        bbox=bbox,
                        label=label,
                    )
                else:
                    seg = client.segment_points(
                        image_path=image_path,
                        points=prop["points"],
                        point_labels=prop["point_labels"],
                        label=label,
                    )

                # Save mask with deterministic name
                mask_fname = make_mask_filename(sample_id, idx)
                mask_path = str(masks_dir / mask_fname)

                # If backend returned a mask_path, copy/rename it
                if seg.mask_path and Path(seg.mask_path).exists():
                    import shutil
                    shutil.copy2(seg.mask_path, mask_path)

                # If mock mode (no mask_path), create bbox mask
                if not Path(mask_path).exists():
                    mask_arr = np.zeros((h, w), dtype=np.uint8)
                    x1, y1, x2, y2 = bbox
                    mask_arr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
                    from PIL import Image as PILImage
                    PILImage.fromarray(mask_arr, mode="L").save(mask_path, "PNG")

                # Build result record pinned to metadata
                result = {
                    "sample_id": sample_id,
                    "annotator": clip.get("annotator", ""),
                    "image_path": image_path,
                    "sheet_25k": clip.get("sheet_25k", ""),
                    "sheet_5k": clip.get("sheet_5k", ""),
                    "province": clip.get("province", ""),
                    "position_on_25k_sheet": clip.get("position_on_25k_sheet", ""),
                    "target_present": clip.get("target_present", "true"),
                    "target_count_claimed": clip.get("target_count_claimed", ""),
                    "contains_necropolis": clip.get("contains_necropolis", "false"),
                    "contains_single_mound": clip.get("contains_single_mound", "false"),
                    "contains_hard_negative": clip.get("contains_hard_negative", "false"),
                    "relief_original": clip.get("relief_original", ""),
                    "relief_normalized": clip.get("relief_normalized", ""),
                    "uncertainty": clip.get("uncertainty", "false"),
                    # SAM2 result fields
                    "obj_index": idx,
                    "label": label,
                    "prompt_type": prompt_type,
                    "bbox_x_min": seg.bbox[0] if seg.bbox else bbox[0],
                    "bbox_y_min": seg.bbox[1] if seg.bbox else bbox[1],
                    "bbox_x_max": seg.bbox[2] if seg.bbox else bbox[2],
                    "bbox_y_max": seg.bbox[3] if seg.bbox else bbox[3],
                    "polygon": seg.polygon,
                    "mask_path": mask_path,
                    "confidence": seg.confidence,
                    "source": seg.source,
                    "status": seg.status,
                }
                clip_results.append(result)
                total_masks += 1

            except Exception as e:
                logger.warning("SAM2 failed for %s obj%d: %s", sample_id, idx, e)

        if clip_results:
            all_results.extend(clip_results)
            success_count += 1
        else:
            fail_count += 1

    # Write annotation CSV pinned to metadata
    ann_csv_path = out_root / "sam2_annotations.csv"
    if all_results:
        fieldnames = [
            "sample_id", "annotator", "image_path", "sheet_25k", "sheet_5k",
            "province", "position_on_25k_sheet", "target_present",
            "target_count_claimed", "contains_necropolis", "contains_single_mound",
            "contains_hard_negative", "relief_original", "relief_normalized",
            "uncertainty", "obj_index", "label", "prompt_type",
            "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
            "polygon", "mask_path", "confidence", "source", "status",
        ]
        with open(ann_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row = dict(r)
                row["polygon"] = json.dumps(row.get("polygon", []))
                writer.writerow(row)
        logger.info("Wrote %d annotation rows to %s", len(all_results), ann_csv_path)

    # Write JSON manifest
    manifest = {
        "run_id": f"sam2_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "input_csv": str(csv_path),
        "output_dir": str(out_root),
        "backend_url": backend_url,
        "backend_health": health,
        "clips_processed": len(clips),
        "clips_with_masks": success_count,
        "clips_failed": fail_count,
        "total_masks": total_masks,
        "max_proposals_per_image": max_proposals,
        "annotation_csv": str(ann_csv_path),
        "masks_dir": str(masks_dir),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Wrote manifest to %s", manifest_path)

    # Render review overlays for first few clips
    render_reviews(in_root, out_root, all_results, max_reviews=20)

    return manifest


def render_reviews(
    input_dir: Path,
    output_dir: Path,
    results: list[dict[str, Any]],
    max_reviews: int = 20,
) -> None:
    """Render review images with SAM2 boxes overlaid."""
    from PIL import ImageDraw, ImageFont

    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    # Group by sample_id
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_sample.setdefault(r["sample_id"], []).append(r)

    label_colors = {
        "mound": "lime",
        "hard_negative_symbol": "orange",
        "uncertain_ignore": "red",
    }

    count = 0
    for sample_id, objs in by_sample.items():
        if count >= max_reviews:
            break

        image_path = objs[0]["image_path"]
        if not Path(image_path).exists():
            continue

        try:
            img = Image.open(image_path).copy()
        except Exception:
            continue

        draw = ImageDraw.Draw(img)

        # Red border
        draw.rectangle(
            [0, 0, img.width - 1, img.height - 1],
            outline="red",
            width=2,
        )

        # Draw boxes
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        for obj in objs:
            x1, y1 = obj["bbox_x_min"], obj["bbox_y_min"]
            x2, y2 = obj["bbox_x_max"], obj["bbox_y_max"]
            color = label_colors.get(obj["label"], "cyan")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            txt = f"{obj['label']} {obj['confidence']:.2f}"
            draw.text((x1 + 2, max(0, y1 - 14)), txt, fill=color, font=font)

        # Info bar
        info = f"{sample_id} | {len(objs)} objs"
        draw.rectangle(
            [0, img.height - 24, min(500, img.width), img.height],
            fill="black",
        )
        draw.text((4, img.height - 22), info, fill="white", font=font)

        review_path = review_dir / f"review_{sample_id}"
        img.save(str(review_path), "PNG")
        count += 1

    logger.info("Rendered %d review images to %s", count, review_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate proposals and run SAM2 segmentation on extracted clips"
    )
    parser.add_argument(
        "--input-dir",
        default="data/sam2_test",
        help="Path to extraction output (with raw/images/ and annotations/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sam2_test/automated_mcp",
        help="Path to write SAM2 results",
    )
    parser.add_argument(
        "--backend-url",
        default="mock",
        help="SAM2 backend URL or 'mock' (default: mock)",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=10,
        help="Max proposals per image (default: 10)",
    )
    args = parser.parse_args()

    summary = run_proposals(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backend_url=args.backend_url,
        max_proposals=args.max_proposals,
    )

    print("\n=== SAM2 Proposal Run Summary ===")
    for k, v in summary.items():
        if k not in ("backend_health",):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {json.dumps(v, indent=4)}")


if __name__ == "__main__":
    main()
