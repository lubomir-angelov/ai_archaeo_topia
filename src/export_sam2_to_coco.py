#!/usr/bin/env python3
"""Export SAM2 annotations to COCO format for CVAT import.

COCO format is more robust for import than CVAT XML.

Usage:
    python src/export_sam2_to_coco.py \
        --csv data/sam2_test/automated_mcp_live_manual/sam2_annotations.csv \
        --images data/sam2_test/raw/images \
        --output data/sam2_test/coco_import/annotations.json \
        --min-confidence 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM2 annotations to COCO format")
    parser.add_argument("--csv", required=True, help="Path to sam2_annotations.csv")
    parser.add_argument("--images", required=True, help="Path to raw images root")
    parser.add_argument("--output", required=True, help="Output directory for COCO JSON + images")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min confidence threshold")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_root = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["confidence"]) >= args.min_confidence:
                annotations.append(row)
    logger.info("Loaded %d annotations (>= %.2f confidence)", len(annotations), args.min_confidence)

    # Collect images
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, dict] = {}
    img_id = 0
    for ann in annotations:
        sid = ann["sample_id"]
        if sid in image_map:
            continue
        # Find image
        found = False
        for img_path in sorted(images_root.rglob(sid)):
            if img_path.suffix.lower() == ".png":
                img = Image.open(img_path)
                shutil.copy2(img_path, img_dir / sid)
                image_map[sid] = {
                    "id": img_id,
                    "file_name": sid,
                    "width": img.width,
                    "height": img.height,
                }
                img_id += 1
                found = True
                break
        if not found:
            logger.warning("Image not found: %s", sid)

    # Build COCO
    coco = {
        "info": {
            "description": "SAM2 annotations for archaeological map clips",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "mound"},
            {"id": 2, "name": "hard_negative_symbol"},
            {"id": 3, "name": "uncertain_ignore"},
        ],
        "images": list(image_map.values()),
        "annotations": [],
    }

    cat_map = {"mound": 1, "hard_negative_symbol": 2, "uncertain_ignore": 3}
    ann_id = 1
    for ann in annotations:
        sid = ann["sample_id"]
        if sid not in image_map:
            continue

        polygon_str = ann.get("polygon", "[]")
        try:
            points = json.loads(polygon_str)
        except (json.JSONDecodeError, TypeError):
            continue
        if not points or len(points) < 3:
            continue

        # COCO polygon is flat [x1,y1,x2,y2,...]
        flat_poly = [round(v, 1) for pt in points for v in pt]

        # Bounding box [x, y, w, h]
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

        coco["annotations"].append({
            "id": ann_id,
            "image_id": image_map[sid]["id"],
            "category_id": cat_map.get(ann["label"], 1),
            "segmentation": [flat_poly],
            "bbox": [round(v, 1) for v in bbox],
            "area": round(bbox[2] * bbox[3], 1),
            "iscrowd": 0,
        })
        ann_id += 1

    # Write JSON
    json_path = output_dir / "annotations.json"
    json_path.write_text(json.dumps(coco, indent=2))
    logger.info("Wrote COCO JSON to %s (%d annotations, %d images)", json_path, len(coco["annotations"]), len(coco["images"]))

    print(f"\n=== COCO Export Summary ===")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Output: {json_path}")
    print(f"\n  To import into CVAT:")
    print(f"    1. Create task with images from: {img_dir}/")
    print(f"    2. Upload annotations.json")
    print(f"    3. Format: COCO")


if __name__ == "__main__":
    main()
