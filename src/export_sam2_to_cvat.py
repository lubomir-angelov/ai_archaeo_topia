#!/usr/bin/env python3
"""Export SAM2 annotations to CVAT for images 1.1 XML format.

Filters the SAM2 annotation CSV by confidence threshold, builds a CVAT
compatible XML file, and copies images into a flat directory for import.

Usage:
    python src/export_sam2_to_cvat.py \
        --csv data/sam2_test/automated_mcp_live_manual/sam2_annotations.csv \
        --images data/sam2_test/raw/images \
        --output data/sam2_test/cvat_import \
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
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_annotations(csv_path: Path, min_confidence: float) -> list[dict]:
    """Load and filter annotations from CSV."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row["confidence"])
            if conf >= min_confidence:
                rows.append(row)
    logger.info("Loaded %d annotations (>= %.2f confidence)", len(rows), min_confidence)
    return rows


def collect_images(images_root: Path) -> dict[str, dict]:
    """Collect all image files from subdirectories.

    Returns dict mapping filename -> {path, width, height}.
    """
    from PIL import Image

    images: dict[str, dict] = {}
    for img_path in sorted(images_root.rglob("*.png")):
        img = Image.open(img_path)
        images[img_path.name] = {
            "path": str(img_path),
            "width": img.width,
            "height": img.height,
        }
    logger.info("Found %d images in %s", len(images), images_root)
    return images


def build_cvat_xml(
    annotations: list[dict],
    images: dict[str, dict],
) -> str:
    """Build CVAT for images 1.1 XML string."""
    root = Element("annotations")
    root.set("verified", "no")

    # Meta
    meta = SubElement(root, "meta")
    SubElement(meta, "image_quality").text = "good"
    SubElement(meta, "source").text = "sam2_mcp"

    # Labels
    labels_elem = SubElement(meta, "labels")
    label_names = sorted(set(a["label"] for a in annotations))
    for label in label_names:
        label_elem = SubElement(labels_elem, "label")
        label_elem.set("name", label)
        label_elem.set("priority", "high")

    # Images
    images_elem = SubElement(root, "images")
    # Determine which images have annotations
    annotated_images = sorted(set(a["sample_id"] for a in annotations))
    frame_map: dict[str, int] = {}

    for idx, filename in enumerate(annotated_images):
        if filename not in images:
            logger.warning("Image not found: %s", filename)
            continue
        img_info = images[filename]
        img_elem = SubElement(images_elem, "image")
        img_elem.set("id", str(idx))
        fname_elem = SubElement(img_elem, "FileName")
        fname_elem.text = filename
        frame_map[filename] = idx

    # Annotations
    ann_root = SubElement(root, "annotations")
    for ann in annotations:
        sample_id = ann["sample_id"]
        if sample_id not in frame_map:
            continue

        polygon_str = ann.get("polygon", "[]")
        try:
            points = json.loads(polygon_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid polygon for %s obj%d, skipping", sample_id, ann["obj_index"])
            continue

        if not points or len(points) < 3:
            logger.warning("Polygon too small for %s obj%d, skipping", sample_id, ann["obj_index"])
            continue

        poly_elem = SubElement(ann_root, "polygon")
        poly_elem.set("label", ann["label"])
        poly_elem.set("source", "manual")
        poly_elem.set("occluded", "0")
        poly_elem.set("deleted", "false")

        frame_elem = SubElement(poly_elem, "frame_id")
        frame_elem.text = str(frame_map[sample_id])

        points_elem = SubElement(poly_elem, "points")
        for pt in points:
            pt_elem = SubElement(points_elem, "point")
            SubElement(pt_elem, "x").text = str(int(round(pt[0])))
            SubElement(pt_elem, "y").text = str(int(round(pt[1])))

    # Pretty print
    raw_xml = tostring(root, encoding="unicode")
    dom = parseString(raw_xml)
    pretty = dom.toprettyxml(indent="  ", encoding=None)
    # Remove extra XML declaration if present
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" ?>'
    return "\n".join(lines)


def prepare_import_dir(
    output_dir: Path,
    annotations: list[dict],
    images: dict[str, dict],
) -> None:
    """Copy annotated images into flat output directory."""
    import_dir = output_dir / "images"
    import_dir.mkdir(parents=True, exist_ok=True)

    annotated_images = sorted(set(a["sample_id"] for a in annotations))
    copied = 0
    for filename in annotated_images:
        if filename not in images:
            continue
        src = Path(images[filename]["path"])
        dst = import_dir / filename
        shutil.copy2(src, dst)
        copied += 1
    logger.info("Copied %d images to %s", copied, import_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM2 annotations to CVAT XML")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to sam2_annotations.csv",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to raw images root (with subdirectories)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for CVAT import package",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_root = Path(args.images)
    output_dir = Path(args.output)

    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    # Load and filter
    annotations = load_annotations(csv_path, args.min_confidence)
    if not annotations:
        logger.error("No annotations passed the confidence filter")
        sys.exit(1)

    # Collect images
    images = collect_images(images_root)

    # Prepare import directory
    prepare_import_dir(output_dir, annotations, images)

    # Build XML
    xml_str = build_cvat_xml(annotations, images)
    xml_path = output_dir / "annotations.xml"
    xml_path.write_text(xml_str, encoding="utf-8")
    logger.info("Wrote CVAT XML to %s", xml_path)

    # Summary
    unique_samples = len(set(a["sample_id"] for a in annotations))
    print(f"\n=== Export Summary ===")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Unique samples: {unique_samples}")
    print(f"  Min confidence: {args.min_confidence}")
    print(f"  Output: {output_dir}")
    print(f"  XML: {xml_path}")
    print(f"\n  To import into CVAT:")
    print(f"    1. Create a task in CVAT with images from: {output_dir}/images/")
    print(f"    2. Or upload annotations.xml via Task -> Upload annotations")
    print(f"       Format: 'CVAT for images 1.1'")


if __name__ == "__main__":
    main()
