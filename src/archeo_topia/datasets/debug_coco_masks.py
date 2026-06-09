#!/usr/bin/env python3
"""Debug overlay for raw COCO RLE masks.

Decodes COCO RLE segmentations directly from the CVAT export and overlays
them on the source image to verify that masks align with annotations before
the MapSAM conversion step.

Usage:
    python -m archeo_topia.datasets.debug_coco_masks \\
        --coco-json data/curated/datasets/cvat/v0.0.1/annotations/instances_default.json \\
        --images-dir data/curated/datasets/cvat/v0.0.1/images/default \\
        --output-dir artifacts/debug/raw_coco_masks \\
        --image-name K-34-35-B-g_1.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from archeo_topia.datasets.prepare_mapsam_coco import (
    decode_uncompressed_coco_rle,
)

logger = logging.getLogger(__name__)


def draw_overlay(
    image_path: Path,
    annotations: list[dict],
    categories: dict[int, str],
    output_path: Path,
) -> None:
    """Save a composite debug image with decoded RLE overlays.

    Draws each annotation as:
    - Decoded mask as semi-transparent color overlay
    - COCO bbox in white
    - Annotation ID and category label

    Args:
        image_path: Path to the source image.
        annotations: List of COCO annotation dicts for this image.
        categories: Mapping of category ID to name.
        output_path: Destination PNG path.
    """
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    category_colors = {
        1: (255, 0, 0, 100),
        2: (0, 255, 0, 80),
        3: (0, 0, 255, 80),
    }
    outline_colors = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        cat_name = categories.get(cat_id, "unknown")
        seg = ann.get("segmentation")
        bbox = ann.get("bbox")
        ann_id = ann.get("id", "?")

        overlay_data = np.zeros((height, width, 4), dtype=np.uint8)
        color = category_colors.get(cat_id, (128, 128, 128, 100))

        if isinstance(seg, dict) and "counts" in seg:
            try:
                mask_arr = decode_uncompressed_coco_rle(seg, height, width)
                fg = mask_arr > 0
                overlay_data[fg] = color
            except ValueError as exc:
                logger.warning("Failed to decode RLE for ann_id=%s: %s", ann_id, exc)

        composite = Image.fromarray(overlay_data, "RGBA")
        overlay = Image.alpha_composite(overlay, composite)

    final = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(final)

    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        cat_name = categories.get(cat_id, "unknown")
        bbox = ann.get("bbox")
        ann_id = ann.get("id", "?")
        color = outline_colors.get(cat_id, (128, 128, 128))

        if bbox:
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            label = f"{ann_id}:{cat_name}"
            draw.text((x, max(0, y - 14)), label, fill=color)

    final.convert("RGB").save(output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Debug overlay for raw COCO RLE masks.",
    )
    parser.add_argument(
        "--coco-json",
        required=True,
        help="Path to COCO instances_default.json",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Path to directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save debug overlay images",
    )
    parser.add_argument(
        "--image-name",
        required=True,
        help="Image filename to visualize (e.g. K-34-35-B-g_1.png)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import json

    coco_path = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_data = json.loads(coco_path.read_text(encoding="utf-8"))

    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    image = None
    for img in coco_data["images"]:
        if img.get("file_name") == args.image_name:
            image = img
            break

    if image is None:
        logger.error("Image '%s' not found in COCO JSON", args.image_name)
        return

    image_id = image["id"]
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

    image_path = images_dir / args.image_name
    if not image_path.exists():
        logger.error("Image file not found: %s", image_path)
        return

    out_path = output_dir / args.image_name
    draw_overlay(image_path, annotations, categories, out_path)
    logger.info(
        "Saved overlay %s (%d annotations)",
        out_path,
        len(annotations),
    )


if __name__ == "__main__":
    main()
