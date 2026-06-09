#!/usr/bin/env python3
"""Convert CVAT COCO export to MapSAM-ready binary mound segmentation dataset.

Reads a CVAT COCO JSON annotation file and its associated images, then produces
a MapSAM dataset layout with binary positive masks (mound), ignore masks
(uncertain_ignore), and hard-negative tracking (hard_negative_symbol).

Images are split by map sheet to prevent data leakage. Each image receives
exactly one positive mask and one ignore mask.

Usage:
    python -m src.archeo_topia.datasets.prepare_mapsam_coco \\
        --coco-json data/curated/datasets/cvat/annotations/instances_default.json \\
        --images-dir data/curated/datasets/cvat/images \\
        --output-dir data/curated/datasets/mapsam_v0 \\
        --positive-label mound \\
        --ignore-label uncertain_ignore \\
        --split-by sheet
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------


def load_coco(path: Path) -> dict[str, Any]:
    """Load and minimally validate a COCO JSON file.

    Args:
        path: Path to the COCO JSON file.

    Returns:
        Parsed COCO dictionary.

    Raises:
        ValueError: If required top-level keys are missing.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for key in ("images", "annotations"):
        if key not in data:
            raise ValueError(f"COCO JSON missing required key: {key}")

    return data


def find_category_id(categories: list[dict[str, Any]], name: str) -> int | None:
    """Return the category ID for *name*, or ``None``.

    Args:
        categories: COCO ``categories`` list.
        name: Category name to look up.

    Returns:
        Category ID integer or ``None``.
    """
    for cat in categories:
        if cat.get("name") == name:
            return cat["id"]
    return None


# ---------------------------------------------------------------------------
# Sheet extraction
# ---------------------------------------------------------------------------


def extract_sheet_id(filename: str) -> str:
    """Derive a sheet identifier from an image filename.

    Strips the extension and the trailing ``_<index>`` segment.
    E.g. ``K-35-8-G-a_1.png`` → ``K-35-8-G-a``.

    Args:
        filename: Original image filename.

    Returns:
        Sheet identifier string.
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else stem


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_splits(
    sheet_ids: list[str],
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Deterministically assign each sheet to a split.

    Sheets are sorted alphabetically, then distributed according to
    *ratios*.  Every sheet appears in exactly one split.

    Args:
        sheet_ids: Unique sheet identifiers.
        ratios: Mapping of split name → fraction.  Defaults to 70/15/15.

    Returns:
        Mapping of sheet ID → split name.
    """
    ratios = ratios or SPLIT_RATIOS
    sorted_sheets = sorted(set(sheet_ids))
    total = len(sorted_sheets)

    sheet_to_split: dict[str, str] = {}
    idx = 0

    for split_name in SPLIT_ORDER:
        fraction = ratios.get(split_name, 0.0)
        count = int(round(total * fraction))
        for _ in range(count):
            if idx < total:
                sheet_to_split[sorted_sheets[idx]] = split_name
                idx += 1

    last_split = SPLIT_ORDER[-1]
    while idx < total:
        sheet_to_split[sorted_sheets[idx]] = last_split
        idx += 1

    return sheet_to_split


# ---------------------------------------------------------------------------
# Segmentation decoding
# ---------------------------------------------------------------------------


def decode_polygon_mask(polygon: list[float], height: int, width: int) -> Image.Image:
    """Fill a COCO polygon (flat [x, y, ...]) into a binary PIL mask.

    Args:
        polygon: Flat list of alternating x, y coordinates.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Binary ``L`` mode Image (0/255).
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if len(polygon) >= 6:
        draw.polygon(polygon, fill=255)
    return mask


def decode_rle_mask(rle: dict[str, Any], height: int, width: int) -> Image.Image:
    """Decode a COCO RLE segmentation dict into a binary PIL mask.

    Supports both the compact dict format (``{"size": [w, h], "counts": "..."}``)
    and raw counts.  Falls back to an empty mask on error.

    Args:
        rle: RLE dictionary with ``size`` and ``counts`` keys.
        height: Expected image height.
        width: Expected image width.

    Returns:
        Binary ``L`` mode Image (0/255).
    """
    mask = Image.new("L", (width, height), 0)
    try:
        size = rle.get("size", [width, height])
        w, h = size[0], size[1]
        counts = rle.get("counts", "")

        counts_bytes = counts.encode("ascii") if isinstance(counts, str) else counts

        pixels = bytearray(w * h)
        val = 0
        pos = 0
        for byte in counts_bytes:
            run = byte
            while run == 255:
                pos += 255
                if pos >= len(counts_bytes):
                    break
                run = counts_bytes[pos]
                pos += 1
            val = 1 - val
            end = min(pos + run, w * h)
            for i in range(pos, end):
                pixels[i] = val * 255
            pos = end

        img = Image.frombytes("L", (w, h), bytes(pixels))
        if w != width or h != height:
            img = img.resize((width, height), Image.NEAREST)
        return img
    except Exception as exc:  # noqa: BLE001
        logger.warning("RLE decode failed: %s", exc)
        return mask


def decode_bbox_mask(bbox: list[float], height: int, width: int) -> Image.Image:
    """Create a filled-rectangle mask from a COCO bbox [x, y, w, h].

    Args:
        bbox: Bounding box as [x, y, width, height].
        height: Image height.
        width: Image width.

    Returns:
        Binary ``L`` mode Image (0/255).
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    x, y, bw, bh = bbox
    draw.rectangle([x, y, x + bw, y + bh], fill=255)
    return mask


def decode_segmentation(
    seg: Any, bbox: list[float] | None, height: int, width: int
) -> Image.Image:
    """Decode a single COCO segmentation entry into a binary mask.

    Handles polygon (list of lists of floats), RLE dict, and falls back
    to bbox fill when segmentation is empty but bbox exists.

    Args:
        seg: COCO segmentation field (polygon list, RLE dict, or empty).
        bbox: Fallback bbox [x, y, w, h], or ``None``.
        height: Image height.
        width: Image width.

    Returns:
        Binary ``L`` mode Image (0/255).
    """
    if seg is None or seg == []:
        if bbox:
            logger.debug("Empty segmentation, falling back to bbox fill")
            return decode_bbox_mask(bbox, height, width)
        return Image.new("L", (width, height), 0)

    if isinstance(seg, dict):
        return decode_rle_mask(seg, height, width)

    if isinstance(seg, list):
        if len(seg) == 0:
            if bbox:
                return decode_bbox_mask(bbox, height, width)
            return Image.new("L", (width, height), 0)

        first = seg[0]
        if isinstance(first, (int, float)):
            return decode_polygon_mask(seg, height, width)

        if isinstance(first, list):
            mask = Image.new("L", (width, height), 0)
            for poly in seg:
                if len(poly) >= 6:
                    sub = decode_polygon_mask(poly, height, width)
                    mask = _max_image(mask, sub)
            return mask

    if bbox:
        return decode_bbox_mask(bbox, height, width)

    return Image.new("L", (width, height), 0)


def _max_image(a: Image.Image, b: Image.Image) -> Image.Image:
    """Pixel-wise maximum of two binary masks.

    Args:
        a: First mask.
        b: Second mask.

    Returns:
        Combined mask.
    """
    import numpy as np  # noqa: PLC0415

    arr_a = np.array(a)
    arr_b = np.array(b)
    combined = np.maximum(arr_a, arr_b)
    return Image.fromarray(combined, "L")


# ---------------------------------------------------------------------------
# Image / mask pipeline
# ---------------------------------------------------------------------------


def build_image_index(
    coco_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a lookup from image filename to image metadata.

    Args:
        coco_data: Parsed COCO dictionary.

    Returns:
        Mapping of filename → image record.
    """
    index: dict[str, dict[str, Any]] = {}
    for img in coco_data.get("images", []):
        fname = img.get("file_name") or img.get("name", "")
        index[fname] = img
    return index


def process_image(
    image_record: dict[str, Any],
    annotations: list[dict[str, Any]],
    positive_id: int | None,
    ignore_id: int | None,
    hard_negative_id: int | None,
    images_dir: Path,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, int]]:
    """Generate masks for a single image.

    Args:
        image_record: COCO image record dict.
        annotations: All COCO annotations.
        positive_id: Category ID for positive (mound) class.
        ignore_id: Category ID for ignore class.
        hard_negative_id: Category ID for hard-negative class.
        images_dir: Root directory containing source images.

    Returns:
        Tuple of (source_image, positive_mask, ignore_mask, counts_dict).
    """
    fname = image_record["file_name"]
    img_path = images_dir / fname

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    source = Image.open(img_path).convert("L")
    height, width = source.size[1], source.size[0]

    positive_mask = Image.new("L", (width, height), 0)
    ignore_mask = Image.new("L", (width, height), 0)

    counts = {
        "mound": 0,
        "uncertain_ignore": 0,
        "hard_negative": 0,
    }

    img_id = image_record["id"]
    for ann in annotations:
        if ann["image_id"] != img_id:
            continue

        cat_id = ann.get("category_id")
        seg = ann.get("segmentation")
        bbox = ann.get("bbox")

        if cat_id == positive_id:
            counts["mound"] += 1
            sub = decode_segmentation(seg, bbox, height, width)
            positive_mask = _max_image(positive_mask, sub)

        elif cat_id == ignore_id:
            counts["uncertain_ignore"] += 1
            sub = decode_segmentation(seg, bbox, height, width)
            ignore_mask = _max_image(ignore_mask, sub)

        elif cat_id == hard_negative_id:
            counts["hard_negative"] += 1

    return source, positive_mask, ignore_mask, counts


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def build_splits_json(
    sheet_to_split: dict[str, str],
    image_index: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Build the splits metadata structure.

    Args:
        sheet_to_split: Sheet ID → split name mapping.
        image_index: Filename → image record mapping.

    Returns:
        Dictionary with split names as keys and lists of
        ``{"image": ..., "sheet": ...}`` entries as values.
    """
    splits: dict[str, list[dict[str, str]]] = {s: [] for s in SPLIT_ORDER}
    for fname in sorted(image_index):
        sheet = extract_sheet_id(fname)
        split = sheet_to_split.get(sheet)
        if split:
            splits[split].append({"image": fname, "sheet": sheet})
    return splits


def build_dataset_summary(
    splits: dict[str, list[dict[str, str]]],
    all_counts: dict[str, dict[str, int]],
    total_annotations_by_cat: dict[str, int],
) -> dict[str, Any]:
    """Build the dataset summary metadata.

    Args:
        splits: Split name → entries mapping.
        all_counts: Filename → per-image counts mapping.
        total_annotations_by_cat: Category name → total count mapping.

    Returns:
        Summary dictionary suitable for JSON serialization.
    """
    images_with_no_mound = sum(1 for c in all_counts.values() if c["mound"] == 0)

    summary: dict[str, Any] = {
        "total_images": sum(len(v) for v in splits.values()),
        "total_sheets": len(
            set(entry["sheet"] for entries in splits.values() for entry in entries)
        ),
        "images_per_split": {k: len(v) for k, v in splits.items()},
        "sheets_per_split": {k: len(set(e["sheet"] for e in v)) for k, v in splits.items()},
        "annotation_counts_by_category": total_annotations_by_cat,
        "mound_count_per_split": {},
        "hard_negative_count_per_split": {},
        "uncertain_ignore_count_per_split": {},
        "images_with_no_mound": images_with_no_mound,
    }

    for split_name in SPLIT_ORDER:
        mound_total = 0
        hn_total = 0
        ign_total = 0
        for entry in splits[split_name]:
            c = all_counts.get(entry["image"], {})
            mound_total += c.get("mound", 0)
            hn_total += c.get("hard_negative", 0)
            ign_total += c.get("uncertain_ignore", 0)
        summary["mound_count_per_split"][split_name] = mound_total
        summary["hard_negative_count_per_split"][split_name] = hn_total
        summary["uncertain_ignore_count_per_split"][split_name] = ign_total

    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    coco_json: Path,
    images_dir: Path,
    output_dir: Path,
    positive_label: str = "mound",
    ignore_label: str = "uncertain_ignore",
    hard_negative_label: str = "hard_negative_symbol",
    split_by: str = "sheet",
) -> None:
    """Execute the full CVAT → MapSAM conversion pipeline.

    Args:
        coco_json: Path to the COCO annotations JSON.
        images_dir: Path to the directory containing source images.
        output_dir: Destination directory for the MapSAM dataset.
        positive_label: Category name for positive (foreground) masks.
        ignore_label: Category name for ignore masks.
        hard_negative_label: Category name for hard-negative tracking.
        split_by: Split strategy (currently only ``"sheet"`` is supported).
    """
    if split_by != "sheet":
        logger.warning("Only --split-by sheet is supported; ignoring %s", split_by)

    logger.info("Loading COCO JSON from %s", coco_json)
    coco_data = load_coco(coco_json)

    categories = coco_data.get("categories", [])
    annotations = coco_data.get("annotations", [])

    positive_id = find_category_id(categories, positive_label)
    ignore_id = find_category_id(categories, ignore_label)
    hard_negative_id = find_category_id(categories, hard_negative_label)

    if positive_id is None:
        raise ValueError(
            f"Positive label '{positive_label}' not found in COCO categories. "
            f"Available: {[c.get('name') for c in categories]}"
        )

    logger.info(
        "Category IDs — positive=%s, ignore=%s, hard_negative=%s",
        positive_id,
        ignore_id,
        hard_negative_id,
    )

    image_index = build_image_index(coco_data)
    logger.info("Found %d images in COCO index", len(image_index))

    sheet_ids = [extract_sheet_id(fname) for fname in image_index]
    sheet_to_split = assign_splits(sheet_ids)
    logger.info(
        "Split assignment: %s",
        {s: len([sh for sh, sp in sheet_to_split.items() if sp == s]) for s in SPLIT_ORDER},
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_ORDER:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "masks" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "ignore_masks" / split_name).mkdir(parents=True, exist_ok=True)

    all_counts: dict[str, dict[str, int]] = {}
    total_cat_counts: dict[str, int] = {
        positive_label: 0,
        ignore_label: 0,
        hard_negative_label: 0,
    }

    for fname, img_record in sorted(image_index.items()):
        sheet = extract_sheet_id(fname)
        split = sheet_to_split.get(sheet)
        if not split:
            logger.warning("No split assigned for sheet %s (image %s)", sheet, fname)
            continue

        source, pos_mask, ign_mask, counts = process_image(
            img_record, annotations, positive_id, ignore_id, hard_negative_id, images_dir
        )

        shutil.copy2(images_dir / fname, output_dir / "images" / split / fname)
        pos_mask.save(output_dir / "masks" / split / fname, "PNG")
        ign_mask.save(output_dir / "ignore_masks" / split / fname, "PNG")

        all_counts[fname] = counts
        total_cat_counts[positive_label] += counts["mound"]
        total_cat_counts[ignore_label] += counts["uncertain_ignore"]
        total_cat_counts[hard_negative_label] += counts["hard_negative"]

        logger.debug(
            "%s → %s (mound=%d, ignore=%d, hard_neg=%d)",
            fname,
            split,
            counts["mound"],
            counts["uncertain_ignore"],
            counts["hard_negative"],
        )

    splits_meta = build_splits_json(sheet_to_split, image_index)
    summary = build_dataset_summary(splits_meta, all_counts, total_cat_counts)

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits_meta, f, indent=2, ensure_ascii=False)

    with open(metadata_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Dataset written to %s", output_dir)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Convert CVAT COCO export to MapSAM binary segmentation dataset.",
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
        help="Output directory for the MapSAM dataset",
    )
    parser.add_argument(
        "--positive-label",
        default="mound",
        help="Category name for positive (foreground) class (default: mound)",
    )
    parser.add_argument(
        "--ignore-label",
        default="uncertain_ignore",
        help="Category name for ignore region class (default: uncertain_ignore)",
    )
    parser.add_argument(
        "--hard-negative-label",
        default="hard_negative_symbol",
        help="Category name for hard-negative tracking (default: hard_negative_symbol)",
    )
    parser.add_argument(
        "--split-by",
        default="sheet",
        choices=["sheet"],
        help="Split strategy (default: sheet)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run(
        coco_json=Path(args.coco_json),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        positive_label=args.positive_label,
        ignore_label=args.ignore_label,
        hard_negative_label=args.hard_negative_label,
        split_by=args.split_by,
    )


if __name__ == "__main__":
    main()
