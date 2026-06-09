#!/usr/bin/env python3
"""Generate MapSAM-style training prompts from binary mound masks.

Reads binary mound masks and ignore masks from a MapSAM dataset layout,
finds connected components in each mask, and produces JSONL training
samples with bounding box and center point prompts for SAM training.

Usage:
    python -m archeo_topia.datasets.generate_mapsam_prompts \\
        --dataset-root data_lake/curated/datasets/mapsam_v0 \\
        --output-path data_lake/curated/datasets/mapsam_v0/metadata/training_samples.jsonl \\
        --min-component-area 20 \\
        --bbox-padding 4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SPLIT_ORDER = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Sheet ID parsing
# ---------------------------------------------------------------------------


def parse_sheet_id(filename: str) -> str:
    """Derive a sheet identifier from an image filename.

    Strips the extension and the trailing ``_<integer>`` suffix.
    E.g. ``K-35-8-G-a_1.png`` → ``K-35-8-G-a``.

    Args:
        filename: Original image filename (with or without extension).

    Returns:
        Sheet identifier string.
    """
    stem = Path(filename).stem
    match = re.match(r"^(.+)_\d+$", stem)
    if match:
        return match.group(1)
    return stem


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a binary mask image as a uint8 numpy array.

    Args:
        path: Path to the mask PNG file.

    Returns:
        2D numpy array of dtype uint8 with values 0 or 255.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the image cannot be converted to a binary mask.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {arr.ndim}D array from {path}")

    return arr


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def find_connected_components(
    mask: np.ndarray,
    min_area: int = 0,
) -> list[dict[str, Any]]:
    """Find connected components in a binary mask using 8-connectivity.

    Uses a simple flood-fill BFS approach to avoid hard dependency on
    OpenCV when it is not available.

    Args:
        mask: 2D binary mask array (foreground = 255, background = 0).
        min_area: Minimum number of foreground pixels to include a
            component. Components smaller than this are skipped.

    Returns:
        List of component dicts with keys ``index``, ``area``, ``bbox``,
        and ``center_point``.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    components: list[dict[str, Any]] = []

    eight_neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for start_y in range(height):
        for start_x in range(width):
            if mask[start_y, start_x] == 255 and not visited[start_y, start_x]:
                queue = [(start_y, start_x)]
                visited[start_y, start_x] = True
                pixels: list[tuple[int, int]] = []

                head = 0
                while head < len(queue):
                    cy, cx = queue[head]
                    head += 1
                    pixels.append((cx, cy))

                    for dy, dx in eight_neighbors:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask[ny, nx] == 255
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            queue.append((ny, nx))

                area = len(pixels)
                if area < min_area:
                    continue

                xs = [p[0] for p in pixels]
                ys = [p[1] for p in pixels]

                bbox = [min(xs), min(ys), max(xs), max(ys)]
                center = [
                    (min(xs) + max(xs)) // 2,
                    (min(ys) + max(ys)) // 2,
                ]

                components.append(
                    {
                        "index": len(components) + 1,
                        "area": area,
                        "bbox": bbox,
                        "center_point": center,
                    }
                )

    return components


# ---------------------------------------------------------------------------
# Bounding box utilities
# ---------------------------------------------------------------------------


def component_to_bbox(
    component: dict[str, Any],
    image_width: int,
    image_height: int,
    padding: int = 0,
) -> dict[str, Any]:
    """Apply padding to a component bounding box and clamp to image bounds.

    Args:
        component: Component dict with ``bbox`` key.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        padding: Number of pixels to expand the bounding box by on each side.

    Returns:
        Updated component dict with padded and clamped ``bbox`` and
        ``center_point``.
    """
    x_min, y_min, x_max, y_max = component["bbox"]

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width - 1, x_max + padding)
    y_max = min(image_height - 1, y_max + padding)

    bbox = [x_min, y_min, x_max, y_max]
    center = [
        (x_min + x_max) // 2,
        (y_min + y_max) // 2,
    ]

    result = dict(component)
    result["bbox"] = bbox
    result["center_point"] = center
    return result


# ---------------------------------------------------------------------------
# Training sample construction
# ---------------------------------------------------------------------------


def build_training_sample(
    *,
    split: str,
    sheet_id: str,
    filename: str,
    component: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """Build a single JSONL-ready training sample record.

    Args:
        split: Dataset split name (``train``, ``val``, ``test``).
        sheet_id: Map sheet identifier.
        filename: Original image filename.
        component: Component dict with ``index``, ``area``, ``bbox``,
            and ``center_point`` keys.
        dataset_root: Root path of the dataset (used for relative paths).

    Returns:
        Dictionary representing one JSONL row.
    """
    component_idx = component["index"]
    sample_id = f"{split}_{sheet_id}_{component_idx:06d}"

    return {
        "sample_id": sample_id,
        "split": split,
        "sheet_id": sheet_id,
        "image_path": f"images/{split}/{filename}",
        "mask_path": f"masks/{split}/{filename}",
        "ignore_mask_path": f"ignore_masks/{split}/{filename}",
        "component_index": component_idx,
        "component_area": component["area"],
        "bbox": component["bbox"],
        "center_point": component["center_point"],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset_root(dataset_root: Path) -> None:
    """Validate that the dataset root exists and has required split folders.

    Args:
        dataset_root: Root directory of the MapSAM dataset.

    Raises:
        FileNotFoundError: If *dataset_root* does not exist.
        ValueError: If required split directories are missing.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    for split in SPLIT_ORDER:
        for subdir in ("images", "masks"):
            expected = dataset_root / subdir / split
            if not expected.is_dir():
                raise ValueError(
                    f"Missing expected directory: {expected.relative_to(dataset_root)}"
                )


def validate_image_mask_pair(
    image_path: Path,
    mask_path: Path,
    ignore_mask_path: Path | None,
) -> tuple[int, int]:
    """Validate that an image and its mask exist and have matching sizes.

    Args:
        image_path: Path to the image file.
        mask_path: Path to the mound mask file.
        ignore_mask_path: Path to the ignore mask file, or ``None``.

    Returns:
        Tuple of ``(width, height)`` of the image.

    Raises:
        FileNotFoundError: If image or mask file is missing.
        ValueError: If image and mask sizes do not match.
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found for image {image_path.name}: {mask_path}")

    mask = Image.open(mask_path)
    if mask.size != (img_w, img_h):
        raise ValueError(
            f"Mask size {mask.size} does not match image size {img.size} for {image_path.name}"
        )

    if ignore_mask_path and ignore_mask_path.exists():
        ign = Image.open(ignore_mask_path)
        if ign.size != (img_w, img_h):
            raise ValueError(
                f"Ignore mask size {ign.size} does not match image size "
                f"{img.size} for {image_path.name}"
            )

    return img_w, img_h


# ---------------------------------------------------------------------------
# Sample generation pipeline
# ---------------------------------------------------------------------------


def generate_samples(
    dataset_root: Path,
    min_component_area: int = 20,
    bbox_padding: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate training samples from all splits in the dataset.

    For each image in each split, loads the binary mound mask, finds
    connected components, and produces a training sample per component.

    Args:
        dataset_root: Root directory of the MapSAM dataset.
        min_component_area: Minimum pixel count for a component to be
            included.
        bbox_padding: Padding in pixels to apply to each bounding box.

    Returns:
        Tuple of ``(samples, summary)`` where *samples* is the list of
        JSONL-ready dicts and *summary* is the metadata summary dict.
    """
    validate_dataset_root(dataset_root)

    samples: list[dict[str, Any]] = []
    samples_by_split: dict[str, int] = {s: 0 for s in SPLIT_ORDER}
    images_by_split: dict[str, int] = {s: 0 for s in SPLIT_ORDER}
    images_without_mounds_by_split: dict[str, int] = {s: 0 for s in SPLIT_ORDER}
    skipped_too_small = 0

    for split in SPLIT_ORDER:
        image_dir = dataset_root / "images" / split
        mask_dir = dataset_root / "masks" / split
        ignore_dir = dataset_root / "ignore_masks" / split

        image_files = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

        logger.info("Processing split=%s with %d images", split, len(image_files))

        for image_path in image_files:
            images_by_split[split] += 1
            filename = image_path.name
            mask_path = mask_dir / filename
            ignore_mask_path = ignore_dir / filename if ignore_dir.exists() else None

            img_w, img_h = validate_image_mask_pair(image_path, mask_path, ignore_mask_path)

            mask_arr = load_binary_mask(mask_path)
            sheet_id = parse_sheet_id(filename)

            raw_components = find_connected_components(mask_arr, min_area=1)

            valid_components = []
            for comp in raw_components:
                if comp["area"] < min_component_area:
                    skipped_too_small += 1
                    continue
                padded = component_to_bbox(comp, img_w, img_h, padding=bbox_padding)
                valid_components.append(padded)

            if not valid_components:
                images_without_mounds_by_split[split] += 1
                continue

            for comp in valid_components:
                sample = build_training_sample(
                    split=split,
                    sheet_id=sheet_id,
                    filename=filename,
                    component=comp,
                    dataset_root=dataset_root,
                )
                samples.append(sample)
                samples_by_split[split] += 1

    summary = {
        "total_samples": len(samples),
        "samples_by_split": samples_by_split,
        "images_by_split": images_by_split,
        "images_without_mounds_by_split": images_without_mounds_by_split,
        "skipped_components_by_reason": {
            "too_small": skipped_too_small,
        },
        "min_component_area": min_component_area,
        "bbox_padding": bbox_padding,
    }

    return samples, summary


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
        description="Generate MapSAM training prompts from binary mound masks.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of the MapSAM dataset",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for training_samples.jsonl",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=20,
        help="Minimum component area in pixels (default: 20)",
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=4,
        help="Bounding box padding in pixels (default: 4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_path)

    logger.info("Generating prompts from dataset at %s", dataset_root)

    samples, summary = generate_samples(
        dataset_root=dataset_root,
        min_component_area=args.min_component_area,
        bbox_padding=args.bbox_padding,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary_path = output_path.parent / "training_samples_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d samples to %s", len(samples), output_path)
    logger.info("Wrote summary to %s", summary_path)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
