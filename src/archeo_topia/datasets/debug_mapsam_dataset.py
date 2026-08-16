#!/usr/bin/env python3
"""Debug utility for MapSamDataset.

Saves visual overlays for a limited number of samples to verify that
images, masks, bounding boxes, and center points are loaded and resized
correctly before training.

Usage:
    python -m archeo_topia.datasets.debug_mapsam_dataset \\
        --dataset-root data/curated/datasets/mapsam_v0 \\
        --samples-path data/curated/datasets/mapsam_v0/metadata/training_samples.jsonl \\
        --split train \\
        --output-dir artifacts/debug/mapsam_dataset \\
        --max-samples 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from archeo_topia.datasets.mapsam_dataset import MapSamDataset

logger = logging.getLogger(__name__)


def draw_overlay(
    image_tensor,
    target_mask_tensor,
    ignore_mask_tensor,
    box_prompt,
    point_prompt,
    output_path: Path,
) -> None:
    """Save a composite debug image with overlays.

    Creates a single image showing:
    - Original RGB image (resized)
    - Target mask as semi-transparent red overlay
    - Ignore mask as semi-transparent blue overlay
    - Bounding box in green
    - Center point as a yellow dot

    Args:
        image_tensor: Float32 tensor ``(3, H, W)`` in ``[0, 1]``.
        target_mask_tensor: Float32 tensor ``(1, H, W)`` binary.
        ignore_mask_tensor: Float32 tensor ``(1, H, W)`` binary.
        box_prompt: Float32 tensor ``(4,)`` xyxy coordinates.
        point_prompt: Float32 tensor ``(2,)`` xy coordinates.
        output_path: Destination PNG path.
    """
    import torch

    img_np = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np, "RGB")

    target_np = (target_mask_tensor.cpu().numpy()[0] * 255).astype(np.uint8)
    ignore_np = (ignore_mask_tensor.cpu().numpy()[0] * 255).astype(np.uint8)

    target_mask_pil = Image.fromarray(target_np, "L")
    ignore_mask_pil = Image.fromarray(ignore_np, "L")

    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))

    target_rgba = Image.new("RGBA", img_pil.size)
    t_data = target_rgba.load()
    for y in range(target_mask_pil.height):
        for x in range(target_mask_pil.width):
            if target_mask_pil.getpixel((x, y)) > 0:
                t_data[x, y] = (255, 0, 0, 128)
    overlay = Image.alpha_composite(overlay, target_rgba)

    ignore_rgba = Image.new("RGBA", img_pil.size)
    i_data = ignore_rgba.load()
    for y in range(ignore_mask_pil.height):
        for x in range(ignore_mask_pil.width):
            if ignore_mask_pil.getpixel((x, y)) > 0:
                i_data[x, y] = (0, 0, 255, 100)
    overlay = Image.alpha_composite(overlay, ignore_rgba)

    final = Image.composite(overlay, img_pil.convert("RGBA"), overlay.convert("RGBA"))
    draw = ImageDraw.Draw(final)

    if box_prompt is not None:
        if isinstance(box_prompt, torch.Tensor):
            box_prompt = box_prompt.cpu().numpy()
        x_min, y_min, x_max, y_max = box_prompt
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=(0, 255, 0),
            width=2,
        )

    if point_prompt is not None:
        if isinstance(point_prompt, torch.Tensor):
            point_prompt = point_prompt.cpu().numpy()
        px, py = point_prompt
        r = 6
        draw.ellipse(
            [(px - r, py - r), (px + r, py + r)],
            fill=(255, 255, 0),
            outline=(255, 0, 0),
        )

    final.convert("RGB").save(output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Debug visualizer for MapSamDataset samples.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of the MapSAM dataset",
    )
    parser.add_argument(
        "--samples-path",
        required=True,
        help="Path to training_samples.jsonl",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to inspect (default: train)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save debug overlay images",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum number of samples to visualize (default: 8)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating MapSamDataset for split='%s'", args.split)
    dataset = MapSamDataset(
        dataset_root=args.dataset_root,
        samples_path=args.samples_path,
        split=args.split,
    )

    limit = min(args.max_samples, len(dataset))
    logger.info("Visualizing %d samples", limit)

    for i in range(limit):
        sample = dataset[i]
        sample_id = sample["sample_id"]

        out_path = output_dir / f"{sample_id}.png"
        draw_overlay(
            image_tensor=sample["image"],
            target_mask_tensor=sample["target_mask"],
            ignore_mask_tensor=sample["ignore_mask"],
            box_prompt=sample["box_prompt"],
            point_prompt=sample["point_prompt"],
            output_path=out_path,
        )
        logger.info("Saved overlay %s", out_path)

    logger.info("Done. %d overlays saved to %s", limit, output_dir)


if __name__ == "__main__":
    main()
