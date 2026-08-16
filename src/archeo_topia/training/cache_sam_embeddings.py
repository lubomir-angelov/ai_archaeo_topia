#!/usr/bin/env python3
"""Cache SAM image embeddings for all unique images in a dataset split.

Pre-computes frozen image encoder outputs so training only runs the
prompt encoder + mask decoder per prompted instance.

Usage::

    python -m archeo_topia.training.cache_sam_embeddings \\
        --config configs/mapsam/mapsam_v0_1_decoder_only.yaml \\
        --split train
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from archeo_topia.datasets.mapsam_dataset import load_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Cache SAM image embeddings")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-compute embeddings even if cache file exists",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def prepare_image_for_sam(
    image: torch.Tensor,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    target_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize and resize image to SAM's expected input format.

    Args:
        image: Image tensor ``(3, H, W)`` in [0, 1].
        pixel_mean: SAM pixel mean tensor.
        pixel_std: SAM pixel std tensor.
        target_size: SAM image encoder input size.
        device: Target device.

    Returns:
        Normalized tensor ``(3, target_size, target_size)``.
    """
    img = image.to(device)
    pm = pixel_mean.view(3, 1, 1).to(device)
    ps = pixel_std.view(3, 1, 1).to(device)
    img = (img * 255.0 - pm) / ps

    if img.shape[1] != target_size or img.shape[2] != target_size:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return img


def collect_unique_images(
    dataset_root: Path,
    samples_path: Path,
    split: str,
) -> list[Path]:
    """Return deduplicated image paths for a split.

    Args:
        dataset_root: Root of the dataset.
        samples_path: Path to training_samples.jsonl.
        split: Dataset split to process.

    Returns:
        List of absolute image paths, deduplicated.
    """
    rows = load_jsonl(samples_path)
    split_rows = [r for r in rows if r.get("split") == split]

    seen: set[str] = set()
    unique: list[Path] = []
    for row in split_rows:
        rel = row["image_path"]
        if rel not in seen:
            seen.add(rel)
            unique.append(dataset_root / rel)

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for embedding caching."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    dataset_root = Path(dataset_cfg["root"])
    samples_rel = Path(dataset_cfg["samples"])
    samples_path = (
        dataset_root / samples_rel if not samples_rel.is_absolute() else samples_rel
    )
    model_type = model_cfg["model_type"]
    sam_checkpoint = Path(model_cfg["sam_checkpoint"])

    if not sam_checkpoint.exists():
        logger.error("SAM checkpoint not found: %s", sam_checkpoint)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Model type: %s", model_type)

    from segment_anything import sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(sam_checkpoint))
    sam.image_encoder.eval()
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    sam.to(device)

    pixel_mean = sam.pixel_mean.squeeze()
    pixel_std = sam.pixel_std.squeeze()
    target_size = sam.image_encoder.img_size

    unique_images = collect_unique_images(dataset_root, samples_path, args.split)
    logger.info("Found %d unique images for split '%s'", len(unique_images), args.split)

    cache_root = dataset_root / "sam_embeddings" / model_type / args.split
    cache_root.mkdir(parents=True, exist_ok=True)

    skipped = 0
    cached = 0
    failed = 0

    for img_path in tqdm(unique_images, desc=f"Caching {args.split}"):
        stem = img_path.stem
        cache_file = cache_root / f"{stem}.pt"

        if cache_file.exists() and not args.force:
            skipped += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_arr = np.asarray(pil_img)
            img_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1) / 255.0

            orig_h, orig_w = img_tensor.shape[1], img_tensor.shape[2]

            with torch.no_grad():
                prepared = prepare_image_for_sam(
                    img_tensor, pixel_mean, pixel_std, target_size, device
                )
                embedding = sam.image_encoder(prepared.unsqueeze(0))

            payload: dict[str, Any] = {
                "image_embedding": embedding.cpu().squeeze(0),
                "original_size": (orig_h, orig_w),
                "resized_size": (target_size, target_size),
                "image_path": str(img_path),
                "model_type": model_type,
                "checkpoint": str(sam_checkpoint),
            }

            torch.save(payload, str(cache_file))
            cached += 1

        except Exception as exc:
            logger.error("Failed to cache %s: %s", img_path, exc)
            failed += 1

    logger.info(
        "Done — cached: %d, skipped: %d, failed: %d", cached, skipped, failed
    )


if __name__ == "__main__":
    main()
