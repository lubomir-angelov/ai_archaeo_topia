#!/usr/bin/env python3
"""Debug prediction script for MapSAM checkpoints.

Generates overlay images showing ground truth, predictions, and prompts
for visual inspection.

Usage::

    python -m archeo_topia.training.mapsam_predict_debug \\
        --config configs/mapsam/mapsam_v0_1_decoder_only.yaml \\
        --checkpoint artifacts/models/mapsam/v0_1_decoder_only/checkpoints/best.pt \\
        --split val \\
        --output-dir artifacts/models/mapsam/v0_1_decoder_only/debug_predictions \\
        --max-samples 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from archeo_topia.datasets.mapsam_dataset import MapSamDataset
from archeo_topia.training.train_mapsam_v0 import (
    forward_sam,
    load_checkpoint,
    resolve_config,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="MapSAM debug predictions")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True, help="Output directory for overlays")
    parser.add_argument("--max-samples", type=int, default=16, help="Maximum samples to process")
    return parser.parse_args(argv)


def draw_overlay(
    image_np: np.ndarray,
    target_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    ignore_mask_np: np.ndarray,
    box_prompt: list[float],
    point_prompt: list[float],
) -> Image.Image:
    """Create a composite overlay image for debugging.

    Layout (4 panels, left to right):
        1. Original image + bbox + point
        2. Ground truth mask
        3. Predicted mask
        4. Ignore mask

    Args:
        image_np: RGB image array ``(H, W, 3)`` in [0, 255].
        target_mask_np: Binary target mask ``(H, W)`` in [0, 1].
        pred_mask_np: Binary predicted mask ``(H, W)`` in [0, 1].
        ignore_mask_np: Binary ignore mask ``(H, W)`` in [0, 1].
        box_prompt: Bounding box ``[x1, y1, x2, y2]``.
        point_prompt: Point ``[x, y]``.

    Returns:
        PIL Image with the composite overlay.
    """
    h, w = image_np.shape[:2]

    panels: list[Image.Image] = []

    # Panel 1: image + prompts
    p1 = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(p1)
    draw.rectangle(
        [int(box_prompt[0]), int(box_prompt[1]), int(box_prompt[2]), int(box_prompt[3])],
        outline="red",
        width=2,
    )
    cx, cy = int(point_prompt[0]), int(point_prompt[1])
    r = 5
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="green")
    panels.append(p1)

    # Panel 2: GT mask
    gt_arr = np.clip(target_mask_np * 255, 0, 255).astype(np.uint8)
    p2 = Image.fromarray(gt_arr, mode="L").convert("RGB")
    panels.append(p2)

    # Panel 3: Pred mask
    pred_arr = np.clip(pred_mask_np * 255, 0, 255).astype(np.uint8)
    p3 = Image.fromarray(pred_arr, mode="L").convert("RGB")
    panels.append(p3)

    # Panel 4: Ignore mask
    ign_arr = np.clip(ignore_mask_np * 255, 0, 255).astype(np.uint8)
    p4 = Image.fromarray(ign_arr, mode="L").convert("RGB")
    panels.append(p4)

    total_w = w * 4
    composite = Image.new("RGB", (total_w, h))
    for i, panel in enumerate(panels):
        composite.paste(panel, (i * w, 0))

    return composite


def main(argv: list[str] | None = None) -> None:
    """Entry point for debug prediction."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = resolve_config(cfg)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(dataset_cfg["root"])
    samples_path = Path(dataset_cfg["samples"])

    if not dataset_root.exists():
        logger.error("Dataset root not found: %s", dataset_root)
        sys.exit(1)

    if not samples_path.exists():
        logger.error("Samples file not found: %s", samples_path)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    from segment_anything import sam_model_registry

    model_type = model_cfg["model_type"]
    sam = sam_model_registry[model_type]()

    ckpt = load_checkpoint(checkpoint_path, sam, None)  # type: ignore[arg-type]
    sam.to(device)
    sam.eval()

    logger.info("Checkpoint loaded from epoch %d", ckpt.get("epoch", "?"))

    ds = MapSamDataset(
        dataset_root=str(dataset_root),
        samples_path=str(samples_path),
        split=args.split,
        image_size=dataset_cfg["image_size"],
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    count = 0
    for batch in loader:
        if count >= args.max_samples:
            break

        with torch.no_grad():
            logits = forward_sam(
                sam,
                batch["image"],
                batch["box_prompt"],
                batch["point_prompt"],
                batch["point_label"],
                device,
            )

        pred = torch.sigmoid(logits[:, 0:1]) > 0.5
        pred_resized = torch.nn.functional.interpolate(
            pred.float(),
            size=(dataset_cfg["image_size"], dataset_cfg["image_size"]),
            mode="nearest",
        ).squeeze()

        image_np = (batch["image"].squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        )
        target_np = batch["target_mask"].squeeze().cpu().numpy()
        ignore_np = batch["ignore_mask"].squeeze().cpu().numpy()
        pred_np = pred_resized.cpu().numpy()
        box = batch["box_prompt"].squeeze().cpu().tolist()
        point = batch["point_prompt"].squeeze().cpu().tolist()

        overlay = draw_overlay(image_np, target_np, pred_np, ignore_np, box, point)

        sample_id = (
            batch["sample_id"][0]
            if isinstance(batch["sample_id"], (list, torch.Tensor))
            else batch["sample_id"]
        )
        out_path = output_dir / f"{sample_id}_overlay.png"
        overlay.save(str(out_path))
        logger.info("Saved overlay: %s", out_path)

        count += 1

    logger.info("Done. Saved %d overlays to %s", count, output_dir)


if __name__ == "__main__":
    main()
