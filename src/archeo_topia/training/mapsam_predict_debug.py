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

from archeo_topia.datasets.mapsam_dataset import (
    MapSamDataset,
    count_connected_components,
)
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
    sample_id: str = "",
) -> Image.Image:
    """Create a composite overlay image for debugging.

    Layout (5 panels, left to right):
        1. Original image + bbox + point
        2. Ground truth instance mask
        3. Predicted mask (threshold=0.5)
        4. Ignore mask
        5. GT vs Prediction comparison

    Args:
        image_np: RGB image array ``(H, W, 3)`` in [0, 255].
        target_mask_np: Binary target mask ``(H, W)`` in [0, 1].
        pred_mask_np: Binary predicted mask ``(H, W)`` in [0, 1].
        ignore_mask_np: Binary ignore mask ``(H, W)`` in [0, 1].
        box_prompt: Bounding box ``[x1, y1, x2, y2]``.
        point_prompt: Point ``[x, y]``.
        sample_id: Sample identifier for titles.

    Returns:
        PIL Image with the composite overlay.
    """
    h, w = image_np.shape[:2]

    # Compute metrics
    gt_bool = target_mask_np > 0.5
    pred_bool = pred_mask_np > 0.5
    intersection = (gt_bool & pred_bool).sum()
    union = (gt_bool | pred_bool).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0
    dice = (
        (2.0 * intersection) / (gt_bool.sum() + pred_bool.sum())
        if (gt_bool.sum() + pred_bool.sum()) > 0
        else 0.0
    )
    target_pos = int(gt_bool.sum())
    pred_pos = int(pred_bool.sum())
    target_ncc = count_connected_components(target_mask_np)

    # Warn if target has multiple components
    if target_ncc > 1:
        logger.warning(
            "Target mask has %d components for sample_id=%s (expected 1)",
            target_ncc,
            sample_id,
        )

    # Title text
    title = (
        f"{sample_id} | GT={target_pos}px | Pred={pred_pos}px | IoU={iou:.3f} | Dice={dice:.3f}"
    )

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
    _add_title(p1, "Image + Prompts")
    panels.append(p1)

    # Panel 2: GT instance mask
    gt_arr = np.clip(target_mask_np * 255, 0, 255).astype(np.uint8)
    p2 = Image.fromarray(gt_arr, mode="L").convert("RGB")
    _add_title(p2, f"GT instance mask (n={target_ncc})")
    panels.append(p2)

    # Panel 3: Pred mask
    pred_arr = np.clip(pred_mask_np * 255, 0, 255).astype(np.uint8)
    p3 = Image.fromarray(pred_arr, mode="L").convert("RGB")
    _add_title(p3, "Pred mask (threshold=0.5)")
    panels.append(p3)

    # Panel 4: Ignore mask
    ign_arr = np.clip(ignore_mask_np * 255, 0, 255).astype(np.uint8)
    p4 = Image.fromarray(ign_arr, mode="L").convert("RGB")
    _add_title(p4, "Ignore mask")
    panels.append(p4)

    # Panel 5: GT vs Pred comparison
    # Green = GT only, Red = Pred only, Yellow = overlap, Black = neither
    comp_arr = np.zeros((h, w, 3), dtype=np.uint8)
    comp_arr[gt_bool & ~pred_bool] = [0, 255, 0]
    comp_arr[~gt_bool & pred_bool] = [255, 0, 0]
    comp_arr[gt_bool & pred_bool] = [255, 255, 0]
    p5 = Image.fromarray(comp_arr)
    _add_title(p5, "GT vs Pred (G=GT R=Pred Y=overlap)")
    panels.append(p5)

    total_w = w * 5
    title_h = 24
    composite = Image.new("RGB", (total_w, h + title_h), color=(30, 30, 30))
    draw_comp = ImageDraw.Draw(composite)
    draw_comp.text((4, 2), title, fill=(255, 255, 255))
    for i, panel in enumerate(panels):
        composite.paste(panel, (i * w, title_h))

    return composite


def _add_title(img: Image.Image, text: str) -> None:
    """Draw text at the top-left corner of an image."""
    draw = ImageDraw.Draw(img)
    draw.text((4, 2), text, fill=(255, 255, 255))


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

        sample_id = (
            batch["sample_id"][0]
            if isinstance(batch["sample_id"], (list, torch.Tensor))
            else batch["sample_id"]
        )

        overlay = draw_overlay(
            image_np, target_np, pred_np, ignore_np, box, point, sample_id=sample_id
        )
        out_path = output_dir / f"{sample_id}_overlay.png"
        overlay.save(str(out_path))
        logger.info("Saved overlay: %s", out_path)

        count += 1

    logger.info("Done. Saved %d overlays to %s", count, output_dir)


if __name__ == "__main__":
    main()
