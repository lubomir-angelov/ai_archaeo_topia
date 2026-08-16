#!/usr/bin/env python3
"""Debug prediction script for MapSAM checkpoints.

Generates overlay images showing ground truth, predictions, and prompts
for visual inspection.  Supports cached embeddings and prediction
statistics output.

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
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from archeo_topia.datasets.mapsam_dataset import MapSamDataset
from archeo_topia.datasets.mapsam_embedding_dataset import (
    MapSamEmbeddingDataset,
)
from archeo_topia.training.train_mapsam_v0 import (
    compute_prediction_stats,
    forward_sam,
    forward_sam_cached,
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
    parser.add_argument(
        "--use-cached-embeddings",
        action="store_true",
        help="Use pre-computed image embeddings from cache",
    )
    return parser.parse_args(argv)


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font at *size*, falling back to default.

    Args:
        size: Desired font size in pixels.

    Returns:
        PIL font object.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(
    image_np: np.ndarray,
    target_mask_np: np.ndarray,
    pred_mask_np: np.ndarray,
    ignore_mask_np: np.ndarray,
    box_prompt: list[float],
    point_prompt: list[float],
    sample_id: str,
    iou: float,
    dice: float,
    gt_positive_pixels: int,
    pred_positive_pixels: int,
    pred_prob_max: float,
) -> Image.Image:
    """Create a composite overlay image for debugging.

    Layout (5 panels, arranged in grid):
        Row 1: Image+prompts | GT mask | Prediction (threshold=0.5)
        Row 2: Ignore mask   | GT vs Pred overlap

    Each panel includes metrics in the title area.

    Args:
        image_np: RGB image array ``(H, W, 3)`` in [0, 255].
        target_mask_np: Binary target mask ``(H, W)`` in [0, 1].
        pred_mask_np: Binary predicted mask ``(H, W)`` in [0, 1].
        ignore_mask_np: Binary ignore mask ``(H, W)`` in [0, 1].
        box_prompt: Bounding box ``[x1, y1, x2, y2]``.
        point_prompt: Point ``[x, y]``.
        sample_id: Sample identifier for labeling.
        iou: Intersection-over-union score.
        dice: Dice coefficient.
        gt_positive_pixels: Count of positive pixels in GT.
        pred_positive_pixels: Count of positive pixels in prediction.
        pred_prob_max: Maximum prediction probability.

    Returns:
        PIL Image with the composite overlay.
    """
    h, w = image_np.shape[:2]
    small_font = _try_font(11)

    panels: list[tuple[Image.Image, str]] = []

    info_text = (
        f"{sample_id}  IoU={iou:.3f}  Dice={dice:.3f}  "
        f"GT+={gt_positive_pixels}  Pred+={pred_positive_pixels}  Pmax={pred_prob_max:.3f}"
    )

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
    panels.append((p1, "Image + bbox + point"))

    # Panel 2: GT mask
    gt_arr = np.clip(target_mask_np * 255, 0, 255).astype(np.uint8)
    p2 = Image.fromarray(gt_arr, mode="L").convert("RGB")
    panels.append((p2, "Ground truth mask"))

    # Panel 3: Pred mask
    pred_arr = np.clip(pred_mask_np * 255, 0, 255).astype(np.uint8)
    p3 = Image.fromarray(pred_arr, mode="L").convert("RGB")
    panels.append((p3, f"Prediction, threshold=0.5\n{info_text}"))

    # Panel 4: Ignore mask
    ign_arr = np.clip(ignore_mask_np * 255, 0, 255).astype(np.uint8)
    p4 = Image.fromarray(ign_arr, mode="L").convert("RGB")
    panels.append((p4, "Ignore mask"))

    # Panel 5: GT vs Pred overlap
    overlap = np.zeros((h, w, 3), dtype=np.uint8)
    gt_only = (target_mask_np > 0.5) & (pred_mask_np <= 0.5)
    pred_only = (target_mask_np <= 0.5) & (pred_mask_np > 0.5)
    both = (target_mask_np > 0.5) & (pred_mask_np > 0.5)

    overlap[gt_only] = [0, 255, 0]
    overlap[pred_only] = [255, 0, 255]
    overlap[both] = [255, 255, 0]

    p5 = Image.fromarray(overlap)
    panels.append((p5, "GT vs Pred\nyellow=overlap green=GT magenta=Pred"))

    # Layout: 3 panels top, 2 panels bottom
    top_w = w * 3
    bottom_w = w * 2

    top_row = Image.new("RGB", (top_w, h))
    for i, (img, _) in enumerate(panels[:3]):
        top_row.paste(img, (i * w, 0))

    bottom_row = Image.new("RGB", (bottom_w, h))
    for i, (img, _) in enumerate(panels[3:]):
        bottom_row.paste(img, (i * w, 0))

    # Add labels
    label_h = 24
    label_row = Image.new("RGB", (top_w, label_h), color=(30, 30, 30))
    ld = ImageDraw.Draw(label_row)
    panel_labels = [p[1] for p in panels[:3]]
    x = 0
    for label in panel_labels:
        ld.text((x + w // 2 - 40, 4), label, fill="white", font=small_font)
        x += w

    label_row2 = Image.new("RGB", (bottom_w, label_h), color=(30, 30, 30))
    ld2 = ImageDraw.Draw(label_row2)
    panel_labels2 = [p[1] for p in panels[3:]]
    x = 0
    for label in panel_labels2:
        lines = label.split("\n")
        for li, line in enumerate(lines):
            ld2.text((x + w // 2 - 60, 4 + li * 12), line, fill="white", font=small_font)
        x += w

    total_w = max(top_w, bottom_w)
    total_h = label_h * 2 + h * 2
    composite = Image.new("RGB", (total_w, total_h), color=(10, 10, 10))
    composite.paste(top_row, (0, 0))
    composite.paste(label_row, (0, h))
    composite.paste(bottom_row, (0, h + label_h))
    composite.paste(label_row2, (0, h * 2 + label_h))

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

    use_cached = args.use_cached_embeddings
    if use_cached:
        ds = MapSamEmbeddingDataset(
            dataset_root=str(dataset_root),
            samples_path=str(samples_path),
            split=args.split,
            image_size=dataset_cfg["image_size"],
            model_type=model_type,
        )
    else:
        ds = MapSamDataset(
            dataset_root=str(dataset_root),
            samples_path=str(samples_path),
            split=args.split,
            image_size=dataset_cfg["image_size"],
        )

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    all_stats: list[dict[str, Any]] = []
    count = 0
    for batch in loader:
        if count >= args.max_samples:
            break

        with torch.no_grad():
            if use_cached:
                logits = forward_sam_cached(
                    sam,
                    batch["image_embedding"],
                    batch["box_prompt"],
                    batch["point_prompt"],
                    batch["point_label"],
                    device,
                )
            else:
                logits = forward_sam(
                    sam,
                    batch["image"],
                    batch["box_prompt"],
                    batch["point_prompt"],
                    batch["point_label"],
                    device,
                )

        target = batch["target_mask"].to(device)
        ignore = batch["ignore_mask"].to(device)
        box_prompt = batch["box_prompt"].to(device)

        target_r = torch.nn.functional.interpolate(target, size=logits.shape[2:], mode="nearest")
        ignore_r = torch.nn.functional.interpolate(ignore, size=logits.shape[2:], mode="nearest")

        pred = torch.sigmoid(logits[:, 0:1]) > 0.5
        pred_resized = torch.nn.functional.interpolate(
            pred.float(),
            size=(dataset_cfg["image_size"], dataset_cfg["image_size"]),
            mode="nearest",
        ).squeeze()

        if use_cached:
            img_path = dataset_root / batch["image_path"][0]
            from PIL import Image as PilImage

            img_pil = PilImage.open(str(img_path)).convert("RGB")
            img_pil = img_pil.resize(
                (dataset_cfg["image_size"], dataset_cfg["image_size"]),
                PilImage.BILINEAR,
            )
            image_np = np.array(img_pil).astype(np.uint8)
        else:
            image_np = (batch["image"].squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(
                np.uint8
            )
        target_np = batch["target_mask"].squeeze().cpu().numpy()
        ignore_np = batch["ignore_mask"].squeeze().cpu().numpy()
        pred_np = pred_resized.cpu().numpy()
        box = batch["box_prompt"].squeeze().cpu().tolist()
        point = batch["point_prompt"].squeeze().cpu().tolist()

        sample_ids = _get_sample_ids(batch)
        sid = sample_ids[0] if sample_ids else f"sample_{count}"

        stats = compute_prediction_stats(
            logits[:, 0:1], target_r, ignore_r, box_prompt, sample_ids
        )
        all_stats.extend(stats)

        s = stats[0] if stats else {}

        overlay = draw_overlay(
            image_np,
            target_np,
            pred_np,
            ignore_np,
            box,
            point,
            sample_id=sid,
            iou=s.get("iou", 0.0),
            dice=s.get("dice", 0.0),
            gt_positive_pixels=s.get("gt_positive_pixels", 0),
            pred_positive_pixels=s.get("pred_positive_pixels_threshold_0_5", 0),
            pred_prob_max=s.get("pred_probability_max", 0.0),
        )

        out_path = output_dir / f"{sid}_overlay.png"
        overlay.save(str(out_path))
        logger.info(
            "Saved overlay: %s (IoU=%.4f Dice=%.4f)", out_path, s.get("iou", 0), s.get("dice", 0)
        )

        count += 1

    stats_path = output_dir / f"prediction_stats_{args.split}.jsonl"
    with open(stats_path, "w") as f:
        for entry in all_stats:
            f.write(json.dumps(entry) + "\n")
    logger.info("Saved prediction stats: %s", stats_path)

    if all_stats:
        aggregates = {
            "mean_iou": sum(s["iou"] for s in all_stats) / len(all_stats),
            "mean_dice": sum(s["dice"] for s in all_stats) / len(all_stats),
            "mean_gt_positive_pixels": sum(s["gt_positive_pixels"] for s in all_stats)
            / len(all_stats),
            "mean_pred_positive_pixels": sum(
                s["pred_positive_pixels_threshold_0_5"] for s in all_stats
            )
            / len(all_stats),
            "mean_prediction_area_ratio": sum(s["prediction_area_ratio"] for s in all_stats)
            / len(all_stats),
            "mean_target_area_ratio": sum(s["target_area_ratio"] for s in all_stats)
            / len(all_stats),
        }
        agg_path = output_dir / f"prediction_stats_{args.split}_aggregate.json"
        with open(agg_path, "w") as f:
            json.dump(aggregates, f, indent=2)
        logger.info(
            "Aggregate IoU: %.4f  Dice: %.4f", aggregates["mean_iou"], aggregates["mean_dice"]
        )

    logger.info("Done. Saved %d overlays to %s", count, output_dir)


def _get_sample_ids(batch: dict[str, Any]) -> list[str]:
    """Extract sample IDs from a batch.

    Args:
        batch: Batch dictionary from the data loader.

    Returns:
        List of sample ID strings.
    """
    sid = batch.get("sample_id", [])
    if isinstance(sid, str):
        return [sid]
    if isinstance(sid, torch.Tensor):
        return sid.tolist()
    if isinstance(sid, list):
        return sid
    return [f"unknown_{i}" for i in range(batch.get("target_mask", torch.tensor([0])).shape[0])]


if __name__ == "__main__":
    main()
