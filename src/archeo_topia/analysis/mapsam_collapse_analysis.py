#!/usr/bin/env python3
"""Separate genuine prediction failures from metric quantization.

Fold A of the v0.3 leave-one-sheet-out evaluation scores 0.3498 with 19 of its
120 samples at IoU exactly zero, 10 of which predict no pixels at all.  A mean
IoU hides that, and "zero IoU" sounds like the model losing the mound.

It usually is not.  At full-tile resolution a mound covers about five pixels
on SAM's 256x256 decoder grid, so a prediction displaced by one or two pixels
has *no* overlap with the truth and scores zero while still sitting on the
right symbol.  This module measures the displacement instead of the overlap:

* ``centroid_distance`` -- between the predicted mask and the target.
* ``argmax_distance`` -- between the peak logit and the target centroid.  This
  is defined even when nothing crosses the threshold, which is the only way to
  ask where a zero-pixel prediction was *looking*.

Both are reported in source-tile pixels as well as decoder pixels, because
decoder pixels mean a different physical distance in every input window.

A zero-IoU sample is classified ``near_miss`` when it is within
``--near-threshold-px`` source pixels of the target, and ``wrong_object``
otherwise.  The split matters for what to do next: near misses are a
resolution and thresholding problem, and windowing addresses them; wrong
objects are a recognition problem, and it does not.

Usage::

    python -m archeo_topia.analysis.mapsam_collapse_analysis \\
        --config configs/mapsam/mapsam_v0_3_loso_foldA_pw20_cropon.yaml \\
        --checkpoint artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/checkpoints/final.pt \\
        --name fold_a_final --output-dir docs/mapsam/v004/analysis/collapse
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from archeo_topia.datasets.mapsam_dataset import MapSamDataset
from archeo_topia.training.train_mapsam_v0 import (
    forward_sam,
    load_checkpoint,
    load_config,
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
    parser = argparse.ArgumentParser(description="MapSAM zero-IoU failure characterization")
    parser.add_argument("--config", required=True, help="Config of the run being evaluated")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--name", required=True, help="Output file stem")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON output")
    parser.add_argument(
        "--near-threshold-px",
        type=float,
        default=40.0,
        help=(
            "Source-tile pixels within which a zero-IoU prediction counts as a near miss "
            "rather than a wrong object. The default is about twice a mound symbol's width."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def centroid(mask: torch.Tensor) -> tuple[float, float] | None:
    """Return the ``(x, y)`` centroid of a binary mask, or ``None`` if empty.

    Args:
        mask: 2D tensor with foreground > 0.

    Returns:
        The centroid in mask pixels, or ``None`` when nothing is foreground.
    """
    ys, xs = torch.nonzero(mask > 0, as_tuple=True)
    if xs.numel() == 0:
        return None
    return float(xs.float().mean()), float(ys.float().mean())


def argmax_xy(logits: torch.Tensor) -> tuple[float, float]:
    """Return the ``(x, y)`` position of the highest logit.

    Args:
        logits: 2D logit tensor.

    Returns:
        The peak position in mask pixels.
    """
    flat = int(torch.argmax(logits))
    width = logits.shape[1]
    return float(flat % width), float(flat // width)


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    """Euclidean distance between two points, or ``None`` if either is missing.

    Args:
        a: First point.
        b: Second point.

    Returns:
        The distance, or ``None``.
    """
    if a is None or b is None:
        return None
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def analyze_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    window_side_px: float,
    sample_id: str,
    sheet_id: str,
) -> dict[str, Any]:
    """Measure one sample's overlap and its displacement from the target.

    Args:
        logits: Raw logits ``(H, W)`` on the decoder grid.
        target: Binary target ``(H, W)`` on the same grid.
        window_side_px: Side length of the input window in source pixels.
        sample_id: Sample identifier.
        sheet_id: Map sheet identifier.

    Returns:
        A row of per-sample measurements.
    """
    pred = (torch.sigmoid(logits) > 0.5).float()
    gt_c = centroid(target)
    pred_c = centroid(pred)
    peak = argmax_xy(logits)

    intersection = float((pred * target).sum())
    union = float(pred.sum()) + float(target.sum()) - intersection
    iou = intersection / union if union > 0 else 0.0

    px_per_mask_px = window_side_px / logits.shape[0]
    centroid_dist = _distance(pred_c, gt_c)
    argmax_dist = _distance(peak, gt_c)

    return {
        "sample_id": sample_id,
        "sheet_id": sheet_id,
        "iou": round(iou, 6),
        "gt_pixels": int(target.sum()),
        "pred_pixels": int(pred.sum()),
        "logit_max": round(float(logits.max()), 4),
        "centroid_distance_mask_px": round(centroid_dist, 3) if centroid_dist is not None else "",
        "centroid_distance_source_px": round(centroid_dist * px_per_mask_px, 2)
        if centroid_dist is not None
        else "",
        "argmax_distance_mask_px": round(argmax_dist, 3) if argmax_dist is not None else "",
        "argmax_distance_source_px": round(argmax_dist * px_per_mask_px, 2)
        if argmax_dist is not None
        else "",
    }


def classify(row: dict[str, Any], near_threshold_px: float) -> str:
    """Label a sample as a hit, a near miss, or a wrong object.

    A prediction that overlaps the target at all is a ``hit``, however poorly.
    Among zero-IoU samples, the distinguishing question is not whether pixels
    were predicted but whether the model was looking at the right symbol, so
    the peak logit is used when no pixel crossed the threshold.

    Args:
        row: A row from :func:`analyze_sample`.
        near_threshold_px: Source-pixel radius for a near miss.

    Returns:
        One of ``"hit"``, ``"near_miss"`` or ``"wrong_object"``.
    """
    if row["iou"] > 0:
        return "hit"

    if row["pred_pixels"] > 0:
        distance = row["centroid_distance_source_px"]
    else:
        distance = row["argmax_distance_source_px"]

    if distance == "" or distance > near_threshold_px:
        return "wrong_object"
    return "near_miss"


def summarize(rows: list[dict[str, Any]], near_threshold_px: float) -> dict[str, Any]:
    """Aggregate per-sample rows into the headline counts.

    Args:
        rows: Rows from :func:`analyze_sample`, each carrying a ``class``.
        near_threshold_px: The radius the classification used.

    Returns:
        Summary mapping.
    """
    zero = [r for r in rows if r["iou"] == 0.0]
    near = [r for r in zero if r["class"] == "near_miss"]
    wrong = [r for r in zero if r["class"] == "wrong_object"]
    hits = [r for r in rows if r["iou"] > 0]

    def _median(values: list[float]) -> float | None:
        return round(statistics.median(values), 3) if values else None

    return {
        "n": len(rows),
        "near_threshold_px": near_threshold_px,
        "mean_iou": round(sum(r["iou"] for r in rows) / len(rows), 6) if rows else 0.0,
        "n_zero_iou": len(zero),
        "n_zero_prediction": sum(1 for r in zero if r["pred_pixels"] == 0),
        "n_near_miss": len(near),
        "n_wrong_object": len(wrong),
        "median_near_miss_distance_source_px": _median(
            [
                r["centroid_distance_source_px"]
                if r["pred_pixels"] > 0
                else r["argmax_distance_source_px"]
                for r in near
            ]
        ),
        "median_hit_centroid_distance_source_px": _median(
            [r["centroid_distance_source_px"] for r in hits if r["pred_pixels"] > 0]
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for the collapse characterization."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("archeo_topia.datasets.mapsam_dataset").setLevel(logging.WARNING)

    cfg = resolve_config(load_config(args.config))
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    device = torch.device(cfg["training"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available")
        sys.exit(1)

    from segment_anything import sam_model_registry

    base_checkpoint = Path(model_cfg["sam_checkpoint"])
    if not base_checkpoint.exists():
        logger.error("Base SAM checkpoint not found: %s", base_checkpoint)
        sys.exit(1)
    sam = sam_model_registry[model_cfg["model_type"]](checkpoint=str(base_checkpoint))
    ckpt = load_checkpoint(Path(args.checkpoint), sam, None)  # type: ignore[arg-type]
    sam.to(device)
    sam.eval()
    logger.info("Loaded %s (epoch %s)", args.checkpoint, ckpt.get("epoch", "?"))

    dataset = MapSamDataset(
        dataset_root=str(dataset_cfg["root"]),
        samples_path=str(dataset_cfg["samples"]),
        split=args.split,
        image_size=dataset_cfg["image_size"],
        sheets=dataset_cfg.get("eval_sheets"),
        window_px=dataset_cfg.get("window_px"),
    )

    rows: list[dict[str, Any]] = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers):
        with torch.no_grad():
            logits = forward_sam(
                sam,
                batch["image"],
                batch["box_prompt"],
                batch["point_prompt"],
                batch["point_label"],
                device,
            )
        target = torch.nn.functional.interpolate(
            batch["target_mask"].to(device), size=logits.shape[2:], mode="nearest"
        )
        wx0, wy0, wx1, wy1 = batch["window_xyxy"][0].tolist()
        row = analyze_sample(
            logits[0, 0],
            target[0, 0],
            window_side_px=((wx1 - wx0) + (wy1 - wy0)) / 2.0,
            sample_id=batch["sample_id"][0],
            sheet_id=str(batch.get("sheet_id", [""])[0]),
        )
        row["class"] = classify(row, args.near_threshold_px)
        rows.append(row)

    summary = summarize(rows, args.near_threshold_px)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{args.name}_collapse.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / f"{args.name}_collapse.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "window_px": dataset_cfg.get("window_px"),
                "summary": summary,
            },
            f,
            indent=2,
        )

    logger.info(
        "%d samples: %d zero-IoU (%d near miss, %d wrong object), %d predicting nothing",
        summary["n"],
        summary["n_zero_iou"],
        summary["n_near_miss"],
        summary["n_wrong_object"],
        summary["n_zero_prediction"],
    )
    logger.info("Wrote %s and %s", csv_path, json_path)


if __name__ == "__main__":
    main()
