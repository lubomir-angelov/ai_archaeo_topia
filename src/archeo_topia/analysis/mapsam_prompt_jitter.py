#!/usr/bin/env python3
"""Measure how a MapSAM checkpoint degrades as the prompt is displaced.

Every MapSAM number up to v0.3 uses a prompt derived from the annotation, so
it answers *"given the correct mound location, can the mound be segmented"*.
A detector will not localize that exactly.  This matters more for a windowed
run than for a full-tile one, because the window is prompt-centred: a 100 px
localization error is 4% of a 2400 px tile but 20% of a 512 px window, and
past roughly 240 px the mound starts leaving a 512 px window altogether.

The sweep runs an existing checkpoint over a grid of offset magnitudes and
directions.  No training happens, so it can invalidate a window-size
recommendation before any compute is spent on it.

Three arms are worth running:

* the full-tile checkpoint, where jitter is purely a worse prompt;
* the windowed checkpoint, where jitter is a worse prompt *and* a displaced
  input;
* the windowed checkpoint with ``--no-jitter-prompts``, where the window moves
  but SAM still receives the ground-truth prompt.  Comparing this against the
  second arm separates the two costs, which otherwise arrive together.

Report ``abs_error_source_px``.  Decoder IoU is not comparable between window
sizes -- a 512 px window magnifies the target, so the same absolute error
scores a far better IoU -- and the IoU>=0.5 rate inherits that.  Both are
written out, but the IoU columns are only readable *within* one arm, as a
degradation curve against that arm's own zero-offset value.

Usage::

    python -m archeo_topia.analysis.mapsam_prompt_jitter \\
        --config configs/mapsam/mapsam_v0_3_res_c_window512_pw20.yaml \\
        --checkpoint artifacts/models/mapsam/v0_3_res_c_window512_pw20/checkpoints/final.pt \\
        --label window512 --output-dir docs/mapsam/v004/analysis/jitter
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from archeo_topia.datasets.mapsam_dataset import MapSamDataset
from archeo_topia.training.train_mapsam_v0 import (
    compute_prediction_stats,
    forward_sam,
    load_checkpoint,
    load_config,
    resolve_config,
)

logger = logging.getLogger(__name__)

DEFAULT_OFFSETS = (0, 10, 25, 50, 100, 200, 250, 300)

# Eight compass directions.  Diagonals are unit-scaled so that every direction
# at one magnitude displaces the prompt by the same distance; using (1, 1)
# unscaled would make the diagonal arms 41% further out than the axial ones.
_D = 1.0 / math.sqrt(2.0)
DIRECTIONS: tuple[tuple[str, float, float], ...] = (
    ("E", 1.0, 0.0),
    ("NE", _D, -_D),
    ("N", 0.0, -1.0),
    ("NW", -_D, -_D),
    ("W", -1.0, 0.0),
    ("SW", -_D, _D),
    ("S", 0.0, 1.0),
    ("SE", _D, _D),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="MapSAM prompt-jitter robustness sweep")
    parser.add_argument("--config", required=True, help="Config of the run being evaluated")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--label",
        required=True,
        help="Short arm name used in output filenames and the arm column",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON output")
    parser.add_argument(
        "--offsets",
        default=",".join(str(o) for o in DEFAULT_OFFSETS),
        help="Comma-separated offset magnitudes in source pixels",
    )
    parser.add_argument(
        "--no-jitter-prompts",
        action="store_true",
        help="Move the input window only; keep the ground-truth point and box",
    )
    parser.add_argument(
        "--per-sample-dir",
        default=None,
        help=(
            "Where to write the per-sample JSONL. Defaults to the evaluated checkpoint's "
            "run directory, keeping the bulky per-sample rows out of the aggregated tables."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def run_dir_of(checkpoint: Path) -> Path:
    """Return the run directory a checkpoint belongs to.

    Checkpoints are written to ``<outputs.root>/checkpoints/<name>.pt``, so the
    run directory is two levels up. A checkpoint stored anywhere else falls
    back to its own parent.

    Args:
        checkpoint: Path to the checkpoint file.

    Returns:
        The run directory.
    """
    parent = checkpoint.parent
    return parent.parent if parent.name == "checkpoints" else parent


def build_conditions(
    offsets: list[int],
) -> list[tuple[int, str, tuple[float, float] | None]]:
    """Expand offset magnitudes into ``(magnitude, direction, offset)`` cases.

    Zero magnitude has no direction, so it is emitted once rather than eight
    times.

    Args:
        offsets: Offset magnitudes in source pixels.

    Returns:
        List of conditions, each a magnitude, a direction name and the
        ``(dx, dy)`` offset (``None`` at magnitude zero).
    """
    conditions: list[tuple[int, str, tuple[float, float] | None]] = []
    for magnitude in offsets:
        if magnitude == 0:
            conditions.append((0, "none", None))
            continue
        for name, ux, uy in DIRECTIONS:
            conditions.append((magnitude, name, (magnitude * ux, magnitude * uy)))
    return conditions


def summarize(stats: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-sample statistics for one condition.

    Args:
        stats: Per-sample dictionaries from :func:`compute_prediction_stats`.

    Returns:
        Mapping of aggregate name to value.

    Raises:
        ValueError: If *stats* is empty.
    """
    if not stats:
        raise ValueError("No samples to summarize")

    n = len(stats)
    ious = [s["iou"] for s in stats]
    return {
        "n": n,
        "mean_abs_error_source_px": sum(s.get("abs_error_source_px", 0.0) for s in stats) / n,
        "mean_gt_area_source_px": sum(s.get("gt_area_source_px", 0.0) for s in stats) / n,
        "mean_iou": sum(ious) / n,
        "iou_ge_0_5_rate": sum(1 for v in ious if v >= 0.5) / n,
        "zero_iou_rate": sum(1 for v in ious if v == 0.0) / n,
        "zero_prediction_rate": sum(
            1 for s in stats if s["pred_positive_pixels_threshold_0_5"] == 0
        )
        / n,
    }


def evaluate_condition(
    sam: torch.nn.Module,
    dataset: MapSamDataset,
    device: torch.device,
    num_workers: int,
) -> list[dict[str, Any]]:
    """Run the checkpoint over one jittered dataset.

    Args:
        sam: Loaded SAM model in eval mode.
        dataset: Dataset configured with the condition's jitter.
        device: Device to run on.
        num_workers: Data-loader worker count.

    Returns:
        Per-sample statistics for every sample in *dataset*.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    stats: list[dict[str, Any]] = []
    for batch in loader:
        with torch.no_grad():
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
        target_r = torch.nn.functional.interpolate(target, size=logits.shape[2:], mode="nearest")
        ignore_r = torch.nn.functional.interpolate(ignore, size=logits.shape[2:], mode="nearest")
        stats.extend(
            compute_prediction_stats(
                logits[:, 0:1],
                target_r,
                ignore_r,
                batch["box_prompt"].to(device),
                list(batch["sample_id"]),
                window_xyxy=batch.get("window_xyxy"),
                sheet_ids=list(batch.get("sheet_id", [])),
            )
        )
    return stats


def main(argv: list[str] | None = None) -> None:
    """Entry point for the prompt-jitter sweep."""
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

    window_px = dataset_cfg.get("window_px")
    if window_px is None and args.no_jitter_prompts:
        logger.error(
            "--no-jitter-prompts moves the input window only, but this run has no "
            "window (dataset.window_px is unset), so the arm would be a no-op."
        )
        sys.exit(1)

    offsets = [int(v) for v in args.offsets.split(",") if v.strip()]
    conditions = build_conditions(offsets)
    logger.info(
        "Arm '%s': window=%s, jitter_prompts=%s, %d conditions",
        args.label,
        window_px,
        not args.no_jitter_prompts,
        len(conditions),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for magnitude, direction, offset in conditions:
        dataset = MapSamDataset(
            dataset_root=str(dataset_cfg["root"]),
            samples_path=str(dataset_cfg["samples"]),
            split=args.split,
            image_size=dataset_cfg["image_size"],
            sheets=dataset_cfg.get("eval_sheets"),
            window_px=window_px,
            prompt_jitter_xy=offset,
            jitter_prompts=not args.no_jitter_prompts,
        )
        stats = evaluate_condition(sam, dataset, device, args.num_workers)
        for entry in stats:
            per_sample.append(
                {"arm": args.label, "offset_px": magnitude, "direction": direction, **entry}
            )
        row = {
            "arm": args.label,
            "window_px": window_px if window_px is not None else "full_tile",
            "jitter_prompts": not args.no_jitter_prompts,
            "offset_px": magnitude,
            "direction": direction,
            **summarize(stats),
        }
        rows.append(row)
        logger.info(
            "offset=%3d dir=%-4s  err=%7.1f src px  IoU=%.4f  IoU>=0.5=%.2f  zero-pred=%.2f",
            magnitude,
            direction,
            row["mean_abs_error_source_px"],
            row["mean_iou"],
            row["iou_ge_0_5_rate"],
            row["zero_prediction_rate"],
        )

    by_offset: list[dict[str, Any]] = []
    for magnitude in offsets:
        group = [r for r in rows if r["offset_px"] == magnitude]
        if not group:
            continue
        by_offset.append(
            {
                "arm": args.label,
                "offset_px": magnitude,
                "n_directions": len(group),
                **{
                    key: sum(r[key] for r in group) / len(group)
                    for key in (
                        "mean_abs_error_source_px",
                        "mean_gt_area_source_px",
                        "mean_iou",
                        "iou_ge_0_5_rate",
                        "zero_iou_rate",
                        "zero_prediction_rate",
                    )
                },
                "worst_direction_abs_error_source_px": max(
                    r["mean_abs_error_source_px"] for r in group
                ),
            }
        )

    stem = f"jitter_{args.label}"
    _write_csv(output_dir / f"{stem}_by_direction.csv", rows)
    _write_csv(output_dir / f"{stem}_by_offset.csv", by_offset)

    # Per-sample rows go to the evaluated run's own artifact directory, not
    # next to the aggregates.  A full sweep is tens of thousands of rows —
    # regenerable, and far larger than the tables it supports — and the
    # project keeps per-sample outputs under artifacts/ for exactly that
    # reason.  --per-sample-dir overrides it.
    per_sample_dir = (
        Path(args.per_sample_dir) if args.per_sample_dir else run_dir_of(Path(args.checkpoint))
    )
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = per_sample_dir / f"{stem}_per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for entry in per_sample:
            f.write(json.dumps(entry) + "\n")
    logger.info("Wrote %d per-sample rows to %s", len(per_sample), per_sample_path)
    with open(output_dir / f"{stem}_summary.json", "w") as f:
        json.dump(
            {
                "arm": args.label,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "window_px": window_px,
                "jitter_prompts": not args.no_jitter_prompts,
                "by_offset": by_offset,
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s_*.{csv,json,jsonl} to %s", stem, output_dir)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write *rows* as CSV, using the first row's keys as the header.

    Args:
        path: Output path.
        rows: Rows to write; must be non-empty and share a key set.
    """
    if not rows:
        logger.warning("No rows to write to %s", path)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
