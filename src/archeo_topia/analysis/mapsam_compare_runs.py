#!/usr/bin/env python3
"""Aggregate MapSAM runs into one comparison table.

Collects the per-fold and per-arm numbers the v0.3 experiments need, from the
``metrics.json`` and ``prediction_stats_val_e*.jsonl`` files each run writes.

Two things it refuses to do naively.

It recomputes the peak IoU from the epoch history rather than reading
``metrics.json``'s ``best_val_iou``, which is the IoU *at the best-val-loss
epoch* and can differ substantially (0.6464 against a true peak of 0.6838 on
one v0.2 run).

And it reports the final-epoch IoU as the primary figure, with the peak
labelled as an oracle upper bound.  With three sheets and no validation split
there is no leak-free way to pick an epoch, so selecting one by looking at the
evaluation sheet is not a result.

Usage::

    python -m archeo_topia.analysis.mapsam_compare_runs \\
        --runs-dir artifacts/models/mapsam --prefix v0_3_loso \\
        --output-dir docs/mapsam/analysis/v0_3
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

from archeo_topia.analysis.mapsam_size_analysis import (
    load_stats,
    resolve_stats_file,
)

logger = logging.getLogger(__name__)


def peak_from_history(history: list[dict[str, Any]]) -> tuple[float, int | None]:
    """Find the highest validation IoU and the epoch it occurred at.

    Args:
        history: The ``history`` list from a run's ``metrics.json``.

    Returns:
        ``(peak_iou, epoch)``, or ``(0.0, None)`` if nothing was validated.
    """
    validated = [e for e in history if e.get("val") and "mean_iou" in e["val"]]
    if not validated:
        return 0.0, None
    best = max(validated, key=lambda e: e["val"]["mean_iou"])
    return float(best["val"]["mean_iou"]), int(best["epoch"])


def summarise_run(run_dir: Path) -> dict[str, Any] | None:
    """Summarise one training run.

    Args:
        run_dir: The run's ``outputs.root`` directory.

    Returns:
        A flat summary dictionary, or ``None`` if the run has no metrics.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        logger.warning("No metrics.json in %s — skipping", run_dir)
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics.get("history", [])
    peak_iou, peak_epoch = peak_from_history(history)

    config_path = run_dir / "config_resolved.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    dataset_cfg = config.get("dataset", {})

    summary: dict[str, Any] = {
        "run": run_dir.name,
        "eval_sheets": ",".join(dataset_cfg.get("eval_sheets") or []),
        "train_sheets": ",".join(dataset_cfg.get("train_sheets") or []),
        "train_subsample": dataset_cfg.get("train_subsample") or "",
        "window_px": dataset_cfg.get("window_px") or "",
        "epochs": metrics.get("epochs"),
        "final_iou": round(float(metrics.get("final_val_iou") or 0.0), 4),
        "final_dice": round(float(metrics.get("final_val_dice") or 0.0), 4),
        "peak_iou_oracle": round(peak_iou, 4),
        "peak_epoch_oracle": peak_epoch,
    }

    try:
        stats = load_stats(resolve_stats_file(run_dir, "val", None))
    except FileNotFoundError:
        logger.warning("No val prediction stats in %s", run_dir)
        return summary

    ious = [s["iou"] for s in stats]
    summary.update(
        {
            "eval_samples": len(stats),
            "median_iou": round(statistics.median(ious), 4),
            "min_iou": round(min(ious), 4),
            "max_iou": round(max(ious), 4),
            "zero_iou_count": sum(1 for v in ious if v == 0.0),
            "frac_iou_ge_025": round(sum(1 for v in ious if v >= 0.25) / len(ious), 4),
            "frac_iou_ge_050": round(sum(1 for v in ious if v >= 0.50) / len(ious), 4),
            "frac_iou_ge_075": round(sum(1 for v in ious if v >= 0.75) / len(ious), 4),
            "mean_gt_px": round(statistics.fmean(s["gt_positive_pixels"] for s in stats), 2),
            "mean_pred_px": round(
                statistics.fmean(s["pred_positive_pixels_threshold_0_5"] for s in stats), 2
            ),
        }
    )

    gt_total = sum(s["gt_positive_pixels"] for s in stats)
    pred_total = sum(s["pred_positive_pixels_threshold_0_5"] for s in stats)
    summary["pred_gt_area_ratio"] = round(pred_total / gt_total, 4) if gt_total else 0.0

    if all("abs_error_px" in s for s in stats):
        summary["mean_abs_error_px"] = round(statistics.fmean(s["abs_error_px"] for s in stats), 2)
    # The comparable column: decoder-space numbers mean different physical
    # distances in runs with different windows.
    if all("abs_error_source_px" in s for s in stats):
        summary["mean_abs_error_source_px"] = round(
            statistics.fmean(s["abs_error_source_px"] for s in stats), 1
        )
        summary["mean_gt_area_source_px"] = round(
            statistics.fmean(s["gt_area_source_px"] for s in stats), 1
        )
    if all("logit_max" in s for s in stats):
        summary["mean_logit_max"] = round(statistics.fmean(s["logit_max"] for s in stats), 2)
        summary["mean_logit_min"] = round(statistics.fmean(s["logit_min"] for s in stats), 2)
        summary["mean_prob_in_gt"] = round(
            statistics.fmean(s["prob_mean_in_gt"] for s in stats), 4
        )
        summary["mean_prob_in_background"] = round(
            statistics.fmean(s["prob_mean_in_background"] for s in stats), 8
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write summaries to CSV, unioning keys across rows.

    Args:
        path: Destination path.  Parent directories are created.
        rows: Summary dictionaries.
    """
    if not rows:
        logger.warning("Nothing to write to %s", path)
        return
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [k for k in row if k not in fieldnames]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Aggregate MapSAM runs for comparison")
    parser.add_argument("--runs-dir", default="artifacts/models/mapsam")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Only summarise run directories starting with this, e.g. v0_3_loso",
    )
    parser.add_argument("--output-dir", default=None, help="Where to write the CSV")
    parser.add_argument("--name", default=None, help="CSV basename (default: the prefix)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        logger.error("Runs directory not found: %s", runs_dir)
        sys.exit(1)

    candidates = sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(args.prefix)
    )
    rows = [s for d in candidates if (s := summarise_run(d)) is not None]

    if not rows:
        logger.error("No runs matched prefix '%s' under %s", args.prefix, runs_dir)
        sys.exit(1)

    for row in rows:
        logger.info(
            "%-44s eval=%-12s n=%-4s final IoU %.4f (oracle peak %.4f @ e%s)",
            row["run"],
            row["eval_sheets"] or "-",
            row.get("eval_samples", "?"),
            row["final_iou"],
            row["peak_iou_oracle"],
            row["peak_epoch_oracle"],
        )

    if args.output_dir:
        write_csv(Path(args.output_dir) / f"{args.name or args.prefix}.csv", rows)


if __name__ == "__main__":
    main()
