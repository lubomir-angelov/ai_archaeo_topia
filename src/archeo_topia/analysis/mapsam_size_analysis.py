#!/usr/bin/env python3
"""Relate MapSAM segmentation quality to ground-truth target size.

Answers the v0.3 question "do the smallest mound masks systematically perform
worse?" purely from artifacts already on disk: the
``prediction_stats_val_e<N>.jsonl`` files a training run writes when
``debug.log_prediction_stats`` is enabled.  No model, no GPU, no inference.

The decoder emits 256x256 logits, so a mound that covers ~21 px on a ~2400 px
source tile survives as roughly 5 positive pixels.  This module buckets samples
by that surviving pixel count and reports quality per bucket, plus the
correlation between size and IoU.

It also reports *encoder token coverage*: the 256x256 logit grid is upsampled
from the frozen ViT's 64x64 token grid, so token coverage is the tighter
resolution floor and the quantity an input crop actually changes.

Usage::

    python -m archeo_topia.analysis.mapsam_size_analysis \\
        --run-dir artifacts/models/mapsam/v0_2_decoder_only_pw200_cropoff \\
        --epoch 25 \\
        --output-dir docs/mapsam/v003/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Logit resolution the decoder emits, and the ViT token grid it is upsampled
# from.  Both are properties of SAM v1 at a 1024 input, not of this dataset.
LOGIT_RESOLUTION = 256
ENCODER_TOKEN_GRID = 64

DEFAULT_BUCKET_EDGES = (1, 4, 6, 9)


class Bucket(NamedTuple):
    """A half-open target-size bucket.

    Attributes:
        label: Human-readable bucket name, e.g. ``"4-5 px"``.
        low: Inclusive lower bound in ground-truth positive pixels.
        high: Exclusive upper bound, or ``None`` for the open-ended top bucket.
    """

    label: str
    low: int
    high: int | None

    def contains(self, px: int) -> bool:
        """Return whether *px* falls in this bucket.

        Args:
            px: Ground-truth positive pixel count at logit resolution.

        Returns:
            True if the count falls inside the bucket's bounds.
        """
        return px >= self.low and (self.high is None or px < self.high)


def build_buckets(edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES) -> list[Bucket]:
    """Build half-open buckets from ascending lower bounds.

    Args:
        edges: Ascending inclusive lower bounds.  The last edge opens a
            bucket with no upper bound.

    Returns:
        List of buckets covering ``[edges[0], inf)``.

    Raises:
        ValueError: If fewer than two edges are given or they are not
            strictly ascending.
    """
    if len(edges) < 2:
        raise ValueError(f"Need at least two bucket edges, got {edges}")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"Bucket edges must be strictly ascending, got {edges}")

    buckets: list[Bucket] = []
    for low, high in zip(edges, edges[1:], strict=False):
        label = f"{low} px" if high - low == 1 else f"{low}-{high - 1} px"
        buckets.append(Bucket(label, low, high))
    buckets.append(Bucket(f"{edges[-1]}+ px", edges[-1], None))
    return buckets


def sheet_id_from_sample_id(sample_id: str) -> str:
    """Recover the map sheet ID from a sample ID.

    Sample IDs are built as ``{split}_{sheet}_{tile}_{component:06d}`` by
    ``generate_mapsam_prompts.build_training_sample``, and the sheet ID itself
    contains underscores (``K-35-8-G-a``), so the sheet is everything between
    the leading split token and the trailing tile/component tokens.

    Args:
        sample_id: Sample identifier as written into the stats rows.

    Returns:
        The sheet ID, or the whole input if it does not match the pattern.
    """
    parts = sample_id.split("_")
    if len(parts) < 4:
        return sample_id
    return "_".join(parts[1:-2])


def load_stats(path: Path) -> list[dict[str, Any]]:
    """Load a ``prediction_stats_*.jsonl`` file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of per-sample stat rows, each with ``sheet_id`` added.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prediction stats file not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("sheet_id", sheet_id_from_sample_id(row["sample_id"]))
            rows.append(row)
    return rows


def resolve_stats_file(run_dir: Path, split: str, epoch: int | None) -> Path:
    """Find the stats file for a run, defaulting to its last logged epoch.

    Args:
        run_dir: A training run's ``outputs.root`` directory.
        split: ``"val"`` or ``"train"``.
        epoch: Epoch to read, or ``None`` for the highest available.

    Returns:
        Path to the chosen JSONL file.

    Raises:
        FileNotFoundError: If the run has no stats files for *split*.
    """
    if epoch is not None:
        return run_dir / f"prediction_stats_{split}_e{epoch}.jsonl"

    candidates = sorted(
        run_dir.glob(f"prediction_stats_{split}_e*.jsonl"),
        key=lambda p: int(p.stem.rsplit("_e", 1)[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No prediction_stats_{split}_e*.jsonl under {run_dir}")
    return candidates[-1]


def token_coverage(gt_pixels: float) -> float:
    """Convert a logit-resolution area to ViT encoder token coverage.

    Args:
        gt_pixels: Positive pixel count on the 256x256 logit grid.

    Returns:
        Equivalent side length in 64x64 encoder tokens.  A value below 1.0
        means the target is smaller than a single ViT patch, which no
        decoder-side change can recover while the encoder is frozen.
    """
    side_at_logits = math.sqrt(max(gt_pixels, 0.0))
    return side_at_logits * ENCODER_TOKEN_GRID / LOGIT_RESOLUTION


def symmetric_difference_px(row: dict[str, Any]) -> float:
    """Absolute boundary error of one prediction, in logit-resolution pixels.

    IoU is a *ratio*, so the same absolute error scores far worse on a small
    target than a large one: a one-pixel miss on a 3 px mound costs 33 IoU
    points, on a 12 px mound about 15.  The symmetric difference
    ``|pred XOR gt|`` is the same error expressed in absolute pixels, which
    makes size buckets comparable and separates "the model sees less" from
    "the denominator is smaller".

    The intersection is recovered algebraically from the stored IoU and the
    two areas, since the per-sample rows do not persist it directly:
    ``IoU = I / (gt + pred - I)`` gives ``I = IoU * (gt + pred) / (1 + IoU)``.

    Args:
        row: One per-sample stat row.

    Returns:
        Symmetric difference in pixels.
    """
    gt = float(row["gt_positive_pixels"])
    pred = float(row["pred_positive_pixels_threshold_0_5"])
    iou = float(row["iou"])
    intersection = iou * (gt + pred) / (1.0 + iou) if iou > 0 else 0.0
    return gt + pred - 2.0 * intersection


def rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, computed with the standard library only.

    Pearson correlation understates a relationship that is monotone but
    strongly non-linear, which is exactly the shape seen here: quality falls
    off a cliff below ~4 px and is flat above it.  Ranking first makes that
    visible without adding a numerical dependency.

    Args:
        xs: First series.
        ys: Second series, same length as *xs*.

    Returns:
        Spearman coefficient, or ``None`` when it is undefined.
    """
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    try:
        return round(statistics.correlation(_ranks(xs), _ranks(ys)), 4)
    except statistics.StatisticsError:
        return None


def _ranks(values: list[float]) -> list[float]:
    """Rank *values* ascending, averaging ranks within ties.

    Args:
        values: Series to rank.

    Returns:
        Fractional ranks in the original order.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _summarise(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Summarise one group of per-sample stat rows.

    Args:
        rows: Per-sample stat rows.
        label: Name for the group (bucket label, sheet ID or ``"all"``).

    Returns:
        A flat dictionary of summary metrics, suitable for a CSV row.
    """
    n = len(rows)
    if n == 0:
        return {"group": label, "samples": 0}

    ious = [r["iou"] for r in rows]
    dices = [r["dice"] for r in rows]
    gt = [r["gt_positive_pixels"] for r in rows]
    pred = [r["pred_positive_pixels_threshold_0_5"] for r in rows]
    gt_total = sum(gt)

    return {
        "group": label,
        "samples": n,
        "mean_iou": round(statistics.fmean(ious), 4),
        "median_iou": round(statistics.median(ious), 4),
        "min_iou": round(min(ious), 4),
        "max_iou": round(max(ious), 4),
        "mean_dice": round(statistics.fmean(dices), 4),
        "median_dice": round(statistics.median(dices), 4),
        "zero_iou_count": sum(1 for v in ious if v == 0.0),
        "zero_iou_rate": round(sum(1 for v in ious if v == 0.0) / n, 4),
        "frac_iou_ge_025": round(sum(1 for v in ious if v >= 0.25) / n, 4),
        "frac_iou_ge_050": round(sum(1 for v in ious if v >= 0.50) / n, 4),
        "frac_iou_ge_075": round(sum(1 for v in ious if v >= 0.75) / n, 4),
        "mean_gt_px": round(statistics.fmean(gt), 2),
        "mean_pred_px": round(statistics.fmean(pred), 2),
        "pred_gt_area_ratio": round(sum(pred) / gt_total, 4) if gt_total else 0.0,
        "mean_abs_error_px": round(
            statistics.fmean([symmetric_difference_px(r) for r in rows]), 2
        ),
        "mean_encoder_tokens": round(token_coverage(statistics.fmean(gt)), 3),
    }


def size_iou_correlation(rows: list[dict[str, Any]]) -> float | None:
    """Correlate ground-truth target size against IoU.

    Uses :func:`statistics.correlation`, which is in the standard library on
    Python 3.12, rather than pulling a numerical stack into an analysis that
    reads a few hundred rows.

    Args:
        rows: Per-sample stat rows.

    Returns:
        Pearson correlation coefficient, or ``None`` when it is undefined
        (fewer than two samples, or either variable constant).
    """
    if len(rows) < 2:
        return None
    gt = [float(r["gt_positive_pixels"]) for r in rows]
    iou = [float(r["iou"]) for r in rows]
    try:
        return round(statistics.correlation(gt, iou), 4)
    except statistics.StatisticsError:
        # Raised when either series has zero variance — a real outcome here,
        # e.g. an overfit run where every IoU is 1.0.
        return None


def analyse(
    rows: list[dict[str, Any]],
    buckets: list[Bucket],
) -> dict[str, Any]:
    """Run the full size-vs-quality analysis over one set of stat rows.

    Args:
        rows: Per-sample stat rows.
        buckets: Target-size buckets to group by.

    Returns:
        Dictionary with ``overall``, ``buckets``, ``sheets`` and
        ``size_iou_correlation`` entries.
    """
    unbucketed = [r for r in rows if not any(b.contains(r["gt_positive_pixels"]) for b in buckets)]
    if unbucketed:
        logger.warning(
            "%d sample(s) fell outside every bucket (gt px: %s)",
            len(unbucketed),
            sorted({r["gt_positive_pixels"] for r in unbucketed}),
        )

    # Derived here as well as in load_stats so analyse() works on any rows,
    # including ones read straight out of a metrics.json history block.
    for row in rows:
        row.setdefault("sheet_id", sheet_id_from_sample_id(row["sample_id"]))

    sheets = sorted({r["sheet_id"] for r in rows})
    return {
        "overall": _summarise(rows, "all"),
        "buckets": [
            _summarise([r for r in rows if b.contains(r["gt_positive_pixels"])], b.label)
            for b in buckets
        ],
        "sheets": [_summarise([r for r in rows if r["sheet_id"] == s], s) for s in sheets],
        "size_iou_correlation": size_iou_correlation(rows),
        "size_iou_rank_correlation": rank_correlation(
            [float(r["gt_positive_pixels"]) for r in rows],
            [float(r["iou"]) for r in rows],
        ),
        # If this is ~0, absolute boundary error does not grow with target
        # size, so any IoU trend across buckets is the denominator, not the
        # model.
        "size_abs_error_rank_correlation": rank_correlation(
            [float(r["gt_positive_pixels"]) for r in rows],
            [symmetric_difference_px(r) for r in rows],
        ),
    }


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Write summary rows to CSV, skipping empty groups.

    Args:
        path: Destination CSV path.  Parent directories are created.
        summaries: Summary dictionaries as returned by :func:`analyse`.
    """
    populated = [s for s in summaries if s.get("samples", 0) > 0]
    if not populated:
        logger.warning("No populated groups to write to %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(populated[0].keys()))
        writer.writeheader()
        writer.writerows(populated)
    logger.info("Wrote %s (%d rows)", path, len(populated))


def format_markdown(run_name: str, stats_file: Path, result: dict[str, Any]) -> str:
    """Render one run's analysis as a Markdown section.

    Args:
        run_name: Name of the training run.
        stats_file: The stats file the numbers came from.
        result: Analysis result from :func:`analyse`.

    Returns:
        Markdown text, ending in a blank line.
    """
    cols = [
        ("group", "Group"),
        ("samples", "N"),
        ("mean_iou", "mean IoU"),
        ("median_iou", "median IoU"),
        ("mean_dice", "mean Dice"),
        ("zero_iou_rate", "zero-IoU"),
        ("frac_iou_ge_050", "IoU>=0.5"),
        ("mean_gt_px", "GT px"),
        ("mean_pred_px", "pred px"),
        ("mean_abs_error_px", "abs err px"),
        ("mean_encoder_tokens", "tokens"),
    ]

    def table(rows: list[dict[str, Any]]) -> list[str]:
        populated = [r for r in rows if r.get("samples", 0) > 0]
        if not populated:
            return ["_(no samples)_", ""]
        out = [
            "| " + " | ".join(h for _, h in cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        out += ["| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |" for r in populated]
        out.append("")
        return out

    corr = result["size_iou_correlation"]
    lines = [
        f"### {run_name}",
        "",
        f"Source: `{stats_file}`",
        "",
        f"Target size vs IoU correlation: **{'n/a' if corr is None else corr}** "
        f"(Pearson), **{result['size_iou_rank_correlation']}** (Spearman)",
        "",
        "Target size vs *absolute* boundary error (Spearman): "
        f"**{result['size_abs_error_rank_correlation']}**",
        "",
        "By target size:",
        "",
    ]
    lines += table(result["buckets"])
    lines += ["By sheet:", ""]
    lines += table(result["sheets"])
    lines += ["Overall:", ""]
    lines += table([result["overall"]])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="MapSAM target-size vs quality analysis")
    parser.add_argument(
        "--run-dir",
        required=True,
        nargs="+",
        help="One or more training run directories (outputs.root)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to read (default: the highest logged epoch in each run)",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument(
        "--bucket-edges",
        default=",".join(str(e) for e in DEFAULT_BUCKET_EDGES),
        help="Comma-separated ascending lower bounds (default: 1,4,6,9)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV and Markdown artifacts (default: report to stdout only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        buckets = build_buckets(tuple(int(e) for e in args.bucket_edges.split(",")))
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    sections: list[str] = []

    for run in args.run_dir:
        run_dir = Path(run)
        run_name = run_dir.name
        try:
            stats_file = resolve_stats_file(run_dir, args.split, args.epoch)
            rows = load_stats(stats_file)
        except FileNotFoundError as exc:
            logger.error("%s — skipping %s", exc, run_name)
            continue

        if not rows:
            logger.warning("%s is empty — skipping %s", stats_file, run_name)
            continue

        result = analyse(rows, buckets)
        corr = result["size_iou_correlation"]
        logger.info(
            "%s (%s, %d samples): mean IoU %.4f, size/IoU correlation %s",
            run_name,
            stats_file.name,
            result["overall"]["samples"],
            result["overall"]["mean_iou"],
            "n/a" if corr is None else f"{corr:.4f}",
        )
        for b in result["buckets"]:
            if b.get("samples", 0):
                logger.info(
                    "    %-8s n=%-4d mean IoU %.4f  zero-IoU %.2f  abs err %.2f px  tokens %.2f",
                    b["group"],
                    b["samples"],
                    b["mean_iou"],
                    b["zero_iou_rate"],
                    b["mean_abs_error_px"],
                    b["mean_encoder_tokens"],
                )

        sections.append(format_markdown(run_name, stats_file, result))

        if output_dir is not None:
            write_csv(
                output_dir / f"{run_name}_size_analysis.csv",
                result["buckets"] + result["sheets"] + [result["overall"]],
            )
            json_path = output_dir / f"{run_name}_size_analysis.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "run": run_name,
                "stats_file": str(stats_file),
                "logit_resolution": LOGIT_RESOLUTION,
                "encoder_token_grid": ENCODER_TOKEN_GRID,
                **result,
            }
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            logger.info("Wrote %s", json_path)

    if not sections:
        logger.error("No runs produced results")
        sys.exit(1)

    if output_dir is not None:
        md_path = output_dir / "resolution_analysis.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        header = "# MapSAM target size vs segmentation quality\n\n"
        md_path.write_text(header + "\n".join(sections), encoding="utf-8")
        logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
