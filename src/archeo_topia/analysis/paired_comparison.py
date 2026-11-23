#!/usr/bin/env python3
"""Paired discordant counts between two systems on the same annotations (protocol C5).

Two systems scored on the same annotations are compared on the annotations
they disagree about, not on the difference of their rates: the discordant
counts and the exact two-sided binomial p-value (McNemar, exact) are the
pre-registered paired statistic.

Two kinds of input, both read from files an opening wrote:

``detector``
    Two detector ``metrics.json`` files. An annotation is a hit when it is
    matched within the primary recall radius at the headline confidence, pooled.

``end_to_end``
    Two ``end_to_end.json`` pair files. An annotation is a hit when its IoU in
    the chosen prompt arm reaches the chosen threshold, pooled.

The printed counts are the inputs to the wording of ``TEST_RESULTS.md``.

Usage:
    python -m archeo_topia.analysis.paired_comparison detector \\
        --first A/metrics.json --second B/metrics.json --output paired.json

    python -m archeo_topia.analysis.paired_comparison end_to_end \\
        --first A/end_to_end.json --second B/end_to_end.json \\
        --arm fixed --metric iou_0.5 --output paired.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from archeo_topia.analysis.detection_metrics import (
    RECALL_RADII,
    confidence_key,
    paired_discordance,
)
from archeo_topia.analysis.mapsam_end_to_end import BOX_MODES, IOU_THRESHOLDS
from archeo_topia.analysis.runtime import write_record
from archeo_topia.training.train_mound_detector import HEADLINE_CONFIDENCE

#: Distance, in source pixels, within which a detection counts as a hit: the
#: tightest recall radius, which is the primary one.
HIT_RADIUS_PX = RECALL_RADII[0]

#: ``--metric`` values, one per pre-registered IoU threshold.
IOU_METRICS = tuple(f"iou_{threshold:g}" for threshold in IOU_THRESHOLDS)


def detector_hits(metrics: dict[str, Any]) -> dict[str, bool]:
    """Read hit or miss per annotation from a detector ``metrics.json``.

    Args:
        metrics: The parsed document.

    Returns:
        Hit by annotation key.
    """
    pooled = metrics["by_confidence"][confidence_key(HEADLINE_CONFIDENCE)]["pooled"]
    return {
        key: distance is not None and distance <= HIT_RADIUS_PX
        for key, distance in pooled["per_annotation_distance_px"].items()
    }


def end_to_end_hits(pair: dict[str, Any], arm: str, metric: str) -> dict[str, bool]:
    """Read hit or miss per annotation from an ``end_to_end.json`` pair record.

    Args:
        pair: The parsed document.
        arm: Prompt arm.
        metric: One of :data:`IOU_METRICS`.

    Returns:
        Hit by annotation key.

    Raises:
        ValueError: On an unknown arm or metric.
    """
    if arm not in BOX_MODES:
        raise ValueError(f"unknown arm {arm!r}; have {list(BOX_MODES)}")
    if metric not in IOU_METRICS:
        raise ValueError(f"unknown metric {metric!r}; have {list(IOU_METRICS)}")
    threshold = IOU_THRESHOLDS[IOU_METRICS.index(metric)]
    return {
        key: iou >= threshold
        for key, iou in pair["arms"][arm]["pooled"]["per_annotation_iou"].items()
    }


def compare(
    kind: str, first_path: Path, second_path: Path, arm: str | None, metric: str | None
) -> dict[str, Any]:
    """Compare two result files.

    Args:
        kind: ``detector`` or ``end_to_end``.
        first_path: First system's result file.
        second_path: Second system's result file.
        arm: Prompt arm, for ``end_to_end``.
        metric: IoU metric, for ``end_to_end``.

    Returns:
        What was compared and the discordant counts.

    Raises:
        ValueError: On an unknown kind, a missing arm or metric, or two files
            that do not cover the same annotations.
    """
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    if kind == "detector":
        what = {
            "confidence": confidence_key(HEADLINE_CONFIDENCE),
            "hit_radius_px": HIT_RADIUS_PX,
        }
        hits = detector_hits(first), detector_hits(second)
    elif kind == "end_to_end":
        if arm is None or metric is None:
            raise ValueError("an end_to_end comparison needs an arm and a metric")
        what = {"arm": arm, "metric": metric}
        hits = end_to_end_hits(first, arm, metric), end_to_end_hits(second, arm, metric)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return {
        "kind": kind,
        "first": str(first_path),
        "second": str(second_path),
        **what,
        **paired_discordance(*hits),
    }


def main() -> None:
    """Parse arguments, compare, write the result and print it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    kinds = parser.add_subparsers(dest="kind", required=True)
    for kind in ("detector", "end_to_end"):
        sub = kinds.add_parser(kind)
        sub.add_argument("--first", type=Path, required=True)
        sub.add_argument("--second", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if kind == "end_to_end":
            sub.add_argument("--arm", choices=BOX_MODES, required=True)
            sub.add_argument("--metric", choices=IOU_METRICS, required=True)
    args = parser.parse_args()

    result = compare(
        args.kind,
        args.first,
        args.second,
        getattr(args, "arm", None),
        getattr(args, "metric", None),
    )
    write_record(args.output, json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
