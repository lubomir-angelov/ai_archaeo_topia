#!/usr/bin/env python3
"""Paired discordant counts between two systems on the same annotations (protocol C5).

Two systems scored on the same annotations are compared on the annotations
they disagree about, not on the difference of their rates: the discordant
counts and the exact two-sided binomial p-value (McNemar, exact) are the
pre-registered paired statistic.

Three kinds of input, all read from files an opening wrote:

``detector``
    Two detector ``metrics.json`` files. An annotation is a hit when it is
    matched within the primary recall radius at the headline confidence, pooled.

``end_to_end``
    Two ``end_to_end.json`` pair files. An annotation is a hit when its IoU in
    the chosen prompt arm reaches the chosen threshold, pooled.

``openings``
    Both opening directories at once. Checks that the refit's manifest extends
    the baseline's, applies the wording rule fixed in
    ``TEST_PROTOCOL.md`` ("Comparing the refit against the baseline")
    mechanically to pooled recall@5px at the headline confidence, and records
    every descriptive discordance beside it.

The printed counts are the inputs to the wording of ``TEST_RESULTS.md``.

Usage:
    python -m archeo_topia.analysis.paired_comparison detector \\
        --first A/metrics.json --second B/metrics.json --output paired.json

    python -m archeo_topia.analysis.paired_comparison end_to_end \\
        --first A/end_to_end.json --second B/end_to_end.json \\
        --arm fixed --metric iou_0.5 --output paired.json

    python -m archeo_topia.analysis.paired_comparison openings \\
        --baseline artifacts/detection/v0_6_test_baseline \\
        --refit artifacts/detection/v0_6_refit/opening \\
        --output artifacts/detection/v0_6_refit/paired.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from archeo_topia.analysis.detection_metrics import (
    RECALL_RADII,
    confidence_key,
    paired_discordance,
)
from archeo_topia.analysis.frozen_test_opening import REFIT_DETECTOR_LABEL
from archeo_topia.analysis.mapsam_end_to_end import BOX_MODES, IOU_THRESHOLDS
from archeo_topia.analysis.runtime import write_record
from archeo_topia.training.train_mound_detector import HEADLINE_CONFIDENCE, MIN_CONFIDENCE

#: Distance, in source pixels, within which a detection counts as a hit: the
#: tightest recall radius, which is the primary one.
HIT_RADIUS_PX = RECALL_RADII[0]

#: ``--metric`` values, one per pre-registered IoU threshold.
IOU_METRICS = tuple(f"iou_{threshold:g}" for threshold in IOU_THRESHOLDS)

#: Which sentence the detector claim of the refit may carry.
ClaimCase = Literal["improves", "lowers", "no_detectable_difference"]

#: Significance level of every paired p-value in the claim rule.
CLAIM_ALPHA = 0.05

#: The three sentences fixed in advance by ``TEST_PROTOCOL.md``, "Comparing the
#: refit against the baseline". Report text, not processed language.
CLAIM_WORDING: dict[ClaimCase, str] = {
    "improves": "The refit improves recall.",
    "lowers": "The refit lowers recall.",
    "no_detectable_difference": (
        "No difference between the refit and the baseline is detectable on this set."
    ),
}


def detector_hits(metrics: dict[str, Any], confidence: float) -> dict[str, bool]:
    """Read hit or miss per annotation from a detector ``metrics.json``.

    Args:
        metrics: The parsed document.
        confidence: Operating point to read.

    Returns:
        Hit by annotation key.
    """
    pooled = metrics["by_confidence"][confidence_key(confidence)]["pooled"]
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
        hits = (
            detector_hits(first, HEADLINE_CONFIDENCE),
            detector_hits(second, HEADLINE_CONFIDENCE),
        )
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


def refit_claim(baseline: Mapping[str, dict[str, Any]], refit: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered wording rule to the refit's pooled recall@5px.

    The rule reads hit counts, not rates: every figure is scored over the same
    annotations, so a margin compared against the spread is an integer
    comparison with no float boundary.

    Args:
        baseline: The baseline checkpoints' ``metrics.json`` by fold.
        refit: The refit's ``metrics.json``.

    Returns:
        *R*, the baseline values, their spread *S*, the paired discordance
        against each checkpoint, the case and its fixed sentence.

    Raises:
        ValueError: If the documents do not score the same annotation count.
    """
    key = confidence_key(HEADLINE_CONFIDENCE)
    radius = f"{HIT_RADIUS_PX:g}"
    pooled = {name: m["by_confidence"][key]["pooled"] for name, m in baseline.items()}
    refit_pooled = refit["by_confidence"][key]["pooled"]
    sizes = {p["recall"]["n"] for p in [*pooled.values(), refit_pooled]}
    if len(sizes) != 1:
        raise ValueError(f"the systems were scored over different annotation counts: {sizes}")
    hits = {name: p["recall"][radius]["hits"] for name, p in pooled.items()}
    refit_hits = refit_pooled["recall"][radius]["hits"]
    discordance = {
        fold: paired_discordance(
            detector_hits(baseline[fold], HEADLINE_CONFIDENCE),
            detector_hits(refit, HEADLINE_CONFIDENCE),
        )
        for fold in baseline
    }
    significant = all(d["p_value_exact_two_sided"] < CLAIM_ALPHA for d in discordance.values())
    best, worst = max(hits.values()), min(hits.values())
    spread = best - worst
    case: ClaimCase = "no_detectable_difference"
    if refit_hits > best and significant and refit_hits - best >= spread:
        case = "improves"
    elif refit_hits < worst and significant and worst - refit_hits >= spread:
        case = "lowers"
    return {
        "confidence": key,
        "hit_radius_px": HIT_RADIUS_PX,
        "annotations": sizes.pop(),
        "refit": {"hits": refit_hits, "recall": refit_pooled["recall"][radius]["recall"]},
        "baseline": {
            fold: {"hits": hits[fold], "recall": pooled[fold]["recall"][radius]["recall"]}
            for fold in baseline
        },
        "baseline_max_hits": best,
        "baseline_min_hits": worst,
        "spread_hits": spread,
        "discordance": discordance,
        "alpha": CLAIM_ALPHA,
        "case": case,
        "wording": CLAIM_WORDING[case],
    }


def _read_record(directory: Path) -> dict[str, Any]:
    """Read an opening's ``run_record.json``."""
    return json.loads((directory / "run_record.json").read_text(encoding="utf-8"))


def compare_openings(baseline_dir: Path, refit_dir: Path) -> dict[str, Any]:
    """Compare opening 2 against opening 1 under the pre-registered rule.

    Args:
        baseline_dir: Opening 1's output directory.
        refit_dir: Opening 2's output directory.

    Returns:
        The detector claim, the descriptive detector discordance at the second
        operating point, and the paired end-to-end discordance of every fold,
        arm and IoU metric.

    Raises:
        ValueError: If the two directories are not opening 1 and opening 2 of
            one manifest chain over the same folds.
    """
    baseline_record, refit_record = _read_record(baseline_dir), _read_record(refit_dir)
    problems = []
    if baseline_record["opening"] != "v0_6_test_baseline":
        problems.append(f"{baseline_dir} is opening {baseline_record['opening']!r}")
    if refit_record["opening"] != "v0_6_refit":
        problems.append(f"{refit_dir} is opening {refit_record['opening']!r}")
    elif refit_record["manifest"]["base_sha256"] != baseline_record["manifest"]["sha256"]:
        problems.append("the refit's manifest does not extend the manifest opening 1 verified")
    folds = [p["fold"] for p in baseline_record["pairs"]]
    if folds != [p["fold"] for p in refit_record["pairs"]]:
        problems.append("the openings did not score the same folds")
    if problems:
        raise ValueError("not a comparable pair of openings:\n  " + "\n  ".join(problems))

    def metrics(directory: Path, label: str) -> dict[str, Any]:
        path = directory / "detector" / label / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8"))

    baseline = {fold: metrics(baseline_dir, fold) for fold in folds}
    refit = metrics(refit_dir, REFIT_DETECTOR_LABEL)
    # The descriptive operating point is the lowest confidence the detector emits.
    descriptive_key = confidence_key(MIN_CONFIDENCE)
    return {
        "baseline": str(baseline_dir),
        "refit": str(refit_dir),
        "claim": refit_claim(baseline, refit),
        "detector_descriptive": {
            descriptive_key: {
                fold: paired_discordance(
                    detector_hits(baseline[fold], MIN_CONFIDENCE),
                    detector_hits(refit, MIN_CONFIDENCE),
                )
                for fold in folds
            }
        },
        "end_to_end": {
            fold: {
                arm: {
                    metric: compare(
                        "end_to_end",
                        baseline_dir / "end_to_end" / fold / "end_to_end.json",
                        refit_dir / "end_to_end" / fold / "end_to_end.json",
                        arm,
                        metric,
                    )
                    for metric in IOU_METRICS
                }
                for arm in BOX_MODES
            }
            for fold in folds
        },
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
    openings = kinds.add_parser("openings")
    openings.add_argument("--baseline", type=Path, required=True)
    openings.add_argument("--refit", type=Path, required=True)
    openings.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.kind == "openings":
        result = compare_openings(args.baseline, args.refit)
    else:
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
