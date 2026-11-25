#!/usr/bin/env python3
"""Localization-first detection metrics for MapSAM v0.5.

The downstream requirement is a point, not a box. v0.4's prompt-jitter arm
measured what the segmenter does as that point drifts:

    offset   0 px   5 px   10 px   15 px   25 px
    IoU>=0.5  1.00   1.00    0.69    0.02    0.00

So conventional detection metrics select the wrong detector here. A model with
good mAP and occasional 20-30 px centre errors loses to one with mediocre box
IoU that lands within 5 px almost always. These metrics are built around that:
**recall@5px is primary, p90 centre error is the acceptance figure**, and the
rest of the curve is reported for composition with the table above.

Matching is one-to-one and done once, at an association radius of 25 px, with
detections taken in descending confidence. Recall at a tighter radius is then
the fraction of annotations whose matched detection is within it, which keeps
the curve monotone and — more importantly — keeps the 25 px figure honest: 17
of the 180 mounds have a neighbour within 25 px, so greedy many-to-one matching
would let one detection claim credit for the wrong mound.

See ``docs/mapsam/v005/PLAN.md`` for the metric hierarchy this implements.

Scoring is split along the inference / scoring seam. :func:`load_evaluation_set`
is the only constructor of an :class:`EvaluationSet` -- the images, annotations
and regions a run is scored against -- and :func:`score_detections` is the pure
scorer over raw detections, so every caller (leave-one-sheet-out, the frozen
test opening, the reproduction check) goes through the same code.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from archeo_topia.datasets.build_detection_windows import HARD_NEGATIVE, UNCERTAIN, listing_entry

logger = logging.getLogger(__name__)

#: Radii at which recall is reported, in source pixels. 5 is the target.
RECALL_RADII = (5.0, 10.0, 15.0, 25.0)

#: Detections further than this from any annotation are false positives.
ASSOCIATION_RADIUS = 25.0

#: A false positive within this distance of a hard-negative box is attributed
#: to it. Roughly half a symbol width.
ATTRIBUTION_RADIUS = 15.0

#: A sheet with fewer annotations than this has a per-sheet value too noisy to
#: read as a rate; reports mark it.
SMALL_SHEET_N = 50


def confidence_key(threshold: float) -> str:
    """Return the ``by_confidence`` key for a threshold.

    This is the one formatter every writer and reader of a threshold-keyed
    block uses.

    Args:
        threshold: Confidence threshold.

    Returns:
        The threshold at two decimals, for example ``"0.25"``.
    """
    return f"{threshold:.2f}"


def annotation_key(image: str, annotation_id: int) -> str:
    """Return the key that identifies one annotation across a whole export.

    COCO ids are unique within one export, not across label sources, so the
    image is part of the identity.

    Args:
        image: Source clip file name.
        annotation_id: COCO annotation id.

    Returns:
        ``"<image>#<annotation_id>"``.
    """
    return f"{image}#{annotation_id}"


@dataclass(frozen=True)
class Detection:
    """A candidate mound location in source-image coordinates.

    Attributes:
        image: Source clip file name.
        x: Centre x in source pixels.
        y: Centre y in source pixels.
        score: Detector confidence, higher is better.
        box: Optional source-coordinate ``(x0, y0, x1, y1)``. Carried so the
            end-to-end stage can compare MapSAM prompted with the detector's
            own box against MapSAM prompted with a fixed-size box on the same
            centre. v0.4 measured prompt *translation* only, so box scale is an
            untested variable and the two arms separate it.
    """

    image: str
    x: float
    y: float
    score: float
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Annotation:
    """A ground-truth mound annotation.

    Attributes:
        annotation_id: COCO annotation id.
        image: Source clip file name.
        x: Centre x in source pixels.
        y: Centre y in source pixels.
        sheet: Map sheet id.
        merged: True when the annotation shares a connected component with
            another mound, which is where point-based systems tend to collapse
            two mounds into one candidate.
        has_geometry: False for the one annotation carrying a box but no mask;
            it counts for detection recall and cannot be scored end-to-end.
        attributes: Annotation attribute flags, used for subset reporting.
    """

    annotation_id: int
    image: str
    x: float
    y: float
    sheet: str
    merged: bool = False
    has_geometry: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    """An axis-aligned source-coordinate region.

    Attributes:
        image: Source clip file name.
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.
        sheet: Map sheet id the region belongs to.
        negative_type: The ``negative_type`` attribute of a hard-negative
            region, or None when the annotation carries none (and for ignore
            regions, which have no negative type).
    """

    image: str
    x0: float
    y0: float
    x1: float
    y1: float
    sheet: str
    negative_type: str | None

    def contains(self, x: float, y: float) -> bool:
        """Return whether a point falls inside the region.

        Args:
            x: Point x in source pixels.
            y: Point y in source pixels.

        Returns:
            True if the point is inside.
        """
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def distance_to(self, x: float, y: float) -> float:
        """Return the distance from a point to the region, zero if inside.

        Args:
            x: Point x in source pixels.
            y: Point y in source pixels.

        Returns:
            Euclidean distance in source pixels.
        """
        dx = max(self.x0 - x, 0.0, x - self.x1)
        dy = max(self.y0 - y, 0.0, y - self.y1)
        return math.hypot(dx, dy)


def merge_detections(detections: Sequence[Detection], radius: float = 10.0) -> list[Detection]:
    """Collapse duplicate detections of the same symbol across overlapping windows.

    Merging is on centre distance and confidence rather than box IoU: the
    output that matters is a point, and box IoU between two 25 px boxes is a
    noisy quantity. The default radius is safe on this data — no two annotated
    mounds are closer than 15 px — but it is a parameter because that is a
    property of the current three sheets, not a law.

    Args:
        detections: Candidates in source coordinates, any order.
        radius: Centre distance below which two candidates are the same symbol.

    Returns:
        The highest-scoring detection of each cluster, score-sorted.
    """
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: -d.score):
        if any(
            other.image == det.image and math.hypot(other.x - det.x, other.y - det.y) <= radius
            for other in kept
        ):
            continue
        kept.append(det)
    return kept


def _match(
    detections: Sequence[Detection],
    annotations: Sequence[Annotation],
    radius: float,
) -> tuple[dict[tuple[str, int], tuple[Detection, float]], list[Detection]]:
    """Greedily match detections to annotations, one to one.

    Detections are consumed in descending confidence and each claims its
    nearest unclaimed annotation inside *radius*. Descending-confidence greedy
    matching is the convention detection benchmarks use; it is not the
    assignment that minimizes total error, but it is the one that reflects how
    the candidates would actually be consumed downstream.

    Args:
        detections: Candidates in source coordinates.
        annotations: Ground-truth mounds.
        radius: Association radius in source pixels.

    Returns:
        Tuple of a mapping from ``(image, annotation id)`` to its matched
        detection and the distance, and the list of unmatched detections.
    """
    by_image: dict[str, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        by_image[ann.image].append(ann)

    matched: dict[tuple[str, int], tuple[Detection, float]] = {}
    unmatched: list[Detection] = []

    for det in sorted(detections, key=lambda d: -d.score):
        best: Annotation | None = None
        best_distance = radius
        for ann in by_image.get(det.image, ()):
            if (ann.image, ann.annotation_id) in matched:
                continue
            distance = math.hypot(ann.x - det.x, ann.y - det.y)
            if distance <= best_distance:
                best, best_distance = ann, distance
        if best is None:
            unmatched.append(det)
        else:
            matched[(best.image, best.annotation_id)] = (det, best_distance)

    return matched, unmatched


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linear-interpolated percentile.

    Args:
        values: Sample values, any order.
        fraction: Percentile in ``[0, 1]``.

    Returns:
        The percentile, or ``nan`` for an empty sample.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a proportion.

    The normal approximation is unusable here: fold C evaluates on 18 mounds,
    where a recall near 1.0 puts the naive interval above 1.

    Args:
        successes: Number of successes.
        total: Sample size.
        z: Standard-normal quantile.

    Returns:
        Tuple of lower and upper bounds.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def neutralize(
    detections: Sequence[Detection], ignore_regions: Sequence[Region]
) -> tuple[list[Detection], int]:
    """Drop the detections that land inside an ``uncertain_ignore`` region.

    A detection on an ignore region is neither a true nor a false positive.
    Both the detection scorer and the end-to-end scorer go through this one
    helper, so the two stages cannot disagree about what is ignored.

    Args:
        detections: Candidates in source coordinates.
        ignore_regions: ``uncertain_ignore`` regions.

    Returns:
        The surviving detections, as the same objects they were passed in as,
        and the number neutralized.
    """
    live = [
        det
        for det in detections
        if not any(r.image == det.image and r.contains(det.x, det.y) for r in ignore_regions)
    ]
    return live, len(detections) - len(live)


def macro_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize one per-sheet quantity as an unweighted mean, minimum and maximum.

    NaN propagates: a sheet with an undefined value makes the macro undefined
    rather than silently shrinking the denominator to the defined sheets.

    Args:
        values: One value per sheet.

    Returns:
        ``{"mean", "min", "max"}``.

    Raises:
        ValueError: If there are no values.
    """
    if not values:
        raise ValueError("a macro summary needs at least one sheet")
    if any(math.isnan(value) for value in values):
        nan = float("nan")
        return {"mean": nan, "min": nan, "max": nan}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def exact_binomial_two_sided(first_only: int, second_only: int) -> float:
    """Return the exact two-sided p-value of a paired discordant split.

    This is the exact McNemar test: under the null the discordant pairs split
    as Binomial(n, 1/2), and the p-value is twice the smaller tail, capped at 1.
    The sum is exact (integers and a ``Fraction``), so a p-value on the
    boundary is not decided by floating-point rounding.

    Args:
        first_only: Pairs only the first system got right.
        second_only: Pairs only the second system got right.

    Returns:
        The p-value. With no discordant pairs it is 1.0, the binomial's own
        value for n = 0.
    """
    n = first_only + second_only
    tail = sum(math.comb(n, k) for k in range(min(first_only, second_only) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**n)))


def paired_discordance(first: Mapping[str, bool], second: Mapping[str, bool]) -> dict[str, Any]:
    """Count how two systems agree and disagree on the same annotations.

    Args:
        first: Hit or miss per annotation key for the first system.
        second: Hit or miss per annotation key for the second system.

    Returns:
        ``first_only``, ``second_only``, ``both``, ``neither``, ``n`` and the
        exact two-sided ``p_value_exact_two_sided``.

    Raises:
        ValueError: If the two mappings do not cover the identical annotations.
    """
    if first.keys() != second.keys():
        only_first = sorted(first.keys() - second.keys())
        only_second = sorted(second.keys() - first.keys())
        raise ValueError(
            "paired comparison needs identical annotations: "
            f"only in first {only_first}, only in second {only_second}"
        )
    first_only = sum(1 for k in first if first[k] and not second[k])
    second_only = sum(1 for k in first if second[k] and not first[k])
    both = sum(1 for k in first if first[k] and second[k])
    return {
        "first_only": first_only,
        "second_only": second_only,
        "both": both,
        "neither": len(first) - first_only - second_only - both,
        "n": len(first),
        "p_value_exact_two_sided": exact_binomial_two_sided(first_only, second_only),
    }


def evaluate(
    detections: Sequence[Detection],
    annotations: Sequence[Annotation],
    hard_negatives: Sequence[Region] = (),
    ignore_regions: Sequence[Region] = (),
    window_count: int | None = None,
    megapixels: float | None = None,
    association_radius: float = ASSOCIATION_RADIUS,
) -> dict[str, Any]:
    """Score a set of candidate points against the mound annotations.

    Args:
        detections: Candidates in source coordinates, already deduplicated.
        annotations: Ground-truth mounds for the same images.
        hard_negatives: ``hard_negative_symbol`` boxes, for attributing false
            positives to a ``negative_type``.
        ignore_regions: ``uncertain_ignore`` boxes. Detections landing inside
            one are neither true nor false positives and are dropped before
            matching.
        window_count: Number of 512 px windows the detections were drawn from,
            for the per-window false-positive rate.
        megapixels: Source area searched, for the per-megapixel rate.
        association_radius: Distance within which a detection may claim an
            annotation.

    Returns:
        A metrics dict (a ``DetectionBlock``). ``recall`` is keyed by radius as
        a string; ``precision`` is a Wilson-bounded proportion over the
        candidates that were scored; ``false_positives`` separates the ones
        near a hard negative from background, and counts the hard-negative
        ones that carry no ``negative_type`` under ``without_negative_type``.
    """
    live, neutralized_count = neutralize(detections, ignore_regions)

    matched, unmatched = _match(live, annotations, association_radius)
    errors = [distance for _, distance in matched.values()]

    def recall_block(subset: Sequence[Annotation]) -> dict[str, Any]:
        """Compute the recall curve over a subset of annotations."""
        total = len(subset)
        block: dict[str, Any] = {"n": total}
        for radius in RECALL_RADII:
            hits = sum(
                1
                for ann in subset
                if (ann.image, ann.annotation_id) in matched
                and matched[(ann.image, ann.annotation_id)][1] <= radius
            )
            low, high = wilson_interval(hits, total)
            block[f"{radius:g}"] = {
                "recall": hits / total if total else float("nan"),
                "hits": hits,
                "ci95": [low, high],
            }
        return block

    by_negative_type: dict[str, int] = defaultdict(int)
    without_negative_type = 0
    background = 0
    for det in unmatched:
        candidates = [
            region
            for region in hard_negatives
            if region.image == det.image and region.distance_to(det.x, det.y) <= ATTRIBUTION_RADIUS
        ]
        if not candidates:
            background += 1
            continue
        nearest = min(candidates, key=lambda r: r.distance_to(det.x, det.y))
        if nearest.negative_type is None:
            without_negative_type += 1
        else:
            by_negative_type[nearest.negative_type] += 1

    scored_candidates = len(live)
    precision_value = len(matched) / scored_candidates if scored_candidates else float("nan")
    precision_low, precision_high = wilson_interval(len(matched), scored_candidates)
    recall_at_radius = len(matched) / len(annotations) if annotations else float("nan")
    if (
        math.isnan(precision_value)
        or math.isnan(recall_at_radius)
        or (precision_value + recall_at_radius == 0)
    ):
        f1 = float("nan")
    else:
        f1 = 2 * precision_value * recall_at_radius / (precision_value + recall_at_radius)

    result: dict[str, Any] = {
        "detections": len(detections),
        "detections_neutralized_by_ignore": neutralized_count,
        "annotations": len(annotations),
        "association_radius": association_radius,
        "recall": recall_block(annotations),
        "localization_error_px": {
            "n": len(errors),
            "median": _percentile(errors, 0.5),
            "p90": _percentile(errors, 0.9),
            "p99": _percentile(errors, 0.99),
            "max": max(errors) if errors else float("nan"),
        },
        "false_positives": {
            "count": len(unmatched),
            "per_window": len(unmatched) / window_count if window_count else None,
            "per_megapixel": len(unmatched) / megapixels if megapixels else None,
            "near_hard_negative": {
                "by_negative_type": dict(sorted(by_negative_type.items(), key=lambda kv: -kv[1])),
                "without_negative_type": without_negative_type,
            },
            "background": background,
        },
        "precision": {
            "value": precision_value,
            "matched": len(matched),
            "scored_candidates": scored_candidates,
            "ci95": [precision_low, precision_high],
        },
        "f1_at_association_radius": f1,
        "subsets": {
            "merged_component": recall_block([a for a in annotations if a.merged]),
            "has_trig_point": recall_block(
                [a for a in annotations if a.attributes.get("has_trig_point")]
            ),
            "blurred_or_bad_print": recall_block(
                [a for a in annotations if a.attributes.get("blurred_or_bad_print")]
            ),
            "crossed_by_contour": recall_block(
                [a for a in annotations if a.attributes.get("crossed_by_contour")]
            ),
        },
        "per_annotation_distance_px": {
            annotation_key(ann.image, ann.annotation_id): (
                matched[(ann.image, ann.annotation_id)][1]
                if (ann.image, ann.annotation_id) in matched
                else None
            )
            for ann in annotations
        },
    }

    if result["false_positives"]["per_window"] is None:
        logger.debug("no window count supplied; per-window FP rate omitted")
    return result


def format_report(metrics: dict[str, Any], title: str) -> str:
    """Render one detection block as a short markdown report.

    Args:
        metrics: Output of :func:`evaluate`.
        title: Report heading.

    Returns:
        Markdown text.
    """
    recall = metrics["recall"]
    error = metrics["localization_error_px"]
    false_positives = metrics["false_positives"]
    precision = metrics["precision"]
    near = false_positives["near_hard_negative"]

    lines = [
        f"## {title}",
        "",
        f"{metrics['detections']} candidates against {metrics['annotations']} annotations "
        f"(association radius {metrics['association_radius']:g} px, one-to-one).",
        "",
        "| radius | recall | hits | 95% CI |",
        "|---:|---:|---:|---|",
    ]
    for radius in RECALL_RADII:
        block = recall[f"{radius:g}"]
        low, high = block["ci95"]
        marker = " **(primary)**" if radius == 5.0 else ""
        lines.append(
            f"| {radius:g} px{marker} | {block['recall']:.3f} | {block['hits']}/{recall['n']} "
            f"| {low:.3f}–{high:.3f} |"
        )

    p_low, p_high = precision["ci95"]
    lines += [
        "",
        f"Precision {precision['value']:.3f} ({precision['matched']}/"
        f"{precision['scored_candidates']} scored candidates, 95% CI "
        f"{p_low:.3f}–{p_high:.3f}); F1 at the association radius "
        f"{metrics['f1_at_association_radius']:.3f}.",
        "",
        "| localization error | px |",
        "|---|---:|",
        f"| median | {error['median']:.2f} |",
        f"| **p90** | **{error['p90']:.2f}** |",
        f"| p99 | {error['p99']:.2f} |",
        f"| max | {error['max']:.2f} |",
        "",
        f"False positives: {false_positives['count']}",
    ]
    if false_positives["per_window"] is not None:
        lines[-1] += f" ({false_positives['per_window']:.2f} per 512 px window"
        if false_positives["per_megapixel"] is not None:
            lines[-1] += f", {false_positives['per_megapixel']:.2f} per megapixel"
        lines[-1] += ")"
    lines += [
        "",
        "| false positive lands on | n |",
        "|---|---:|",
    ]
    for label, count in near["by_negative_type"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.append(f"| hard negative without a negative_type | {near['without_negative_type']} |")
    lines.append(f"| background | {false_positives['background']} |")

    lines += ["", "| subset | n | recall@5px | recall@15px |", "|---|---:|---:|---:|"]
    for name, block in metrics["subsets"].items():
        if not block["n"]:
            continue
        lines.append(
            f"| {name} | {block['n']} | {block['5']['recall']:.3f} | {block['15']['recall']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def format_metrics_report(metrics: dict[str, Any], title: str) -> str:
    """Render a ``by_confidence`` metrics document as markdown.

    Every confidence block is printed, the headline first and then the lowest
    emitted confidence. Per-sheet values over fewer than ``SMALL_SHEET_N``
    annotations are marked.

    Args:
        metrics: The ``metrics.json`` document, with ``config`` and
            ``by_confidence``.
        title: Report heading.

    Returns:
        Markdown text.
    """
    config = metrics["config"]
    by_confidence = metrics["by_confidence"]
    headline = confidence_key(config["headline_confidence"])
    floor = confidence_key(config["min_confidence"])
    order = [
        headline,
        floor,
        *[key for key in sorted(by_confidence) if key not in (headline, floor)],
    ]
    order = [key for index, key in enumerate(order) if key not in order[:index]]

    lines = [
        f"# {title}",
        "",
        f"Headline confidence {headline}. The 95% intervals treat instances as "
        "independent and understate the uncertainty.",
        "",
    ]
    for key in order:
        block = by_confidence[key]
        lines.append(format_report(block["pooled"], f"{title} - pooled, confidence {key}"))
        lines += [
            f"### Per sheet, confidence {key}",
            "",
            "| sheet | n | recall@5px | precision | FP/window |",
            "|---|---:|---:|---:|---:|",
        ]
        for sheet, sheet_block in block["per_sheet"].items():
            n = sheet_block["annotations"]
            mark = " †" if n < SMALL_SHEET_N else ""
            lines.append(
                f"| {sheet} | {n} | {sheet_block['recall']['5']['recall']:.3f}{mark} "
                f"| {sheet_block['precision']['value']:.3f}{mark} "
                f"| {sheet_block['false_positives']['per_window']:.2f}{mark} |"
            )
        lines += [
            "",
            f"† fewer than {SMALL_SHEET_N} annotations on the sheet.",
            "",
            f"### Macro over sheets, confidence {key}",
            "",
            "| quantity | mean | min | max |",
            "|---|---:|---:|---:|",
        ]
        for name, summary in block["macro"].items():
            lines.append(
                f"| {name} | {summary['mean']:.3f} | {summary['min']:.3f} | {summary['max']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class EvaluationSet:
    """What a detection run is scored against: whole sheets, their windows and labels.

    Built only by :func:`load_evaluation_set`, which enforces that the listing
    covers every declared sheet in full.

    Attributes:
        dataset: Detection dataset root.
        listing: Image list the detector was run over.
        sheets: Declared sheets, in the order given.
        window_ids: Listed window ids, in listing order.
        image_sheet: Sheet of every image that has a listed window.
        windows_by_sheet: Listed window count per sheet.
        megapixels_by_sheet: Source area per sheet, from the furthest window
            edge of each image (the final row and column are clamped flush
            with the image).
        annotations: Mound annotations of the declared sheets.
        hard_negatives: ``hard_negative_symbol`` regions of the declared sheets.
        ignore_regions: ``uncertain_ignore`` regions of the declared sheets.
    """

    dataset: Path
    listing: Path
    sheets: tuple[str, ...]
    window_ids: tuple[str, ...]
    image_sheet: dict[str, str]
    windows_by_sheet: dict[str, int]
    megapixels_by_sheet: dict[str, float]
    annotations: tuple[Annotation, ...]
    hard_negatives: tuple[Region, ...]
    ignore_regions: tuple[Region, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_evaluation_set(dataset: Path, listing: Path, sheets: Sequence[str]) -> EvaluationSet:
    """Build the evaluation set for a listing over declared sheets.

    Sheets are scored in full: the listing must hold every window the dataset
    has for each declared sheet, and nothing else.

    Args:
        dataset: Detection dataset root written by ``build_detection_windows``.
        listing: Image list, usually ``<fold>_val.txt`` or ``test.txt``.
        sheets: Sheets the listing is expected to cover.

    Returns:
        The evaluation set.

    Raises:
        ValueError: If the listing and the declared sheets disagree or an
            entry is not in the builder's canonical form, naming every
            offender.
    """
    metadata = dataset / "metadata"
    declared = tuple(sheets)
    problems: list[str] = []
    if len(set(declared)) != len(declared):
        problems.append(f"declared sheets repeat: {list(declared)}")

    windows = {w["window_id"]: w for w in _read_jsonl(metadata / "windows.jsonl")}
    entries = listing.read_text(encoding="utf-8").split()
    listed = [Path(entry).stem for entry in entries]
    non_canonical = [e for e in entries if e != listing_entry(Path(e).stem)]
    if non_canonical:
        problems.append(f"listing entries not in canonical form: {non_canonical}")
    if len(set(listed)) != len(listed):
        problems.append("the listing names a window more than once")

    unknown = [window_id for window_id in listed if window_id not in windows]
    if unknown:
        problems.append(f"listed windows absent from windows.jsonl: {unknown}")
    known = [windows[window_id] for window_id in listed if window_id in windows]

    foreign = sorted({w["sheet_id"] for w in known} - set(declared))
    if foreign:
        problems.append(f"listed windows belong to undeclared sheets: {foreign}")
    uncovered = sorted(set(declared) - {w["sheet_id"] for w in known})
    if uncovered:
        problems.append(f"declared sheets have no listed window: {uncovered}")
    listed_set = set(listed)
    partial = sorted(
        {
            w["sheet_id"]
            for w in windows.values()
            if w["sheet_id"] in declared and w["window_id"] not in listed_set
        }
    )
    if partial:
        problems.append(
            f"the listing omits windows of declared sheets (partial coverage): {partial}"
        )

    image_sheet = {w["source_image"]: w["sheet_id"] for w in known}
    annotations, negatives, ignores = load_annotations(metadata)
    in_scope = set(declared)
    scoped_annotations = [a for a in annotations if a.sheet in in_scope]
    scoped_negatives = [r for r in negatives if r.sheet in in_scope]
    scoped_ignores = [r for r in ignores if r.sheet in in_scope]
    off_listing = sorted(
        {
            item.image
            for item in (*scoped_annotations, *scoped_negatives, *scoped_ignores)
            if item.image not in image_sheet
        }
    )
    if off_listing:
        problems.append(
            f"annotations or regions lie on images with no listed window: {off_listing}"
        )

    if problems:
        raise ValueError(f"invalid evaluation set for {listing}: " + "; ".join(problems))

    extents: dict[str, tuple[int, int]] = {}
    windows_by_sheet: dict[str, int] = defaultdict(int)
    for window in known:
        _, _, x1, y1 = window["window_xyxy"]
        width, height = extents.get(window["source_image"], (0, 0))
        extents[window["source_image"]] = (max(width, x1), max(height, y1))
        windows_by_sheet[window["sheet_id"]] += 1
    megapixels_by_sheet: dict[str, float] = defaultdict(float)
    for image, (width, height) in extents.items():
        megapixels_by_sheet[image_sheet[image]] += width * height / 1e6

    return EvaluationSet(
        dataset=dataset,
        listing=listing,
        sheets=declared,
        window_ids=tuple(listed),
        image_sheet=image_sheet,
        windows_by_sheet={sheet: windows_by_sheet[sheet] for sheet in declared},
        megapixels_by_sheet={sheet: megapixels_by_sheet[sheet] for sheet in declared},
        annotations=tuple(scoped_annotations),
        hard_negatives=tuple(scoped_negatives),
        ignore_regions=tuple(scoped_ignores),
    )


def macro_inputs(block: dict[str, Any]) -> dict[str, float]:
    """Pick the quantities of a ``DetectionBlock`` that are summarized across sheets.

    Also the quantities the opening summarizes across checkpoints.

    Args:
        block: A block returned by :func:`evaluate`.

    Returns:
        Quantity name to value.
    """
    return {
        "recall_5px": block["recall"]["5"]["recall"],
        "recall_10px": block["recall"]["10"]["recall"],
        "recall_15px": block["recall"]["15"]["recall"],
        "recall_25px": block["recall"]["25"]["recall"],
        "precision": block["precision"]["value"],
        "f1_at_association_radius": block["f1_at_association_radius"],
        "p90_error_px": block["localization_error_px"]["p90"],
        "fp_per_window": block["false_positives"]["per_window"],
        "fp_per_megapixel": block["false_positives"]["per_megapixel"],
    }


def score_detections(
    detections: Sequence[Detection],
    evaluation_set: EvaluationSet,
    thresholds: Sequence[float],
    merge_radius: float,
) -> dict[str, Any]:
    """Score raw detections against an evaluation set at every threshold.

    Per threshold the detections are filtered by score and merged once over the
    pooled set -- merging never crosses images, so restricting the merged set to
    a sheet equals merging that sheet alone -- then scored pooled and per sheet.

    Args:
        detections: Raw source-coordinate detections, before merging.
        evaluation_set: What to score against.
        thresholds: Confidence operating points.
        merge_radius: Centre distance for cross-window deduplication.

    Returns:
        ``{"by_confidence": {confidence_key: {"pooled", "per_sheet", "macro"}}}``.

    Raises:
        ValueError: If a detection lies on an image outside the evaluation set.
    """
    foreign = sorted({d.image for d in detections} - set(evaluation_set.image_sheet))
    if foreign:
        raise ValueError(f"detections on images outside the evaluation set: {foreign}")

    sheets = evaluation_set.sheets
    by_confidence: dict[str, Any] = {}
    for threshold in sorted(thresholds):
        merged = merge_detections(
            [d for d in detections if d.score >= threshold], radius=merge_radius
        )
        pooled = evaluate(
            detections=merged,
            annotations=evaluation_set.annotations,
            hard_negatives=evaluation_set.hard_negatives,
            ignore_regions=evaluation_set.ignore_regions,
            window_count=sum(evaluation_set.windows_by_sheet.values()),
            megapixels=sum(evaluation_set.megapixels_by_sheet.values()),
        )
        per_sheet = {
            sheet: evaluate(
                detections=[d for d in merged if evaluation_set.image_sheet[d.image] == sheet],
                annotations=filter_by_sheet(evaluation_set.annotations, sheet),
                hard_negatives=filter_by_sheet(evaluation_set.hard_negatives, sheet),
                ignore_regions=filter_by_sheet(evaluation_set.ignore_regions, sheet),
                window_count=evaluation_set.windows_by_sheet[sheet],
                megapixels=evaluation_set.megapixels_by_sheet[sheet],
            )
            for sheet in sheets
        }
        inputs = [macro_inputs(per_sheet[sheet]) for sheet in sheets]
        macro = {name: macro_summary([row[name] for row in inputs]) for name in inputs[0]}
        by_confidence[confidence_key(threshold)] = {
            "pooled": pooled,
            "per_sheet": per_sheet,
            "macro": macro,
        }
    return {"by_confidence": by_confidence}


def load_annotations(metadata_dir: Any) -> tuple[list[Annotation], list[Region], list[Region]]:
    """Load annotations and regions from a built detection dataset.

    Regions come from ``regions.jsonl`` only, whose boxes are the full
    source-coordinate COCO boxes; there is no reconstruction from window
    records.

    Args:
        metadata_dir: ``metadata`` directory written by
            ``build_detection_windows``.

    Returns:
        Tuple of annotations, hard-negative regions and ignore regions.

    Raises:
        FileNotFoundError: If ``regions.jsonl`` is missing, as in a dataset
            built before regions were written.
        ValueError: If a ``(source_image, annotation_id, kind)`` repeats or a
            record has an unknown kind.
    """
    metadata_dir = Path(metadata_dir)
    annotations = []
    for record in _read_jsonl(metadata_dir / "annotations.jsonl"):
        x, y = record["center_source_xy"]
        annotations.append(
            Annotation(
                annotation_id=record["annotation_id"],
                image=record["source_image"],
                x=x,
                y=y,
                sheet=record["sheet_id"],
                merged=record["component_annotation_count"] > 1,
                has_geometry=record["has_geometry"],
                attributes=record["attributes"],
            )
        )

    regions_path = metadata_dir / "regions.jsonl"
    if not regions_path.exists():
        raise FileNotFoundError(
            f"{regions_path} not found; rebuild the dataset with build_detection_windows"
        )
    negatives: list[Region] = []
    ignores: list[Region] = []
    seen: set[tuple[str, int, str]] = set()
    for record in _read_jsonl(regions_path):
        identity = (record["source_image"], record["annotation_id"], record["kind"])
        if identity in seen:
            raise ValueError(f"duplicate region {identity} in {regions_path}")
        seen.add(identity)
        x0, y0, x1, y1 = record["bbox_source_xyxy"]
        region = Region(
            record["source_image"],
            x0,
            y0,
            x1,
            y1,
            record["sheet_id"],
            record["negative_type"],
        )
        if record["kind"] == HARD_NEGATIVE:
            negatives.append(region)
        elif record["kind"] == UNCERTAIN:
            ignores.append(region)
        else:
            raise ValueError(f"unknown region kind {record['kind']!r} in {regions_path}")

    return annotations, negatives, ignores


def filter_by_sheet(items: Iterable[Any], sheet: str) -> list[Any]:
    """Keep items belonging to one sheet.

    Args:
        items: Annotations or regions, each carrying its ``sheet``.
        sheet: Sheet id.

    Returns:
        The matching items.
    """
    return [item for item in items if item.sheet == sheet]
