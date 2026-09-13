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
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Radii at which recall is reported, in source pixels. 5 is the target.
RECALL_RADII = (5.0, 10.0, 15.0, 25.0)

#: Detections further than this from any annotation are false positives.
ASSOCIATION_RADIUS = 25.0

#: A false positive within this distance of a hard-negative box is attributed
#: to it. Roughly half a symbol width.
ATTRIBUTION_RADIUS = 15.0


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
        label: Free-form tag, for example a ``negative_type``.
    """

    image: str
    x0: float
    y0: float
    x1: float
    y1: float
    label: str | None = None

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


def merge_detections(
    detections: Sequence[Detection], radius: float = 10.0
) -> list[Detection]:
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
) -> tuple[dict[int, tuple[Detection, float]], list[Detection]]:
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
        Tuple of a mapping from annotation id to its matched detection and the
        distance, and the list of unmatched detections.
    """
    by_image: dict[str, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        by_image[ann.image].append(ann)

    matched: dict[int, tuple[Detection, float]] = {}
    unmatched: list[Detection] = []

    for det in sorted(detections, key=lambda d: -d.score):
        best: Annotation | None = None
        best_distance = radius
        for ann in by_image.get(det.image, ()):
            if ann.annotation_id in matched:
                continue
            distance = math.hypot(ann.x - det.x, ann.y - det.y)
            if distance <= best_distance:
                best, best_distance = ann, distance
        if best is None:
            unmatched.append(det)
        else:
            matched[best.annotation_id] = (det, best_distance)

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


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
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
        A metrics dict. ``recall`` is keyed by radius as a string.
    """
    def is_ignored(det: Detection) -> bool:
        """Return whether a detection falls inside an uncertain_ignore region."""
        return any(r.image == det.image and r.contains(det.x, det.y) for r in ignore_regions)

    live = [det for det in detections if not is_ignored(det)]
    neutralized_count = len(detections) - len(live)

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
                if ann.annotation_id in matched and matched[ann.annotation_id][1] <= radius
            )
            low, high = _wilson(hits, total)
            block[f"{radius:g}"] = {
                "recall": hits / total if total else float("nan"),
                "hits": hits,
                "ci95": [low, high],
            }
        return block

    attribution: dict[str, int] = defaultdict(int)
    for det in unmatched:
        candidates = [
            region
            for region in hard_negatives
            if region.image == det.image
            and region.distance_to(det.x, det.y) <= ATTRIBUTION_RADIUS
        ]
        if candidates:
            nearest = min(candidates, key=lambda r: r.distance_to(det.x, det.y))
            attribution[nearest.label or "untyped"] += 1
        else:
            attribution["background"] += 1

    by_sheet: dict[str, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        by_sheet[ann.sheet].append(ann)

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
            "by_negative_type": dict(sorted(attribution.items(), key=lambda kv: -kv[1])),
        },
        "precision_at_association_radius": (
            len(matched) / len(live) if live else float("nan")
        ),
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
        "per_sheet": {
            sheet: recall_block(subset) for sheet, subset in sorted(by_sheet.items())
        },
    }

    if result["false_positives"]["per_window"] is None:
        logger.debug("no window count supplied; per-window FP rate omitted")
    return result


def format_report(metrics: dict[str, Any], title: str) -> str:
    """Render a metrics dict as a short markdown report.

    Args:
        metrics: Output of :func:`evaluate`.
        title: Report heading.

    Returns:
        Markdown text.
    """
    recall = metrics["recall"]
    error = metrics["localization_error_px"]
    false_positives = metrics["false_positives"]

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

    lines += [
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
    for label, count in false_positives["by_negative_type"].items():
        lines.append(f"| `{label}` | {count} |")

    lines += ["", "| subset | n | recall@5px | recall@15px |", "|---|---:|---:|---:|"]
    for name, block in metrics["subsets"].items():
        if not block["n"]:
            continue
        lines.append(
            f"| {name} | {block['n']} | {block['5']['recall']:.3f} | {block['15']['recall']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def load_annotations(metadata_dir: Any) -> tuple[list[Annotation], list[Region], list[Region]]:
    """Load annotations and regions from a built detection dataset.

    Args:
        metadata_dir: ``metadata`` directory written by
            ``build_detection_windows``.

    Returns:
        Tuple of annotations, hard-negative regions and ignore regions.
    """
    import json
    from pathlib import Path

    metadata_dir = Path(metadata_dir)
    annotations = []
    for line in (metadata_dir / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
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

    negatives: dict[tuple[str, int], Region] = {}
    ignores: dict[tuple[str, int], Region] = {}
    for line in (metadata_dir / "windows.jsonl").read_text(encoding="utf-8").splitlines():
        window = json.loads(line)
        x0, y0, _, _ = window["window_xyxy"]
        image = window["source_image"]
        for negative in window["hard_negatives"]:
            bx0, by0, bx1, by1 = negative["box_window_xyxy"]
            negatives[(image, negative["annotation_id"])] = Region(
                image, bx0 + x0, by0 + y0, bx1 + x0, by1 + y0, negative["negative_type"]
            )
        for ignore in window["uncertain_ignore"]:
            bx0, by0, bx1, by1 = ignore["box_window_xyxy"]
            ignores[(image, ignore["annotation_id"])] = Region(
                image, bx0 + x0, by0 + y0, bx1 + x0, by1 + y0, "uncertain_ignore"
            )

    return annotations, list(negatives.values()), list(ignores.values())


def filter_by_sheet(items: Iterable[Any], sheet: str) -> list[Any]:
    """Keep items belonging to one sheet.

    Args:
        items: Annotations, or regions whose ``image`` names start with the
            sheet id.
        sheet: Sheet id.

    Returns:
        The matching items.
    """
    out = []
    for item in items:
        owner = getattr(item, "sheet", None) or item.image.rsplit("_", 1)[0]
        if owner == sheet:
            out.append(item)
    return out
