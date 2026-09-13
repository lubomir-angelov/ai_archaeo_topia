"""Tests for the localization-first detection metrics."""

from __future__ import annotations

import pytest

from archeo_topia.analysis.detection_metrics import (
    Annotation,
    Detection,
    Region,
    evaluate,
    merge_detections,
)


def _ann(annotation_id: int, x: float, y: float, **kwargs) -> Annotation:
    """Build an annotation on a single test clip."""
    return Annotation(annotation_id, "S-1_1.png", x, y, "S-1", **kwargs)


def test_merge_collapses_near_duplicates_keeping_the_best_score() -> None:
    """Overlapping windows see the same symbol; the strongest survives."""
    detections = [
        Detection("S-1_1.png", 100, 100, 0.6),
        Detection("S-1_1.png", 104, 103, 0.9),
        Detection("S-1_1.png", 400, 400, 0.5),
    ]
    merged = merge_detections(detections, radius=10)
    assert len(merged) == 2
    assert merged[0].score == pytest.approx(0.9)


def test_merge_keeps_distinct_mounds_apart() -> None:
    """18 px is the closest two annotated mounds get; 10 px must not fuse them."""
    detections = [
        Detection("S-1_1.png", 100, 100, 0.9),
        Detection("S-1_1.png", 118, 100, 0.8),
    ]
    assert len(merge_detections(detections, radius=10)) == 2


def test_merge_does_not_cross_images() -> None:
    """Identical coordinates on different clips are different places."""
    detections = [
        Detection("S-1_1.png", 100, 100, 0.9),
        Detection("S-1_2.png", 100, 100, 0.8),
    ]
    assert len(merge_detections(detections, radius=50)) == 2


def test_recall_curve_is_nested_by_radius() -> None:
    """A match at 5 px counts at every larger radius."""
    annotations = [_ann(1, 100, 100), _ann(2, 300, 300), _ann(3, 500, 500)]
    detections = [
        Detection("S-1_1.png", 102, 100, 0.9),  # 2 px
        Detection("S-1_1.png", 312, 300, 0.8),  # 12 px
        Detection("S-1_1.png", 900, 900, 0.7),  # unmatched
    ]
    metrics = evaluate(detections, annotations)

    assert metrics["recall"]["5"]["hits"] == 1
    assert metrics["recall"]["10"]["hits"] == 1
    assert metrics["recall"]["15"]["hits"] == 2
    assert metrics["recall"]["25"]["hits"] == 2
    assert metrics["false_positives"]["count"] == 1


def test_matching_is_one_to_one_for_close_neighbours() -> None:
    """One detection cannot satisfy two mounds 20 px apart."""
    annotations = [_ann(1, 100, 100), _ann(2, 120, 100)]
    metrics = evaluate([Detection("S-1_1.png", 110, 100, 0.9)], annotations)

    assert metrics["recall"]["25"]["hits"] == 1
    assert metrics["recall"]["25"]["recall"] == pytest.approx(0.5)


def test_higher_confidence_detection_claims_first() -> None:
    """Greedy matching consumes detections in descending confidence."""
    annotations = [_ann(1, 100, 100)]
    detections = [
        Detection("S-1_1.png", 103, 100, 0.4),
        Detection("S-1_1.png", 108, 100, 0.9),
    ]
    metrics = evaluate(detections, annotations)

    # The 0.9 detection at 8 px wins the match, so recall@5px is zero even
    # though a 3 px detection exists.
    assert metrics["recall"]["5"]["hits"] == 0
    assert metrics["recall"]["10"]["hits"] == 1
    assert metrics["localization_error_px"]["median"] == pytest.approx(8.0)


def test_ignore_regions_neutralize_detections() -> None:
    """A detection on an uncertain_ignore region is neither TP nor FP."""
    annotations = [_ann(1, 100, 100)]
    detections = [
        Detection("S-1_1.png", 100, 100, 0.9),
        Detection("S-1_1.png", 500, 500, 0.8),
    ]
    ignores = [Region("S-1_1.png", 480, 480, 520, 520, "uncertain_ignore")]
    metrics = evaluate(detections, annotations, ignore_regions=ignores)

    assert metrics["detections_neutralized_by_ignore"] == 1
    assert metrics["false_positives"]["count"] == 0
    assert metrics["recall"]["5"]["recall"] == pytest.approx(1.0)


def test_false_positives_are_attributed_to_negative_type() -> None:
    """An FP on a trig point is reported as such, not lumped into a rate."""
    annotations = [_ann(1, 100, 100)]
    detections = [
        Detection("S-1_1.png", 300, 300, 0.9),
        Detection("S-1_1.png", 900, 900, 0.8),
    ]
    negatives = [Region("S-1_1.png", 290, 290, 310, 310, "trig_point")]
    metrics = evaluate(detections, annotations, hard_negatives=negatives)

    attribution = metrics["false_positives"]["by_negative_type"]
    assert attribution["trig_point"] == 1
    assert attribution["background"] == 1


def test_subsets_and_rates_are_reported() -> None:
    """Merged-component and attribute subsets get their own recall blocks."""
    annotations = [
        _ann(1, 100, 100, merged=True, attributes={"has_trig_point": True}),
        _ann(2, 300, 300),
    ]
    detections = [Detection("S-1_1.png", 101, 100, 0.9)]
    metrics = evaluate(detections, annotations, window_count=10, megapixels=5.0)

    assert metrics["subsets"]["merged_component"]["n"] == 1
    assert metrics["subsets"]["merged_component"]["5"]["recall"] == pytest.approx(1.0)
    assert metrics["subsets"]["has_trig_point"]["n"] == 1
    assert metrics["false_positives"]["per_window"] == pytest.approx(0.0)


def test_empty_detections_give_zero_recall_not_a_crash() -> None:
    """A detector that finds nothing is scored, not an error."""
    metrics = evaluate([], [_ann(1, 100, 100)])
    assert metrics["recall"]["5"]["recall"] == pytest.approx(0.0)
    assert metrics["localization_error_px"]["p90"] != metrics["localization_error_px"]["p90"]


def test_wilson_interval_stays_inside_zero_one() -> None:
    """Fold C evaluates 18 mounds; a normal-approximation CI would exceed 1."""
    annotations = [_ann(i, 100 * i, 100) for i in range(1, 19)]
    detections = [Detection("S-1_1.png", 100 * i, 100, 0.9) for i in range(1, 19)]
    metrics = evaluate(detections, annotations)

    low, high = metrics["recall"]["5"]["ci95"]
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert metrics["recall"]["5"]["recall"] == pytest.approx(1.0)
