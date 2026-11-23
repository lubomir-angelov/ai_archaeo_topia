"""Tests for the localization-first detection metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from archeo_topia.analysis.detection_metrics import (
    Annotation,
    Detection,
    EvaluationSet,
    Region,
    annotation_key,
    confidence_key,
    evaluate,
    exact_binomial_two_sided,
    load_annotations,
    load_evaluation_set,
    macro_summary,
    merge_detections,
    neutralize,
    paired_discordance,
    score_detections,
)
from tests.detection_fixtures import region_record, write_dataset


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
    ignores = [Region("S-1_1.png", 480, 480, 520, 520, "S-1", None)]
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
    negatives = [Region("S-1_1.png", 290, 290, 310, 310, "S-1", "trig_point")]
    metrics = evaluate(detections, annotations, hard_negatives=negatives)

    false_positives = metrics["false_positives"]
    assert false_positives["near_hard_negative"]["by_negative_type"] == {"trig_point": 1}
    assert false_positives["near_hard_negative"]["without_negative_type"] == 0
    assert false_positives["background"] == 1


def test_a_negative_without_a_type_is_counted_not_renamed() -> None:
    """No stand-in type word exists anywhere in the output."""
    detections = [Detection("S-1_1.png", 300, 300, 0.9)]
    negatives = [Region("S-1_1.png", 290, 290, 310, 310, "S-1", None)]
    metrics = evaluate(detections, [_ann(1, 100, 100)], hard_negatives=negatives)

    near = metrics["false_positives"]["near_hard_negative"]
    assert near == {"by_negative_type": {}, "without_negative_type": 1}
    assert "untyped" not in json.dumps(metrics)


def test_a_negative_type_called_background_cannot_collide_with_background() -> None:
    """Background is a structural sibling, not a key among the types."""
    detections = [Detection("S-1_1.png", 300, 300, 0.9), Detection("S-1_1.png", 900, 900, 0.8)]
    negatives = [Region("S-1_1.png", 290, 290, 310, 310, "S-1", "background")]
    metrics = evaluate(detections, [_ann(1, 100, 100)], hard_negatives=negatives)

    assert metrics["false_positives"]["near_hard_negative"]["by_negative_type"] == {
        "background": 1
    }
    assert metrics["false_positives"]["background"] == 1


def test_precision_is_a_wilson_block_over_scored_candidates() -> None:
    """Ignored candidates are not scored; the interval is on matched / scored."""
    annotations = [_ann(1, 100, 100), _ann(2, 300, 300)]
    detections = [
        Detection("S-1_1.png", 100, 100, 0.9),
        Detection("S-1_1.png", 700, 700, 0.8),
        Detection("S-1_1.png", 500, 500, 0.7),
    ]
    ignores = [Region("S-1_1.png", 480, 480, 520, 520, "S-1", None)]
    precision = evaluate(detections, annotations, ignore_regions=ignores)["precision"]

    assert precision["matched"] == 1
    assert precision["scored_candidates"] == 2
    assert precision["value"] == pytest.approx(0.5)
    low, high = precision["ci95"]
    assert 0.0 < low < 0.5 < high < 1.0


def test_f1_is_the_harmonic_mean_and_nan_when_undefined() -> None:
    annotations = [_ann(1, 100, 100), _ann(2, 300, 300)]
    metrics = evaluate([Detection("S-1_1.png", 100, 100, 0.9)], annotations)
    # precision 1.0, recall 0.5
    assert metrics["f1_at_association_radius"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)

    assert math.isnan(evaluate([], annotations)["f1_at_association_radius"])
    assert math.isnan(
        evaluate([Detection("S-1_1.png", 900, 900, 0.9)], annotations)["f1_at_association_radius"]
    )


def test_per_annotation_distance_covers_every_annotation() -> None:
    annotations = [_ann(1, 100, 100), _ann(2, 300, 300)]
    metrics = evaluate([Detection("S-1_1.png", 103, 104, 0.9)], annotations)

    assert metrics["per_annotation_distance_px"] == {
        annotation_key("S-1_1.png", 1): pytest.approx(5.0),
        annotation_key("S-1_1.png", 2): None,
    }
    assert "per_sheet" not in metrics


def test_matching_is_keyed_by_image_and_id() -> None:
    """The same annotation id on two images does not collide."""
    annotations = [
        Annotation(7, "S-1_1.png", 100, 100, "S-1"),
        Annotation(7, "S-1_2.png", 100, 100, "S-1"),
    ]
    detections = [Detection("S-1_1.png", 100, 100, 0.9), Detection("S-1_2.png", 100, 100, 0.8)]
    metrics = evaluate(detections, annotations)

    assert metrics["recall"]["5"]["hits"] == 2
    assert metrics["false_positives"]["count"] == 0


def test_neutralize_returns_the_survivors_and_the_count() -> None:
    inside = Detection("S-1_1.png", 500, 500, 0.9)
    outside = Detection("S-1_1.png", 100, 100, 0.8)
    other_image = Detection("S-1_2.png", 500, 500, 0.7)
    ignores = [Region("S-1_1.png", 480, 480, 520, 520, "S-1", None)]

    live, count = neutralize([inside, outside, other_image], ignores)

    assert count == 1
    assert live == [outside, other_image]
    assert live[0] is outside


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


# ---------------------------------------------------------------------------
# Dataset loading and the evaluation set
# ---------------------------------------------------------------------------


def _evaluation_set(tmp_path: Path, **kwargs: Any) -> EvaluationSet:
    dataset = write_dataset(tmp_path / "ds", **kwargs)
    return load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])


def test_load_annotations_reads_full_source_regions(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds")
    annotations, negatives, ignores = load_annotations(dataset / "metadata")

    assert [a.annotation_id for a in annotations] == [1, 2]
    assert negatives == [Region("S-1_1.png", 290, 290, 310, 310, "S-1", "trig_point")]
    assert ignores == [Region("S-2_1.png", 480, 480, 520, 520, "S-2", None)]


def test_load_annotations_raises_without_regions_jsonl(tmp_path: Path) -> None:
    """A pre-regions dataset is refused, never reconstructed from windows."""
    dataset = write_dataset(tmp_path / "ds")
    (dataset / "metadata" / "regions.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="regions.jsonl"):
        load_annotations(dataset / "metadata")


def test_load_annotations_raises_on_a_duplicate_region(tmp_path: Path) -> None:
    region = region_record("uncertain_ignore", 12, "S-2", "S-2_1.png", [1, 1, 2, 2], None)
    dataset = write_dataset(tmp_path / "ds", regions=[region, region])
    with pytest.raises(ValueError, match="duplicate region"):
        load_annotations(dataset / "metadata")


def test_evaluation_set_collects_per_sheet_facts(tmp_path: Path) -> None:
    es = _evaluation_set(tmp_path)

    assert es.sheets == ("S-1", "S-2")
    assert len(es.window_ids) == 3
    assert es.image_sheet == {"S-1_1.png": "S-1", "S-2_1.png": "S-2"}
    assert es.windows_by_sheet == {"S-1": 2, "S-2": 1}
    # S-1: furthest edge is 896 x 512; S-2: 512 x 512.
    assert es.megapixels_by_sheet == {
        "S-1": pytest.approx(896 * 512 / 1e6),
        "S-2": pytest.approx(512 * 512 / 1e6),
    }
    assert len(es.annotations) == 2
    assert len(es.hard_negatives) == 1
    assert len(es.ignore_regions) == 1


def test_evaluation_set_rejects_a_listed_window_missing_from_the_dataset(tmp_path: Path) -> None:
    ids = [
        "S-1_1_x00000_y00000",
        "S-1_1_x00384_y00000",
        "S-2_1_x00000_y00000",
        "S-2_1_x09999_y00000",
    ]
    dataset = write_dataset(tmp_path / "ds", listing=ids)
    with pytest.raises(ValueError, match="absent from windows.jsonl"):
        load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])


def test_evaluation_set_rejects_a_foreign_sheet(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="undeclared sheets: \\['S-2'\\]"):
        load_evaluation_set(dataset, dataset / "all.txt", ["S-1"])


def test_evaluation_set_rejects_a_declared_sheet_without_windows(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="no listed window: \\['S-3'\\]"):
        load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2", "S-3"])


def test_evaluation_set_rejects_a_listing_entry_not_in_canonical_form(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds")
    (dataset / "all.txt").write_text(
        "images/S-1_1_x00000_y00000.png\n./images/S-1_1_x00384_y00000.png\n"
        "./images/S-2_1_x00000_y00000.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not in canonical form.*images/S-1_1_x00000_y00000.png"):
        load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])


def test_evaluation_set_rejects_partial_sheet_coverage(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path / "ds", listing=["S-1_1_x00000_y00000", "S-2_1_x00000_y00000"]
    )
    with pytest.raises(ValueError, match="partial coverage\\): \\['S-1'\\]"):
        load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])


def test_evaluation_set_rejects_a_region_on_an_unlisted_image(tmp_path: Path) -> None:
    regions = [region_record("uncertain_ignore", 12, "S-2", "S-2_9.png", [1, 1, 2, 2], None)]
    dataset = write_dataset(tmp_path / "ds", regions=regions)
    with pytest.raises(ValueError, match="no listed window: \\['S-2_9.png'\\]"):
        load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])


# ---------------------------------------------------------------------------
# score_detections
# ---------------------------------------------------------------------------


def test_confidence_key_is_two_decimals() -> None:
    assert [confidence_key(t) for t in (0.05, 0.1, 0.25, 0.5, 0.7)] == [
        "0.05",
        "0.10",
        "0.25",
        "0.50",
        "0.70",
    ]


def test_score_detections_is_keyed_by_threshold_with_pooled_per_sheet_and_macro(
    tmp_path: Path,
) -> None:
    es = _evaluation_set(tmp_path)
    detections = [
        Detection("S-1_1.png", 100, 100, 0.9),
        Detection("S-2_1.png", 200, 200, 0.3),
        Detection("S-1_1.png", 800, 400, 0.1),
    ]
    scored = score_detections(detections, es, (0.05, 0.25, 0.5), merge_radius=10.0)

    assert list(scored["by_confidence"]) == ["0.05", "0.25", "0.50"]
    block = scored["by_confidence"]["0.25"]
    assert set(block) == {"pooled", "per_sheet", "macro"}
    assert list(block["per_sheet"]) == ["S-1", "S-2"]
    assert block["pooled"]["recall"]["5"]["hits"] == 2
    assert block["per_sheet"]["S-1"]["recall"]["5"]["hits"] == 1
    assert block["per_sheet"]["S-2"]["recall"]["5"]["hits"] == 1
    assert block["macro"]["recall_5px"] == {"mean": 1.0, "min": 1.0, "max": 1.0}
    # The 0.5 threshold keeps only the first detection.
    half = scored["by_confidence"]["0.50"]
    assert half["macro"]["recall_5px"] == {"mean": 0.5, "min": 0.0, "max": 1.0}
    # Rates use the evaluation set's own window and area counts.
    assert block["pooled"]["false_positives"]["per_window"] == pytest.approx(0 / 3)
    low = scored["by_confidence"]["0.05"]["per_sheet"]["S-1"]["false_positives"]
    assert low["count"] == 1
    assert low["per_window"] == pytest.approx(1 / 2)
    assert low["per_megapixel"] == pytest.approx(1 / (896 * 512 / 1e6))


def test_pooled_equals_the_sheet_when_there_is_one_sheet(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", listing=["S-2_1_x00000_y00000"])
    es = load_evaluation_set(dataset, dataset / "all.txt", ["S-2"])
    scored = score_detections([Detection("S-2_1.png", 203, 200, 0.9)], es, (0.25,), 10.0)

    block = scored["by_confidence"]["0.25"]
    # JSON text, because NaN != NaN as a Python value.
    assert json.dumps(block["pooled"]) == json.dumps(block["per_sheet"]["S-2"])


def test_score_detections_raises_on_a_foreign_image(tmp_path: Path) -> None:
    es = _evaluation_set(tmp_path)
    with pytest.raises(ValueError, match="outside the evaluation set"):
        score_detections([Detection("S-9_1.png", 1, 1, 0.9)], es, (0.25,), 10.0)


def test_macro_propagates_nan_instead_of_shrinking_the_denominator(tmp_path: Path) -> None:
    """A sheet with no scored candidates has undefined precision; so does the macro."""
    es = _evaluation_set(tmp_path)
    scored = score_detections([Detection("S-1_1.png", 100, 100, 0.9)], es, (0.25,), 10.0)

    block = scored["by_confidence"]["0.25"]
    assert math.isnan(block["per_sheet"]["S-2"]["precision"]["value"])
    assert all(math.isnan(v) for v in block["macro"]["precision"].values())
    assert not math.isnan(block["macro"]["recall_5px"]["mean"])


def test_macro_summary() -> None:
    assert macro_summary([0.25, 0.75]) == {"mean": 0.5, "min": 0.25, "max": 0.75}
    assert all(math.isnan(v) for v in macro_summary([0.5, float("nan")]).values())
    with pytest.raises(ValueError, match="at least one sheet"):
        macro_summary([])


def test_exact_binomial_two_sided_known_values() -> None:
    # 0 of 5: 2 * (1 / 32).
    assert exact_binomial_two_sided(0, 5) == 0.0625
    assert exact_binomial_two_sided(5, 0) == 0.0625
    # A balanced split is capped at 1, and no discordant pair gives the n = 0 value.
    assert exact_binomial_two_sided(3, 3) == 1.0
    assert exact_binomial_two_sided(0, 0) == 1.0
    # 2 of 8: 2 * (1 + 8 + 28) / 256.
    assert exact_binomial_two_sided(2, 6) == pytest.approx(74 / 256)


def test_paired_discordance_counts_the_four_cells() -> None:
    first = {"a": True, "b": True, "c": False, "d": False, "e": True}
    second = {"a": True, "b": False, "c": True, "d": False, "e": False}
    result = paired_discordance(first, second)

    assert result == {
        "first_only": 2,
        "second_only": 1,
        "both": 1,
        "neither": 1,
        "n": 5,
        "p_value_exact_two_sided": 1.0,
    }


def test_paired_discordance_raises_unless_both_cover_the_same_annotations() -> None:
    with pytest.raises(ValueError, match="identical annotations"):
        paired_discordance({"a": True, "b": False}, {"a": True, "c": False})
