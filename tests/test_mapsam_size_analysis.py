"""Tests for the MapSAM target-size vs quality analysis."""

from __future__ import annotations

import json

import pytest

from archeo_topia.analysis.mapsam_size_analysis import (
    analyse,
    build_buckets,
    load_stats,
    rank_correlation,
    resolve_stats_file,
    sheet_id_from_sample_id,
    symmetric_difference_px,
    token_coverage,
    write_csv,
)


def _row(sample_id: str, gt: int, pred: int, iou: float, dice: float = 0.0) -> dict:
    """Build a minimal per-sample stat row.

    Args:
        sample_id: Sample identifier.
        gt: Ground-truth positive pixels.
        pred: Predicted positive pixels.
        iou: Intersection over union.
        dice: Dice coefficient.

    Returns:
        A stat row with the fields the analysis reads.
    """
    return {
        "sample_id": sample_id,
        "gt_positive_pixels": gt,
        "pred_positive_pixels_threshold_0_5": pred,
        "iou": iou,
        "dice": dice,
    }


class TestBuildBuckets:
    def test_default_edges_cover_expected_ranges(self):
        buckets = build_buckets()
        assert [b.label for b in buckets] == ["1-3 px", "4-5 px", "6-8 px", "9+ px"]
        assert buckets[-1].high is None

    def test_single_width_bucket_is_labelled_without_a_range(self):
        assert build_buckets((1, 2, 3))[0].label == "1 px"

    def test_buckets_are_contiguous_and_exclusive(self):
        buckets = build_buckets()
        for px in range(1, 20):
            assert sum(b.contains(px) for b in buckets) == 1

    def test_values_below_the_first_edge_fall_outside(self):
        assert not any(b.contains(0) for b in build_buckets())

    def test_rejects_too_few_edges(self):
        with pytest.raises(ValueError, match="at least two"):
            build_buckets((4,))

    def test_rejects_non_ascending_edges(self):
        with pytest.raises(ValueError, match="ascending"):
            build_buckets((1, 6, 4))


class TestSheetIdFromSampleId:
    def test_recovers_sheet_containing_hyphens_and_underscores(self):
        assert sheet_id_from_sample_id("train_K-34-35-B-g_1_000001") == "K-34-35-B-g"

    def test_handles_the_test_split_prefix(self):
        assert sheet_id_from_sample_id("test_K-35-8-G-a_4_000011") == "K-35-8-G-a"

    def test_unrecognised_id_is_returned_unchanged(self):
        assert sheet_id_from_sample_id("batch_0") == "batch_0"


class TestSymmetricDifference:
    def test_perfect_prediction_has_zero_error(self):
        assert symmetric_difference_px(_row("s", 6, 6, 1.0)) == pytest.approx(0.0)

    def test_empty_prediction_error_equals_target_area(self):
        assert symmetric_difference_px(_row("s", 6, 0, 0.0)) == pytest.approx(6.0)

    def test_one_pixel_miss_on_a_small_target(self):
        # 2 of 3 GT pixels hit, one false positive: IoU = 2/4.
        assert symmetric_difference_px(_row("s", 3, 3, 0.5)) == pytest.approx(2.0)

    def test_same_absolute_error_scores_worse_iou_on_a_smaller_target(self):
        # The point of the metric: identical 2 px error, very different IoU.
        small = _row("small", 3, 3, 0.5)
        large = _row("large", 12, 12, 11 / 13)
        assert symmetric_difference_px(small) == pytest.approx(
            symmetric_difference_px(large), abs=0.01
        )
        assert small["iou"] < large["iou"]


class TestTokenCoverage:
    def test_a_typical_five_pixel_target_is_sub_patch(self):
        assert token_coverage(5.0) < 1.0

    def test_coverage_grows_with_area(self):
        assert token_coverage(100.0) > token_coverage(25.0)

    def test_full_logit_grid_maps_to_the_full_token_grid(self):
        assert token_coverage(256.0 * 256.0) == pytest.approx(64.0)

    def test_zero_area_is_handled(self):
        assert token_coverage(0.0) == 0.0


class TestRankCorrelation:
    def test_monotone_non_linear_series_ranks_as_perfect(self):
        assert rank_correlation([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == 1.0

    def test_inverted_series_ranks_as_minus_one(self):
        assert rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0

    def test_constant_series_is_undefined(self):
        assert rank_correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None

    def test_too_short_or_mismatched_series_is_undefined(self):
        assert rank_correlation([1.0], [1.0]) is None
        assert rank_correlation([1.0, 2.0], [1.0]) is None

    def test_ties_do_not_break_the_ranking(self):
        assert rank_correlation([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]) == 1.0


class TestLoadStats:
    def test_reads_rows_and_derives_sheet_id(self, tmp_path):
        path = tmp_path / "prediction_stats_val_e5.jsonl"
        path.write_text(
            "\n".join(json.dumps(_row(f"test_K-35-8-G-a_1_{i:06d}", 5, 5, 1.0)) for i in range(3))
            + "\n",
            encoding="utf-8",
        )
        rows = load_stats(path)
        assert len(rows) == 3
        assert {r["sheet_id"] for r in rows} == {"K-35-8-G-a"}

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "s.jsonl"
        path.write_text(json.dumps(_row("test_A_1_000001", 5, 5, 1.0)) + "\n\n\n", "utf-8")
        assert len(load_stats(path)) == 1

    def test_existing_sheet_id_is_not_overwritten(self, tmp_path):
        path = tmp_path / "s.jsonl"
        row = _row("test_K-35-8-G-a_1_000001", 5, 5, 1.0) | {"sheet_id": "explicit"}
        path.write_text(json.dumps(row) + "\n", "utf-8")
        assert load_stats(path)[0]["sheet_id"] == "explicit"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_stats(tmp_path / "absent.jsonl")


class TestResolveStatsFile:
    def test_explicit_epoch_is_used_verbatim(self, tmp_path):
        assert resolve_stats_file(tmp_path, "val", 25).name == "prediction_stats_val_e25.jsonl"

    def test_defaults_to_the_highest_epoch_numerically_not_lexically(self, tmp_path):
        for epoch in (5, 10, 100):
            (tmp_path / f"prediction_stats_val_e{epoch}.jsonl").write_text("", "utf-8")
        assert resolve_stats_file(tmp_path, "val", None).name == "prediction_stats_val_e100.jsonl"

    def test_does_not_pick_up_the_other_split(self, tmp_path):
        (tmp_path / "prediction_stats_train_e5.jsonl").write_text("", "utf-8")
        with pytest.raises(FileNotFoundError):
            resolve_stats_file(tmp_path, "val", None)


class TestAnalyse:
    def _rows(self):
        # Same 2 px absolute error at every size: IoU must fall for small
        # targets while absolute error stays flat.
        return [
            _row("test_S_1_000001", 3, 3, 0.5),
            _row("test_S_1_000002", 4, 4, 3 / 5),
            _row("test_S_1_000003", 7, 7, 6 / 8),
            _row("test_S_1_000004", 12, 12, 11 / 13),
        ]

    def test_buckets_partition_every_sample(self):
        result = analyse(load_rows := self._rows(), build_buckets())
        assert sum(b["samples"] for b in result["buckets"]) == len(load_rows)

    def test_overall_summary_counts_all_samples(self):
        assert analyse(self._rows(), build_buckets())["overall"]["samples"] == 4

    def test_absolute_error_is_flat_while_iou_is_not(self):
        result = analyse(self._rows(), build_buckets())
        errors = [b["mean_abs_error_px"] for b in result["buckets"] if b["samples"]]
        ious = [b["mean_iou"] for b in result["buckets"] if b["samples"]]
        assert max(errors) - min(errors) < 0.01
        assert max(ious) - min(ious) > 0.3
        # The IoU spread is entirely the denominator, so a size-vs-IoU rank
        # correlation appears where a size-vs-error one has nothing to find.
        assert result["size_iou_rank_correlation"] == 1.0

    def test_sheets_are_grouped_separately(self):
        rows = [
            _row("test_A_1_000001", 5, 5, 1.0),
            _row("test_B_1_000001", 5, 5, 0.5),
        ]
        groups = {s["group"]: s for s in analyse(rows, build_buckets())["sheets"]}
        assert set(groups) == {"A", "B"}
        assert groups["A"]["mean_iou"] == 1.0

    def test_zero_iou_and_threshold_fractions(self):
        rows = [
            _row("test_A_1_000001", 5, 0, 0.0),
            _row("test_A_1_000002", 5, 5, 1.0),
        ]
        overall = analyse(rows, build_buckets())["overall"]
        assert overall["zero_iou_count"] == 1
        assert overall["zero_iou_rate"] == 0.5
        assert overall["frac_iou_ge_050"] == 0.5

    def test_samples_outside_every_bucket_are_reported(self, caplog):
        rows = [_row("test_A_1_000001", 0, 0, 0.0)]
        result = analyse(rows, build_buckets())
        assert all(b["samples"] == 0 for b in result["buckets"])
        assert "outside every bucket" in caplog.text

    def test_empty_group_summary_has_no_metrics(self):
        assert _summary_keys(analyse([], build_buckets())["overall"]) == {"group", "samples"}


def _summary_keys(summary: dict) -> set[str]:
    """Return the keys of a summary dictionary.

    Args:
        summary: Summary dictionary.

    Returns:
        Its key set.
    """
    return set(summary)


class TestWriteCsv:
    def test_writes_only_populated_groups(self, tmp_path):
        summaries = analyse([_row("test_A_1_000001", 5, 5, 1.0)], build_buckets())
        out = tmp_path / "nested" / "out.csv"
        write_csv(out, summaries["buckets"] + [summaries["overall"]])
        lines = out.read_text("utf-8").strip().splitlines()
        assert len(lines) == 3  # header + the 4-5 px bucket + overall
        assert "mean_abs_error_px" in lines[0]

    def test_no_populated_groups_writes_nothing(self, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(out, [{"group": "all", "samples": 0}])
        assert not out.exists()
