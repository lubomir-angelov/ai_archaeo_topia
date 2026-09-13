"""Tests for separating near misses from genuine zero-IoU failures."""

from __future__ import annotations

import torch

from archeo_topia.analysis.mapsam_collapse_analysis import (
    analyze_sample,
    argmax_xy,
    centroid,
    classify,
    summarize,
)

MASK_SIZE = 256
WINDOW_SIDE = 2400.0
SOURCE_PX_PER_MASK_PX = WINDOW_SIDE / MASK_SIZE  # 9.375


def _blob(cx: int, cy: int, size: int = 2) -> torch.Tensor:
    """Square blob of *size* pixels a side, centred near ``(cx, cy)``."""
    mask = torch.zeros(MASK_SIZE, MASK_SIZE)
    mask[cy : cy + size, cx : cx + size] = 1.0
    return mask


def _logits_from(mask: torch.Tensor, peak: float = 10.0) -> torch.Tensor:
    """Logits that threshold to *mask*, with *peak* inside it."""
    return torch.where(mask > 0, torch.full_like(mask, peak), torch.full_like(mask, -10.0))


class TestGeometryHelpers:
    def test_centroid_of_empty_mask_is_none(self):
        assert centroid(torch.zeros(8, 8)) is None

    def test_centroid_is_in_xy_order(self):
        mask = torch.zeros(8, 8)
        mask[2, 5] = 1.0
        assert centroid(mask) == (5.0, 2.0)

    def test_argmax_is_in_xy_order(self):
        logits = torch.full((8, 8), -5.0)
        logits[2, 5] = 3.0
        assert argmax_xy(logits) == (5.0, 2.0)


class TestAnalyzeSample:
    def test_perfect_prediction_scores_one_and_zero_distance(self):
        target = _blob(100, 100)
        row = analyze_sample(_logits_from(target), target, WINDOW_SIDE, "s", "SHEET")
        assert row["iou"] == 1.0
        assert row["centroid_distance_mask_px"] == 0.0
        assert row["argmax_distance_source_px"] < SOURCE_PX_PER_MASK_PX

    def test_one_pixel_displacement_scores_zero_but_stays_close(self):
        target = _blob(100, 100)
        prediction = _blob(102, 100)
        row = analyze_sample(_logits_from(prediction), target, WINDOW_SIDE, "s", "SHEET")
        # No overlap at all, on a target only two pixels a side.
        assert row["iou"] == 0.0
        assert row["centroid_distance_mask_px"] == 2.0
        assert row["centroid_distance_source_px"] == round(2.0 * SOURCE_PX_PER_MASK_PX, 2)

    def test_empty_prediction_still_reports_where_the_peak_was(self):
        target = _blob(100, 100)
        logits = torch.full((MASK_SIZE, MASK_SIZE), -10.0)
        logits[100, 103] = -0.5  # below threshold, so nothing is predicted
        row = analyze_sample(logits, target, WINDOW_SIDE, "s", "SHEET")
        assert row["pred_pixels"] == 0
        assert row["centroid_distance_mask_px"] == ""
        assert row["argmax_distance_mask_px"] > 0.0

    def test_source_distances_scale_with_the_window(self):
        target = _blob(100, 100)
        prediction = _blob(104, 100)
        wide = analyze_sample(_logits_from(prediction), target, 2400.0, "s", "SHEET")
        tight = analyze_sample(_logits_from(prediction), target, 512.0, "s", "SHEET")
        assert wide["centroid_distance_mask_px"] == tight["centroid_distance_mask_px"]
        assert wide["centroid_distance_source_px"] > tight["centroid_distance_source_px"]


class TestClassify:
    def _row(self, **kwargs):
        base = {
            "iou": 0.0,
            "pred_pixels": 2,
            "centroid_distance_source_px": 10.0,
            "argmax_distance_source_px": 10.0,
        }
        base.update(kwargs)
        return base

    def test_any_overlap_is_a_hit(self):
        assert classify(self._row(iou=0.01), near_threshold_px=40.0) == "hit"

    def test_close_prediction_is_a_near_miss(self):
        assert classify(self._row(), near_threshold_px=40.0) == "near_miss"

    def test_distant_prediction_is_a_wrong_object(self):
        row = self._row(centroid_distance_source_px=900.0)
        assert classify(row, near_threshold_px=40.0) == "wrong_object"

    def test_empty_prediction_is_judged_by_its_peak_logit(self):
        near = self._row(
            pred_pixels=0, centroid_distance_source_px="", argmax_distance_source_px=8.0
        )
        far = self._row(
            pred_pixels=0, centroid_distance_source_px="", argmax_distance_source_px=800.0
        )
        assert classify(near, near_threshold_px=40.0) == "near_miss"
        assert classify(far, near_threshold_px=40.0) == "wrong_object"


class TestSummarize:
    def test_counts_split_zero_iou_by_class(self):
        rows = [
            {
                "iou": 0.6,
                "pred_pixels": 4,
                "centroid_distance_source_px": 5.0,
                "argmax_distance_source_px": 5.0,
                "class": "hit",
            },
            {
                "iou": 0.0,
                "pred_pixels": 0,
                "centroid_distance_source_px": "",
                "argmax_distance_source_px": 20.0,
                "class": "near_miss",
            },
            {
                "iou": 0.0,
                "pred_pixels": 3,
                "centroid_distance_source_px": 900.0,
                "argmax_distance_source_px": 900.0,
                "class": "wrong_object",
            },
        ]
        summary = summarize(rows, near_threshold_px=40.0)
        assert summary["n"] == 3
        assert summary["n_zero_iou"] == 2
        assert summary["n_zero_prediction"] == 1
        assert summary["n_near_miss"] == 1
        assert summary["n_wrong_object"] == 1
        assert summary["median_near_miss_distance_source_px"] == 20.0
        assert summary["median_hit_centroid_distance_source_px"] == 5.0
