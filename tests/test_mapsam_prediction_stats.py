"""Tests for per-sample prediction statistics, including scale conversion."""

from __future__ import annotations

import pytest
import torch

from archeo_topia.training.train_mapsam_v0 import compute_prediction_stats

MASK = 256


def _batch(gt_px: int, pred_px: int, overlap: int):
    """Build logits and masks with an exact pixel overlap.

    Args:
        gt_px: Ground-truth positive pixels.
        pred_px: Predicted positive pixels.
        overlap: Pixels where the two agree.

    Returns:
        ``(logits, target, ignore, box)`` tensors.
    """
    target = torch.zeros(1, 1, MASK, MASK)
    target.view(-1)[:gt_px] = 1.0

    logits = torch.full((1, 1, MASK, MASK), -20.0)
    flat = logits.view(-1)
    flat[:overlap] = 20.0  # inside the target
    flat[gt_px : gt_px + (pred_px - overlap)] = 20.0  # outside it

    return logits, target, torch.zeros(1, 1, MASK, MASK), torch.tensor([[0.0, 0.0, 10.0, 10.0]])


class TestPredictionStats:
    def test_perfect_prediction(self):
        logits, target, ignore, box = _batch(10, 10, 10)
        s = compute_prediction_stats(logits, target, ignore, box, ["a"])[0]
        assert s["iou"] == pytest.approx(1.0, abs=1e-4)
        assert s["abs_error_px"] == pytest.approx(0.0)

    def test_absolute_error_counts_both_kinds_of_disagreement(self):
        # 8 of 10 hit, plus 3 false positives: 2 missed + 3 spurious = 5.
        logits, target, ignore, box = _batch(10, 11, 8)
        s = compute_prediction_stats(logits, target, ignore, box, ["a"])[0]
        assert s["abs_error_px"] == pytest.approx(5.0)

    def test_same_absolute_error_scores_a_better_iou_on_a_bigger_target(self):
        small = compute_prediction_stats(*_batch(6, 6, 5), ["a"])[0]
        large = compute_prediction_stats(*_batch(60, 60, 59), ["b"])[0]
        assert small["abs_error_px"] == large["abs_error_px"]
        assert large["iou"] > small["iou"] + 0.2

    def test_window_conversion_makes_the_two_comparable(self):
        # A 6 px target in a 512 px window and a 60 px target in a 1620 px
        # window are the same mound at two zooms; a 2 px disagreement is the
        # same physical error and must convert to a similar source area.
        tight = compute_prediction_stats(
            *_batch(6, 6, 5), ["a"], window_xyxy=torch.tensor([[0.0, 0.0, 512.0, 512.0]])
        )[0]
        assert tight["source_px_per_mask_px"] == pytest.approx(4.0)
        assert tight["abs_error_source_px"] == pytest.approx(2.0 * 4.0)

    def test_source_conversion_uses_the_window_side(self):
        wide = compute_prediction_stats(
            *_batch(6, 6, 5), ["a"], window_xyxy=torch.tensor([[0.0, 0.0, 2400.0, 2200.0]])
        )[0]
        assert wide["window_side_px"] == pytest.approx(2300.0)
        assert wide["source_px_per_mask_px"] > 80.0

    def test_source_fields_are_absent_without_a_window(self):
        s = compute_prediction_stats(*_batch(6, 6, 5), ["a"])[0]
        assert "abs_error_source_px" not in s
        assert "source_px_per_mask_px" not in s

    def test_sheet_id_is_carried_through(self):
        s = compute_prediction_stats(*_batch(6, 6, 5), ["a"], sheet_ids=["K-35-8-G-a"])[0]
        assert s["sheet_id"] == "K-35-8-G-a"

    def test_logits_are_recorded_because_probabilities_saturate(self):
        logits, target, ignore, box = _batch(10, 10, 10)
        s = compute_prediction_stats(logits, target, ignore, box, ["a"])[0]
        assert s["pred_probability_max"] == pytest.approx(1.0)  # saturated
        assert s["logit_max"] == pytest.approx(20.0)  # the real signal
        assert s["logit_min"] == pytest.approx(-20.0)

    def test_probability_is_split_by_region(self):
        logits, target, ignore, box = _batch(10, 10, 10)
        s = compute_prediction_stats(logits, target, ignore, box, ["a"])[0]
        assert s["prob_mean_in_gt"] > 0.99
        assert s["prob_mean_in_pred"] > 0.99
        assert s["prob_mean_in_background"] < 0.01

    def test_empty_prediction_does_not_divide_by_zero(self):
        logits = torch.full((1, 1, MASK, MASK), -20.0)
        target = torch.zeros(1, 1, MASK, MASK)
        target.view(-1)[:5] = 1.0
        s = compute_prediction_stats(
            logits,
            target,
            torch.zeros(1, 1, MASK, MASK),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            ["a"],
        )[0]
        assert s["iou"] == 0.0
        assert s["prob_mean_in_pred"] == 0.0
        assert s["abs_error_px"] == pytest.approx(5.0)
