#!/usr/bin/env python3
"""Tests for MapSAM loss functions."""

from __future__ import annotations

import pytest
import torch

from archeo_topia.training.mapsam_losses import (
    build_bbox_valid_mask,
    combined_bce_dice_loss,
    dice_loss,
    masked_bce_with_logits_loss,
)


class TestMaskedBceWithLogitsLoss:
    """Tests for masked_bce_with_logits_loss."""

    def test_returns_scalar(self) -> None:
        logits = torch.randn(2, 1, 8, 8)
        target = torch.zeros(2, 1, 8, 8)
        ignore = torch.zeros(2, 1, 8, 8)
        loss = masked_bce_with_logits_loss(logits, target, ignore)
        assert loss.ndim == 0

    def test_perfect_prediction_low_loss(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * 10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = masked_bce_with_logits_loss(logits, target, ignore)
        assert loss.item() < 1e-4

    def test_wrong_prediction_higher_loss(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * -10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = masked_bce_with_logits_loss(logits, target, ignore)
        assert loss.item() > 5.0

    def test_ignore_mask_excludes_pixels(self) -> None:
        logits = torch.zeros(1, 1, 4, 4)
        logits[0, 0, 0, 0] = -100.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        ignore[0, 0, 0, 0] = 1.0
        loss = masked_bce_with_logits_loss(logits, target, ignore)
        assert loss.item() < 1.0

    def test_empty_valid_region_no_crash(self) -> None:
        logits = torch.randn(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        ignore = torch.ones(1, 1, 4, 4)
        loss = masked_bce_with_logits_loss(logits, target, ignore)
        assert loss.item() == 0.0

    def test_gradients_flow(self) -> None:
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        target = torch.rand(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = masked_bce_with_logits_loss(logits, target, ignore)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_foreground_weighted_returns_scalar(self) -> None:
        logits = torch.randn(2, 1, 8, 8)
        target = torch.zeros(2, 1, 8, 8)
        ignore = torch.zeros(2, 1, 8, 8)
        loss = masked_bce_with_logits_loss(logits, target, ignore, foreground_bce_pos_weight=20.0)
        assert loss.ndim == 0

    def test_foreground_weight_increases_loss_for_missed_positives(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * -5.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss_unweighted = masked_bce_with_logits_loss(logits, target, ignore)
        loss_weighted = masked_bce_with_logits_loss(
            logits, target, ignore, foreground_bce_pos_weight=20.0
        )
        assert loss_weighted.item() > loss_unweighted.item()

    def test_foreground_weight_gradients_flow(self) -> None:
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        target = torch.rand(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = masked_bce_with_logits_loss(logits, target, ignore, foreground_bce_pos_weight=10.0)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_pos_weight_no_device_mismatch(self) -> None:
        if torch.cuda.is_available():
            logits = torch.randn(1, 1, 4, 4, device="cuda")
            target = torch.rand(1, 1, 4, 4, device="cuda")
            ignore = torch.zeros(1, 1, 4, 4, device="cuda")
            loss = masked_bce_with_logits_loss(
                logits, target, ignore, foreground_bce_pos_weight=20.0
            )
            assert loss.device.type == "cuda"


class TestDiceLoss:
    """Tests for dice_loss."""

    def test_returns_scalar(self) -> None:
        logits = torch.randn(2, 1, 8, 8)
        target = torch.zeros(2, 1, 8, 8)
        ignore = torch.zeros(2, 1, 8, 8)
        loss = dice_loss(logits, target, ignore)
        assert loss.ndim == 0

    def test_perfect_prediction_low_loss(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * 10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = dice_loss(logits, target, ignore)
        assert loss.item() < 1e-4

    def test_wrong_prediction_higher_loss(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * -10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = dice_loss(logits, target, ignore)
        assert loss.item() > 0.9

    def test_ignore_mask_excludes_pixels(self) -> None:
        """Ignoring a bad prediction should lower the Dice loss."""
        logits = torch.ones(1, 1, 4, 4) * 10.0
        logits[0, 0, 0, 0] = -10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        ignore[0, 0, 0, 0] = 1.0
        loss_with_ignore = dice_loss(logits, target, ignore)
        loss_no_ignore = dice_loss(logits, target, torch.zeros(1, 1, 4, 4))
        assert loss_with_ignore.item() < loss_no_ignore.item()

    def test_empty_valid_region_no_crash(self) -> None:
        logits = torch.randn(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        ignore = torch.ones(1, 1, 4, 4)
        loss = dice_loss(logits, target, ignore)
        assert loss.item() == 0.0

    def test_gradients_flow(self) -> None:
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        target = torch.rand(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = dice_loss(logits, target, ignore)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestBboxValidMask:
    """Tests for build_bbox_valid_mask."""

    def test_shape(self) -> None:
        bbox = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
        mask = build_bbox_valid_mask(bbox, (100, 100), margin=0, device="cpu")
        assert mask.shape == (1, 1, 100, 100)
        assert mask.dtype == torch.bool

    def test_batch_shape(self) -> None:
        bbox = torch.tensor(
            [
                [10.0, 20.0, 50.0, 60.0],
                [30.0, 40.0, 70.0, 80.0],
            ]
        )
        mask = build_bbox_valid_mask(bbox, (100, 100), margin=0, device="cpu")
        assert mask.shape == (2, 1, 100, 100)

    def test_clamps_to_boundaries(self) -> None:
        bbox = torch.tensor([[-10.0, -20.0, 110.0, 120.0]])
        mask = build_bbox_valid_mask(bbox, (100, 100), margin=0, device="cpu")
        assert mask.shape == (1, 1, 100, 100)
        assert mask.sum() == 100 * 100

    def test_respects_margin(self) -> None:
        bbox = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
        mask_no_margin = build_bbox_valid_mask(bbox, (100, 100), margin=0, device="cpu")
        mask_with_margin = build_bbox_valid_mask(bbox, (100, 100), margin=10, device="cpu")
        assert mask_with_margin.sum() > mask_no_margin.sum()

    def test_inside_bbox_is_true(self) -> None:
        bbox = torch.tensor([[20.0, 30.0, 40.0, 50.0]])
        mask = build_bbox_valid_mask(bbox, (100, 100), margin=0, device="cpu")
        assert mask[0, 0, 35, 30].item() is True

    def test_outside_bbox_is_false(self) -> None:
        bbox = torch.tensor([[20.0, 30.0, 40.0, 50.0]])
        mask = build_bbox_valid_mask(bbox, (100, 100), margin=0, device="cpu")
        assert mask[0, 0, 0, 0].item() is False

    def test_empty_bbox_no_crash(self) -> None:
        bbox = torch.tensor([[50.0, 50.0, 50.0, 50.0]])
        mask = build_bbox_valid_mask(bbox, (100, 100), margin=0, device="cpu")
        assert mask.sum() == 0


class TestCombinedBceDiceLoss:
    """Tests for combined_bce_dice_loss."""

    def test_returns_dict(self) -> None:
        logits = torch.randn(2, 1, 8, 8)
        target = torch.zeros(2, 1, 8, 8)
        ignore = torch.zeros(2, 1, 8, 8)
        result = combined_bce_dice_loss(logits, target, ignore)
        assert "loss" in result
        assert "bce_loss" in result
        assert "dice_loss" in result

    def test_loss_is_scalar(self) -> None:
        logits = torch.randn(2, 1, 8, 8)
        target = torch.zeros(2, 1, 8, 8)
        ignore = torch.zeros(2, 1, 8, 8)
        result = combined_bce_dice_loss(logits, target, ignore)
        assert result["loss"].ndim == 0

    def test_perfect_prediction_low_loss(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * 10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        result = combined_bce_dice_loss(logits, target, ignore)
        assert result["loss"].item() < 1e-3

    def test_gradients_flow(self) -> None:
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        target = torch.rand(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        result = combined_bce_dice_loss(logits, target, ignore)
        result["loss"].backward()
        assert logits.grad is not None

    def test_custom_weights(self) -> None:
        logits = torch.randn(1, 1, 4, 4)
        target = torch.rand(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        r1 = combined_bce_dice_loss(logits, target, ignore, bce_weight=2.0, dice_weight=0.0)
        r2 = combined_bce_dice_loss(logits, target, ignore, bce_weight=1.0, dice_weight=0.0)
        assert abs(r1["loss"].item() - 2.0 * r2["loss"].item()) < 1e-6

    def test_empty_valid_region_no_crash(self) -> None:
        logits = torch.randn(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        ignore = torch.ones(1, 1, 4, 4)
        result = combined_bce_dice_loss(logits, target, ignore)
        assert result["loss"].item() == 0.0

    def test_empty_valid_region_raises_when_requested(self) -> None:
        logits = torch.randn(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        ignore = torch.ones(1, 1, 4, 4)
        with pytest.raises(ValueError, match="zero valid pixels"):
            combined_bce_dice_loss(logits, target, ignore, raise_on_zero_valid=True)

    def test_bbox_crop_ignores_pixels_outside(self) -> None:
        logits = torch.zeros(1, 1, 10, 10)
        target = torch.zeros(1, 1, 10, 10)
        ignore = torch.zeros(1, 1, 10, 10)
        target[0, 0, 4:6, 4:6] = 1.0
        bbox = torch.tensor([[3.0, 3.0, 7.0, 7.0]])

        logits_mod = logits.clone()
        logits_mod[0, 0, 0:2, 0:2] = 100.0

        crop_base = combined_bce_dice_loss(
            logits,
            target,
            ignore,
            use_bbox_loss_crop=True,
            bbox_xyxy=bbox,
            bbox_loss_crop_margin=0,
        )
        crop_mod = combined_bce_dice_loss(
            logits_mod,
            target,
            ignore,
            use_bbox_loss_crop=True,
            bbox_xyxy=bbox,
            bbox_loss_crop_margin=0,
        )
        assert abs(crop_base["loss"].item() - crop_mod["loss"].item()) < 1e-6

    def test_ignore_mask_and_bbox_crop_combine(self) -> None:
        logits = torch.zeros(1, 1, 10, 10)
        target = torch.zeros(1, 1, 10, 10)
        ignore = torch.zeros(1, 1, 10, 10)
        target[0, 0, 4:6, 4:6] = 1.0
        ignore[0, 0, 4:6, 4:6] = 1.0
        bbox = torch.tensor([[3.0, 3.0, 7.0, 7.0]])

        logits_mod = logits.clone()
        logits_mod[0, 0, 4:6, 4:6] = 100.0

        base = combined_bce_dice_loss(
            logits,
            target,
            ignore,
            use_bbox_loss_crop=True,
            bbox_xyxy=bbox,
            bbox_loss_crop_margin=0,
        )
        mod = combined_bce_dice_loss(
            logits_mod,
            target,
            ignore,
            use_bbox_loss_crop=True,
            bbox_xyxy=bbox,
            bbox_loss_crop_margin=0,
        )
        assert abs(base["loss"].item() - mod["loss"].item()) < 1e-6

    def test_empty_bbox_crop_no_crash(self) -> None:
        logits = torch.randn(1, 1, 10, 10)
        target = torch.zeros(1, 1, 10, 10)
        ignore = torch.zeros(1, 1, 10, 10)
        bbox = torch.tensor([[5.0, 5.0, 5.0, 5.0]])

        result = combined_bce_dice_loss(
            logits,
            target,
            ignore,
            use_bbox_loss_crop=True,
            bbox_xyxy=bbox,
            bbox_loss_crop_margin=0,
        )
        assert result["loss"].item() == 0.0

    def test_foreground_weight_with_bbox_crop(self) -> None:
        logits = torch.zeros(1, 1, 10, 10)
        target = torch.zeros(1, 1, 10, 10)
        ignore = torch.zeros(1, 1, 10, 10)
        target[0, 0, 4:6, 4:6] = 1.0
        bbox = torch.tensor([[3.0, 3.0, 7.0, 7.0]])

        r_no_weight = combined_bce_dice_loss(
            logits,
            target,
            ignore,
            use_bbox_loss_crop=True,
            bbox_xyxy=bbox,
            bbox_loss_crop_margin=0,
            foreground_bce_pos_weight=None,
        )
        r_weighted = combined_bce_dice_loss(
            logits,
            target,
            ignore,
            use_bbox_loss_crop=True,
            bbox_xyxy=bbox,
            bbox_loss_crop_margin=0,
            foreground_bce_pos_weight=20.0,
        )
        assert r_weighted["bce_loss"].item() > r_no_weight["bce_loss"].item()

    def test_dice_with_ignore_mask(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * 10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        result = combined_bce_dice_loss(logits, target, ignore)
        assert result["dice_loss"].item() < 1e-4
