#!/usr/bin/env python3
"""Tests for MapSAM loss functions."""

from __future__ import annotations

import torch

from archeo_topia.training.mapsam_losses import (
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


class TestCombinedBceDiceLoss:
    """Tests for combined_bce_dice_loss."""

    def test_returns_scalar(self) -> None:
        logits = torch.randn(2, 1, 8, 8)
        target = torch.zeros(2, 1, 8, 8)
        ignore = torch.zeros(2, 1, 8, 8)
        loss = combined_bce_dice_loss(logits, target, ignore)
        assert loss.ndim == 0

    def test_perfect_prediction_low_loss(self) -> None:
        logits = torch.ones(1, 1, 4, 4) * 10.0
        target = torch.ones(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = combined_bce_dice_loss(logits, target, ignore)
        assert loss.item() < 1e-3

    def test_gradients_flow(self) -> None:
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        target = torch.rand(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss = combined_bce_dice_loss(logits, target, ignore)
        loss.backward()
        assert logits.grad is not None

    def test_custom_weights(self) -> None:
        logits = torch.randn(1, 1, 4, 4)
        target = torch.rand(1, 1, 4, 4)
        ignore = torch.zeros(1, 1, 4, 4)
        loss1 = combined_bce_dice_loss(logits, target, ignore, bce_weight=2.0, dice_weight=0.0)
        loss2 = combined_bce_dice_loss(logits, target, ignore, bce_weight=1.0, dice_weight=0.0)
        assert abs(loss1.item() - 2.0 * loss2.item()) < 1e-6

    def test_empty_valid_region_no_crash(self) -> None:
        logits = torch.randn(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        ignore = torch.ones(1, 1, 4, 4)
        loss = combined_bce_dice_loss(logits, target, ignore)
        assert loss.item() == 0.0
