#!/usr/bin/env python3
"""Loss functions for MapSAM mask-decoder fine-tuning.

Provides BCE, Dice, and combined BCE+Dice losses that respect an
ignore mask so that ambiguous regions do not contribute to the gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy with logits, ignoring masked pixels.

    Computes BCE only over pixels where *ignore_mask* is 0.  Returns 0.0
    when no valid pixels remain.

    Args:
        logits: Raw prediction logits of shape ``(B, 1, H, W)``.
        target: Binary target mask of shape ``(B, 1, H, W)`` with values 0/1.
        ignore_mask: Binary mask of shape ``(B, 1, H, W)``. Pixels with
            value 1 are excluded from the loss.

    Returns:
        Scalar ``torch.Tensor`` containing the mean BCE loss over valid
        pixels across the batch.
    """
    valid = (ignore_mask == 0).float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = bce * valid
    n_valid = valid.sum()
    if n_valid < 1:
        return logits.sum() * 0.0
    return weighted.sum() / n_valid


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss with ignore-mask support.

    Computes the Dice coefficient between sigmoid(logits) and the target,
    then returns ``1 - dice``.  Only pixels where *ignore_mask* is 0
    participate in the computation.

    Args:
        logits: Raw prediction logits of shape ``(B, 1, H, W)``.
        target: Binary target mask of shape ``(B, 1, H, W)`` with values 0/1.
        ignore_mask: Binary mask of shape ``(B, 1, H, W)``. Pixels with
            value 1 are excluded from the loss.
        eps: Small constant for numerical stability.

    Returns:
        Scalar ``torch.Tensor`` containing the Dice loss (1 - Dice) over
        valid pixels across the batch.
    """
    valid = (ignore_mask == 0).float()
    pred = torch.sigmoid(logits)
    pred_clipped = pred * valid
    target_clipped = target * valid

    intersection = (pred_clipped * target_clipped).sum()
    union = pred_clipped.sum() + target_clipped.sum()

    dice_coef = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice_coef


def combined_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted combination of masked BCE and Dice loss.

    Args:
        logits: Raw prediction logits of shape ``(B, 1, H, W)``.
        target: Binary target mask of shape ``(B, 1, H, W)`` with values 0/1.
        ignore_mask: Binary mask of shape ``(B, 1, H, W)``. Pixels with
            value 1 are excluded from the loss.
        bce_weight: Multiplicative weight for the BCE component.
        dice_weight: Multiplicative weight for the Dice component.
        eps: Epsilon passed to :func:`dice_loss`.

    Returns:
        Scalar ``torch.Tensor`` equal to
        ``bce_weight * BCE + dice_weight * Dice``.
    """
    bce = masked_bce_with_logits_loss(logits, target, ignore_mask)
    dice = dice_loss(logits, target, ignore_mask, eps=eps)
    return bce_weight * bce + dice_weight * dice
