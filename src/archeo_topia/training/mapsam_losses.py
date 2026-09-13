#!/usr/bin/env python3
"""Loss functions for MapSAM mask-decoder fine-tuning.

Provides BCE, Dice, and combined BCE+Dice losses that respect an
ignore mask so that ambiguous regions do not contribute to the gradient.

Supports foreground-weighted BCE and bbox-local loss crops to combat
background domination when target objects are small.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
    foreground_bce_pos_weight: float | None = None,
    raise_on_zero_valid: bool = False,
) -> torch.Tensor:
    """Binary cross-entropy with logits, ignoring masked pixels.

    Computes BCE only over pixels where *ignore_mask* is 0.  Returns 0.0
    when no valid pixels remain (unless *raise_on_zero_valid* is True).

    Optionally weights positive-class pixels via *foreground_bce_pos_weight*
    to counter background domination when targets are small.

    Args:
        logits: Raw prediction logits of shape ``(B, 1, H, W)``.
        target: Binary target mask of shape ``(B, 1, H, W)`` with values 0/1.
        ignore_mask: Binary mask of shape ``(B, 1, H, W)``. Pixels with
            value 1 are excluded from the loss.
        foreground_bce_pos_weight: If provided, used as ``pos_weight`` in
            BCE to up-weight positive-class pixels.
        raise_on_zero_valid: If True, raise when no valid pixels remain
            instead of silently returning 0.0.

    Returns:
        Scalar ``torch.Tensor`` containing the mean BCE loss over valid
        pixels across the batch.

    Raises:
        ValueError: If *raise_on_zero_valid* and no valid pixels remain.
    """
    valid = (ignore_mask == 0).float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    if foreground_bce_pos_weight is not None and foreground_bce_pos_weight > 0:
        pos_weight = torch.tensor(
            foreground_bce_pos_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, target, weight=None, pos_weight=pos_weight, reduction="none"
        )

    weighted = bce * valid
    n_valid = valid.sum()
    if n_valid < 1:
        if raise_on_zero_valid:
            raise ValueError(
                f"BCE loss has zero valid pixels. "
                f"ignore_mask sum={ignore_mask.sum().item()}, "
                f"shape={ignore_mask.shape}"
            )
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


def build_bbox_valid_mask(
    bbox_xyxy: torch.Tensor,
    mask_shape: tuple[int, int],
    margin: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean mask that is True inside bbox + margin.

    Creates a spatial mask of shape ``(B, 1, H, W)`` that is ``True``
    for pixels within the bounding box expanded by *margin* pixels in
    each direction, clamped to image boundaries.

    Args:
        bbox_xyxy: Bounding boxes of shape ``(B, 4)`` in ``[x1, y1, x2, y2]``
            pixel coordinates.
        mask_shape: ``(height, width)`` of the target mask.
        margin: Number of pixels to expand the bbox in each direction.
        device: Target device for the output tensor.

    Returns:
        Boolean tensor of shape ``(B, 1, H, W)``.
    """
    height, width = mask_shape
    batch_size = bbox_xyxy.shape[0]
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=device)

    x1 = torch.clamp(bbox_xyxy[:, 0] - margin, min=0).long()
    y1 = torch.clamp(bbox_xyxy[:, 1] - margin, min=0).long()
    x2 = torch.clamp(bbox_xyxy[:, 2] + margin, max=width).long()
    y2 = torch.clamp(bbox_xyxy[:, 3] + margin, max=height).long()

    for b in range(batch_size):
        r1, c1 = int(y1[b]), int(x1[b])
        r2, c2 = int(y2[b]), int(x2[b])
        if r2 > r1 and c2 > c1:
            mask[b, 0, r1:r2, c1:c2] = True

    return mask


def combined_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    eps: float = 1e-6,
    foreground_bce_pos_weight: float | None = None,
    bbox_xyxy: torch.Tensor | None = None,
    bbox_loss_crop_margin: int | None = None,
    use_bbox_loss_crop: bool = False,
    raise_on_zero_valid: bool = False,
) -> dict[str, Any]:
    """Weighted combination of masked BCE and Dice loss.

    Supports foreground-weighted BCE and optional bbox-local loss crops
    to focus gradient computation on the region around the target object.

    Args:
        logits: Raw prediction logits of shape ``(B, 1, H, W)``.
        target: Binary target mask of shape ``(B, 1, H, W)`` with values 0/1.
        ignore_mask: Binary mask of shape ``(B, 1, H, W)``. Pixels with
            value 1 are excluded from the loss.
        bce_weight: Multiplicative weight for the BCE component.
        dice_weight: Multiplicative weight for the Dice component.
        eps: Epsilon passed to :func:`dice_loss`.
        foreground_bce_pos_weight: If provided, up-weights positive-class
            pixels in BCE.
        bbox_xyxy: Bounding boxes ``(B, 4)`` for optional loss crop.
        bbox_loss_crop_margin: Margin in pixels for bbox crop.
        use_bbox_loss_crop: If True, restrict loss computation to bbox +
            margin region.
        raise_on_zero_valid: If True, raise when no valid pixels remain
            instead of silently returning 0.0.

    Returns:
        Dictionary with keys ``loss``, ``bce_loss``, and ``dice_loss``.

    Raises:
        ValueError: If *raise_on_zero_valid* and no valid pixels remain.
    """
    effective_ignore = ignore_mask.clone()

    if use_bbox_loss_crop and bbox_xyxy is not None and bbox_loss_crop_margin is not None:
        bbox_mask = build_bbox_valid_mask(
            bbox_xyxy,
            mask_shape=logits.shape[2:],
            margin=bbox_loss_crop_margin,
            device=logits.device,
        )
        effective_ignore[~bbox_mask] = 1

    bce = masked_bce_with_logits_loss(
        logits,
        target,
        effective_ignore,
        foreground_bce_pos_weight=foreground_bce_pos_weight,
        raise_on_zero_valid=raise_on_zero_valid,
    )
    dice = dice_loss(logits, target, effective_ignore, eps=eps)

    total = bce_weight * bce + dice_weight * dice

    return {
        "loss": total,
        "bce_loss": bce,
        "dice_loss": dice,
    }
