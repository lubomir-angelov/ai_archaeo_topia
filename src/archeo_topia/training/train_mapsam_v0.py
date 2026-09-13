#!/usr/bin/env python3
"""MapSAM v0.1 — frozen image encoder, train mask decoder only.

Fine-tunes only the SAM mask decoder on archaeological mound-symbol
prompts.  Image encoder and prompt encoder remain frozen.

Supports overfit-test mode, prediction statistics, and foreground/bbox-aware
loss to diagnose and improve training quality.

Usage::

    python -m archeo_topia.training.train_mapsam_v0 \\
        --config configs/mapsam/mapsam_v0_1_decoder_only.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from archeo_topia.datasets.mapsam_dataset import MapSamDataset
from archeo_topia.datasets.mapsam_embedding_dataset import (
    MapSamEmbeddingDataset,
)
from archeo_topia.training.mapsam_losses import combined_bce_dice_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Nested dictionary with configuration values.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand relative paths and fill defaults.

    Args:
        cfg: Raw configuration dictionary.

    Returns:
        Configuration dictionary with resolved paths and defaults.
    """
    result = json.loads(json.dumps(cfg))

    ds = result.get("dataset", {})
    root = ds.get("root", "")
    samples = ds.get("samples", "")
    if not Path(samples).is_absolute():
        ds["samples"] = str(Path(root) / samples)
    result["dataset"] = ds

    mdl = result.get("model", {})
    ckpt = mdl.get("sam_checkpoint", "")
    if not Path(ckpt).is_absolute():
        mdl["sam_checkpoint"] = str(Path.cwd() / ckpt)
    result["model"] = mdl

    out = result.get("outputs", {})
    out_root = out.get("root", "")
    if not Path(out_root).is_absolute():
        out["root"] = str(Path.cwd() / out_root)
    result["outputs"] = out

    trn = result.get("training", {})
    trn.setdefault("seed", 42)
    trn.setdefault("max_train_batches", None)
    trn.setdefault("max_val_batches", None)
    trn.setdefault("save_every_epochs", 5)
    trn.setdefault("validate_every_epochs", 1)
    result["training"] = trn

    loss_cfg = result.get("loss", {})
    loss_cfg.setdefault("eps", 1e-6)
    loss_cfg.setdefault("bce_weight", 1.0)
    loss_cfg.setdefault("dice_weight", 1.0)
    loss_cfg.setdefault("foreground_bce_pos_weight", None)
    loss_cfg.setdefault("use_bbox_loss_crop", False)
    loss_cfg.setdefault("bbox_loss_crop_margin", 32)
    result["loss"] = loss_cfg

    outputs_cfg = result.get("outputs", {})
    outputs_cfg.setdefault("save_debug_predictions", True)
    outputs_cfg.setdefault("debug_prediction_count", 8)
    result["outputs"] = outputs_cfg

    debug_cfg = result.get("debug", {})
    debug_cfg.setdefault("overfit_mode", False)
    debug_cfg.setdefault("overfit_sample_count", 5)
    debug_cfg.setdefault("overfit_repeat_factor", 200)
    debug_cfg.setdefault("log_prediction_stats", False)
    result["debug"] = debug_cfg

    return result


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Overfit dataset wrapper
# ---------------------------------------------------------------------------


class OverfitDataset(Dataset):
    """Wraps a dataset to repeat a tiny subset for overfit testing.

    Selects the first *N* samples and repeats them *repeat_factor* times
    so the effective dataset size is ``N * repeat_factor``.

    Args:
        base_dataset: The original dataset.
        sample_count: Number of unique samples to use.
        repeat_factor: How many times to repeat each sample.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        sample_count: int,
        repeat_factor: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.sample_count = min(sample_count, len(base_dataset))
        self.repeat_factor = repeat_factor

        self._indices: list[int] = []
        for i in range(self.sample_count):
            for _ in range(self.repeat_factor):
                self._indices.append(i)

        sample_ids = [base_dataset._samples[i]["sample_id"] for i in range(self.sample_count)]  # noqa: SLF001
        logger.info(
            "OverfitDataset: %d unique samples x %d repeats = %d effective size",
            self.sample_count,
            self.repeat_factor,
            len(self._indices),
        )
        logger.info("Overfit sample IDs: %s", sample_ids)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.base_dataset[self._indices[idx]]


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_sam_model(cfg: dict[str, Any]) -> nn.Module:
    """Load SAM from checkpoint and apply freezing policy.

    Args:
        cfg: Resolved configuration dictionary.

    Returns:
        SAM model with appropriate parameters frozen.
    """
    model_type = cfg["model"]["model_type"]
    checkpoint_path = cfg["model"]["sam_checkpoint"]

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        logger.error("SAM checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        logger.error(
            "segment_anything not installed. Install with:\n"
            '  python -m pip install "git+https://github.com/facebookresearch/segment-anything.git"'
        )
        raise SystemExit(1) from exc

    logger.info("Loading SAM model: %s from %s", model_type, checkpoint_path)
    sam = sam_model_registry[model_type](checkpoint=str(ckpt))

    if cfg["model"].get("freeze_image_encoder", True):
        logger.info("Freezing image encoder")
        sam.image_encoder.eval()
        for p in sam.image_encoder.parameters():
            p.requires_grad = False

    if not cfg["model"].get("train_prompt_encoder", False):
        logger.info("Freezing prompt encoder")
        sam.prompt_encoder.eval()
        for p in sam.prompt_encoder.parameters():
            p.requires_grad = False

    if not cfg["model"].get("train_mask_decoder", True):
        logger.info("Freezing mask decoder")
        sam.mask_decoder.eval()
        for p in sam.mask_decoder.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in sam.parameters() if not p.requires_grad)
    logger.info("Trainable parameters: %d", trainable)
    logger.info("Frozen parameters: %d", frozen)

    if trainable == 0:
        logger.error("No trainable parameters found. Check model config.")
        sys.exit(1)

    return sam


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------


def prepare_image_for_sam(
    image: torch.Tensor,
    sam: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Normalize and resize image to SAM's expected input format.

    The dataset returns images in [0, 1] range. SAM expects images
    normalized with its own pixel_mean and pixel_std, then resized to
    the image encoder's input size.

    Args:
        image: Image tensor of shape ``(B, 3, H, W)`` or ``(3, H, W)``
            in ``[0, 1]``.
        sam: Loaded SAM model.
        device: Target device.

    Returns:
        Normalized tensor of shape ``(B, 3, img_size, img_size)`` ready
        for ``sam.image_encoder``.
    """
    pixel_mean = sam.pixel_mean.squeeze()
    pixel_std = sam.pixel_std.squeeze()

    img = image.to(device)
    if img.dim() == 3:
        img = img.unsqueeze(0)

    pm = pixel_mean.view(1, 3, 1, 1).to(device)
    ps = pixel_std.view(1, 3, 1, 1).to(device)
    img = (img * 255.0 - pm) / ps

    target_size = sam.image_encoder.img_size
    if img.shape[2] != target_size or img.shape[3] != target_size:
        img = torch.nn.functional.interpolate(
            img,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

    return img


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def forward_sam(
    sam: nn.Module,
    image: torch.Tensor,
    box_prompt: torch.Tensor,
    point_prompt: torch.Tensor,
    point_label: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run a single SAM forward pass and return low-res mask logits.

    Uses ``multimask_output=False`` so the decoder produces a single
    mask per prompt.  Frozen sub-modules (image encoder, prompt encoder)
    are evaluated inside ``torch.no_grad()``.

    Args:
        sam: Loaded SAM model.
        image: Image tensor ``(B, 3, H, W)`` or ``(3, H, W)`` in [0, 1].
        box_prompt: Bounding box ``(B, 4)`` or ``(4,)`` in xyxy.
        point_prompt: Point ``(B, 2)`` or ``(2,)`` in xy.
        point_label: Label ``(B, 1)`` or ``(1,)``.
        device: Target device.

    Returns:
        Mask logits of shape ``(B, 1, 256, 256)``.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if box_prompt.dim() == 1:
        box_prompt = box_prompt.unsqueeze(0)
    if point_prompt.dim() == 1:
        point_prompt = point_prompt.unsqueeze(0)
    if point_label.dim() == 1:
        point_label = point_label.unsqueeze(0)

    img_input = prepare_image_for_sam(image, sam, device)

    with torch.no_grad():
        image_embeddings = sam.image_encoder(img_input)

    point_coords = point_prompt.to(device).unsqueeze(1)
    point_labels = point_label.to(device)
    boxes = box_prompt.to(device)

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=boxes,
            masks=None,
        )

    low_res_logits, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    return low_res_logits


def forward_sam_cached(
    sam: nn.Module,
    image_embedding: torch.Tensor,
    box_prompt: torch.Tensor,
    point_prompt: torch.Tensor,
    point_label: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run SAM forward pass using a pre-computed image embedding.

    Skips the image encoder entirely.  Prompt encoder is still evaluated
    inside ``torch.no_grad()`` since it is frozen.

    Args:
        sam: Loaded SAM model.
        image_embedding: Pre-computed embedding ``(256, 64, 64)`` or
            ``(B, 256, 64, 64)``.
        box_prompt: Bounding box ``(B, 4)`` or ``(4,)`` in xyxy.
        point_prompt: Point ``(B, 2)`` or ``(2,)`` in xy.
        point_label: Label ``(B, 1)`` or ``(1,)``.
        device: Target device.

    Returns:
        Mask logits of shape ``(B, 1, 256, 256)``.
    """
    if image_embedding.dim() == 3:
        image_embedding = image_embedding.unsqueeze(0)

    if box_prompt.dim() == 1:
        box_prompt = box_prompt.unsqueeze(0)
    if point_prompt.dim() == 1:
        point_prompt = point_prompt.unsqueeze(0)
    if point_label.dim() == 1:
        point_label = point_label.unsqueeze(0)

    image_embeddings = image_embedding.to(device)

    point_coords = point_prompt.to(device).unsqueeze(1)
    point_labels = point_label.to(device)
    boxes = box_prompt.to(device)

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=boxes,
            masks=None,
        )

    low_res_logits, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    return low_res_logits


# ---------------------------------------------------------------------------
# Metrics and prediction statistics
# ---------------------------------------------------------------------------


def compute_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute IoU and Dice metrics on thresholded predictions.

    Args:
        logits: Raw logits ``(B, 1, H, W)``.
        target: Binary target ``(B, 1, H, W)``.
        ignore_mask: Ignore mask ``(B, 1, H, W)``.
        threshold: Threshold for binarizing predictions.

    Returns:
        Dictionary with ``mean_iou`` and ``mean_dice``.
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    valid = (ignore_mask == 0).float()

    pred_valid = pred * valid
    target_valid = target * valid

    intersection = (pred_valid * target_valid).sum(dim=[1, 2, 3])
    union = pred_valid.sum(dim=[1, 2, 3]) + target_valid.sum(dim=[1, 2, 3])

    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    dice_val = (2.0 * intersection + 1e-6) / (union + 1e-6)

    return {
        "mean_iou": iou.mean().item(),
        "mean_dice": dice_val.mean().item(),
    }


def compute_prediction_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_mask: torch.Tensor,
    box_prompt: torch.Tensor,
    sample_ids: list[str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Compute per-sample prediction statistics.

    Args:
        logits: Raw logits ``(B, 1, H, W)``.
        target: Binary target ``(B, 1, H, W)``.
        ignore_mask: Ignore mask ``(B, 1, H, W)``.
        box_prompt: Bounding boxes ``(B, 4)`` in xyxy.
        sample_ids: List of sample IDs for each batch element.
        threshold: Threshold for binarizing predictions.

    Returns:
        List of per-sample statistic dictionaries.
    """
    prob = torch.sigmoid(logits)
    pred_bin = (prob > threshold).float()
    valid = (ignore_mask == 0).float()

    stats_list: list[dict[str, Any]] = []
    for b in range(logits.shape[0]):
        v = valid[b, 0]
        t = target[b, 0]
        p = pred_bin[b, 0]
        pr = prob[b, 0]

        gt_pos = (t * v).sum().item()
        pred_pos = (p * v).sum().item()
        valid_count = v.sum().item()
        prob_valid = pr * v

        prob_mean = prob_valid.mean().item() if valid_count > 0 else 0.0
        prob_max = prob_valid.max().item() if valid_count > 0 else 0.0
        prob_min = prob_valid.min().item() if valid_count > 0 else 0.0

        x1, y1, x2, y2 = box_prompt[b].tolist()
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

        target_area_ratio = gt_pos / valid_count if valid_count > 0 else 0.0
        prediction_area_ratio = pred_pos / valid_count if valid_count > 0 else 0.0

        intersection = (p * t * v).sum().item()
        union = (p * v).sum().item() + (t * v).sum().item()
        iou = intersection / (union - intersection + 1e-6) if union > 0 else 0.0
        dice = 2 * intersection / (union + 1e-6) if union > 0 else 0.0

        stats_list.append(
            {
                "sample_id": sample_ids[b] if b < len(sample_ids) else f"batch_{b}",
                "gt_positive_pixels": int(gt_pos),
                "pred_positive_pixels_threshold_0_5": int(pred_pos),
                "pred_probability_mean": round(prob_mean, 6),
                "pred_probability_max": round(prob_max, 6),
                "pred_probability_min": round(prob_min, 6),
                "iou": round(iou, 6),
                "dice": round(dice, 6),
                "bbox_area": round(bbox_area, 2),
                "target_area_ratio": round(target_area_ratio, 6),
                "prediction_area_ratio": round(prediction_area_ratio, 6),
                "valid_pixels": int(valid_count),
            }
        )

    return stats_list


def log_prediction_warnings(stats_list: list[dict[str, Any]]) -> None:
    """Log warnings if predictions appear to be all-background.

    Args:
        stats_list: List of per-sample statistic dictionaries.
    """
    if not stats_list:
        return

    empty_count = sum(1 for s in stats_list if s["pred_positive_pixels_threshold_0_5"] == 0)
    if empty_count > len(stats_list) * 0.5:
        logger.warning(
            "Predictions are nearly empty at threshold 0.5 (%d/%d samples). "
            "Model may be learning all-background.",
            empty_count,
            len(stats_list),
        )

    mean_pred_ratio = sum(s["prediction_area_ratio"] for s in stats_list) / len(stats_list)
    mean_gt_ratio = sum(s["target_area_ratio"] for s in stats_list) / len(stats_list)
    if mean_pred_ratio < mean_gt_ratio * 0.1 and mean_gt_ratio > 0.001:
        logger.warning(
            "Mean prediction area ratio (%.4f) is much smaller than "
            "mean target area ratio (%.4f). Model under-predicts foreground.",
            mean_pred_ratio,
            mean_gt_ratio,
        )


def compute_stats_aggregates(
    stats_list: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute aggregate statistics from per-sample stats.

    Args:
        stats_list: List of per-sample statistic dictionaries.

    Returns:
        Dictionary with mean aggregate values.
    """
    if not stats_list:
        return {}

    n = len(stats_list)
    return {
        "mean_gt_positive_pixels": sum(s["gt_positive_pixels"] for s in stats_list) / n,
        "mean_pred_positive_pixels": sum(
            s["pred_positive_pixels_threshold_0_5"] for s in stats_list
        )
        / n,
        "mean_pred_probability_mean": sum(s["pred_probability_mean"] for s in stats_list) / n,
        "mean_pred_probability_max": sum(s["pred_probability_max"] for s in stats_list) / n,
        "mean_target_area_ratio": sum(s["target_area_ratio"] for s in stats_list) / n,
        "mean_prediction_area_ratio": sum(s["prediction_area_ratio"] for s in stats_list) / n,
        "mean_iou": sum(s["iou"] for s in stats_list) / n,
        "mean_dice": sum(s["dice"] for s in stats_list) / n,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _log_first_batch_diagnostics(
    target: torch.Tensor,
    ignore: torch.Tensor,
    box_prompt: torch.Tensor,
    logits: torch.Tensor,
    loss_cfg: dict[str, Any],
    image_size: int = 1024,
) -> None:
    """Log diagnostic info for the first batch to catch silent bugs.

    Args:
        target: Target mask ``(B, 1, H, W)`` at logit resolution.
        ignore: Ignore mask ``(B, 1, H, W)`` at logit resolution.
        box_prompt: Bounding boxes ``(B, 4)`` in image-space.
        logits: Logits ``(B, 1, H, W)``.
        loss_cfg: Loss configuration.
        image_size: Original image dimension for bbox scaling.
    """
    logit_h, logit_w = logits.shape[2:]
    valid = (ignore == 0).float()
    logger.info("DIAG — target sum: %.0f", target.sum().item())
    logger.info("DIAG — ignore sum: %.0f", ignore.sum().item())
    logger.info("DIAG — valid full sum: %.0f", valid.sum().item())
    logger.info("DIAG — bbox (image-space): %s", box_prompt.squeeze().tolist())

    use_crop = loss_cfg.get("use_bbox_loss_crop", False)
    if use_crop:
        margin = loss_cfg.get("bbox_loss_crop_margin", 32)
        from archeo_topia.training.mapsam_losses import build_bbox_valid_mask

        bbox_scaled = _scale_bbox_to_logits(box_prompt, image_size, logit_h)
        logger.info("DIAG — bbox (logit-space): %s", bbox_scaled.squeeze().tolist())
        bbox_mask = build_bbox_valid_mask(
            bbox_scaled,
            mask_shape=(logit_h, logit_w),
            margin=margin,
            device=logits.device,
        )
        logger.info("DIAG — bbox valid sum: %.0f", bbox_mask.sum().item())

        final_valid = (valid.bool() & bbox_mask).float()
        logger.info("DIAG — final valid sum: %.0f", final_valid.sum().item())

        target_pos_in_valid = ((target > 0.5) & final_valid.bool()).sum().item()
        logger.info("DIAG — target positives in final valid: %.0f", target_pos_in_valid)

        if final_valid.sum().item() == 0:
            logger.error(
                "DIAG — ZERO valid pixels after bbox crop! "
                "Loss will be 0. Check bbox coordinates and crop logic."
            )
        if target_pos_in_valid == 0:
            logger.error(
                "DIAG — ZERO target positives in valid region! "
                "Bbox crop may not cover the target. Check coordinate scaling."
            )
    else:
        logger.info("DIAG — bbox crop disabled, using full valid region")


def _get_sample_ids(batch: dict[str, Any]) -> list[str]:
    """Extract sample IDs from a batch.

    Args:
        batch: Batch dictionary from the data loader.

    Returns:
        List of sample ID strings.
    """
    sid = batch.get("sample_id", [])
    if isinstance(sid, str):
        return [sid]
    if isinstance(sid, torch.Tensor):
        return sid.tolist()
    if isinstance(sid, list):
        return sid
    return [f"unknown_{i}" for i in range(batch.get("target_mask", torch.tensor([0])).shape[0])]


def _forward_and_get_logits(
    sam: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    use_cached: bool,
) -> torch.Tensor:
    """Run forward pass and return logits.

    Args:
        sam: SAM model.
        batch: Batch dictionary.
        device: Target device.
        use_cached: Whether to use cached embeddings.

    Returns:
        Logits tensor ``(B, 1, 256, 256)``.
    """
    if use_cached:
        return forward_sam_cached(
            sam,
            batch["image_embedding"],
            batch["box_prompt"],
            batch["point_prompt"],
            batch["point_label"],
            device,
        )
    return forward_sam(
        sam,
        batch["image"],
        batch["box_prompt"],
        batch["point_prompt"],
        batch["point_label"],
        device,
    )


def _resize_target_and_ignore(
    target: torch.Tensor,
    ignore: torch.Tensor,
    logit_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize target and ignore masks to logit spatial size.

    Args:
        target: Target mask ``(B, 1, H, W)``.
        ignore: Ignore mask ``(B, 1, H, W)``.
        logit_size: ``(height, width)`` of the logits.

    Returns:
        Resized ``(target, ignore)`` tuple.
    """
    target_r = torch.nn.functional.interpolate(target, size=logit_size, mode="nearest")
    ignore_r = torch.nn.functional.interpolate(ignore, size=logit_size, mode="nearest")
    return target_r, ignore_r


def _scale_bbox_to_logits(
    box_prompt: torch.Tensor,
    image_size: int,
    logit_size: int,
) -> torch.Tensor:
    """Scale bbox from image-space coordinates to logit-space.

    Args:
        box_prompt: Bounding boxes ``(B, 4)`` in image-space.
        image_size: Original image dimension (e.g., 1024).
        logit_size: Logit spatial dimension (e.g., 256).

    Returns:
        Scaled bounding boxes in logit-space.
    """
    scale = logit_size / image_size
    return box_prompt * scale


def _compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore: torch.Tensor,
    box_prompt: torch.Tensor,
    loss_cfg: dict[str, Any],
    image_size: int = 1024,
    raise_on_zero_valid: bool = False,
) -> dict[str, Any]:
    """Compute the combined loss for a batch.

    Args:
        logits: Logits ``(B, 1, H, W)``.
        target: Target mask ``(B, 1, H, W)``.
        ignore: Ignore mask ``(B, 1, H, W)``.
        box_prompt: Bounding boxes ``(B, 4)`` in image-space coords.
        loss_cfg: Loss configuration.
        image_size: Original image dimension for bbox scaling.
        raise_on_zero_valid: If True, raise on zero valid pixels.

    Returns:
        Dictionary with ``loss``, ``bce_loss``, ``dice_loss``.
    """
    logit_h, logit_w = logits.shape[2:]
    use_crop = loss_cfg.get("use_bbox_loss_crop", False)

    bbox_for_loss = box_prompt
    if use_crop:
        bbox_for_loss = _scale_bbox_to_logits(box_prompt, image_size, logit_h)

    return combined_bce_dice_loss(
        logits,
        target,
        ignore,
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        eps=loss_cfg.get("eps", 1e-6),
        foreground_bce_pos_weight=loss_cfg.get("foreground_bce_pos_weight"),
        bbox_xyxy=bbox_for_loss if use_crop else None,
        bbox_loss_crop_margin=loss_cfg.get("bbox_loss_crop_margin"),
        use_bbox_loss_crop=use_crop,
        raise_on_zero_valid=raise_on_zero_valid,
    )


def train_epoch(
    sam: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: dict[str, Any],
    max_batches: int | None = None,
    use_cached: bool = False,
    log_stats: bool = False,
    image_size: int = 1024,
    raise_on_zero_valid: bool = False,
) -> dict[str, Any]:
    """Run one training epoch.

    Args:
        sam: SAM model.
        loader: Training data loader.
        optimizer: Optimizer.
        device: Target device.
        loss_cfg: Loss configuration sub-dictionary.
        max_batches: Maximum batches per epoch. ``None`` for full epoch.
        use_cached: If True, batch contains ``image_embedding`` instead of
            ``image`` and ``forward_sam_cached`` is used.
        log_stats: If True, collect and return prediction statistics.
        image_size: Original image dimension for bbox scaling.
        raise_on_zero_valid: If True, raise on zero valid pixels.

    Returns:
        Dictionary with epoch metrics and optional ``prediction_stats``.
    """
    sam.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    n_batches = 0

    all_stats: list[dict[str, Any]] = []

    iterator = tqdm(loader, desc="Train", leave=False)
    for batch in iterator:
        logits = _forward_and_get_logits(sam, batch, device, use_cached)

        target = batch["target_mask"].to(device)
        ignore = batch["ignore_mask"].to(device)
        box_prompt = batch["box_prompt"].to(device)

        target_r, ignore_r = _resize_target_and_ignore(target, ignore, logits.shape[2:])

        if n_batches == 0:
            _log_first_batch_diagnostics(
                target_r, ignore_r, box_prompt, logits, loss_cfg, image_size
            )

        loss_result = _compute_loss(
            logits[:, 0:1],
            target_r,
            ignore_r,
            box_prompt,
            loss_cfg,
            image_size=image_size,
            raise_on_zero_valid=raise_on_zero_valid,
        )

        optimizer.zero_grad()
        loss_result["loss"].backward()
        optimizer.step()

        total_loss += loss_result["loss"].item()
        total_bce += loss_result["bce_loss"].item()
        total_dice += loss_result["dice_loss"].item()
        n_batches += 1

        iterator.set_postfix(loss=f"{loss_result['loss'].item():.4f}")

        if log_stats:
            sample_ids = _get_sample_ids(batch)
            stats = compute_prediction_stats(
                logits[:, 0:1], target_r, ignore_r, box_prompt, sample_ids
            )
            all_stats.extend(stats)

        if max_batches is not None and n_batches >= max_batches:
            break

    result: dict[str, Any] = {
        "loss": total_loss / max(n_batches, 1),
        "bce_loss": total_bce / max(n_batches, 1),
        "dice_loss": total_dice / max(n_batches, 1),
    }
    if log_stats and all_stats:
        result["prediction_stats"] = all_stats
        result["prediction_stats_aggregate"] = compute_stats_aggregates(all_stats)
        log_prediction_warnings(all_stats)

    return result


@torch.no_grad()
def validate(
    sam: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    max_batches: int | None = None,
    use_cached: bool = False,
    log_stats: bool = False,
    image_size: int = 1024,
) -> dict[str, Any]:
    """Run validation.

    Args:
        sam: SAM model.
        loader: Validation data loader.
        device: Target device.
        loss_cfg: Loss configuration sub-dictionary.
        max_batches: Maximum batches. ``None`` for full validation.
        use_cached: If True, batch contains ``image_embedding`` instead of
            ``image`` and ``forward_sam_cached`` is used.
        log_stats: If True, collect and return prediction statistics.
        image_size: Original image dimension for bbox scaling.

    Returns:
        Dictionary with validation metrics and optional ``prediction_stats``.
    """
    sam.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_dice_metric = 0.0
    n_batches = 0

    all_stats: list[dict[str, Any]] = []

    iterator = tqdm(loader, desc="Val", leave=False)
    for batch in iterator:
        logits = _forward_and_get_logits(sam, batch, device, use_cached)

        target = batch["target_mask"].to(device)
        ignore = batch["ignore_mask"].to(device)
        box_prompt = batch["box_prompt"].to(device)

        target_r, ignore_r = _resize_target_and_ignore(target, ignore, logits.shape[2:])

        loss_result = _compute_loss(
            logits[:, 0:1],
            target_r,
            ignore_r,
            box_prompt,
            loss_cfg,
            image_size=image_size,
        )

        metrics = compute_metrics(logits[:, 0:1], target_r, ignore_r)

        total_loss += loss_result["loss"].item()
        total_bce += loss_result["bce_loss"].item()
        total_dice += loss_result["dice_loss"].item()
        total_iou += metrics["mean_iou"]
        total_dice_metric += metrics["mean_dice"]
        n_batches += 1

        if log_stats:
            sample_ids = _get_sample_ids(batch)
            stats = compute_prediction_stats(
                logits[:, 0:1], target_r, ignore_r, box_prompt, sample_ids
            )
            all_stats.extend(stats)

        if max_batches is not None and n_batches >= max_batches:
            break

    result: dict[str, Any] = {
        "loss": total_loss / max(n_batches, 1),
        "bce_loss": total_bce / max(n_batches, 1),
        "dice_loss": total_dice / max(n_batches, 1),
        "mean_iou": total_iou / max(n_batches, 1),
        "mean_dice": total_dice_metric / max(n_batches, 1),
    }
    if log_stats and all_stats:
        result["prediction_stats"] = all_stats
        result["prediction_stats_aggregate"] = compute_stats_aggregates(all_stats)
        log_prediction_warnings(all_stats)

    return result


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    sam: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_type: str,
    sam_checkpoint: str,
    train_loss: float,
    val_loss: float,
    config: dict[str, Any],
) -> None:
    """Save a training checkpoint.

    Args:
        path: Destination file path.
        sam: SAM model.
        optimizer: Optimizer.
        epoch: Current epoch number (1-indexed).
        model_type: SAM model type identifier.
        sam_checkpoint: Original SAM checkpoint path.
        train_loss: Latest training loss.
        val_loss: Latest validation loss.
        config: Resolved configuration dictionary.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_type": model_type,
            "sam_checkpoint": sam_checkpoint,
            "model_state_dict": sam.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
        },
        str(path),
    )


def load_checkpoint(
    path: str | Path,
    sam: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        sam: SAM model to restore weights into.
        optimizer: Optimizer to restore state into. May be ``None``
            when loading for inference only.

    Returns:
        Checkpoint dictionary.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sam.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="MapSAM v0.1 decoder-only training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--use-cached-embeddings",
        action="store_true",
        help="Use pre-computed image embeddings from cache",
    )
    parser.add_argument(
        "--overfit-mode",
        action="store_true",
        default=None,
        help="Override config: enable overfit-test mode",
    )
    parser.add_argument(
        "--overfit-count",
        type=int,
        default=None,
        help="Override config: number of samples for overfit test",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for training."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = resolve_config(load_config(args.config))

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    outputs_cfg = cfg["outputs"]
    debug_cfg = cfg["debug"]

    if args.overfit_mode is not None:
        debug_cfg["overfit_mode"] = args.overfit_mode
    if args.overfit_count is not None:
        debug_cfg["overfit_sample_count"] = args.overfit_count
    cfg["debug"] = debug_cfg

    overfit_mode = debug_cfg["overfit_mode"]
    overfit_count = debug_cfg["overfit_sample_count"]
    overfit_repeat = debug_cfg["overfit_repeat_factor"]
    log_stats = debug_cfg["log_prediction_stats"]

    dataset_root = Path(dataset_cfg["root"])
    if not dataset_root.exists():
        logger.error("Dataset root not found: %s", dataset_root)
        sys.exit(1)

    samples_path = Path(dataset_cfg["samples"])
    if not samples_path.exists():
        logger.error("Samples file not found: %s", samples_path)
        sys.exit(1)

    sam_ckpt = Path(model_cfg["sam_checkpoint"])
    if not sam_ckpt.exists():
        logger.error("SAM checkpoint not found: %s", sam_ckpt)
        sys.exit(1)

    device_str = training_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but not available")
        sys.exit(1)
    device = torch.device(device_str)

    logger.info("Device: %s", device)
    logger.info("Model type: %s", model_cfg["model_type"])
    logger.info("SAM checkpoint: %s", sam_ckpt)

    set_seed(training_cfg["seed"])

    output_root = Path(outputs_cfg["root"])
    checkpoints_dir = output_root / "checkpoints"
    debug_dir = output_root / "debug_predictions"
    logs_dir = output_root / "logs"
    for d in [checkpoints_dir, debug_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(output_root / "config_resolved.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    if overfit_mode:
        logger.warning("=" * 60)
        logger.warning("OVERFIT-TEST MODE ENABLED")
        logger.warning("  Samples: %d, Repeats: %d", overfit_count, overfit_repeat)
        logger.warning("=" * 60)

    use_cached = args.use_cached_embeddings or model_cfg.get("use_cached_embeddings", False)
    if use_cached:
        logger.info("Using cached image embeddings")
        train_ds = MapSamEmbeddingDataset(
            dataset_root=str(dataset_root),
            samples_path=str(samples_path),
            split="train",
            image_size=dataset_cfg["image_size"],
            model_type=model_cfg["model_type"],
        )
        val_split = dataset_cfg.get("val_split", "val")
        val_ds = MapSamEmbeddingDataset(
            dataset_root=str(dataset_root),
            samples_path=str(samples_path),
            split=val_split,
            image_size=dataset_cfg["image_size"],
            model_type=model_cfg["model_type"],
        )
    else:
        train_ds = MapSamDataset(
            dataset_root=str(dataset_root),
            samples_path=str(samples_path),
            split="train",
            image_size=dataset_cfg["image_size"],
        )
        val_split = dataset_cfg.get("val_split", "val")
        val_ds = MapSamDataset(
            dataset_root=str(dataset_root),
            samples_path=str(samples_path),
            split=val_split,
            image_size=dataset_cfg["image_size"],
        )

    if overfit_mode:
        train_ds = OverfitDataset(train_ds, overfit_count, overfit_repeat)
        val_ds = OverfitDataset(val_ds, overfit_count, overfit_repeat)
        logger.info("Overfit train samples: %d (effective)", len(train_ds))
        logger.info("Overfit val samples: %d (effective)", len(val_ds))
    else:
        logger.info("Train samples: %d", len(train_ds))
        logger.info("Val samples: %d", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=not overfit_mode,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=device_str == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=device_str == "cuda",
    )

    sam = build_sam_model(cfg)
    sam.to(device)

    trainable_params = [p for p in sam.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    metrics_history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_iou: float | None = None

    epochs = training_cfg["epochs"]
    max_train = training_cfg.get("max_train_batches")
    max_val = training_cfg.get("max_val_batches")
    save_every = training_cfg.get("save_every_epochs", 5)
    val_every = training_cfg.get("validate_every_epochs", 1)

    logger.info(
        "Loss config: bce_w=%.2f dice_w=%.2f fg_pos_w=%s bbox_crop=%s margin=%s",
        loss_cfg.get("bce_weight", 1.0),
        loss_cfg.get("dice_weight", 1.0),
        loss_cfg.get("foreground_bce_pos_weight"),
        loss_cfg.get("use_bbox_loss_crop", False),
        loss_cfg.get("bbox_loss_crop_margin"),
    )

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)

        log_train_stats = log_stats and (epoch % 10 == 0 or epoch == epochs)
        train_metrics = train_epoch(
            sam,
            train_loader,
            optimizer,
            device,
            loss_cfg,
            max_batches=max_train,
            use_cached=use_cached,
            log_stats=log_train_stats,
            image_size=dataset_cfg["image_size"],
            raise_on_zero_valid=overfit_mode,
        )
        logger.info(
            "Epoch %d train — loss: %.4f  bce: %.4f  dice: %.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["bce_loss"],
            train_metrics["dice_loss"],
        )

        if "prediction_stats_aggregate" in train_metrics:
            agg = train_metrics["prediction_stats_aggregate"]
            logger.info(
                "  train stats — iou: %.4f  dice: %.4f  "
                "pred_pos_px: %.0f  gt_pos_px: %.0f  "
                "pred_ratio: %.4f  gt_ratio: %.4f",
                agg["mean_iou"],
                agg["mean_dice"],
                agg["mean_pred_positive_pixels"],
                agg["mean_gt_positive_pixels"],
                agg["mean_prediction_area_ratio"],
                agg["mean_target_area_ratio"],
            )

        val_metrics: dict[str, Any] | None = None
        if epoch % val_every == 0:
            log_val_stats = log_stats or overfit_mode
            val_metrics = validate(
                sam,
                val_loader,
                device,
                loss_cfg,
                max_batches=max_val,
                use_cached=use_cached,
                log_stats=log_val_stats,
                image_size=dataset_cfg["image_size"],
            )
            logger.info(
                "Epoch %d val — loss: %.4f  bce: %.4f  dice: %.4f  iou: %.4f  dice_m: %.4f",
                epoch,
                val_metrics["loss"],
                val_metrics["bce_loss"],
                val_metrics["dice_loss"],
                val_metrics["mean_iou"],
                val_metrics["mean_dice"],
            )

            if "prediction_stats_aggregate" in val_metrics:
                agg = val_metrics["prediction_stats_aggregate"]
                logger.info(
                    "  val stats — iou: %.4f  dice: %.4f  "
                    "pred_pos_px: %.0f  gt_pos_px: %.0f  "
                    "pred_ratio: %.4f  gt_ratio: %.4f",
                    agg["mean_iou"],
                    agg["mean_dice"],
                    agg["mean_pred_positive_pixels"],
                    agg["mean_gt_positive_pixels"],
                    agg["mean_prediction_area_ratio"],
                    agg["mean_target_area_ratio"],
                )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_iou = val_metrics.get("mean_iou")
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    sam,
                    optimizer,
                    epoch,
                    model_cfg["model_type"],
                    str(sam_ckpt),
                    train_metrics["loss"],
                    val_metrics["loss"],
                    cfg,
                )
                logger.info(
                    "Saved best checkpoint (val_loss=%.4f, val_iou=%.4f)",
                    best_val_loss,
                    best_val_iou if best_val_iou else 0,
                )

        if epoch % save_every == 0:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch}.pt",
                sam,
                optimizer,
                epoch,
                model_cfg["model_type"],
                str(sam_ckpt),
                train_metrics["loss"],
                val_metrics["loss"] if val_metrics else float("inf"),
                cfg,
            )

        record = {
            "epoch": epoch,
            "train": {k: v for k, v in train_metrics.items() if k != "prediction_stats"},
            "val": val_metrics,
        }
        metrics_history.append(record)

        if log_stats and "prediction_stats" in train_metrics:
            stats_path = output_root / f"prediction_stats_train_e{epoch}.jsonl"
            with open(stats_path, "w") as f:
                for entry in train_metrics["prediction_stats"]:
                    f.write(json.dumps(entry) + "\n")

        if log_stats and val_metrics and "prediction_stats" in val_metrics:
            stats_path = output_root / f"prediction_stats_val_e{epoch}.jsonl"
            with open(stats_path, "w") as f:
                for entry in val_metrics["prediction_stats"]:
                    f.write(json.dumps(entry) + "\n")

    save_checkpoint(
        checkpoints_dir / "final.pt",
        sam,
        optimizer,
        epochs,
        model_cfg["model_type"],
        str(sam_ckpt),
        train_metrics["loss"],
        val_metrics["loss"] if val_metrics else float("inf"),
        cfg,
    )

    final_record = {
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "best_val_iou": best_val_iou,
        "final_train_loss": train_metrics["loss"],
        "final_val_loss": val_metrics["loss"] if val_metrics else None,
        "final_val_iou": val_metrics["mean_iou"] if val_metrics else None,
        "final_val_dice": val_metrics["mean_dice"] if val_metrics else None,
        "overfit_mode": overfit_mode,
        "history": metrics_history,
    }
    with open(output_root / "metrics.json", "w") as f:
        json.dump(final_record, f, indent=2)

    logger.info("Training complete. Outputs in %s", output_root)
    logger.info("Best val loss: %.4f", best_val_loss)
    if val_metrics:
        logger.info(
            "Final val IoU: %.4f  Dice: %.4f", val_metrics["mean_iou"], val_metrics["mean_dice"]
        )


if __name__ == "__main__":
    main()
