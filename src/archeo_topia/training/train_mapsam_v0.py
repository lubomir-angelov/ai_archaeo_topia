#!/usr/bin/env python3
"""MapSAM v0.1 — frozen image encoder, train mask decoder only.

Fine-tunes only the SAM mask decoder on archaeological mound-symbol
prompts.  Image encoder and prompt encoder remain frozen.

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
from torch.utils.data import DataLoader
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
    result["loss"] = loss_cfg

    outputs_cfg = result.get("outputs", {})
    outputs_cfg.setdefault("save_debug_predictions", True)
    outputs_cfg.setdefault("debug_prediction_count", 8)
    result["outputs"] = outputs_cfg

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
# Metrics
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    sam: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_cfg: dict[str, Any],
    max_batches: int | None = None,
    use_cached: bool = False,
) -> dict[str, float]:
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

    Returns:
        Dictionary with epoch metrics.
    """
    sam.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    n_batches = 0

    iterator = tqdm(loader, desc="Train", leave=False)
    for batch in iterator:
        if use_cached:
            logits = forward_sam_cached(
                sam,
                batch["image_embedding"],
                batch["box_prompt"],
                batch["point_prompt"],
                batch["point_label"],
                device,
            )
        else:
            logits = forward_sam(
                sam,
                batch["image"],
                batch["box_prompt"],
                batch["point_prompt"],
                batch["point_label"],
                device,
            )

        target = batch["target_mask"].to(device)
        ignore = batch["ignore_mask"].to(device)

        target_resized = torch.nn.functional.interpolate(
            target,
            size=logits.shape[2:],
            mode="nearest",
        )
        ignore_resized = torch.nn.functional.interpolate(
            ignore,
            size=logits.shape[2:],
            mode="nearest",
        )

        bce_loss = combined_bce_dice_loss(
            logits[:, 0:1],
            target_resized,
            ignore_resized,
            bce_weight=1.0,
            dice_weight=0.0,
            eps=loss_cfg.get("eps", 1e-6),
        )
        dice_l = combined_bce_dice_loss(
            logits[:, 0:1],
            target_resized,
            ignore_resized,
            bce_weight=0.0,
            dice_weight=1.0,
            eps=loss_cfg.get("eps", 1e-6),
        )
        loss = combined_bce_dice_loss(
            logits[:, 0:1],
            target_resized,
            ignore_resized,
            bce_weight=loss_cfg.get("bce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            eps=loss_cfg.get("eps", 1e-6),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_dice += dice_l.item()
        n_batches += 1

        iterator.set_postfix(loss=f"{loss.item():.4f}")

        if max_batches is not None and n_batches >= max_batches:
            break

    return {
        "loss": total_loss / max(n_batches, 1),
        "bce_loss": total_bce / max(n_batches, 1),
        "dice_loss": total_dice / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    sam: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
    max_batches: int | None = None,
    use_cached: bool = False,
) -> dict[str, float]:
    """Run validation.

    Args:
        sam: SAM model.
        loader: Validation data loader.
        device: Target device.
        loss_cfg: Loss configuration sub-dictionary.
        max_batches: Maximum batches. ``None`` for full validation.
        use_cached: If True, batch contains ``image_embedding`` instead of
            ``image`` and ``forward_sam_cached`` is used.

    Returns:
        Dictionary with validation metrics.
    """
    sam.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_dice_metric = 0.0
    n_batches = 0

    iterator = tqdm(loader, desc="Val", leave=False)
    for batch in iterator:
        if use_cached:
            logits = forward_sam_cached(
                sam,
                batch["image_embedding"],
                batch["box_prompt"],
                batch["point_prompt"],
                batch["point_label"],
                device,
            )
        else:
            logits = forward_sam(
                sam,
                batch["image"],
                batch["box_prompt"],
                batch["point_prompt"],
                batch["point_label"],
                device,
            )

        target = batch["target_mask"].to(device)
        ignore = batch["ignore_mask"].to(device)

        target_resized = torch.nn.functional.interpolate(
            target,
            size=logits.shape[2:],
            mode="nearest",
        )
        ignore_resized = torch.nn.functional.interpolate(
            ignore,
            size=logits.shape[2:],
            mode="nearest",
        )

        bce_l = combined_bce_dice_loss(
            logits[:, 0:1],
            target_resized,
            ignore_resized,
            bce_weight=1.0,
            dice_weight=0.0,
            eps=loss_cfg.get("eps", 1e-6),
        )
        dice_l = combined_bce_dice_loss(
            logits[:, 0:1],
            target_resized,
            ignore_resized,
            bce_weight=0.0,
            dice_weight=1.0,
            eps=loss_cfg.get("eps", 1e-6),
        )
        loss = combined_bce_dice_loss(
            logits[:, 0:1],
            target_resized,
            ignore_resized,
            bce_weight=loss_cfg.get("bce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            eps=loss_cfg.get("eps", 1e-6),
        )

        metrics = compute_metrics(logits[:, 0:1], target_resized, ignore_resized)

        total_loss += loss.item()
        total_bce += bce_l.item()
        total_dice += dice_l.item()
        total_iou += metrics["mean_iou"]
        total_dice_metric += metrics["mean_dice"]
        n_batches += 1

        if max_batches is not None and n_batches >= max_batches:
            break

    return {
        "loss": total_loss / max(n_batches, 1),
        "bce_loss": total_bce / max(n_batches, 1),
        "dice_loss": total_dice / max(n_batches, 1),
        "mean_iou": total_iou / max(n_batches, 1),
        "mean_dice": total_dice_metric / max(n_batches, 1),
    }


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

    use_cached = args.use_cached_embeddings
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

    logger.info("Train samples: %d", len(train_ds))
    logger.info("Val samples: %d", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
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

    epochs = training_cfg["epochs"]
    max_train = training_cfg.get("max_train_batches")
    max_val = training_cfg.get("max_val_batches")
    save_every = training_cfg.get("save_every_epochs", 5)
    val_every = training_cfg.get("validate_every_epochs", 1)

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)

        train_metrics = train_epoch(
            sam,
            train_loader,
            optimizer,
            device,
            loss_cfg,
            max_batches=max_train,
            use_cached=use_cached,
        )
        logger.info(
            "Epoch %d train — loss: %.4f  bce: %.4f  dice: %.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["bce_loss"],
            train_metrics["dice_loss"],
        )

        val_metrics: dict[str, float] | None = None
        if epoch % val_every == 0:
            val_metrics = validate(
                sam,
                val_loader,
                device,
                loss_cfg,
                max_batches=max_val,
                use_cached=use_cached,
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

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
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
                logger.info("Saved best checkpoint (val_loss=%.4f)", best_val_loss)

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
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics_history.append(record)

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

    with open(output_root / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.info("Training complete. Outputs in %s", output_root)


if __name__ == "__main__":
    main()
