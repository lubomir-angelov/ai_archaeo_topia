#!/usr/bin/env python3
"""PyTorch Dataset for MapSAM fine-tuning.

Reads ``training_samples.jsonl`` produced by the prompt-generation step
and returns SAM-ready training samples: resized RGB image, binary target
mask, binary ignore mask, bounding-box prompt, and center-point prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_VALID_SPLITS = {"train", "val", "test"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of row dictionaries.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Samples file not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rgb_image(path: str | Path) -> torch.Tensor:
    """Load an image as a float32 RGB tensor in [0, 1].

    Args:
        path: Path to the image file.

    Returns:
        Tensor of shape ``(3, H, W)`` with values in ``[0.0, 1.0]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path).convert("RGB")
    arr = torch.tensor(np_array(img), dtype=torch.float32) / 255.0
    return arr.permute(2, 0, 1)


def load_binary_mask(path: str | Path) -> torch.Tensor:
    """Load a binary mask as a float32 tensor with values 0.0 or 1.0.

    Any pixel value > 0 is treated as foreground (1.0).

    Args:
        path: Path to the mask file.

    Returns:
        Tensor of shape ``(1, H, W)`` with binary values.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    img = Image.open(path).convert("L")
    arr = torch.tensor(np_array(img), dtype=torch.float32)
    return (arr > 0).float().unsqueeze(0)


def np_array(img: Image.Image) -> Any:
    """Convert a PIL image to a numpy array without importing numpy eagerly.

    Args:
        img: PIL Image instance.

    Returns:
        Numpy array representation of the image.
    """
    import numpy as np

    return np.array(img)


def resize_image_and_masks(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize image and masks to a square of *size* x *size*.

    Uses bilinear interpolation for the image and nearest-neighbor for
    masks to preserve binary values.

    Args:
        image: Tensor of shape ``(3, H, W)``.
        target_mask: Tensor of shape ``(1, H, W)``.
        ignore_mask: Tensor of shape ``(1, H, W)``.
        size: Target square dimension.

    Returns:
        Tuple of ``(resized_image, resized_target, resized_ignore)``
        each with spatial dimensions ``(size, size)``.
    """
    image_resized = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    target_resized = torch.nn.functional.interpolate(
        target_mask.unsqueeze(0),
        size=(size, size),
        mode="nearest",
    ).squeeze(0)

    ignore_resized = torch.nn.functional.interpolate(
        ignore_mask.unsqueeze(0),
        size=(size, size),
        mode="nearest",
    ).squeeze(0)

    return image_resized, target_resized, ignore_resized


def scale_bbox(
    bbox: list[int],
    orig_h: int,
    orig_w: int,
    new_size: int,
) -> torch.Tensor:
    """Scale a bounding box from original image coordinates to resized space.

    Args:
        bbox: Original bounding box as ``[x_min, y_min, x_max, y_max]``.
        orig_h: Original image height.
        orig_w: Original image width.
        new_size: Target square dimension.

    Returns:
        Tensor of shape ``(4,)`` with scaled coordinates.
    """
    scale_x = new_size / orig_w
    scale_y = new_size / orig_h

    x_min, y_min, x_max, y_max = bbox
    return torch.tensor(
        [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y],
        dtype=torch.float32,
    )


def scale_point(
    point: list[int],
    orig_h: int,
    orig_w: int,
    new_size: int,
) -> torch.Tensor:
    """Scale a point from original image coordinates to resized space.

    Args:
        point: Original point as ``[x, y]``.
        orig_h: Original image height.
        orig_w: Original image width.
        new_size: Target square dimension.

    Returns:
        Tensor of shape ``(2,)`` with scaled coordinates.
    """
    scale_x = new_size / orig_w
    scale_y = new_size / orig_h

    x, y = point
    return torch.tensor([x * scale_x, y * scale_y], dtype=torch.float32)


def validate_sample_row(row: dict[str, Any]) -> None:
    """Validate that a JSONL row has the required fields and structure.

    Args:
        row: Parsed JSONL row dictionary.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    required = ["sample_id", "split", "image_path", "mask_path", "bbox", "center_point"]
    for key in required:
        if key not in row:
            raise ValueError(f"Missing required field '{key}' in sample row")

    bbox = row["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Malformed bbox (expected [x_min, y_min, x_max, y_max]): {bbox}")

    point = row["center_point"]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"Malformed center_point (expected [x, y]): {point}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MapSamDataset(Dataset):
    """PyTorch Dataset for MapSAM fine-tuning.

    Loads training samples from a JSONL manifest and returns SAM-ready
    tensors: resized RGB image, binary target mask, binary ignore mask,
    bounding-box prompt, and center-point prompt.

    Args:
        dataset_root: Root directory containing ``images/``, ``masks/``,
            and ``ignore_masks/`` subdirectories.
        samples_path: Path to the ``training_samples.jsonl`` manifest.
        split: Dataset split to load (``train``, ``val``, or ``test``).
        image_size: Target square dimension for resizing. Defaults to 1024.
        return_original_size: Include original ``(H, W)`` in returned dict.
            Defaults to ``True``.

    Raises:
        FileNotFoundError: If *dataset_root* or *samples_path* does not exist.
        ValueError: If *split* is invalid or no samples match the split.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        samples_path: str | Path,
        split: str,
        image_size: int = 1024,
        return_original_size: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples_path = Path(samples_path)
        self.split = split
        self.image_size = image_size
        self.return_original_size = return_original_size

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        if not self.samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {self.samples_path}")

        if split not in _VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {sorted(_VALID_SPLITS)}")

        all_rows = load_jsonl(self.samples_path)
        self._samples: list[dict[str, Any]] = [
            row for row in all_rows if row.get("split") == split
        ]

        if not self._samples:
            raise ValueError(f"No samples found for split '{split}'")

        logger.info(
            "MapSamDataset: loaded %d samples for split '%s' from %d total rows",
            len(self._samples),
            split,
            len(all_rows),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._samples[idx]
        validate_sample_row(row)

        image_path = self.dataset_root / row["image_path"]
        mask_path = self.dataset_root / row["mask_path"]
        ignore_mask_path = self.dataset_root / row.get("ignore_mask_path", row["mask_path"])

        image = load_rgb_image(image_path)
        target_mask = load_binary_mask(mask_path)
        ignore_mask = load_binary_mask(ignore_mask_path)

        orig_h = image.shape[1]
        orig_w = image.shape[2]

        if target_mask.shape[1:] != (orig_h, orig_w):
            raise ValueError(
                f"Target mask size {target_mask.shape[1:]} does not match "
                f"image size ({orig_h}, {orig_w}) for {row['sample_id']}"
            )

        if ignore_mask.shape[1:] != (orig_h, orig_w):
            raise ValueError(
                f"Ignore mask size {ignore_mask.shape[1:]} does not match "
                f"image size ({orig_h}, {orig_w}) for {row['sample_id']}"
            )

        image_r, target_r, ignore_r = resize_image_and_masks(
            image, target_mask, ignore_mask, self.image_size
        )

        box_prompt = scale_bbox(row["bbox"], orig_h, orig_w, self.image_size)
        point_prompt = scale_point(row["center_point"], orig_h, orig_w, self.image_size)
        point_label = torch.tensor([1], dtype=torch.int64)

        result: dict[str, Any] = {
            "image": image_r,
            "target_mask": target_r,
            "ignore_mask": ignore_r,
            "box_prompt": box_prompt,
            "point_prompt": point_prompt,
            "point_label": point_label,
            "sample_id": row["sample_id"],
            "image_path": str(row["image_path"]),
        }

        if self.return_original_size:
            result["original_size"] = (orig_h, orig_w)
            result["resized_size"] = (self.image_size, self.image_size)

        return result
