#!/usr/bin/env python3
"""PyTorch Dataset that loads cached SAM image embeddings.

Pairs each cached embedding with the per-instance prompts and masks
from ``training_samples.jsonl``.  Returns the embedding tensor instead
 of the raw image, so the training loop skips the frozen image encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from archeo_topia.datasets.mapsam_dataset import (
    _VALID_SPLITS,
    load_binary_mask,
    load_jsonl,
    resize_image_and_masks,
    scale_bbox,
    scale_point,
    validate_sample_row,
)

logger = logging.getLogger(__name__)


class MapSamEmbeddingDataset(Dataset):
    """Dataset that loads pre-computed SAM image embeddings.

    Args:
        dataset_root: Root directory containing ``images/``, ``masks/``,
            ``ignore_masks/``, and ``sam_embeddings/``.
        samples_path: Path to ``training_samples.jsonl``.
        split: Dataset split (``train``, ``val``, or ``test``).
        image_size: Target square dimension for resizing masks.
        model_type: SAM model type (determines cache subdirectory).
        return_original_size: Include original ``(H, W)`` in dict.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        samples_path: str | Path,
        split: str,
        image_size: int = 1024,
        model_type: str = "vit_b",
        return_original_size: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples_path = Path(samples_path)
        self.split = split
        self.image_size = image_size
        self.model_type = model_type
        self.return_original_size = return_original_size

        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {sorted(_VALID_SPLITS)}"
            )

        all_rows = load_jsonl(self.samples_path)
        self._samples: list[dict[str, Any]] = [
            row for row in all_rows if row.get("split") == split
        ]

        if not self._samples:
            raise ValueError(f"No samples found for split '{split}'")

        logger.info(
            "MapSamEmbeddingDataset: loaded %d samples for split '%s' "
            "(embeddings from %s/%s/%s)",
            len(self._samples),
            split,
            model_type,
            split,
            self.dataset_root / "sam_embeddings" / model_type / split,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._samples[idx]
        validate_sample_row(row)

        image_rel = row["image_path"]
        stem = Path(image_rel).stem
        cache_file = (
            self.dataset_root
            / "sam_embeddings"
            / self.model_type
            / self.split
            / f"{stem}.pt"
        )

        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cached embedding not found for {image_rel}: {cache_file}"
            )

        payload = torch.load(str(cache_file), map_location="cpu", weights_only=False)
        image_embedding = payload["image_embedding"]
        original_size = payload["original_size"]
        orig_h, orig_w = original_size

        mask_path = self.dataset_root / row["mask_path"]
        ignore_mask_path = self.dataset_root / row.get(
            "ignore_mask_path", row["mask_path"]
        )

        target_mask = load_binary_mask(mask_path)
        ignore_mask = load_binary_mask(ignore_mask_path)

        target_r, ignore_r = resize_image_and_masks(
            torch.zeros(3, orig_h, orig_w),
            target_mask,
            ignore_mask,
            self.image_size,
        )[1:]

        box_prompt = scale_bbox(row["bbox"], orig_h, orig_w, self.image_size)
        point_prompt = scale_point(row["center_point"], orig_h, orig_w, self.image_size)
        point_label = torch.tensor([1], dtype=torch.int64)

        result: dict[str, Any] = {
            "image_embedding": image_embedding,
            "target_mask": target_r,
            "ignore_mask": ignore_r,
            "box_prompt": box_prompt,
            "point_prompt": point_prompt,
            "point_label": point_label,
            "sample_id": row["sample_id"],
            "image_path": str(row["image_path"]),
        }

        if self.return_original_size:
            result["original_size"] = original_size
            result["resized_size"] = (self.image_size, self.image_size)

        return result
