#!/usr/bin/env python3
"""PyTorch Dataset that loads cached SAM image embeddings.

Pairs each cached embedding with the per-instance prompts and masks
from ``training_samples.jsonl``.  Returns the embedding tensor instead
 of the raw image, so the training loop skips the frozen image encoder.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from archeo_topia.datasets.mapsam_dataset import (
    _VALID_SPLITS,
    label_mask_file,
    load_binary_mask_cached,
    load_jsonl,
    scale_bbox,
    scale_point,
    select_instance_mask,
    select_samples,
    validate_sample_row,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _load_embedding_cached(cache_file: str) -> dict[str, Any]:
    """Load a cached SAM embedding payload, memoised by path.

    Each sheet embedding is shared by every sample from that sheet, so the
    uncached load re-read the same multi-megabyte tensor once per sample.

    Args:
        cache_file: Path to the ``.pt`` embedding payload.

    Returns:
        The payload dict.  Shared across callers, so treat it as read-only.
    """
    return torch.load(cache_file, map_location="cpu", weights_only=False)


def _resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour resize a ``(1, H, W)`` mask to ``(1, size, size)``.

    Args:
        mask: Mask tensor of shape ``(1, H, W)``.
        size: Target square dimension.

    Returns:
        Resized mask tensor of shape ``(1, size, size)``.
    """
    return torch.nn.functional.interpolate(
        mask.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0)


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
        sheets: Sheet IDs to select instead of filtering on *split*.
        subsample: Cap on the number of samples kept, for size-matched folds.
        seed: Seed for the *subsample* draw.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        samples_path: str | Path,
        split: str,
        image_size: int = 1024,
        model_type: str = "vit_b",
        return_original_size: bool = True,
        sheets: list[str] | None = None,
        subsample: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples_path = Path(samples_path)
        self.split = split
        self.image_size = image_size
        self.model_type = model_type
        self.return_original_size = return_original_size

        if split not in _VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {sorted(_VALID_SPLITS)}")

        all_rows = load_jsonl(self.samples_path)
        self.sheets = sheets
        self._samples: list[dict[str, Any]] = select_samples(
            all_rows, split, sheets=sheets, subsample=subsample, seed=seed
        )

        logger.info(
            "MapSamEmbeddingDataset: loaded %d samples (embeddings from %s)",
            len(self._samples),
            self.dataset_root / "sam_embeddings" / model_type,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._samples[idx]
        validate_sample_row(row)

        image_rel = row["image_path"]
        stem = Path(image_rel).stem
        # Keyed on the row's own split, not self.split: a leave-one-sheet-out
        # fold draws rows from whichever split directory the sheet was
        # originally exported into.
        row_split = row.get("split", self.split)
        cache_file = (
            self.dataset_root / "sam_embeddings" / self.model_type / row_split / f"{stem}.pt"
        )

        if not cache_file.exists():
            raise FileNotFoundError(f"Cached embedding not found for {image_rel}: {cache_file}")

        payload = _load_embedding_cached(str(cache_file))
        image_embedding = payload["image_embedding"]
        original_size = payload["original_size"]
        orig_h, orig_w = original_size

        mask_path = self.dataset_root / row["mask_path"]
        ignore_mask_path = self.dataset_root / row.get("ignore_mask_path", row["mask_path"])

        ignore_mask = load_binary_mask_cached(str(ignore_mask_path))

        # Select the prompted instance at ORIGINAL resolution, before any
        # resize, for the same reason as in MapSamDataset: downsampling a
        # ~459 px mound symbol before labelling can fragment or erase it.
        # Foreground is derived from the cached label array, so the sheet
        # mask itself never has to be decoded here.
        labeled_components = label_mask_file(str(mask_path))
        instance_mask_np = select_instance_mask(
            semantic_mask=labeled_components[0] > 0,
            center_point_xy=(
                float(row["center_point"][0]),
                float(row["center_point"][1]),
            ),
            bbox_xyxy=(
                float(row["bbox"][0]),
                float(row["bbox"][1]),
                float(row["bbox"][2]),
                float(row["bbox"][3]),
            ),
            sample_id=row["sample_id"],
            labeled_components=labeled_components,
        )
        instance_mask = torch.from_numpy(instance_mask_np).to(torch.float32).unsqueeze(0)

        target_r = _resize_mask(instance_mask, self.image_size)
        ignore_r = _resize_mask(ignore_mask, self.image_size)

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
            "sheet_id": str(row.get("sheet_id", "")),
            "image_path": str(row["image_path"]),
        }

        if self.return_original_size:
            result["original_size"] = original_size
            result["resized_size"] = (self.image_size, self.image_size)

        return result
