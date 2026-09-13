#!/usr/bin/env python3
"""Tests for MapSAM instance-target mask selection.

Verifies that the dataset returns only the prompted mound instance,
not the full semantic mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from archeo_topia.datasets.mapsam_dataset import (
    MapSamDataset,
    _label_foreground,
    count_connected_components,
    label_mask_file,
    select_instance_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_multi_component_dataset(
    root: Path,
    image_w: int = 100,
    image_h: int = 100,
) -> None:
    """Build dataset with two separated mound components per image."""
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        (root / "ignore_masks" / split).mkdir(parents=True, exist_ok=True)

        # RGB image
        img = np.random.randint(0, 256, (image_h, image_w, 3), dtype=np.uint8)
        Image.fromarray(img, "RGB").save(root / "images" / split / "multi.png")

        # Mask with two components: A at top-left, B at bottom-right
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # Component A, center ~ (20, 20)
        mask[60:80, 60:80] = 255  # Component B, center ~ (70, 70)
        Image.fromarray(mask, "L").save(root / "masks" / split / "multi.png")

        # Empty ignore mask
        ign = np.zeros((image_h, image_w), dtype=np.uint8)
        Image.fromarray(ign, "L").save(root / "ignore_masks" / split / "multi.png")


# ---------------------------------------------------------------------------
# count_connected_components
# ---------------------------------------------------------------------------


class TestCountConnectedComponents:
    def test_zero_components(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert count_connected_components(mask) == 0

    def test_single_component(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255
        assert count_connected_components(mask) == 1

    def test_two_components(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[12:15, 12:15] = 255
        assert count_connected_components(mask) == 2

    def test_diagonal_connection_8connectivity(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3, 3] = 255
        mask[4, 4] = 255
        assert count_connected_components(mask, connectivity=8) == 1

    def test_diagonal_separate_4connectivity(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3, 3] = 255
        mask[4, 4] = 255
        assert count_connected_components(mask, connectivity=4) == 2


# ---------------------------------------------------------------------------
# select_instance_mask
# ---------------------------------------------------------------------------


class TestSelectInstanceMask:
    def _make_two_component_mask(self) -> np.ndarray:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # Component A, center ~(20, 20)
        mask[60:80, 60:80] = 255  # Component B, center ~(70, 70)
        return mask

    def test_point_in_component_a(self) -> None:
        mask = self._make_two_component_mask()
        result = select_instance_mask(mask, center_point_xy=(20, 20), bbox_xyxy=(10, 10, 30, 30))
        assert count_connected_components(result) == 1
        assert result[15, 15] == 1
        assert result[70, 70] == 0

    def test_point_in_component_b(self) -> None:
        mask = self._make_two_component_mask()
        result = select_instance_mask(mask, center_point_xy=(70, 70), bbox_xyxy=(60, 60, 80, 80))
        assert count_connected_components(result) == 1
        assert result[70, 70] == 1
        assert result[20, 20] == 0

    def test_point_on_background_fallback_to_bbox(self) -> None:
        mask = self._make_two_component_mask()
        # Point at (50, 50) is background, but bbox covers component B
        result = select_instance_mask(
            mask,
            center_point_xy=(50, 50),
            bbox_xyxy=(60, 60, 80, 80),
        )
        assert count_connected_components(result) == 1
        assert result[70, 70] == 1
        assert result[20, 20] == 0

    def test_point_on_background_fallback_selects_largest_intersection(self) -> None:
        mask = self._make_two_component_mask()
        # Point on background, bbox overlaps both components slightly
        # but overlaps component B more
        result = select_instance_mask(
            mask,
            center_point_xy=(50, 50),
            bbox_xyxy=(55, 55, 85, 85),
        )
        assert count_connected_components(result) == 1
        assert result[70, 70] == 1
        assert result[20, 20] == 0

    def test_other_components_removed(self) -> None:
        mask = self._make_two_component_mask()
        result = select_instance_mask(mask, center_point_xy=(20, 20), bbox_xyxy=(10, 10, 30, 30))
        # Component B must be fully removed
        assert result[60:80, 60:80].sum() == 0

    def test_single_component_unchanged(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        result = select_instance_mask(mask, center_point_xy=(25, 25), bbox_xyxy=(10, 10, 40, 40))
        np.testing.assert_array_equal(result, (mask > 0).astype(np.uint8))

    def test_empty_mask_raises(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="No foreground"):
            select_instance_mask(mask, center_point_xy=(5, 5), bbox_xyxy=(0, 0, 10, 10))

    def test_result_has_one_component(self) -> None:
        mask = self._make_two_component_mask()
        result = select_instance_mask(mask, center_point_xy=(20, 20), bbox_xyxy=(10, 10, 30, 30))
        assert count_connected_components(result) == 1

    def test_sample_id_in_error(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="test_sample_42"):
            select_instance_mask(
                mask,
                center_point_xy=(5, 5),
                bbox_xyxy=(0, 0, 10, 10),
                sample_id="test_sample_42",
            )

    def test_float_coordinates(self) -> None:
        mask = self._make_two_component_mask()
        result = select_instance_mask(
            mask,
            center_point_xy=(20.7, 20.3),
            bbox_xyxy=(10.1, 10.1, 30.9, 30.9),
        )
        assert count_connected_components(result) == 1
        assert result[15, 15] == 1
        assert result[70, 70] == 0

    def test_coordinate_clamping(self) -> None:
        mask = self._make_two_component_mask()
        result = select_instance_mask(
            mask,
            center_point_xy=(-5, -5),
            bbox_xyxy=(-10, -10, 35, 35),
        )
        assert count_connected_components(result) == 1

    def test_small_synthetic_mask(self) -> None:
        # 10x10 mask: component A at rows 1:3, cols 1:3; B at rows 6:8, cols 6:8
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        mask[6:8, 6:8] = 255

        # Select A with center point (1.5, 1.5)
        result_a = select_instance_mask(mask, center_point_xy=(1.5, 1.5), bbox_xyxy=(1, 1, 3, 3))
        assert result_a[1, 1] == 1
        assert result_a[6, 6] == 0

        # Select B with center point (6.5, 6.5)
        result_b = select_instance_mask(mask, center_point_xy=(6.5, 6.5), bbox_xyxy=(6, 6, 8, 8))
        assert result_b[6, 6] == 1
        assert result_b[1, 1] == 0

    def test_bbox_fallback_rounding(self) -> None:
        # Point slightly off foreground due to rounding
        mask = self._make_two_component_mask()
        result = select_instance_mask(
            mask,
            center_point_xy=(9, 9),  # just outside component A (starts at 10)
            bbox_xyxy=(10, 10, 30, 30),
        )
        assert count_connected_components(result) == 1
        assert result[20, 20] == 1
        assert result[70, 70] == 0


# ---------------------------------------------------------------------------
# MapSamDataset integration
# ---------------------------------------------------------------------------


class TestMapSamDatasetInstanceTarget:
    def _build_dataset(self, tmp_path: Path, point_a: bool = True) -> MapSamDataset:
        _build_multi_component_dataset(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        center = [20, 20] if point_a else [70, 70]
        bbox = [10, 10, 30, 30] if point_a else [60, 60, 80, 80]
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "multi_001",
                    "split": "train",
                    "image_path": "images/train/multi.png",
                    "mask_path": "masks/train/multi.png",
                    "ignore_mask_path": "ignore_masks/train/multi.png",
                    "bbox": bbox,
                    "center_point": center,
                }
            ],
        )
        return MapSamDataset(tmp_path, samples_path, "train", image_size=64)

    def test_returns_instance_mask_not_semantic(self, tmp_path: Path) -> None:
        ds = self._build_dataset(tmp_path, point_a=True)
        sample = ds[0]
        target = sample["target_mask"].squeeze().numpy()

        ncc = count_connected_components(target)
        assert ncc == 1, f"Expected 1 component, got {ncc}"

    def test_selects_component_a(self, tmp_path: Path) -> None:
        ds = self._build_dataset(tmp_path, point_a=True)
        sample = ds[0]
        target = sample["target_mask"].squeeze().numpy()

        # Component A is at original rows/cols 10:30 -> scaled to ~6:19 in 64x64
        # Component B is at original rows/cols 60:80 -> scaled to ~37:50 in 64x64
        assert target.sum() > 0, "Target mask should not be empty"
        assert count_connected_components(target) == 1

    def test_selects_component_b(self, tmp_path: Path) -> None:
        ds = self._build_dataset(tmp_path, point_a=False)
        sample = ds[0]
        target = sample["target_mask"].squeeze().numpy()
        assert count_connected_components(target) == 1

    def test_ignore_mask_not_filtered(self, tmp_path: Path) -> None:
        # Build dataset with non-empty ignore mask
        root = tmp_path
        _build_multi_component_dataset(root)

        # Overwrite ignore mask with some foreground
        ign = np.zeros((100, 100), dtype=np.uint8)
        ign[5:20, 5:20] = 255
        ign[50:90, 50:90] = 255
        Image.fromarray(ign, "L").save(root / "ignore_masks" / "train" / "multi.png")

        samples_path = root / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "ign_test",
                    "split": "train",
                    "image_path": "images/train/multi.png",
                    "mask_path": "masks/train/multi.png",
                    "ignore_mask_path": "ignore_masks/train/multi.png",
                    "bbox": [10, 10, 30, 30],
                    "center_point": [20, 20],
                }
            ],
        )
        ds = MapSamDataset(root, samples_path, "train", image_size=64)
        sample = ds[0]
        ignore = sample["ignore_mask"].squeeze().numpy()

        # Ignore mask should still have foreground from both regions
        # (it's not component-filtered)
        assert ignore.sum() > 0

    def test_target_mask_values_binary(self, tmp_path: Path) -> None:
        ds = self._build_dataset(tmp_path)
        sample = ds[0]
        unique = sample["target_mask"].unique().tolist()
        for v in unique:
            assert v in (0.0, 1.0), f"Unexpected value {v} in target mask"


# ---------------------------------------------------------------------------
# Rounding edge case: bbox fallback selects correct component
# ---------------------------------------------------------------------------


class TestRoundingFallback:
    def test_point_off_by_one_selects_via_bbox(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[60:80, 60:80] = 255

        result = select_instance_mask(
            mask,
            center_point_xy=(9, 9),
            bbox_xyxy=(10, 10, 30, 30),
        )
        assert result[20, 20] == 1
        assert result[70, 70] == 0
        assert count_connected_components(result) == 1


class TestLabelCacheEquivalence:
    """The cached label path must match the uncached path exactly."""

    def _write_mask(self, path: Path, arr: np.ndarray) -> None:
        Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(str(path))

    def test_cached_labels_match_uncached(self, tmp_path: Path) -> None:
        arr = np.zeros((40, 40), dtype=np.uint8)
        arr[2:8, 2:8] = 1
        arr[20:28, 20:28] = 1
        arr[30:34, 5:9] = 1
        mask_file = tmp_path / "mask.png"
        self._write_mask(mask_file, arr)

        label_mask_file.cache_clear()
        labeled, n = label_mask_file(str(mask_file))
        assert n == 3

        direct, n_direct = _label_foreground(arr > 0, 8)
        assert n_direct == n
        assert np.array_equal(labeled, direct)

    def test_selection_identical_with_and_without_cache(self, tmp_path: Path) -> None:
        arr = np.zeros((40, 40), dtype=np.uint8)
        arr[2:8, 2:8] = 1
        arr[20:28, 20:28] = 1
        mask_file = tmp_path / "mask.png"
        self._write_mask(mask_file, arr)

        label_mask_file.cache_clear()
        cached = label_mask_file(str(mask_file))

        point = (23.0, 23.0)
        bbox = (20.0, 20.0, 28.0, 28.0)

        without = select_instance_mask(arr, point, bbox, sample_id="s")
        with_cache = select_instance_mask(
            arr, point, bbox, sample_id="s", labeled_components=cached
        )
        assert np.array_equal(without, with_cache)
        assert with_cache[23, 23] == 1
        assert with_cache[5, 5] == 0

    def test_cached_array_is_read_only(self, tmp_path: Path) -> None:
        arr = np.zeros((20, 20), dtype=np.uint8)
        arr[5:10, 5:10] = 1
        mask_file = tmp_path / "mask.png"
        self._write_mask(mask_file, arr)

        label_mask_file.cache_clear()
        labeled, _ = label_mask_file(str(mask_file))
        with pytest.raises(ValueError):
            labeled[0, 0] = 99
