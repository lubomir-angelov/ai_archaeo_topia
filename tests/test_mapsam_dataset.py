"""Tests for MapSamDataset and helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from archeo_topia.datasets.mapsam_dataset import (
    MapSamDataset,
    load_binary_mask,
    load_jsonl,
    load_rgb_image,
    resize_image_and_masks,
    scale_bbox,
    scale_point,
    validate_sample_row,
)


def _make_rgb_image(w: int = 100, h: int = 80) -> np.ndarray:
    """Create a synthetic RGB image array."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_binary_mask(w: int = 100, h: int = 80, seed: int = 42) -> np.ndarray:
    """Create a synthetic binary mask with a foreground patch."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = rng.randint(0, h - 10, 2)
    x1, x2 = rng.randint(0, w - 10, 2)
    y1, y2 = sorted([y1, y2])
    x1, x2 = sorted([x1, x2])
    arr[y1:y2, x1:x2] = 255
    return arr


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_minimal_dataset(
    root: Path,
    image_w: int = 100,
    image_h: int = 80,
) -> None:
    """Create a minimal valid dataset layout with one image per split."""
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        (root / "ignore_masks" / split).mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(_make_rgb_image(image_w, image_h), "RGB")
        mask = Image.fromarray(_make_binary_mask(image_w, image_h), "L")
        ign = Image.fromarray(np.zeros((image_h, image_w), dtype=np.uint8), "L")

        fname = f"sheet1_{split}.png"
        img.save(root / "images" / split / fname)
        mask.save(root / "masks" / split / fname)
        ign.save(root / "ignore_masks" / split / fname)


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_loads_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, [{"a": 1}, {"a": 2}])
        rows = load_jsonl(path)
        assert len(rows) == 2
        assert rows[0] == {"a": 1}

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_jsonl(path) == []

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "gaps.jsonl"
        path.write_text('{"a":1}\n\n{"a":2}\n')
        rows = load_jsonl(path)
        assert len(rows) == 2

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_jsonl("/nonexistent/file.jsonl")


# ---------------------------------------------------------------------------
# load_rgb_image
# ---------------------------------------------------------------------------


class TestLoadRgbImage:
    def test_loads_rgb(self, tmp_path: Path) -> None:
        arr = np.full((10, 10, 3), [100, 150, 200], dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(tmp_path / "img.png")
        tensor = load_rgb_image(tmp_path / "img.png")
        assert tensor.shape == (3, 10, 10)
        assert tensor.dtype == torch.float32
        assert abs(tensor[0, 0, 0].item() - 100 / 255) < 1e-5
        assert abs(tensor[1, 0, 0].item() - 150 / 255) < 1e-5
        assert abs(tensor[2, 0, 0].item() - 200 / 255) < 1e-5

    def test_rgba_converted_to_rgb(self, tmp_path: Path) -> None:
        arr = np.full((10, 10, 4), [100, 150, 200, 255], dtype=np.uint8)
        Image.fromarray(arr, "RGBA").save(tmp_path / "img.png")
        tensor = load_rgb_image(tmp_path / "img.png")
        assert tensor.shape == (3, 10, 10)

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_rgb_image("/nonexistent/img.png")


# ---------------------------------------------------------------------------
# load_binary_mask
# ---------------------------------------------------------------------------


class TestLoadBinaryMask:
    def test_loads_binary(self, tmp_path: Path) -> None:
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[3:7, 3:7] = 255
        Image.fromarray(arr, "L").save(tmp_path / "mask.png")
        tensor = load_binary_mask(tmp_path / "mask.png")
        assert tensor.shape == (1, 10, 10)
        assert tensor.dtype == torch.float32
        assert tensor[0, 5, 5].item() == 1.0
        assert tensor[0, 0, 0].item() == 0.0

    def test_threshold_defensive(self, tmp_path: Path) -> None:
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[3:7, 3:7] = 10
        Image.fromarray(arr, "L").save(tmp_path / "mask.png")
        tensor = load_binary_mask(tmp_path / "mask.png")
        assert tensor[0, 5, 5].item() == 1.0

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_binary_mask("/nonexistent/mask.png")


# ---------------------------------------------------------------------------
# resize_image_and_masks
# ---------------------------------------------------------------------------


class TestResizeImageAndMasks:
    def test_square_resize(self) -> None:
        img = torch.rand(3, 100, 100)
        mask = torch.zeros(1, 100, 100)
        ign = torch.zeros(1, 100, 100)
        img_r, mask_r, ign_r = resize_image_and_masks(img, mask, ign, 50)
        assert img_r.shape == (3, 50, 50)
        assert mask_r.shape == (1, 50, 50)
        assert ign_r.shape == (1, 50, 50)

    def test_binary_mask_preserved(self) -> None:
        mask = torch.zeros(1, 100, 100)
        mask[0, 25:75, 25:75] = 1.0
        ign = torch.zeros(1, 100, 100)
        img = torch.rand(3, 100, 100)
        _, mask_r, _ = resize_image_and_masks(img, mask, ign, 50)
        assert mask_r.unique().tolist() == [0.0, 1.0]

    def test_non_square_resize(self) -> None:
        img = torch.rand(3, 80, 120)
        mask = torch.zeros(1, 80, 120)
        ign = torch.zeros(1, 80, 120)
        img_r, mask_r, ign_r = resize_image_and_masks(img, mask, ign, 64)
        assert img_r.shape == (3, 64, 64)
        assert mask_r.shape == (1, 64, 64)


# ---------------------------------------------------------------------------
# scale_bbox
# ---------------------------------------------------------------------------


class TestScaleBbox:
    def test_basic_scaling(self) -> None:
        result = scale_bbox([10, 20, 30, 40], 100, 100, 200)
        expected = torch.tensor([20.0, 40.0, 60.0, 80.0])
        torch.testing.assert_close(result, expected)

    def test_non_square_scaling(self) -> None:
        result = scale_bbox([10, 10, 50, 50], 100, 200, 100)
        expected = torch.tensor([5.0, 10.0, 25.0, 50.0])
        torch.testing.assert_close(result, expected)

    def test_downscale(self) -> None:
        result = scale_bbox([50, 50, 100, 100], 200, 200, 100)
        expected = torch.tensor([25.0, 25.0, 50.0, 50.0])
        torch.testing.assert_close(result, expected)

    def test_dtype(self) -> None:
        result = scale_bbox([0, 0, 100, 100], 100, 100, 100)
        assert result.dtype == torch.float32
        assert result.shape == (4,)


# ---------------------------------------------------------------------------
# scale_point
# ---------------------------------------------------------------------------


class TestScalePoint:
    def test_basic_scaling(self) -> None:
        result = scale_point([50, 50], 100, 100, 200)
        expected = torch.tensor([100.0, 100.0])
        torch.testing.assert_close(result, expected)

    def test_non_square_scaling(self) -> None:
        result = scale_point([50, 25], 50, 100, 100)
        expected = torch.tensor([50.0, 50.0])
        torch.testing.assert_close(result, expected)

    def test_dtype(self) -> None:
        result = scale_point([0, 0], 100, 100, 100)
        assert result.dtype == torch.float32
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# validate_sample_row
# ---------------------------------------------------------------------------


class TestValidateSampleRow:
    def test_valid_row(self) -> None:
        validate_sample_row(
            {
                "sample_id": "s1",
                "split": "train",
                "image_path": "img.png",
                "mask_path": "mask.png",
                "bbox": [0, 0, 10, 10],
                "center_point": [5, 5],
            }
        )

    def test_missing_field(self) -> None:
        with pytest.raises(ValueError, match="Missing required field"):
            validate_sample_row({"sample_id": "s1"})

    def test_malformed_bbox(self) -> None:
        with pytest.raises(ValueError, match="Malformed bbox"):
            validate_sample_row(
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "i.png",
                    "mask_path": "m.png",
                    "bbox": [1, 2],
                    "center_point": [5, 5],
                }
            )

    def test_malformed_point(self) -> None:
        with pytest.raises(ValueError, match="Malformed center_point"):
            validate_sample_row(
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "i.png",
                    "mask_path": "m.png",
                    "bbox": [0, 0, 10, 10],
                    "center_point": [5],
                }
            )


# ---------------------------------------------------------------------------
# MapSamDataset constructor
# ---------------------------------------------------------------------------


class TestMapSamDatasetInit:
    def test_missing_dataset_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            MapSamDataset(
                dataset_root="/nonexistent/root",
                samples_path=tmp_path / "samples.jsonl",
                split="train",
            )

    def test_missing_samples_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            MapSamDataset(
                dataset_root=tmp_path,
                samples_path=tmp_path / "nonexistent.jsonl",
                split="train",
            )

    def test_invalid_split(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(samples_path, [{"split": "train", "sample_id": "s1"}])

        with pytest.raises(ValueError, match="Invalid split"):
            MapSamDataset(
                dataset_root=tmp_path,
                samples_path=samples_path,
                split="production",
            )

    def test_filters_by_split(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        rows = []
        for split in ("train", "val", "test"):
            fname = f"sheet1_{split}.png"
            rows.append(
                {
                    "sample_id": f"s_{split}",
                    "split": split,
                    "image_path": f"images/{split}/{fname}",
                    "mask_path": f"masks/{split}/{fname}",
                    "ignore_mask_path": f"ignore_masks/{split}/{fname}",
                    "bbox": [10, 10, 50, 50],
                    "center_point": [30, 30],
                }
            )
        _write_jsonl(samples_path, rows)

        ds = MapSamDataset(tmp_path, samples_path, "val")
        assert len(ds) == 1

    def test_empty_split_fails(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(samples_path, [{"split": "train", "sample_id": "s1"}])

        with pytest.raises(ValueError, match="No samples found"):
            MapSamDataset(tmp_path, samples_path, "test")


# ---------------------------------------------------------------------------
# MapSamDataset __getitem__
# ---------------------------------------------------------------------------


class TestMapSamDatasetGetItem:
    def _prepare(self, tmp_path: Path, **kwargs) -> MapSamDataset:
        """Build a minimal dataset and return the Dataset instance."""
        image_w = kwargs.get("image_w", 100)
        image_h = kwargs.get("image_h", 80)
        _build_minimal_dataset(tmp_path, image_w, image_h)

        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "test_001",
                    "split": "train",
                    "image_path": "images/train/sheet1_train.png",
                    "mask_path": "masks/train/sheet1_train.png",
                    "ignore_mask_path": "ignore_masks/train/sheet1_train.png",
                    "bbox": [10, 10, 50, 50],
                    "center_point": [30, 30],
                }
            ],
        )
        return MapSamDataset(tmp_path, samples_path, "train", image_size=64)

    def test_tensor_shapes(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path)
        sample = ds[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["target_mask"].shape == (1, 64, 64)
        assert sample["ignore_mask"].shape == (1, 64, 64)
        assert sample["box_prompt"].shape == (4,)
        assert sample["point_prompt"].shape == (2,)
        assert sample["point_label"].shape == (1,)

    def test_image_dtype_and_range(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path)
        sample = ds[0]
        assert sample["image"].dtype == torch.float32
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_mask_binary_values(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path)
        sample = ds[0]
        unique_target = sample["target_mask"].unique().tolist()
        unique_ignore = sample["ignore_mask"].unique().tolist()
        for v in unique_target:
            assert v in (0.0, 1.0)
        for v in unique_ignore:
            assert v in (0.0, 1.0)

    def test_bbox_scaling(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path)
        sample = ds[0]
        expected = scale_bbox([10, 10, 50, 50], 80, 100, 64)
        torch.testing.assert_close(sample["box_prompt"], expected)

    def test_point_scaling(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path)
        sample = ds[0]
        expected = scale_point([30, 30], 80, 100, 64)
        torch.testing.assert_close(sample["point_prompt"], expected)

    def test_point_label(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path)
        sample = ds[0]
        assert sample["point_label"].dtype == torch.int64
        assert sample["point_label"].item() == 1

    def test_metadata(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path)
        sample = ds[0]
        assert sample["sample_id"] == "test_001"
        assert sample["image_path"] == "images/train/sheet1_train.png"
        assert sample["original_size"] == (80, 100)
        assert sample["resized_size"] == (64, 64)

    def test_missing_image_fails(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        (tmp_path / "images" / "train" / "sheet1_train.png").unlink()
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "images/train/sheet1_train.png",
                    "mask_path": "masks/train/sheet1_train.png",
                    "ignore_mask_path": "ignore_masks/train/sheet1_train.png",
                    "bbox": [10, 10, 50, 50],
                    "center_point": [30, 30],
                }
            ],
        )
        ds = MapSamDataset(tmp_path, samples_path, "train")
        with pytest.raises(FileNotFoundError, match="not found"):
            ds[0]

    def test_missing_mask_fails(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        (tmp_path / "masks" / "train" / "sheet1_train.png").unlink()
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "images/train/sheet1_train.png",
                    "mask_path": "masks/train/sheet1_train.png",
                    "ignore_mask_path": "ignore_masks/train/sheet1_train.png",
                    "bbox": [10, 10, 50, 50],
                    "center_point": [30, 30],
                }
            ],
        )
        ds = MapSamDataset(tmp_path, samples_path, "train")
        with pytest.raises(FileNotFoundError, match="not found"):
            ds[0]

    def test_missing_ignore_mask_fails(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        (tmp_path / "ignore_masks" / "train" / "sheet1_train.png").unlink()
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "images/train/sheet1_train.png",
                    "mask_path": "masks/train/sheet1_train.png",
                    "ignore_mask_path": "ignore_masks/train/sheet1_train.png",
                    "bbox": [10, 10, 50, 50],
                    "center_point": [30, 30],
                }
            ],
        )
        ds = MapSamDataset(tmp_path, samples_path, "train")
        with pytest.raises(FileNotFoundError, match="not found"):
            ds[0]

    def test_size_mismatch_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "ds"
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)
            (root / "ignore_masks" / split).mkdir(parents=True, exist_ok=True)

        Image.fromarray(_make_rgb_image(100, 80), "RGB").save(
            root / "images" / "train" / "img.png"
        )
        Image.fromarray(_make_binary_mask(50, 50), "L").save(root / "masks" / "train" / "img.png")
        Image.fromarray(np.zeros((80, 100), dtype=np.uint8), "L").save(
            root / "ignore_masks" / "train" / "img.png"
        )

        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "images/train/img.png",
                    "mask_path": "masks/train/img.png",
                    "ignore_mask_path": "ignore_masks/train/img.png",
                    "bbox": [10, 10, 50, 50],
                    "center_point": [30, 30],
                }
            ],
        )
        ds = MapSamDataset(root, samples_path, "train")
        with pytest.raises(ValueError, match="does not match"):
            ds[0]

    def test_multiple_samples_same_image(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "images/train/sheet1_train.png",
                    "mask_path": "masks/train/sheet1_train.png",
                    "ignore_mask_path": "ignore_masks/train/sheet1_train.png",
                    "bbox": [10, 10, 30, 30],
                    "center_point": [20, 20],
                },
                {
                    "sample_id": "s2",
                    "split": "train",
                    "image_path": "images/train/sheet1_train.png",
                    "mask_path": "masks/train/sheet1_train.png",
                    "ignore_mask_path": "ignore_masks/train/sheet1_train.png",
                    "bbox": [50, 50, 80, 80],
                    "center_point": [65, 65],
                },
            ],
        )
        ds = MapSamDataset(tmp_path, samples_path, "train", image_size=64)
        assert len(ds) == 2

        s1 = ds[0]
        s2 = ds[1]
        assert s1["sample_id"] == "s1"
        assert s2["sample_id"] == "s2"
        torch.testing.assert_close(s1["box_prompt"], scale_bbox([10, 10, 30, 30], 80, 100, 64))
        torch.testing.assert_close(s2["box_prompt"], scale_bbox([50, 50, 80, 80], 80, 100, 64))

    def test_nonsquare_original(self, tmp_path: Path) -> None:
        ds = self._prepare(tmp_path, image_w=200, image_h=100)
        sample = ds[0]
        assert sample["original_size"] == (100, 200)
        assert sample["resized_size"] == (64, 64)
        assert sample["image"].shape == (3, 64, 64)
        assert sample["target_mask"].shape == (1, 64, 64)

    def test_return_original_size_false(self, tmp_path: Path) -> None:
        _build_minimal_dataset(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        _write_jsonl(
            samples_path,
            [
                {
                    "sample_id": "s1",
                    "split": "train",
                    "image_path": "images/train/sheet1_train.png",
                    "mask_path": "masks/train/sheet1_train.png",
                    "ignore_mask_path": "ignore_masks/train/sheet1_train.png",
                    "bbox": [10, 10, 50, 50],
                    "center_point": [30, 30],
                }
            ],
        )
        ds = MapSamDataset(
            tmp_path, samples_path, "train", image_size=64, return_original_size=False
        )
        sample = ds[0]
        assert "original_size" not in sample
        assert "resized_size" not in sample
