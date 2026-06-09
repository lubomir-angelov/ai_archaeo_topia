"""Tests for generate_mapsam_prompts pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from archeo_topia.datasets.generate_mapsam_prompts import (
    build_training_sample,
    component_to_bbox,
    find_connected_components,
    generate_samples,
    load_binary_mask,
    parse_sheet_id,
    validate_dataset_root,
    validate_image_mask_pair,
)

# ---------------------------------------------------------------------------
# Unit: parse_sheet_id
# ---------------------------------------------------------------------------


class TestParseSheetId:
    def test_standard_filename(self) -> None:
        assert parse_sheet_id("K-35-8-G-a_1.png") == "K-35-8-G-a"

    def test_another_sheet(self) -> None:
        assert parse_sheet_id("K-34-35-B-g_2.png") == "K-34-35-B-g"

    def test_higher_index(self) -> None:
        assert parse_sheet_id("K-35-51-B-a_4.png") == "K-35-51-B-a"

    def test_no_suffix(self) -> None:
        assert parse_sheet_id("SINGLE.png") == "SINGLE"

    def test_no_extension(self) -> None:
        assert parse_sheet_id("K-35-8-G-a_1") == "K-35-8-G-a"

    def test_multiple_underscores(self) -> None:
        assert parse_sheet_id("A_B_C_5.png") == "A_B_C"

    def test_single_digit_suffix(self) -> None:
        assert parse_sheet_id("Sheet_0.png") == "Sheet"


# ---------------------------------------------------------------------------
# Unit: load_binary_mask
# ---------------------------------------------------------------------------


class TestLoadBinaryMask:
    def test_loads_binary_mask(self, tmp_path: Path) -> None:
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[3:7, 3:7] = 255
        img = Image.fromarray(arr, "L")
        path = tmp_path / "mask.png"
        img.save(path)

        result = load_binary_mask(path)
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8
        assert result[5, 5] == 255
        assert result[0, 0] == 0

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="Mask file not found"):
            load_binary_mask(Path("/nonexistent/mask.png"))


# ---------------------------------------------------------------------------
# Unit: find_connected_components
# ---------------------------------------------------------------------------


class TestFindConnectedComponents:
    def test_single_component(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        components = find_connected_components(mask)

        assert len(components) == 1
        assert components[0]["area"] == 16
        assert components[0]["bbox"] == [2, 2, 5, 5]
        assert components[0]["center_point"] == [3, 3]

    def test_two_separate_components(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[12:15, 12:15] = 255
        components = find_connected_components(mask)

        assert len(components) == 2
        assert components[0]["area"] == 9
        assert components[1]["area"] == 9

    def test_empty_mask(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        components = find_connected_components(mask)
        assert len(components) == 0

    def test_single_pixel_component(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 255
        components = find_connected_components(mask)

        assert len(components) == 1
        assert components[0]["area"] == 1
        assert components[0]["bbox"] == [5, 5, 5, 5]

    def test_diagonal_connectivity(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2, 2] = 255
        mask[3, 3] = 255
        components = find_connected_components(mask)

        assert len(components) == 1
        assert components[0]["area"] == 2

    def test_min_area_filtering(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        mask[10:15, 10:15] = 255
        components = find_connected_components(mask, min_area=10)

        assert len(components) == 1
        assert components[0]["area"] == 25

    def test_component_indexing(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[12:15, 12:15] = 255
        mask[2, 15] = 255
        components = find_connected_components(mask)

        assert len(components) == 3
        assert components[0]["index"] == 1
        assert components[1]["index"] == 2
        assert components[2]["index"] == 3


# ---------------------------------------------------------------------------
# Unit: component_to_bbox
# ---------------------------------------------------------------------------


class TestComponentToBbox:
    def test_no_padding(self) -> None:
        comp = {
            "index": 1,
            "area": 16,
            "bbox": [10, 10, 20, 20],
            "center_point": [15, 15],
        }
        result = component_to_bbox(comp, 100, 100, padding=0)
        assert result["bbox"] == [10, 10, 20, 20]
        assert result["center_point"] == [15, 15]

    def test_padding_expands_box(self) -> None:
        comp = {
            "index": 1,
            "area": 16,
            "bbox": [10, 10, 20, 20],
            "center_point": [15, 15],
        }
        result = component_to_bbox(comp, 100, 100, padding=5)
        assert result["bbox"] == [5, 5, 25, 25]
        assert result["center_point"] == [15, 15]

    def test_clamp_left_boundary(self) -> None:
        comp = {
            "index": 1,
            "area": 4,
            "bbox": [2, 2, 4, 4],
            "center_point": [3, 3],
        }
        result = component_to_bbox(comp, 100, 100, padding=10)
        assert result["bbox"] == [0, 0, 14, 14]

    def test_clamp_right_boundary(self) -> None:
        comp = {
            "index": 1,
            "area": 4,
            "bbox": [95, 95, 98, 98],
            "center_point": [96, 96],
        }
        result = component_to_bbox(comp, 100, 100, padding=10)
        assert result["bbox"] == [85, 85, 99, 99]

    def test_clamp_both_boundaries(self) -> None:
        comp = {
            "index": 1,
            "area": 1,
            "bbox": [1, 1, 2, 2],
            "center_point": [1, 1],
        }
        result = component_to_bbox(comp, 10, 10, padding=100)
        assert result["bbox"] == [0, 0, 9, 9]

    def test_does_not_mutate_input(self) -> None:
        comp = {
            "index": 1,
            "area": 16,
            "bbox": [10, 10, 20, 20],
            "center_point": [15, 15],
        }
        original_bbox = list(comp["bbox"])
        component_to_bbox(comp, 100, 100, padding=5)
        assert comp["bbox"] == original_bbox


# ---------------------------------------------------------------------------
# Unit: build_training_sample
# ---------------------------------------------------------------------------


class TestBuildTrainingSample:
    def test_sample_structure(self) -> None:
        comp = {
            "index": 1,
            "area": 574,
            "bbox": [622, 256, 648, 284],
            "center_point": [635, 270],
        }
        sample = build_training_sample(
            split="train",
            sheet_id="K-35-8-G-a",
            filename="K-35-8-G-a_1.png",
            component=comp,
            dataset_root=Path("/data"),
        )

        assert sample["sample_id"] == "train_K-35-8-G-a_1_000001"
        assert sample["split"] == "train"
        assert sample["sheet_id"] == "K-35-8-G-a"
        assert sample["image_path"] == "images/train/K-35-8-G-a_1.png"
        assert sample["mask_path"] == "masks/train/K-35-8-G-a_1.png"
        assert sample["ignore_mask_path"] == "ignore_masks/train/K-35-8-G-a_1.png"
        assert sample["component_index"] == 1
        assert sample["component_area"] == 574
        assert sample["bbox"] == [622, 256, 648, 284]
        assert sample["center_point"] == [635, 270]


# ---------------------------------------------------------------------------
# Unit: validate_dataset_root
# ---------------------------------------------------------------------------


class TestValidateDatasetRoot:
    def test_missing_root(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            validate_dataset_root(Path("/nonexistent"))

    def test_missing_split_folder(self, tmp_path: Path) -> None:
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "masks" / "train").mkdir(parents=True)
        with pytest.raises(ValueError, match="Missing expected directory"):
            validate_dataset_root(tmp_path)

    def test_valid_root(self, tmp_path: Path) -> None:
        for split in ("train", "val", "test"):
            (tmp_path / "images" / split).mkdir(parents=True)
            (tmp_path / "masks" / split).mkdir(parents=True)
        validate_dataset_root(tmp_path)


# ---------------------------------------------------------------------------
# Unit: validate_image_mask_pair
# ---------------------------------------------------------------------------


class TestValidateImageMaskPair:
    def test_matching_sizes(self, tmp_path: Path) -> None:
        img = Image.new("L", (100, 80), 128)
        mask = Image.new("L", (100, 80), 0)
        img.save(tmp_path / "img.png")
        mask.save(tmp_path / "mask.png")

        w, h = validate_image_mask_pair(tmp_path / "img.png", tmp_path / "mask.png", None)
        assert w == 100
        assert h == 80

    def test_size_mismatch(self, tmp_path: Path) -> None:
        img = Image.new("L", (100, 80), 128)
        mask = Image.new("L", (50, 50), 0)
        img.save(tmp_path / "img.png")
        mask.save(tmp_path / "mask.png")

        with pytest.raises(ValueError, match="does not match"):
            validate_image_mask_pair(tmp_path / "img.png", tmp_path / "mask.png", None)

    def test_missing_mask(self, tmp_path: Path) -> None:
        img = Image.new("L", (100, 80), 128)
        img.save(tmp_path / "img.png")

        with pytest.raises(FileNotFoundError, match="Mask not found"):
            validate_image_mask_pair(
                tmp_path / "img.png",
                tmp_path / "mask.png",
                None,
            )

    def test_ignore_mask_size_mismatch(self, tmp_path: Path) -> None:
        img = Image.new("L", (100, 80), 128)
        mask = Image.new("L", (100, 80), 0)
        ign = Image.new("L", (50, 50), 0)
        img.save(tmp_path / "img.png")
        mask.save(tmp_path / "mask.png")
        ign.save(tmp_path / "ign.png")

        with pytest.raises(ValueError, match="Ignore mask size"):
            validate_image_mask_pair(
                tmp_path / "img.png",
                tmp_path / "mask.png",
                tmp_path / "ign.png",
            )


# ---------------------------------------------------------------------------
# Integration: generate_samples
# ---------------------------------------------------------------------------


class TestGenerateSamples:
    def _create_dataset(
        self,
        root: Path,
        active_splits: list[str] | None = None,
    ) -> None:
        """Create a minimal valid dataset layout.

        Always creates all three split directories so validation passes.
        *active_splits* controls which splits receive test images.
        """
        active_splits = active_splits or ["train"]
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "masks" / split).mkdir(parents=True)
            (root / "ignore_masks" / split).mkdir(parents=True)

    def test_basic_generation(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (100, 100), 128)
        mask = Image.new("L", (100, 100), 0)
        mask_arr = np.array(mask)
        mask_arr[20:30, 20:30] = 255
        Image.fromarray(mask_arr, "L").save(root / "masks" / "train" / "S1_1.png")
        img.save(root / "images" / "train" / "S1_1.png")
        Image.new("L", (100, 100), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        samples, summary = generate_samples(root, min_component_area=1)

        assert len(samples) == 1
        assert samples[0]["split"] == "train"
        assert samples[0]["sheet_id"] == "S1"
        assert samples[0]["component_area"] == 100
        assert summary["total_samples"] == 1
        assert summary["images_by_split"]["train"] == 1

    def test_multiple_components(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (200, 200), 128)
        mask = Image.new("L", (200, 200), 0)
        mask_arr = np.array(mask)
        mask_arr[10:20, 10:20] = 255
        mask_arr[80:90, 80:90] = 255
        Image.fromarray(mask_arr, "L").save(root / "masks" / "train" / "S1_1.png")
        img.save(root / "images" / "train" / "S1_1.png")
        Image.new("L", (200, 200), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        samples, summary = generate_samples(root, min_component_area=1)

        assert len(samples) == 2
        assert samples[0]["component_index"] == 1
        assert samples[1]["component_index"] == 2

    def test_min_component_area_filtering(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (200, 200), 128)
        mask = Image.new("L", (200, 200), 0)
        mask_arr = np.array(mask)
        mask_arr[10:12, 10:12] = 255
        mask_arr[50:60, 50:60] = 255
        Image.fromarray(mask_arr, "L").save(root / "masks" / "train" / "S1_1.png")
        img.save(root / "images" / "train" / "S1_1.png")
        Image.new("L", (200, 200), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        samples, summary = generate_samples(root, min_component_area=20)

        assert len(samples) == 1
        assert samples[0]["component_area"] == 100
        assert summary["skipped_components_by_reason"]["too_small"] == 1

    def test_bbox_padding_applied(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (100, 100), 128)
        mask = Image.new("L", (100, 100), 0)
        mask_arr = np.array(mask)
        mask_arr[40:45, 40:45] = 255
        Image.fromarray(mask_arr, "L").save(root / "masks" / "train" / "S1_1.png")
        img.save(root / "images" / "train" / "S1_1.png")
        Image.new("L", (100, 100), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        samples, _ = generate_samples(root, min_component_area=1, bbox_padding=10)

        assert samples[0]["bbox"] == [30, 30, 54, 54]

    def test_image_without_mounds(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (100, 100), 128)
        mask = Image.new("L", (100, 100), 0)
        img.save(root / "images" / "train" / "S1_1.png")
        mask.save(root / "masks" / "train" / "S1_1.png")
        Image.new("L", (100, 100), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        samples, summary = generate_samples(root)

        assert len(samples) == 0
        assert summary["images_without_mounds_by_split"]["train"] == 1

    def test_all_splits_processed(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train", "val", "test"])

        for split in ("train", "val", "test"):
            img = Image.new("L", (50, 50), 128)
            mask = Image.new("L", (50, 50), 0)
            mask_arr = np.array(mask)
            mask_arr[10:20, 10:20] = 255
            Image.fromarray(mask_arr, "L").save(root / "masks" / split / "S1_1.png")
            img.save(root / "images" / split / "S1_1.png")
            Image.new("L", (50, 50), 0).save(root / "ignore_masks" / split / "S1_1.png")

        samples, summary = generate_samples(root, min_component_area=1)

        assert len(samples) == 3
        assert summary["samples_by_split"]["train"] == 1
        assert summary["samples_by_split"]["val"] == 1
        assert summary["samples_by_split"]["test"] == 1

    def test_jsonl_row_valid(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (100, 100), 128)
        mask = Image.new("L", (100, 100), 0)
        mask_arr = np.array(mask)
        mask_arr[20:30, 20:30] = 255
        Image.fromarray(mask_arr, "L").save(root / "masks" / "train" / "S1_1.png")
        img.save(root / "images" / "train" / "S1_1.png")
        Image.new("L", (100, 100), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        samples, _ = generate_samples(root, min_component_area=1)

        for sample in samples:
            row = json.dumps(sample)
            parsed = json.loads(row)
            assert "sample_id" in parsed
            assert "split" in parsed
            assert "sheet_id" in parsed
            assert "image_path" in parsed
            assert "mask_path" in parsed
            assert "ignore_mask_path" in parsed
            assert "component_index" in parsed
            assert "component_area" in parsed
            assert "bbox" in parsed
            assert "center_point" in parsed

    def test_summary_includes_params(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (50, 50), 128)
        mask = Image.new("L", (50, 50), 0)
        mask_arr = np.array(mask)
        mask_arr[10:20, 10:20] = 255
        Image.fromarray(mask_arr, "L").save(root / "masks" / "train" / "S1_1.png")
        img.save(root / "images" / "train" / "S1_1.png")
        Image.new("L", (50, 50), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        _, summary = generate_samples(root, min_component_area=20, bbox_padding=5)

        assert summary["min_component_area"] == 20
        assert summary["bbox_padding"] == 5

    def test_image_mask_size_mismatch_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (100, 100), 128)
        mask = Image.new("L", (50, 50), 0)
        img.save(root / "images" / "train" / "S1_1.png")
        mask.save(root / "masks" / "train" / "S1_1.png")
        Image.new("L", (50, 50), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        with pytest.raises(ValueError, match="does not match"):
            generate_samples(root)

    def test_paths_are_relative(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        self._create_dataset(root, ["train"])

        img = Image.new("L", (50, 50), 128)
        mask = Image.new("L", (50, 50), 0)
        mask_arr = np.array(mask)
        mask_arr[10:20, 10:20] = 255
        Image.fromarray(mask_arr, "L").save(root / "masks" / "train" / "S1_1.png")
        img.save(root / "images" / "train" / "S1_1.png")
        Image.new("L", (50, 50), 0).save(root / "ignore_masks" / "train" / "S1_1.png")

        samples, _ = generate_samples(root, min_component_area=1)

        for sample in samples:
            assert not Path(sample["image_path"]).is_absolute()
            assert not Path(sample["mask_path"]).is_absolute()
            assert not Path(sample["ignore_mask_path"]).is_absolute()
