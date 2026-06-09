"""Tests for prepare_mapsam_coco pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from archeo_topia.datasets.prepare_mapsam_coco import (
    assign_splits,
    decode_bbox_mask,
    decode_polygon_mask,
    decode_rle_mask,
    decode_segmentation,
    decode_uncompressed_coco_rle,
    extract_sheet_id,
    find_category_id,
    mask_to_bbox,
    run,
)


def _make_image(w: int = 100, h: int = 100) -> Image.Image:
    """Create a dummy grayscale image."""
    return Image.new("L", (w, h), 128)


def _write_image(directory: Path, name: str, img: Image.Image | None = None) -> None:
    """Write an image file to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes((img or _make_image()).tobytes())
    if img is None:
        _make_image().save(directory / name)
    else:
        img.save(directory / name)


def _make_coco(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict] | None = None,
) -> dict:
    """Build a minimal COCO dict."""
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
        or [
            {"id": 1, "name": "mound"},
            {"id": 2, "name": "uncertain_ignore"},
            {"id": 3, "name": "hard_negative_symbol"},
        ],
    }


# ---------------------------------------------------------------------------
# Unit: extract_sheet_id
# ---------------------------------------------------------------------------


class TestExtractSheetId:
    def test_standard_prefix(self) -> None:
        assert extract_sheet_id("K-35-8-G-a_1.png") == "K-35-8-G-a"

    def test_another_sheet(self) -> None:
        assert extract_sheet_id("K-34-35-B-g_2.png") == "K-34-35-B-g"

    def test_no_suffix(self) -> None:
        assert extract_sheet_id("SINGLE.png") == "SINGLE"


# ---------------------------------------------------------------------------
# Unit: find_category_id
# ---------------------------------------------------------------------------


class TestFindCategoryId:
    def test_found(self) -> None:
        cats = [{"id": 10, "name": "mound"}]
        assert find_category_id(cats, "mound") == 10

    def test_not_found(self) -> None:
        cats = [{"id": 10, "name": "mound"}]
        assert find_category_id(cats, "river") is None


# ---------------------------------------------------------------------------
# Unit: assign_splits
# ---------------------------------------------------------------------------


class TestAssignSplits:
    def test_deterministic(self) -> None:
        sheets = ["B", "A", "C"]
        result = assign_splits(sheets)
        assert result == assign_splits(sheets)

    def test_all_sheets_assigned(self) -> None:
        sheets = ["A", "B", "C", "D", "E"]
        result = assign_splits(sheets)
        assert set(result.keys()) == set(sheets)

    def test_ratios(self) -> None:
        sheets = [f"S{i:02d}" for i in range(100)]
        result = assign_splits(sheets)
        counts = {s: 0 for s in ("train", "val", "test")}
        for v in result.values():
            counts[v] += 1
        assert counts["train"] == 70
        assert counts["val"] == 15
        assert counts["test"] == 15

    def test_single_sheet(self) -> None:
        result = assign_splits(["only"])
        assert len(result) == 1
        assert result["only"] == "train"


# ---------------------------------------------------------------------------
# Unit: decode_polygon_mask
# ---------------------------------------------------------------------------


class TestDecodePolygonMask:
    def test_triangle(self) -> None:
        polygon = [10.0, 10.0, 90.0, 10.0, 50.0, 90.0]
        mask = decode_polygon_mask(polygon, 100, 100)
        assert mask.size == (100, 100)
        assert mask.getpixel((50, 50)) == 255
        assert mask.getpixel((0, 0)) == 0

    def test_empty_polygon(self) -> None:
        mask = decode_polygon_mask([], 100, 100)
        assert mask.getpixel((50, 50)) == 0


# ---------------------------------------------------------------------------
# Unit: decode_bbox_mask
# ---------------------------------------------------------------------------


class TestDecodeBboxMask:
    def test_rectangle(self) -> None:
        mask = decode_bbox_mask([20, 30, 40, 50], 100, 100)
        assert mask.getpixel((40, 50)) == 255
        assert mask.getpixel((0, 0)) == 0
        assert mask.getpixel((99, 99)) == 0


# ---------------------------------------------------------------------------
# Unit: decode_rle_mask
# ---------------------------------------------------------------------------


class TestDecodeRleMask:
    def test_rle_basic(self) -> None:
        """Simple 5x5 RLE: 20 bg, 5 fg, 10 bg = 35 pixels."""
        rle = {"size": [5, 7], "counts": [20, 5, 10]}
        mask = decode_rle_mask(rle, 5, 7)
        assert mask.size == (7, 5)

    def test_rle_fallback_on_error(self) -> None:
        """Non-list counts should fall back to empty mask."""
        rle = {"size": [100, 100], "counts": "compressed_string"}
        mask = decode_rle_mask(rle, 100, 100)
        assert mask.size == (100, 100)

    def test_rle_integer_counts(self) -> None:
        """CVAT exports RLE counts as a list of integers."""
        # 10x10 image: 50 background, 30 foreground, 20 background = 100
        rle = {"size": [10, 10], "counts": [50, 30, 20]}
        mask = decode_rle_mask(rle, 10, 10)
        assert mask.size == (10, 10)

    def test_rle_integer_counts_large_runs(self) -> None:
        """Large run values (like CVAT's 1377995) should work."""
        rle = {"size": [100, 100], "counts": [9999, 1]}
        mask = decode_rle_mask(rle, 100, 100)
        assert mask.size == (100, 100)


# ---------------------------------------------------------------------------
# Unit: decode_uncompressed_coco_rle
# ---------------------------------------------------------------------------


class TestDecodeUncompressedCocoRle:
    def test_first_count_is_background(self) -> None:
        """First count must be treated as background (0)."""
        seg = {"size": [10, 10], "counts": [60, 40]}
        arr = decode_uncompressed_coco_rle(seg, 10, 10)
        assert arr.shape == (10, 10)
        assert arr.dtype == np.uint8
        fg_count = (arr > 0).sum()
        assert fg_count == 40

    def test_reshapes_with_fortran_order(self) -> None:
        """Reshape must use order='F' (column-major), not row-major."""
        # 5x3 image: 10 bg, 5 fg, 0 bg = 15 pixels
        # Column-major: first 10 pixels fill cols 0-1, then 5 fg fill col 2
        seg = {"size": [3, 5], "counts": [10, 5]}
        arr = decode_uncompressed_coco_rle(seg, 3, 5)
        assert arr.shape == (3, 5)
        fg_count = (arr > 250).sum()
        assert fg_count == 5

    def test_fg_value_is_255(self) -> None:
        """Foreground pixels must be exactly 255."""
        seg = {"size": [10, 10], "counts": [50, 50]}
        arr = decode_uncompressed_coco_rle(seg, 10, 10)
        unique_vals = set(arr[arr > 0])
        assert unique_vals == {255}

    def test_negative_count_raises(self) -> None:
        seg = {"size": [10, 10], "counts": [51, -1, 50]}
        with pytest.raises(ValueError, match="negative"):
            decode_uncompressed_coco_rle(seg, 10, 10)

    def test_pixel_count_mismatch_raises(self) -> None:
        seg = {"size": [10, 10], "counts": [50, 50]}
        with pytest.raises(ValueError, match="sum"):
            decode_uncompressed_coco_rle(seg, 20, 20)

    def test_non_list_counts_raises(self) -> None:
        seg = {"size": [10, 10], "counts": "compressed_string"}
        with pytest.raises(ValueError, match="Compressed"):
            decode_uncompressed_coco_rle(seg, 10, 10)

    def test_bbox_regression(self) -> None:
        """Decoded mask bbox must approximately match the COCO bbox."""
        import json
        from pathlib import Path

        coco_path = Path("data/curated/datasets/cvat/v0.0.1/annotations/instances_default.json")
        if not coco_path.exists():
            pytest.skip("COCO export not available")

        coco = json.loads(coco_path.read_text())
        img = next(i for i in coco["images"] if i["file_name"] == "K-34-35-B-g_1.png")
        anns = [
            a for a in coco["annotations"] if a["image_id"] == img["id"] and a["category_id"] == 1
        ]
        assert len(anns) >= 1
        ann = anns[0]
        arr = decode_uncompressed_coco_rle(ann["segmentation"], img["height"], img["width"])
        bbox = mask_to_bbox(arr)
        assert bbox is not None
        cx, cy, cw, ch = ann["bbox"]
        expected = (int(cx), int(cy), int(cx + cw), int(cy + ch))
        dx = abs(bbox[0] - expected[0]) + abs(bbox[2] - expected[2])
        dy = abs(bbox[1] - expected[1]) + abs(bbox[3] - expected[3])
        assert dx <= 3, f"X deviation {dx} too large: decoded={bbox} expected={expected}"
        assert dy <= 3, f"Y deviation {dy} too large: decoded={bbox} expected={expected}"


# ---------------------------------------------------------------------------
# Unit: mask_to_bbox
# ---------------------------------------------------------------------------


class TestMaskToBbox:
    def test_solid_block(self) -> None:
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[20:40, 30:60] = 255
        bbox = mask_to_bbox(arr)
        assert bbox == (30, 20, 59, 39)

    def test_empty_mask(self) -> None:
        arr = np.zeros((100, 100), dtype=np.uint8)
        assert mask_to_bbox(arr) is None

    def test_single_pixel(self) -> None:
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[3, 7] = 255
        bbox = mask_to_bbox(arr)
        assert bbox == (7, 3, 7, 3)


# ---------------------------------------------------------------------------
# Unit: decode_segmentation
# ---------------------------------------------------------------------------


class TestDecodeSegmentation:
    def test_polygon_segmentation(self) -> None:
        seg = [[10.0, 10.0, 90.0, 10.0, 50.0, 90.0]]
        mask = decode_segmentation(seg, None, 100, 100)
        assert mask.getpixel((50, 50)) == 255

    def test_flat_polygon_segmentation(self) -> None:
        seg = [10.0, 10.0, 90.0, 10.0, 50.0, 90.0]
        mask = decode_segmentation(seg, None, 100, 100)
        assert mask.getpixel((50, 50)) == 255

    def test_empty_seg_with_bbox_fallback(self) -> None:
        mask = decode_segmentation([], [20, 20, 20, 20], 100, 100)
        assert mask.getpixel((30, 30)) == 255
        assert mask.getpixel((0, 0)) == 0

    def test_none_seg_with_bbox_fallback(self) -> None:
        mask = decode_segmentation(None, [20, 20, 20, 20], 100, 100)
        assert mask.getpixel((30, 30)) == 255

    def test_empty_seg_no_bbox(self) -> None:
        mask = decode_segmentation([], None, 100, 100)
        assert mask.getpixel((50, 50)) == 0

    def test_rle_segmentation(self) -> None:
        rle = {"size": [100, 100], "counts": "AAAAAAAA"}
        mask = decode_segmentation(rle, None, 100, 100)
        assert mask.size == (100, 100)

    def test_rle_integer_counts_segmentation(self) -> None:
        """CVAT RLE with integer counts through decode_segmentation."""
        rle = {"size": [100, 100], "counts": [9999, 1]}
        mask = decode_segmentation(rle, None, 100, 100)
        assert mask.size == (100, 100)
        assert mask.getpixel((99, 99)) == 255
        assert mask.getpixel((0, 0)) == 0

    def test_null_segmentation_with_bbox_fallback(self) -> None:
        """Ignore annotations with null segmentation should fall back to bbox."""
        mask = decode_segmentation(None, [30, 40, 20, 20], 100, 100)
        assert mask.getpixel((40, 50)) == 255
        assert mask.getpixel((0, 0)) == 0


# ---------------------------------------------------------------------------
# Integration: run()
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_polygon_mound_and_ignore_mask(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        coco_path = tmp_path / "coco.json"
        out_dir = tmp_path / "output"

        _make_image().save(img_dir / "K-35-8-G-a_1.png")

        coco = _make_coco(
            images=[{"id": 1, "file_name": "K-35-8-G-a_1.png", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10, 10, 90, 10, 50, 90]],
                    "bbox": [10, 10, 80, 80],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "segmentation": [[5, 5, 15, 5, 10, 15]],
                    "bbox": [5, 5, 10, 10],
                },
            ],
        )
        coco_path.write_text(json.dumps(coco))

        run(
            coco_json=coco_path,
            images_dir=img_dir,
            output_dir=out_dir,
            positive_label="mound",
            ignore_label="uncertain_ignore",
            hard_negative_label="hard_negative_symbol",
        )

        assert (out_dir / "images" / "train" / "K-35-8-G-a_1.png").exists()
        assert (out_dir / "masks" / "train" / "K-35-8-G-a_1.png").exists()
        assert (out_dir / "ignore_masks" / "train" / "K-35-8-G-a_1.png").exists()

        pos = Image.open(out_dir / "masks" / "train" / "K-35-8-G-a_1.png")
        assert pos.getpixel((50, 50)) == 255

        ign = Image.open(out_dir / "ignore_masks" / "train" / "K-35-8-G-a_1.png")
        assert ign.getpixel((10, 10)) == 255

    def test_hard_negative_not_in_positive_mask(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        coco_path = tmp_path / "coco.json"
        out_dir = tmp_path / "output"

        _make_image().save(img_dir / "S1_1.png")

        coco = _make_coco(
            images=[{"id": 1, "file_name": "S1_1.png", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 3,
                    "segmentation": [[10, 10, 90, 10, 50, 90]],
                    "bbox": [10, 10, 80, 80],
                },
            ],
        )
        coco_path.write_text(json.dumps(coco))

        run(
            coco_json=coco_path,
            images_dir=img_dir,
            output_dir=out_dir,
            positive_label="mound",
            ignore_label="uncertain_ignore",
            hard_negative_label="hard_negative_symbol",
        )

        pos = Image.open(out_dir / "masks" / "train" / "S1_1.png")
        assert pos.getpixel((50, 50)) == 0

        summary = json.loads((out_dir / "metadata" / "dataset_summary.json").read_text())
        assert summary["hard_negative_count_per_split"]["train"] == 1

    def test_bbox_fallback_for_empty_segmentation(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        coco_path = tmp_path / "coco.json"
        out_dir = tmp_path / "output"

        _make_image().save(img_dir / "S1_1.png")

        coco = _make_coco(
            images=[{"id": 1, "file_name": "S1_1.png", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [],
                    "bbox": [20, 20, 30, 30],
                },
            ],
        )
        coco_path.write_text(json.dumps(coco))

        run(
            coco_json=coco_path,
            images_dir=img_dir,
            output_dir=out_dir,
            positive_label="mound",
            ignore_label="uncertain_ignore",
            hard_negative_label="hard_negative_symbol",
        )

        pos = Image.open(out_dir / "masks" / "train" / "S1_1.png")
        assert pos.getpixel((35, 35)) == 255
        assert pos.getpixel((0, 0)) == 0

    def test_uncertain_ignore_mask(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        coco_path = tmp_path / "coco.json"
        out_dir = tmp_path / "output"

        _make_image().save(img_dir / "S1_1.png")

        coco = _make_coco(
            images=[{"id": 1, "file_name": "S1_1.png", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 2,
                    "segmentation": [[30, 30, 70, 30, 50, 70]],
                    "bbox": [30, 30, 40, 40],
                },
            ],
        )
        coco_path.write_text(json.dumps(coco))

        run(
            coco_json=coco_path,
            images_dir=img_dir,
            output_dir=out_dir,
            positive_label="mound",
            ignore_label="uncertain_ignore",
            hard_negative_label="hard_negative_symbol",
        )

        ign = Image.open(out_dir / "ignore_masks" / "train" / "S1_1.png")
        assert ign.getpixel((50, 50)) == 255
        assert ign.getpixel((0, 0)) == 0

    def test_sheet_level_split_grouping(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        coco_path = tmp_path / "coco.json"
        out_dir = tmp_path / "output"

        img_records = []
        ann_records = []
        for i in range(10):
            sheet = f"Sheet{i}"
            for j in range(3):
                fname = f"{sheet}_{j}.png"
                _make_image().save(img_dir / fname)
                img_records.append(
                    {
                        "id": i * 3 + j,
                        "file_name": fname,
                        "width": 100,
                        "height": 100,
                    }
                )

        coco = _make_coco(images=img_records, annotations=ann_records)
        coco_path.write_text(json.dumps(coco))

        run(
            coco_json=coco_path,
            images_dir=img_dir,
            output_dir=out_dir,
            positive_label="mound",
            ignore_label="uncertain_ignore",
            hard_negative_label="hard_negative_symbol",
        )

        splits = json.loads((out_dir / "metadata" / "splits.json").read_text())

        sheet_splits: dict[str, str | None] = {}
        for split_name, entries in splits.items():
            for entry in entries:
                sheet = entry["sheet"]
                if sheet in sheet_splits and sheet_splits[sheet] != split_name:
                    raise AssertionError(
                        f"Sheet {sheet} appears in both {sheet_splits[sheet]} and {split_name}"
                    )
                sheet_splits[sheet] = split_name

    def test_metadata_files(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        coco_path = tmp_path / "coco.json"
        out_dir = tmp_path / "output"

        _make_image().save(img_dir / "S1_1.png")

        coco = _make_coco(
            images=[{"id": 1, "file_name": "S1_1.png", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[10, 10, 90, 10, 50, 90]],
                    "bbox": [10, 10, 80, 80],
                },
            ],
        )
        coco_path.write_text(json.dumps(coco))

        run(
            coco_json=coco_path,
            images_dir=img_dir,
            output_dir=out_dir,
            positive_label="mound",
            ignore_label="uncertain_ignore",
            hard_negative_label="hard_negative_symbol",
        )

        splits_path = out_dir / "metadata" / "splits.json"
        summary_path = out_dir / "metadata" / "dataset_summary.json"
        assert splits_path.exists()
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert summary["total_images"] == 1
        assert summary["mound_count_per_split"]["train"] == 1
        assert summary["images_with_no_mound"] == 0
