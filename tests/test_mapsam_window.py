"""Tests for prompt-centred input windows."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from archeo_topia.datasets.mapsam_dataset import MapSamDataset
from archeo_topia.datasets.mapsam_window import (
    Window,
    bbox_into_window,
    compute_window,
    crop_to_window,
    point_into_window,
    source_pixels_per_mask_pixel,
)

IMAGE_H, IMAGE_W = 220, 240


class TestComputeWindow:
    def test_centred_prompt_puts_the_window_around_it(self):
        window = compute_window((120.0, 110.0), (IMAGE_H, IMAGE_W), 100)
        assert (window.x0, window.y0, window.size) == (70, 60, 100)
        assert window.x1 == 170 and window.y1 == 160

    def test_window_always_lies_inside_the_image(self):
        for cx, cy in [
            (120, 110),  # centre
            (0, 110),  # left edge
            (IMAGE_W, 110),  # right edge
            (120, 0),  # top edge
            (120, IMAGE_H),  # bottom edge
            (0, 0),  # corners
            (IMAGE_W, 0),
            (0, IMAGE_H),
            (IMAGE_W, IMAGE_H),
        ]:
            window = compute_window((float(cx), float(cy)), (IMAGE_H, IMAGE_W), 100)
            assert window.x0 >= 0 and window.y0 >= 0
            assert window.x1 <= IMAGE_W and window.y1 <= IMAGE_H

    def test_edge_prompt_clamps_rather_than_padding(self):
        assert compute_window((5.0, 110.0), (IMAGE_H, IMAGE_W), 100).x0 == 0
        assert compute_window((235.0, 110.0), (IMAGE_H, IMAGE_W), 100).x1 == IMAGE_W

    def test_window_larger_than_the_image_shrinks_to_fit(self):
        window = compute_window((120.0, 110.0), (IMAGE_H, IMAGE_W), 9999)
        assert window.size == IMAGE_H  # the shorter side caps the square
        assert window.y0 == 0 and window.y1 == IMAGE_H  # pinned on that side
        assert window.x0 == (IMAGE_W - IMAGE_H) // 2  # still centred on the other

    def test_is_deterministic(self):
        args = ((120.0, 110.0), (IMAGE_H, IMAGE_W), 100)
        assert compute_window(*args) == compute_window(*args)

    def test_rejects_a_non_positive_window(self):
        with pytest.raises(ValueError, match="must be positive"):
            compute_window((10.0, 10.0), (IMAGE_H, IMAGE_W), 0)

    def test_rejects_an_empty_image(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_window((10.0, 10.0), (0, 10), 4)


class TestCropToWindow:
    def _tensor(self, channels: int = 3) -> torch.Tensor:
        values = torch.arange(IMAGE_H * IMAGE_W, dtype=torch.float32).reshape(IMAGE_H, IMAGE_W)
        return values.unsqueeze(0).repeat(channels, 1, 1)

    def test_crop_has_the_window_shape(self):
        cropped = crop_to_window(self._tensor(), Window(70, 60, 100))
        assert cropped.shape == (3, 100, 100)

    def test_crop_takes_the_right_pixels(self):
        cropped = crop_to_window(self._tensor(1), Window(70, 60, 100))
        assert cropped[0, 0, 0].item() == 60 * IMAGE_W + 70

    def test_identical_transform_across_image_and_masks(self):
        window = Window(70, 60, 100)
        image = self._tensor(3)
        target = self._tensor(1)
        ignore = self._tensor(1)
        crops = [crop_to_window(t, window) for t in (image, target, ignore)]
        assert {c.shape[1:] for c in crops} == {(100, 100)}
        assert torch.equal(crops[1], crops[2])
        assert torch.equal(crops[0][0], crops[1][0])

    def test_rejects_a_non_chw_tensor(self):
        with pytest.raises(ValueError, match=r"\(C, H, W\)"):
            crop_to_window(torch.zeros(10, 10), Window(0, 0, 5))

    def test_rejects_a_window_that_does_not_fit(self):
        with pytest.raises(ValueError, match="does not fit"):
            crop_to_window(self._tensor(), Window(200, 200, 100))


class TestCoordinateMapping:
    def test_point_maps_to_the_window_centre_when_not_clamped(self):
        window = compute_window((120.0, 110.0), (IMAGE_H, IMAGE_W), 100)
        mapped = point_into_window((120.0, 110.0), window, 256)
        assert mapped.tolist() == pytest.approx([128.0, 128.0])

    def test_point_is_off_centre_when_the_window_was_clamped(self):
        window = compute_window((10.0, 110.0), (IMAGE_H, IMAGE_W), 100)
        mapped = point_into_window((10.0, 110.0), window, 256)
        assert mapped[0].item() == pytest.approx(25.6)
        assert mapped[0].item() < 128.0

    def test_mapped_point_stays_inside_the_output(self):
        for cx, cy in [(0, 0), (IMAGE_W, IMAGE_H), (120, 110), (5, 215)]:
            window = compute_window((float(cx), float(cy)), (IMAGE_H, IMAGE_W), 100)
            mapped = point_into_window((float(cx), float(cy)), window, 256)
            clamped_x = min(max(cx, window.x0), window.x1)
            clamped_y = min(max(cy, window.y0), window.y1)
            assert 0.0 <= mapped[0].item() <= 256.0 or cx != clamped_x
            assert 0.0 <= mapped[1].item() <= 256.0 or cy != clamped_y

    def test_bbox_maps_consistently_with_its_own_centre(self):
        window = compute_window((120.0, 110.0), (IMAGE_H, IMAGE_W), 100)
        box = bbox_into_window((110.0, 100.0, 130.0, 120.0), window, 256)
        point = point_into_window((120.0, 110.0), window, 256)
        assert ((box[0] + box[2]) / 2).item() == pytest.approx(point[0].item())
        assert ((box[1] + box[3]) / 2).item() == pytest.approx(point[1].item())

    def test_bbox_is_clamped_into_the_output(self):
        window = compute_window((5.0, 5.0), (IMAGE_H, IMAGE_W), 100)
        box = bbox_into_window((-20.0, -20.0, 10.0, 10.0), window, 256)
        assert box[0].item() == 0.0 and box[1].item() == 0.0
        assert all(0.0 <= v <= 256.0 for v in box.tolist())

    def test_a_tighter_window_magnifies_the_prompt(self):
        wide = compute_window((120.0, 110.0), (IMAGE_H, IMAGE_W), 200)
        tight = compute_window((120.0, 110.0), (IMAGE_H, IMAGE_W), 50)
        box = (115.0, 105.0, 125.0, 115.0)
        wide_w = (bbox_into_window(box, wide, 256)[2] - bbox_into_window(box, wide, 256)[0]).item()
        tight_box = bbox_into_window(box, tight, 256)
        assert (tight_box[2] - tight_box[0]).item() > wide_w


class TestSourcePixelsPerMaskPixel:
    def test_a_tighter_window_means_finer_mask_pixels(self):
        assert source_pixels_per_mask_pixel(Window(0, 0, 512), 256) < source_pixels_per_mask_pixel(
            Window(0, 0, 2400), 256
        )

    def test_known_value(self):
        # A 512 px window on a 256 px mask: each mask pixel is 2x2 source px.
        assert source_pixels_per_mask_pixel(Window(0, 0, 512), 256) == pytest.approx(4.0)

    def test_rejects_a_non_positive_mask_size(self):
        with pytest.raises(ValueError, match="must be positive"):
            source_pixels_per_mask_pixel(Window(0, 0, 512), 0)


def _write_dataset(root, mounds: list[tuple[int, int]], window_center: tuple[int, int]):
    """Write a minimal on-disk dataset with square mounds.

    Args:
        root: Dataset root directory.
        mounds: ``(x, y)`` centres of 9x9 square mounds.
        window_center: Which mound centre the single sample prompts.

    Returns:
        Path to the samples manifest.
    """
    for sub in ("images/train", "masks/train", "ignore_masks/train"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    rgb = np.full((IMAGE_H, IMAGE_W, 3), 40, dtype=np.uint8)
    mask = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)
    for x, y in mounds:
        rgb[y - 4 : y + 5, x - 4 : x + 5] = 200
        mask[y - 4 : y + 5, x - 4 : x + 5] = 255

    Image.fromarray(rgb).save(root / "images/train/SHEET_1.png")
    Image.fromarray(mask).save(root / "masks/train/SHEET_1.png")
    Image.fromarray(np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)).save(
        root / "ignore_masks/train/SHEET_1.png"
    )

    cx, cy = window_center
    manifest = root / "samples.jsonl"
    manifest.write_text(
        '{"sample_id": "train_SHEET_1_000001", "split": "train", "sheet_id": "SHEET", '
        '"image_path": "images/train/SHEET_1.png", "mask_path": "masks/train/SHEET_1.png", '
        '"ignore_mask_path": "ignore_masks/train/SHEET_1.png", "component_index": 1, '
        '"component_area": 81, '
        f'"bbox": [{cx - 4}, {cy - 4}, {cx + 5}, {cy + 5}], "center_point": [{cx}, {cy}]}}\n',
        encoding="utf-8",
    )
    return manifest


class TestDatasetWindowing:
    def test_window_off_reproduces_the_v0_2_shapes(self, tmp_path):
        manifest = _write_dataset(tmp_path, [(120, 110)], (120, 110))
        sample = MapSamDataset(tmp_path, manifest, "train", image_size=128)[0]
        assert sample["image"].shape == (3, 128, 128)
        assert sample["window_xyxy"].tolist() == [0, 0, IMAGE_W, IMAGE_H]

    def test_window_magnifies_the_target(self, tmp_path):
        manifest = _write_dataset(tmp_path, [(120, 110)], (120, 110))
        without = MapSamDataset(tmp_path, manifest, "train", image_size=128)[0]
        with_window = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=40)[0]
        assert with_window["target_mask"].sum() > without["target_mask"].sum()

    def test_window_is_recorded_for_mapping_predictions_back(self, tmp_path):
        manifest = _write_dataset(tmp_path, [(120, 110)], (120, 110))
        sample = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=40)[0]
        assert sample["window_xyxy"].tolist() == [100.0, 90.0, 140.0, 130.0]

    def test_target_stays_non_empty_and_single_component(self, tmp_path):
        manifest = _write_dataset(tmp_path, [(120, 110)], (120, 110))
        sample = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=60)[0]
        assert sample["target_mask"].sum() > 0
        from archeo_topia.datasets.mapsam_dataset import count_connected_components

        assert count_connected_components(sample["target_mask"].squeeze(0).numpy() > 0.5) == 1

    def test_a_neighbouring_mound_inside_the_window_is_not_in_the_target(self, tmp_path):
        # Two mounds 20 px apart, both well inside an 80 px window.
        manifest = _write_dataset(tmp_path, [(110, 110), (130, 110)], (110, 110))
        sample = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=80)[0]
        from archeo_topia.datasets.mapsam_dataset import count_connected_components

        assert count_connected_components(sample["target_mask"].squeeze(0).numpy() > 0.5) == 1

    def test_image_target_and_ignore_share_the_window(self, tmp_path):
        manifest = _write_dataset(tmp_path, [(120, 110)], (120, 110))
        sample = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=60)[0]
        assert sample["image"].shape[1:] == sample["target_mask"].shape[1:]
        assert sample["target_mask"].shape == sample["ignore_mask"].shape

    def test_prompt_lands_on_the_target_after_windowing(self, tmp_path):
        manifest = _write_dataset(tmp_path, [(120, 110)], (120, 110))
        sample = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=60)[0]
        px, py = (int(round(v)) for v in sample["point_prompt"].tolist())
        assert sample["target_mask"][0, py, px] > 0.5

    @pytest.mark.parametrize(
        "center",
        [(120, 110), (10, 110), (230, 110), (120, 10), (120, 210), (10, 10), (230, 210)],
    )
    def test_prompt_lands_on_the_target_at_every_boundary(self, tmp_path, center):
        manifest = _write_dataset(tmp_path, [center], center)
        sample = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=60)[0]
        px, py = (int(round(v)) for v in sample["point_prompt"].tolist())
        assert 0 <= px < 128 and 0 <= py < 128
        assert sample["target_mask"][0, py, px] > 0.5

    def test_windowed_samples_are_deterministic(self, tmp_path):
        manifest = _write_dataset(tmp_path, [(120, 110)], (120, 110))
        first = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=60)[0]
        second = MapSamDataset(tmp_path, manifest, "train", image_size=128, window_px=60)[0]
        assert torch.equal(first["image"], second["image"])
        assert torch.equal(first["target_mask"], second["target_mask"])
        assert first["sample_id"] == second["sample_id"]
