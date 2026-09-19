"""Tests for the unlabelled-sheet detector sweep (MapSAM v0.6 step 1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from archeo_topia.analysis.sweep_sheets import (
    discover,
    load_raster,
    to_ground,
    valid_fraction,
)
from archeo_topia.datasets.build_detection_windows import plan_windows

RUN = Path("artifacts/detection/v0_5_yolo26s_gtfix")
DATASET = Path("data/curated/datasets/mapsam_det_v1")
IMAGES = Path("data/curated/datasets/mapsam_v02/images")


class TestValidFraction:
    """The alpha filter that keeps transparent windows out of the denominator."""

    def test_no_alpha_is_fully_valid(self) -> None:
        assert valid_fraction(np.zeros((8, 8, 3), dtype=np.uint8)) == 1.0

    def test_fully_transparent(self) -> None:
        assert valid_fraction(np.zeros((8, 8, 4), dtype=np.uint8)) == 0.0

    def test_half_transparent(self) -> None:
        crop = np.zeros((8, 8, 4), dtype=np.uint8)
        crop[:4, :, 3] = 255
        assert valid_fraction(crop) == 0.5


class TestToGround:
    """Pixel to EPSG:25835, using the same geotransform convention as sheet_clips."""

    def test_without_geotransform(self) -> None:
        assert to_ground(None, 10.0, 20.0) is None

    def test_origin_maps_to_geotransform_origin(self) -> None:
        gt = [499888.86, 2.14, 0.0, 4881099.53, 0.0, -2.14]
        assert to_ground(gt, 0.0, 0.0) == pytest.approx((499888.86, 4881099.53))

    def test_north_up_geotransform(self) -> None:
        gt = [500000.0, 2.0, 0.0, 4880000.0, 0.0, -2.0]
        assert to_ground(gt, 100.0, 50.0) == pytest.approx((500200.0, 4879900.0))

    def test_round_trips_through_the_inverse(self) -> None:
        """A candidate's ground position must come back to the same pixel.

        This is the property the frozen test set depends on: its clips are PNG
        and carry no CRS, so the only route from a detection to a map is the
        recorded geotransform.
        """
        gt = [
            499888.86038794264,
            2.140145991885338,
            0.0,
            4881099.532670771,
            0.0,
            -2.140145991885408,
        ]
        for x, y in ((0.0, 0.0), (1234.5, 987.25), (4681.0, 4327.0)):
            east, north = to_ground(gt, x, y)
            assert (east - gt[0]) / gt[1] == pytest.approx(x)
            assert (north - gt[3]) / gt[5] == pytest.approx(y)


class TestDiscover:
    """Sheet discovery, and the guard around the blind set."""

    def test_single_file(self, tmp_path: Path) -> None:
        raster = tmp_path / "K-35-1-A-a_clipped.tif"
        raster.touch()
        assert discover(raster) == [raster]

    def test_skips_the_frozen_pool(self, tmp_path: Path) -> None:
        """Frozen sheets must never have detector output produced for them.

        Every one of them is going to blind annotation, and the contamination
        is irreversible: once a reviewer has seen a proposal on a sheet, it can
        never serve as an uncontaminated test set again.
        """
        (tmp_path / "01_maps_test").mkdir()
        (tmp_path / "_frozen" / "01_maps_test").mkdir(parents=True)
        working = tmp_path / "01_maps_test" / "K-35-1-A-a_clipped.tif"
        working.touch()
        (tmp_path / "_frozen" / "01_maps_test" / "K-35-22-A-v_clipped.tif").touch()
        assert discover(tmp_path) == [working]


class TestTilingParity:
    """The sweep must lay the same windows as the labelled dataset builder."""

    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [(4971, 4618, 156), (4753, 4351, 143), (4682, 4328, 132)],
    )
    def test_matches_the_input_inventory(self, width: int, height: int, expected: int) -> None:
        """Window counts must match docs/mapsam/v006/INPUT_INVENTORY.md."""
        assert len(plan_windows(width, height, 512, 384)) == expected

    def test_last_window_sits_flush_with_the_edge(self) -> None:
        windows = plan_windows(4971, 4618, 512, 384)
        assert max(w.x0 + w.size for w in windows) == 4971
        assert max(w.y0 + w.size for w in windows) == 4618


@pytest.mark.skipif(
    not (RUN / "foldB" / "weights" / "last.pt").exists() or not DATASET.exists(),
    reason="needs the v0.5 detector artifacts and the mapsam_det_v1 dataset",
)
class TestRegressionAgainstV05:
    """Does the sweep reproduce a number the project already trusts?

    The sweep reaches the same pixels by a different route from
    ``build_detection_windows``: it tiles the clip in memory rather than reading
    pre-cut window PNGs off disk, and hands arrays to Ultralytics instead of
    file paths. Agreement on a known answer is therefore a real test of the new
    path, and it is the reason the 56-sheet numbers can be believed at all.

    Fold B is used because it is the cleanest case -- 34 of 34 mounds found,
    zero false positives at every threshold above 0.05.
    """

    def test_fold_b_reproduces_exactly(self) -> None:
        from ultralytics import YOLO

        from archeo_topia.analysis.detection_metrics import (
            evaluate,
            filter_by_sheet,
            load_annotations,
            merge_detections,
        )
        from archeo_topia.analysis.sweep_sheets import sweep

        sheet = "K-35-8-G-a"
        annotations, negatives, ignores = load_annotations(DATASET / "metadata")
        windows = [
            json.loads(line)
            for line in (DATASET / "metadata" / "windows.jsonl").read_text().splitlines()
        ]
        eval_windows = [w for w in windows if w["sheet_id"] == sheet]
        clips = {p.name: p for p in IMAGES.rglob("*.png")}

        model = YOLO(str(RUN / "foldB" / "weights" / "last.pt"))
        raw = []
        swept = 0
        for name in sorted({w["source_image"] for w in eval_windows}):
            _, sheet_raw, stats = sweep(
                load_raster(clips[name]),
                model,
                window=512,
                stride=384,
                imgsz=512,
                confidence=0.05,
                min_valid_fraction=0.05,
                merge_radius=10.0,
            )
            raw.extend(sheet_raw)
            swept += stats["windows_swept"]

        assert swept == len(eval_windows)

        extents: dict[str, tuple[int, int]] = {}
        for window in eval_windows:
            _, _, x1, y1 = window["window_xyxy"]
            width, height = extents.get(window["source_image"], (0, 0))
            extents[window["source_image"]] = (max(width, x1), max(height, y1))

        reference = json.loads((RUN / "foldB" / "metrics.json").read_text())
        points = {round(p["confidence"], 2): p for p in reference["operating_points"]}

        for threshold, expected in points.items():
            kept = merge_detections([d for d in raw if d.score >= threshold], radius=10.0)
            metrics = evaluate(
                detections=kept,
                annotations=filter_by_sheet(annotations, sheet),
                hard_negatives=filter_by_sheet(negatives, sheet),
                ignore_regions=filter_by_sheet(ignores, sheet),
                window_count=len(eval_windows),
                megapixels=sum(w * h for w, h in extents.values()) / 1e6,
            )
            assert metrics["recall"]["5"]["recall"] == expected["recall_5px"]
            assert metrics["localization_error_px"]["p90"] == expected["p90_error_px"]
            assert metrics["false_positives"]["count"] == expected["false_positives"]
