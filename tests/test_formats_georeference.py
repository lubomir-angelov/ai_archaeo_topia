"""Tests for SheetReference -- the pixel-to-ground hop the package is built on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archeo_topia.formats import SheetReference
from archeo_topia.formats.georeference import Clip

# A real geotransform, from L-35-139-V-v's clips.json.
GT = (
    499888.86038794264,
    2.140145991885338,
    0.0,
    4881099.532670771,
    0.0,
    -2.140145991885408,
)


def manifest(tmp_path: Path) -> Path:
    """Write a clips.json shaped exactly like sheet_clips writes one."""
    document = {
        "sheet_id": "K-35-1-A-a",
        "parent": "K-35-1-A-a_clipped.tif",
        "parent_size": [200, 160],
        "parent_geotransform": list(GT),
        "crs": "EPSG:25835",
        "method": "programmatic 2x2 grid",
        "clips": [
            {"file": "K-35-1-A-a_1.png", "offset_xy": [0, 0], "size": [100, 80]},
            {"file": "K-35-1-A-a_2.png", "offset_xy": [100, 0], "size": [100, 80]},
            {"file": "K-35-1-A-a_3.png", "offset_xy": [0, 80], "size": [100, 80]},
            {"file": "K-35-1-A-a_4.png", "offset_xy": [100, 80], "size": [100, 80]},
        ],
    }
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(document))
    return path


class TestFromClipsJson:
    """Reading the manifest sheet_clips writes."""

    def test_reads_sheet_geometry(self, tmp_path: Path) -> None:
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        assert ref.sheet_id == "K-35-1-A-a"
        assert ref.size == (200, 160)
        assert ref.crs == "EPSG:25835"
        assert ref.epsg == 25835
        assert ref.georeferenced is True
        assert [c.file for c in ref.clips] == [f"K-35-1-A-a_{n}.png" for n in (1, 2, 3, 4)]

    def test_accepts_a_directory(self, tmp_path: Path) -> None:
        manifest(tmp_path)
        assert SheetReference.from_clips_json(tmp_path).sheet_id == "K-35-1-A-a"

    def test_a_sheet_without_a_geotransform_says_so(self) -> None:
        ref = SheetReference(sheet_id="x", size=(10, 10))
        assert ref.georeferenced is False
        assert ref.to_ground(1, 1) is None
        assert ref.epsg is None


class TestTheThreeHops:
    """clip pixel -> sheet pixel -> ground, and back."""

    def test_clip_to_sheet_adds_the_offset(self, tmp_path: Path) -> None:
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        assert ref.clip_to_sheet("K-35-1-A-a_4.png", 50.0, 40.0) == (150.0, 120.0)

    def test_sheet_to_clip_finds_the_right_clip(self, tmp_path: Path) -> None:
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        assert ref.sheet_to_clip(150.0, 120.0) == ("K-35-1-A-a_4.png", 50.0, 40.0)

    def test_a_point_outside_every_clip_returns_none(self, tmp_path: Path) -> None:
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        assert ref.sheet_to_clip(999.0, 999.0) is None

    def test_clip_boundaries_do_not_double_count(self, tmp_path: Path) -> None:
        """A pixel on a seam belongs to exactly one clip."""
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        assert ref.sheet_to_clip(100.0, 80.0) == ("K-35-1-A-a_4.png", 0.0, 0.0)

    def test_pixel_zero_lands_on_the_geotransform_origin(self, tmp_path: Path) -> None:
        """The classic silent GIS error, asserted against.

        These GeoTIFFs carry AREA_OR_POINT=Area, so the geotransform maps pixel
        *corners*. A half-pixel shift here would displace every detection by
        about a metre on the ground and nothing would fail loudly.
        """
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        assert ref.to_ground(0.0, 0.0) == pytest.approx((GT[0], GT[3]))

    def test_ground_round_trips_back_to_the_same_pixel(self, tmp_path: Path) -> None:
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        for x, y in ((0.0, 0.0), (123.5, 87.25), (199.0, 159.0)):
            east, north = ref.to_ground(x, y)
            assert ref.from_ground(east, north) == pytest.approx((x, y))

    def test_clip_to_ground_matches_the_two_step_path(self, tmp_path: Path) -> None:
        """The shortcut and the long way round must agree."""
        ref = SheetReference.from_clips_json(manifest(tmp_path))
        direct = ref.clip_to_ground("K-35-1-A-a_4.png", 50.0, 40.0)
        stepwise = ref.to_ground(*ref.clip_to_sheet("K-35-1-A-a_4.png", 50.0, 40.0))
        assert direct == pytest.approx(stepwise)

    def test_a_clips_own_geotransform_is_preferred(self, tmp_path: Path) -> None:
        """A clip separated from its parent must still resolve.

        This is the property the frozen test set depends on: its clips are PNG
        and carry no CRS of their own.
        """
        ref = SheetReference(
            sheet_id="s",
            size=(200, 160),
            geotransform=None,
            crs="EPSG:25835",
            clips=(Clip("a.png", (100, 80), (100, 80), GT),),
        )
        assert ref.georeferenced is False
        assert ref.clip_to_ground("a.png", 0.0, 0.0) == pytest.approx((GT[0], GT[3]))


class TestRotatedGeotransform:
    """The inverse must be the full six-term one, not a north-up shortcut."""

    def test_round_trip_survives_rotation(self) -> None:
        rotated = (500000.0, 2.0, 0.5, 4880000.0, 0.3, -2.0)
        ref = SheetReference(sheet_id="r", size=(100, 100), geotransform=rotated, crs="EPSG:25835")
        for x, y in ((10.0, 20.0), (99.0, 1.0)):
            east, north = ref.to_ground(x, y)
            assert ref.from_ground(east, north) == pytest.approx((x, y))

    def test_a_degenerate_transform_is_rejected(self) -> None:
        ref = SheetReference(
            sheet_id="d", size=(10, 10), geotransform=(0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
        )
        with pytest.raises(ValueError, match="not invertible"):
            ref.from_ground(1.0, 1.0)


SWEEP = Path("artifacts/detection/v0_6_sweep/foldC_last")
LAKE = Path("/mnt/c/Users/lubom/ai_archaeo_topia/data_lake")


@pytest.mark.skipif(not SWEEP.exists(), reason="needs the v0.6 sweep artifacts")
class TestAgainstRealSweepData:
    """Two code paths, one answer.

    sweep_sheets computed easting/northing from the parent GeoTIFF at write
    time. SheetReference recomputes it from the same raster. They must agree
    exactly, or one of them is wrong.
    """

    def test_reproduces_recorded_ground_coordinates(self) -> None:
        sheet_dir = next(d for d in sorted(SWEEP.iterdir()) if d.is_dir())
        records = [
            json.loads(line)
            for line in (sheet_dir / "candidates.jsonl").read_text().splitlines()
            if line
        ]
        raster = next(
            (
                p
                for p in (LAKE / "raw/mound_test_20260915").rglob(f"{sheet_dir.name}_clipped.tif")
                if "_frozen" not in p.parts
            ),
            None,
        )
        if raster is None:
            pytest.skip("parent raster not reachable")

        ref = SheetReference.from_raster(raster)
        assert ref.crs == records[0]["crs"]
        for record in records[:25]:
            east, north = ref.to_ground(record["x"], record["y"])
            assert round(east, 3) == record["easting"]
            assert round(north, 3) == record["northing"]
