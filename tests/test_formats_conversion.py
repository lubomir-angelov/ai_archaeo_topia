"""COCO to features and back, and the guard keeping osgeo out of the package."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from archeo_topia.formats import CocoDocument, FeatureCollection, LabelSchema, SheetReference
from archeo_topia.formats.coco import MOUND

GT = (500000.0, 2.0, 0.0, 4880000.0, 0.0, -2.0)
PACKAGE = Path("src/archeo_topia/formats")


def reference(tmp_path: Path) -> SheetReference:
    """A two-by-two sheet with a north-up transform."""
    manifest = {
        "sheet_id": "K-35-1-A-a",
        "parent_size": [200, 160],
        "parent_geotransform": list(GT),
        "crs": "EPSG:25835",
        "clips": [
            {"file": "K-35-1-A-a_1.png", "offset_xy": [0, 0], "size": [100, 80]},
            {"file": "K-35-1-A-a_2.png", "offset_xy": [100, 0], "size": [100, 80]},
            {"file": "K-35-1-A-a_3.png", "offset_xy": [0, 80], "size": [100, 80]},
            {"file": "K-35-1-A-a_4.png", "offset_xy": [100, 80], "size": [100, 80]},
        ],
    }
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(manifest))
    return SheetReference.from_clips_json(path)


def document() -> CocoDocument:
    """Two mounds, one per clip, with polygons."""
    coco = CocoDocument.empty("test")
    for n in (1, 4):
        coco.add_image(f"K-35-1-A-a_{n}.png", 100, 80)
    coco.add_annotation(
        1,
        MOUND,
        [10.0, 10.0, 20.0, 20.0],
        [[10, 10, 30, 10, 30, 30, 10, 30]],
        {"blurred_or_bad_print": True},
    )
    coco.add_annotation(
        2,
        MOUND,
        [40.0, 30.0, 20.0, 20.0],
        [[40, 30, 60, 30, 60, 50, 40, 50]],
        {"blurred_or_bad_print": False},
    )
    return coco


class TestTheReferenceIsRequired:
    """COCO is in pixels and features are on the ground."""

    def test_an_ungeoreferenced_sheet_refuses_rather_than_guessing(self) -> None:
        bare = SheetReference(sheet_id="x", size=(10, 10))
        with pytest.raises(ValueError, match="carries no geotransform"):
            document().to_features(bare)

    def test_an_unknown_kind_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be 'point' or 'polygon'"):
            document().to_features(reference(tmp_path), kind="line")

    def test_features_cannot_be_placed_without_clips(self) -> None:
        bare = SheetReference(sheet_id="x", size=(10, 10), geotransform=GT, crs="EPSG:25835")
        collection = FeatureCollection(crs="EPSG:25835")
        with pytest.raises(ValueError, match="no clips recorded"):
            CocoDocument.from_features(collection, bare)


class TestToFeatures:
    """Projecting annotations onto the ground."""

    def test_points_land_where_the_transform_says(self, tmp_path: Path) -> None:
        ref = reference(tmp_path)
        collection = document().to_features(ref, kind="point")
        assert collection.crs == "EPSG:25835"
        assert collection.name == "mound_points"
        assert len(collection) == 2
        # clip 1 box centre (20, 20) is sheet (20, 20)
        assert collection.features[0].geometry["coordinates"] == pytest.approx(
            [GT[0] + 20 * 2.0, GT[3] - 20 * 2.0]
        )

    def test_the_second_clips_offset_is_applied(self, tmp_path: Path) -> None:
        ref = reference(tmp_path)
        collection = document().to_features(ref, kind="point")
        # clip 4 box centre (50, 40) is sheet (150, 120)
        assert collection.features[1].geometry["coordinates"] == pytest.approx(
            [GT[0] + 150 * 2.0, GT[3] - 120 * 2.0]
        )

    def test_mound_ids_are_stable_and_prefixed(self, tmp_path: Path) -> None:
        collection = document().to_features(reference(tmp_path), kind="point")
        assert [f.properties["mound_id"] for f in collection.features] == [
            "K-35-1-A-a-00001",
            "K-35-1-A-a-00002",
        ]

    def test_attributes_are_coerced_through_the_schema(self, tmp_path: Path) -> None:
        schema = LabelSchema.load("annotation/cvat/labels.json")
        collection = document().to_features(reference(tmp_path), kind="point", schema=schema)
        properties = collection.features[0].properties
        assert properties["blurred_or_bad_print"] is True
        assert properties["review_status"] == "unreviewed"
        assert properties["water_line_crossing"] == "none"

    def test_polygons_become_closed_rings_on_the_ground(self, tmp_path: Path) -> None:
        collection = document().to_features(reference(tmp_path), kind="polygon")
        assert collection.name == "mound_symbols"
        assert len(collection) == 2
        ring = collection.features[0].geometry["coordinates"][0]
        assert ring[0] == ring[-1]
        collection.validate()

    def test_the_two_layers_join_on_mound_id(self, tmp_path: Path) -> None:
        ref = reference(tmp_path)
        coco = document()
        points = coco.to_features(ref, kind="point")
        symbols = coco.to_features(ref, kind="polygon")
        assert [f.properties["mound_id"] for f in points.features] == [
            f.properties["mound_id"] for f in symbols.features
        ]


class TestFromFeatures:
    """The return path, after a reviewer has edited in QGIS or ArcGIS."""

    def test_round_trip_returns_the_same_pixels(self, tmp_path: Path) -> None:
        ref = reference(tmp_path)
        original = document()
        collection = original.to_features(ref, kind="point")
        back = CocoDocument.from_features(collection, ref)

        assert len(back.annotations) == 2
        names = {image["id"]: image["file_name"] for image in back.images}
        first = back.annotations[0]
        assert names[first["image_id"]] == "K-35-1-A-a_1.png"
        x, y, w, h = first["bbox"]
        assert (x + w / 2, y + h / 2) == pytest.approx((20.0, 20.0))

    def test_review_status_survives_the_round_trip(self, tmp_path: Path) -> None:
        ref = reference(tmp_path)
        schema = LabelSchema.load("annotation/cvat/labels.json")
        collection = document().to_features(ref, kind="point", schema=schema)
        collection.features[0].properties["review_status"] = "confirmed"
        collection.features[1].properties["review_status"] = "rejected"
        back = CocoDocument.from_features(collection, ref, schema=schema)
        assert [a["attributes"]["review_status"] for a in back.annotations] == [
            "confirmed",
            "rejected",
        ]

    def test_a_feature_outside_every_clip_is_dropped_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A reviewer can drag a point off the sheet; that must not crash."""
        ref = reference(tmp_path)
        collection = FeatureCollection(crs="EPSG:25835")
        collection.add(
            {"type": "Point", "coordinates": [GT[0] + 10_000, GT[3] - 10_000]},
            mound_id="stray",
        )
        with caplog.at_level("WARNING"):
            back = CocoDocument.from_features(collection, ref)
        assert back.annotations == []
        assert "outside every clip" in caplog.text


class TestNoGdalBindings:
    """The formats package must never import osgeo.

    The Python bindings must match the system libgdal exactly and be built
    against an installed numpy, or they import and then fail with
    `no module named _gdal_array`. README.md documents the recovery. This
    package shells out to gdal-bin instead, and this test stops that decision
    quietly eroding.
    """

    def test_no_module_imports_osgeo(self) -> None:
        offenders = []
        for path in sorted(PACKAGE.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name.split(".")[0] == "osgeo" for name in names):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], f"osgeo imported at {offenders}"
