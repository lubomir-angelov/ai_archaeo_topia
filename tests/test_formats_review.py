"""Tests for the return path: a reviewed GeoPackage becoming evidence."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from archeo_topia.formats import FeatureCollection, SheetReference
from archeo_topia.formats.features import point
from archeo_topia.formats.ingest_review import (
    ADDED,
    CONFIRMED,
    CORRECTED,
    REJECTED,
    check,
    to_coco,
)
from archeo_topia.formats.labels import REVIEW_STATUS, LabelSchema

HAS_GDAL_CLI = shutil.which("ogr2ogr") is not None
GT = (500000.0, 2.0, 0.0, 4880000.0, 0.0, -2.0)


def reference(tmp_path: Path) -> SheetReference:
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


def delivered() -> FeatureCollection:
    """Four proposals, as exported."""
    collection = FeatureCollection(crs="EPSG:25835", name="mound_points")
    for index, (x, y) in enumerate([(10, 10), (30, 30), (50, 50), (70, 70)], start=1):
        collection.add(
            point(GT[0] + x * 2.0, GT[3] - y * 2.0),
            mound_id=f"K-35-1-A-a-{index:05d}",
            sheet_id="K-35-1-A-a",
            detector_confidence=0.9,
            review_status="unreviewed",
        )
    return collection


class TestVerdicts:
    """The four outcomes a reviewer can record."""

    def test_a_clean_review_counts_every_verdict(self) -> None:
        original = delivered()
        returned = copy.deepcopy(original)
        for feature, status in zip(
            returned.features, [CONFIRMED, CONFIRMED, REJECTED, CORRECTED], strict=True
        ):
            feature.properties[REVIEW_STATUS] = status
        report = check(returned, original)
        assert report.sound
        assert report.counts == {CONFIRMED: 2, CORRECTED: 1, REJECTED: 1}
        assert report.precision == pytest.approx(0.75)

    def test_an_added_feature_is_recognised_without_an_id(self) -> None:
        original = delivered()
        returned = copy.deepcopy(original)
        for feature in returned.features:
            feature.properties[REVIEW_STATUS] = CONFIRMED
        returned.add(
            point(GT[0] + 180.0, GT[3] - 180.0), sheet_id="K-35-1-A-a", review_status=ADDED
        )
        report = check(returned, original)
        assert report.counts[ADDED] == 1
        assert report.recall_floor == pytest.approx(4 / 5)

    def test_recall_is_reported_as_a_floor(self) -> None:
        """Reviewers see proposals, so they do not search exhaustively."""
        original = delivered()
        returned = copy.deepcopy(original)
        for feature in returned.features:
            feature.properties[REVIEW_STATUS] = CONFIRMED
        report = check(returned, original)
        assert "floor" in report.as_dict()["scope"]
        assert "frozen blind test" in report.as_dict()["scope"]


class TestIntegrity:
    """A hand-edited file will have surprises; count nothing until they are checked."""

    def test_a_deleted_feature_is_reported_not_silently_lost(self) -> None:
        """The failure mode review_status exists to prevent."""
        original = delivered()
        returned = copy.deepcopy(original)
        returned.features.pop()
        for feature in returned.features:
            feature.properties[REVIEW_STATUS] = CONFIRMED
        report = check(returned, original)
        assert not report.sound
        assert any("absent from the return" in p for p in report.problems)
        assert any("rejected rather than deleting" in p for p in report.problems)

    def test_a_duplicated_mound_id_is_rejected(self) -> None:
        original = delivered()
        returned = copy.deepcopy(original)
        returned.features.append(copy.deepcopy(returned.features[0]))
        for feature in returned.features:
            feature.properties[REVIEW_STATUS] = CONFIRMED
        report = check(returned, original)
        assert any("duplicated mound_id" in p for p in report.problems)

    def test_a_changed_crs_is_caught(self) -> None:
        original = delivered()
        returned = copy.deepcopy(original)
        returned.crs = "EPSG:4326"
        report = check(returned, original)
        assert any("CRS changed on save" in p for p in report.problems)

    def test_unreviewed_features_are_flagged_and_excluded(self) -> None:
        original = delivered()
        returned = copy.deepcopy(original)
        returned.features[0].properties[REVIEW_STATUS] = CONFIRMED
        report = check(returned, original)
        assert any("still marked unreviewed" in p for p in report.problems)
        assert report.precision == pytest.approx(1.0)

    def test_a_feature_moved_off_the_sheet_is_caught(self, tmp_path: Path) -> None:
        original = delivered()
        returned = copy.deepcopy(original)
        for feature in returned.features:
            feature.properties[REVIEW_STATUS] = CONFIRMED
        returned.features[0].geometry = point(GT[0] + 10_000.0, GT[3] - 10_000.0)
        report = check(returned, original, reference(tmp_path))
        assert any("outside the sheet" in p for p in report.problems)

    def test_a_nudged_geometry_is_recorded_not_treated_as_an_error(self) -> None:
        original = delivered()
        returned = copy.deepcopy(original)
        for feature in returned.features:
            feature.properties[REVIEW_STATUS] = CONFIRMED
        east, north = returned.features[1].geometry["coordinates"]
        returned.features[1].geometry = point(east + 40.0, north)
        report = check(returned, original)
        assert report.moved == {"K-35-1-A-a-00002": pytest.approx(40.0)}
        assert report.sound


class TestToCoco:
    """Verdicts landing in CVAT beside the existing annotations."""

    def test_only_accepted_verdicts_become_annotations(self, tmp_path: Path) -> None:
        schema = LabelSchema.load("annotation/cvat/labels.json")
        original = delivered()
        returned = copy.deepcopy(original)
        for feature, status in zip(
            returned.features, [CONFIRMED, REJECTED, CORRECTED, "unreviewed"], strict=True
        ):
            feature.properties[REVIEW_STATUS] = status
        document = to_coco(returned, reference(tmp_path), schema)
        assert len(document.annotations) == 2
        assert {a["attributes"][REVIEW_STATUS] for a in document.annotations} == {
            CONFIRMED,
            CORRECTED,
        }

    def test_annotations_land_in_the_right_clip(self, tmp_path: Path) -> None:
        schema = LabelSchema.load("annotation/cvat/labels.json")
        returned = delivered()
        for feature in returned.features:
            feature.properties[REVIEW_STATUS] = CONFIRMED
        document = to_coco(returned, reference(tmp_path), schema)
        names = {image["id"]: image["file_name"] for image in document.images}
        # every delivered point is inside clip 1 (sheet pixels 10..70)
        assert {names[a["image_id"]] for a in document.annotations} == {"K-35-1-A-a_1.png"}


@pytest.mark.skipif(not HAS_GDAL_CLI, reason="needs gdal-bin on PATH")
class TestThroughRealFiles:
    """The whole loop, through GeoPackages as the team will actually send them."""

    def test_export_edit_and_ingest(self, tmp_path: Path) -> None:
        sent = tmp_path / "sent.gpkg"
        delivered().to_geopackage(sent)

        # Simulate the reviewer: confirm two, reject one, correct one, add one.
        edited = FeatureCollection.from_geopackage(sent, "mound_points")
        for feature, status in zip(
            edited.features, [CONFIRMED, CONFIRMED, REJECTED, CORRECTED], strict=True
        ):
            feature.properties[REVIEW_STATUS] = status
        edited.add(
            point(GT[0] + 180.0, GT[3] - 180.0),
            mound_id="",
            sheet_id="K-35-1-A-a",
            detector_confidence=0.0,
            review_status=ADDED,
        )
        back = tmp_path / "back.gpkg"
        edited.to_geopackage(back)

        report = check(
            FeatureCollection.from_geopackage(back, "mound_points"),
            FeatureCollection.from_geopackage(sent, "mound_points"),
            reference(tmp_path),
        )
        assert report.sound, report.problems
        assert report.counts == {ADDED: 1, CONFIRMED: 2, CORRECTED: 1, REJECTED: 1}
        assert report.precision == pytest.approx(0.75)
        assert report.recall_floor == pytest.approx(0.75)
