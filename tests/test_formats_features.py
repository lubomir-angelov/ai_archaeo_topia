"""Tests for FeatureCollection, GeoJSON and the GeoPackage round-trip."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from archeo_topia.formats import FeatureCollection
from archeo_topia.formats.features import (
    GdalUnavailableError,
    point,
    polygon,
    require_gdal_cli,
)

HAS_GDAL_CLI = shutil.which("ogr2ogr") is not None
needs_gdal = pytest.mark.skipif(not HAS_GDAL_CLI, reason="needs gdal-bin on PATH")

EAST, NORTH = 148562.312, 4870439.009


def sample() -> FeatureCollection:
    """A collection shaped like the real deliverable."""
    collection = FeatureCollection(crs="EPSG:25835", name="mound_points")
    collection.add(
        point(EAST, NORTH),
        mound_id="K-34-10-A-g-00001",
        sheet_id="K-34-10-A-g",
        detector_confidence=0.878,
        blurred_or_bad_print=False,
        crossed_by_forestation_line="none",
        review_status="unreviewed",
    )
    return collection


class TestGeometryHelpers:
    """A Polygon's coordinates are a list of rings, not a list of positions."""

    def test_polygon_wraps_its_ring(self) -> None:
        geometry = polygon([(0, 0), (1, 0), (1, 1)])
        assert geometry["type"] == "Polygon"
        assert isinstance(geometry["coordinates"][0][0], list)

    def test_polygon_closes_an_open_ring(self) -> None:
        geometry = polygon([(0, 0), (1, 0), (1, 1)])
        ring = geometry["coordinates"][0]
        assert ring[0] == ring[-1]
        assert len(ring) == 4

    def test_an_already_closed_ring_is_not_closed_twice(self) -> None:
        ring = polygon([(0, 0), (1, 0), (1, 1), (0, 0)])["coordinates"][0]
        assert len(ring) == 4


class TestValidation:
    """Invalid geometry must fail here, not silently fail to draw in a GIS."""

    def test_a_valid_collection_passes(self) -> None:
        sample().validate()

    def test_a_point_needs_two_coordinates(self) -> None:
        collection = FeatureCollection(crs="EPSG:25835")
        collection.add({"type": "Point", "coordinates": [1.0]})
        with pytest.raises(ValueError, match="two coordinates"):
            collection.validate()

    def test_an_unclosed_ring_is_rejected(self) -> None:
        collection = FeatureCollection(crs="EPSG:25835")
        collection.add({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]})
        with pytest.raises(ValueError, match="not closed"):
            collection.validate()

    def test_a_flat_ring_is_rejected(self) -> None:
        """The exact shape services/sam2_mcp/schemas.py emits."""
        collection = FeatureCollection(crs="EPSG:25835")
        collection.add({"type": "Polygon", "coordinates": [[0, 0], [1, 0], [1, 1], [0, 0]]})
        with pytest.raises(ValueError, match="at least four positions|non-list"):
            collection.validate()

    def test_an_unsupported_type_is_rejected(self) -> None:
        collection = FeatureCollection(crs="EPSG:25835")
        collection.add({"type": "LineString", "coordinates": [[0, 0], [1, 1]]})
        with pytest.raises(ValueError, match="unsupported geometry type"):
            collection.validate()


class TestGeoJson:
    """Written with the standard library, so the crs member is ours to control."""

    def test_projected_output_declares_its_crs(self, tmp_path: Path) -> None:
        """Without this member a projected file is read as WGS84.

        RFC 7946 removed it, but the alternative is data that lands thousands
        of kilometres from Bulgaria, so a non-standard member beats silently
        wrong coordinates. to_geojson_wgs84 is the conformant path.
        """
        out = sample().to_geojson(tmp_path / "a.geojson")
        document = json.loads(out.read_text())
        assert document["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::25835"
        assert document["features"][0]["geometry"]["coordinates"] == [EAST, NORTH]

    def test_wgs84_collections_omit_the_crs_member(self, tmp_path: Path) -> None:
        collection = FeatureCollection(crs="EPSG:4326", name="m")
        collection.add(point(22.6, 43.9))
        document = json.loads(collection.to_geojson(tmp_path / "b.geojson").read_text())
        assert "crs" not in document

    def test_round_trip_preserves_crs_and_properties(self, tmp_path: Path) -> None:
        out = sample().to_geojson(tmp_path / "c.geojson")
        back = FeatureCollection.from_geojson(out)
        assert back.crs == "EPSG:25835"
        assert len(back) == 1
        assert back.features[0].properties["mound_id"] == "K-34-10-A-g-00001"
        assert back.features[0].properties["detector_confidence"] == 0.878


class TestGdalPreflight:
    """A missing binary must say what to install."""

    def test_message_names_the_fix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("archeo_topia.formats.features.shutil.which", lambda _: None)
        with pytest.raises(GdalUnavailableError) as excinfo:
            require_gdal_cli()
        message = str(excinfo.value)
        assert "make system-deps" in message
        assert "README" in message
        assert "Do not pip install GDAL" in message


@needs_gdal
class TestGeoPackage:
    """The working format, exercised through ogr2ogr as the team's tools do."""

    def test_round_trip_preserves_everything(self, tmp_path: Path) -> None:
        path = sample().to_geopackage(tmp_path / "d.gpkg")
        back = FeatureCollection.from_geopackage(path, "mound_points")
        assert back.crs == "EPSG:25835"
        assert len(back) == 1
        properties = back.features[0].properties
        assert properties["mound_id"] == "K-34-10-A-g-00001"
        assert properties["detector_confidence"] == pytest.approx(0.878)
        assert properties["blurred_or_bad_print"] in (False, 0)
        assert properties["crossed_by_forestation_line"] == "none"
        assert back.features[0].geometry["coordinates"][:2] == pytest.approx([EAST, NORTH])

    def test_field_names_survive_at_full_length(self, tmp_path: Path) -> None:
        """Shapefile truncates these to ten characters; GeoPackage does not."""
        path = sample().to_geopackage(tmp_path / "e.gpkg")
        properties = FeatureCollection.from_geopackage(path, "mound_points").features[0].properties
        assert "crossed_by_forestation_line" in properties
        assert "review_status" in properties

    def test_two_layers_in_one_file(self, tmp_path: Path) -> None:
        path = tmp_path / "f.gpkg"
        sample().to_geopackage(path)
        symbols = FeatureCollection(crs="EPSG:25835", name="mound_symbols")
        symbols.add(
            polygon([(EAST, NORTH), (EAST + 20, NORTH), (EAST + 20, NORTH + 20)]),
            mound_id="K-34-10-A-g-00001",
            symbol_area_px=472.0,
        )
        symbols.to_geopackage(path, append=True)
        assert set(FeatureCollection.layers(path)) == {"mound_points", "mound_symbols"}

    def test_wgs84_export_is_rfc7946(self, tmp_path: Path) -> None:
        """RFC7946=YES is a layer option; as a datasource option it is ignored."""
        out = sample().to_geojson_wgs84(tmp_path / "g.geojson")
        document = json.loads(out.read_text())
        assert "crs" not in document
        lon, lat = document["features"][0]["geometry"]["coordinates"][:2]
        assert lon == pytest.approx(22.6238, abs=1e-3)
        assert lat == pytest.approx(43.9032, abs=1e-3)
