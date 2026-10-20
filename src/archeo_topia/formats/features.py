#!/usr/bin/env python3
"""Vector features, and the GIS formats the team reads and writes.

**GeoJSON is written by the standard library.** It is JSON, so a driver buys
nothing, and writing it directly gives exact control over the ``crs`` member --
which matters, because that member is the difference between a file landing in
Bulgaria and landing in the Gulf of Guinea.

**GDAL is a subprocess, never an import.** ``ogr2ogr`` converts GeoJSON to
GeoPackage, reprojects, and reads a returned file back. The Python bindings
must match the system ``libgdal`` exactly and be built against an installed
numpy, or they import and then fail with ``no module named _gdal_array``; the
GDAL section of ``README.md`` documents the recovery. ``gdal-bin`` is installed
by ``make system-deps`` and verified by ``make check-gdal``.

**Why GeoPackage is the working format, measured rather than assumed.**

* It holds ``mound_points`` and ``mound_symbols`` in one file; GeoJSON is
  single-layer.
* It declares EPSG:25835 unambiguously. GDAL's GeoJSON writer emits a legacy
  ``crs`` member that RFC 7946 removed: QGIS honours it, ArcGIS generally does
  not and assumes WGS84.
* It keeps full-length field names. Shapefile truncates at ten characters, so
  ``crossed_by_contour``, ``crossed_by_forestation_line``, ``crossed_by_grid``,
  ``crossed_by_powerline`` and ``crossed_by_road`` all collapse to
  ``crossed_by``, ``crossed__1`` … ``crossed__4``. Shapefile is not offered.
* It round-trips booleans, as ``Integer(Boolean)``, and floats exactly.

The portable copy is GeoJSON reprojected to WGS84 with ``RFC7946=YES``, which
drops the legacy member and is read correctly by both tools.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The CLI tools this module needs, all from the ``gdal-bin`` package.
REQUIRED_BINARIES = ("ogr2ogr", "ogrinfo")

#: WGS84, the only CRS RFC 7946 permits.
WGS84 = "EPSG:4326"


class GdalUnavailableError(RuntimeError):
    """The GDAL command-line tools are not on PATH."""


def require_gdal_cli() -> None:
    """Check the GDAL command-line tools are available.

    Raises a message naming the fix, rather than letting ``subprocess`` raise a
    bare ``FileNotFoundError`` that says nothing about what to install.

    Raises:
        GdalUnavailableError: If any required binary is missing.
    """
    missing = [name for name in REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        raise GdalUnavailableError(
            f"{', '.join(missing)} not found on PATH. Install the GDAL command-line tools "
            "with `make system-deps` (apt: gdal-bin). Do not pip install GDAL to fix this -- "
            "see the GDAL section of README.md."
        )


@dataclass
class Feature:
    """One vector feature.

    Attributes:
        geometry: A GeoJSON geometry dict, ``Point`` or ``Polygon``.
        properties: Flat attributes. Values must be JSON scalars so ``ogr2ogr``
            can infer a column type.
    """

    geometry: dict[str, Any]
    properties: dict[str, Any] = field(default_factory=dict)

    def as_geojson(self) -> dict[str, Any]:
        """Return the feature as a GeoJSON Feature.

        Returns:
            The feature dict.
        """
        return {"type": "Feature", "properties": dict(self.properties), "geometry": self.geometry}


def point(x: float, y: float) -> dict[str, Any]:
    """Build a GeoJSON Point.

    Args:
        x: Easting or longitude.
        y: Northing or latitude.

    Returns:
        The geometry.
    """
    return {"type": "Point", "coordinates": [x, y]}


def polygon(ring: Iterable[tuple[float, float]]) -> dict[str, Any]:
    """Build a GeoJSON Polygon from one exterior ring.

    The ring is closed if it is not already. A Polygon's coordinates are a list
    *of rings*, not a list of positions -- a distinction this project has got
    wrong before, in ``services/sam2_mcp/schemas.py``.

    Args:
        ring: Exterior ring positions.

    Returns:
        The geometry.
    """
    coords = [[float(x), float(y)] for x, y in ring]
    if coords and coords[0] != coords[-1]:
        coords.append(list(coords[0]))
    return {"type": "Polygon", "coordinates": [coords]}


@dataclass
class FeatureCollection:
    """A set of features sharing one CRS.

    Attributes:
        features: The features.
        crs: The CRS these coordinates are in, for example ``EPSG:25835``.
            Never implicit: ``AGENTS.md`` requires that CRS metadata is never
            dropped silently.
        name: Layer name, used when writing a GeoPackage layer.
    """

    features: list[Feature] = field(default_factory=list)
    crs: str | None = None
    name: str = "features"

    def __len__(self) -> int:
        """Return the feature count."""
        return len(self.features)

    def add(self, geometry: dict[str, Any], **properties: Any) -> Feature:
        """Append a feature.

        Args:
            geometry: A GeoJSON geometry.
            **properties: Feature attributes.

        Returns:
            The feature added.
        """
        feature = Feature(geometry=geometry, properties=properties)
        self.features.append(feature)
        return feature

    # ---- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Check every geometry before export.

        Required by ``AGENTS.md``: *"validate geometry validity before export
        where applicable"*. A malformed ring reaches a GIS as a feature that
        silently fails to draw, which is far harder to diagnose than an
        exception here.

        Raises:
            ValueError: On the first invalid feature.
        """
        for index, feature in enumerate(self.features):
            geometry = feature.geometry
            kind = geometry.get("type")
            coords = geometry.get("coordinates")
            if kind == "Point":
                if not isinstance(coords, list) or len(coords) < 2:
                    raise ValueError(f"feature {index}: Point needs two coordinates")
            elif kind == "Polygon":
                if not isinstance(coords, list) or not coords:
                    raise ValueError(f"feature {index}: Polygon needs at least one ring")
                for ring_index, ring in enumerate(coords):
                    if not isinstance(ring, list) or len(ring) < 4:
                        raise ValueError(
                            f"feature {index} ring {ring_index}: a closed ring needs "
                            f"at least four positions, got {len(ring) if isinstance(ring, list) else 'a non-list'}"
                        )
                    if ring[0] != ring[-1]:
                        raise ValueError(f"feature {index} ring {ring_index}: ring is not closed")
            else:
                raise ValueError(f"feature {index}: unsupported geometry type {kind!r}")

    # ---- GeoJSON, standard library only -------------------------------------

    def as_geojson(self) -> dict[str, Any]:
        """Return the collection as a GeoJSON FeatureCollection.

        The ``crs`` member is written when the CRS is not WGS84. RFC 7946
        removed that member, but without it a projected file is read as WGS84
        and lands thousands of kilometres away, so the choice is between a
        non-standard member and silently wrong data. Use
        :meth:`to_geojson_wgs84` for a strictly conformant file.

        Returns:
            The collection dict.
        """
        document: dict[str, Any] = {"type": "FeatureCollection", "name": self.name}
        if self.crs and self.crs.upper() != WGS84:
            code = self.crs.split(":", 1)[1]
            document["crs"] = {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{code}"},
            }
        document["features"] = [feature.as_geojson() for feature in self.features]
        return document

    def to_geojson(self, path: str | Path) -> Path:
        """Write the collection as GeoJSON, in its own CRS.

        Args:
            path: Destination.

        Returns:
            The path written.
        """
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_geojson(), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_geojson(cls, path: str | Path, name: str | None = None) -> FeatureCollection:
        """Read a GeoJSON file.

        Args:
            path: The file.
            name: Layer name override.

        Returns:
            The collection, carrying whatever CRS the file declared.
        """
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_geojson_dict(document, name=name)

    @classmethod
    def from_geojson_dict(
        cls, document: dict[str, Any], name: str | None = None
    ) -> FeatureCollection:
        """Build a collection from a parsed GeoJSON document.

        Args:
            document: The parsed FeatureCollection.
            name: Layer name override.

        Returns:
            The collection.
        """
        crs = None
        member = document.get("crs")
        if isinstance(member, dict):
            declared = member.get("properties", {}).get("name", "")
            if "EPSG" in declared:
                crs = "EPSG:" + declared.rsplit(":", 1)[-1]
        return cls(
            features=[
                Feature(
                    geometry=f.get("geometry") or {}, properties=dict(f.get("properties") or {})
                )
                for f in document.get("features", [])
            ],
            crs=crs,
            name=name or document.get("name", "features"),
        )

    # ---- GeoPackage and reprojection, via ogr2ogr ---------------------------

    def to_geopackage(self, path: str | Path, append: bool = False) -> Path:
        """Write or append this collection as a GeoPackage layer.

        Args:
            path: Destination ``.gpkg``.
            append: Add a layer to an existing file rather than creating one.

        Returns:
            The path written.
        """
        require_gdal_cli()
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "layer.geojson"
            source.write_text(json.dumps(self.as_geojson()), encoding="utf-8")
            command = ["ogr2ogr", "-f", "GPKG"]
            if append and path.exists():
                command += ["-update", "-append"]
            if self.crs:
                command += ["-a_srs", self.crs]
            command += ["-nln", self.name, str(path), str(source)]
            _run(command)
        logger.info("wrote %s layer %s (%d features)", path, self.name, len(self.features))
        return path

    def to_geojson_wgs84(self, path: str | Path) -> Path:
        """Write a strictly RFC 7946 copy, reprojected to WGS84.

        The reprojection is explicit and logged, per the ``AGENTS.md`` rule
        *"reproject explicitly"*. ``RFC7946=YES`` is a **layer** creation
        option; passed as a datasource option GDAL ignores it with a warning
        and silently writes the projected coordinates instead.

        Args:
            path: Destination ``.geojson``.

        Returns:
            The path written.
        """
        require_gdal_cli()
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "layer.geojson"
            source.write_text(json.dumps(self.as_geojson()), encoding="utf-8")
            command = ["ogr2ogr", "-f", "GeoJSON"]
            if self.crs:
                command += ["-s_srs", self.crs]
            command += ["-t_srs", WGS84, "-lco", "RFC7946=YES", str(path), str(source)]
            _run(command)
        logger.info("reprojected %s -> %s, wrote %s", self.crs, WGS84, path)
        return path

    @classmethod
    def from_geopackage(cls, path: str | Path, layer: str) -> FeatureCollection:
        """Read one GeoPackage layer.

        Args:
            path: The ``.gpkg``.
            layer: Layer name.

        Returns:
            The collection.
        """
        require_gdal_cli()
        result = _run(["ogr2ogr", "-f", "GeoJSON", "/vsistdout/", str(path), layer])
        return cls.from_geojson_dict(json.loads(result.stdout), name=layer)

    @staticmethod
    def layers(path: str | Path) -> list[str]:
        """List the layers in a GeoPackage.

        Args:
            path: The ``.gpkg``.

        Returns:
            Layer names.
        """
        require_gdal_cli()
        result = _run(["ogrinfo", "-q", str(path)])
        names = []
        for line in result.stdout.splitlines():
            # ogrinfo -q prints "1: layer_name (Point)"
            if ":" in line:
                names.append(line.split(":", 1)[1].strip().split(" ")[0])
        return names


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a GDAL command, raising with its stderr on failure.

    Args:
        command: The argv.

    Returns:
        The completed process.

    Raises:
        RuntimeError: If the command fails.
    """
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result
