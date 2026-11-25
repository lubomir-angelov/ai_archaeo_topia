#!/usr/bin/env python3
"""The georeferencing that connects pixel space to the ground.

COCO annotations are in image pixels and GIS features are on the ground, so
there is no conversion between them without an affine transform and a CRS.
This class is that transform, and every conversion in this package takes one
as a required argument rather than an optional one.

**Three coordinate spaces, in order.**

``clip pixel``
    Inside one of the four PNG clips a sheet is cut into. What CVAT annotates.
``sheet pixel``
    Inside the parent GeoTIFF. What the detector sweep reports.
``ground``
    Easting and northing in the sheet's own CRS, EPSG:25835 throughout this
    project. What a GIS reads.

The clip-to-ground path has been described in ``sheet_clips.py``, specified in
the v0.6 plan and tested in ``tests/test_sweep_sheets.py`` since it was
written, but until now no code walked it: ``sweep_sheets.load_raster``
discards georeferencing for a ``.png``, and ``export_proposals_coco`` reads
``clips.json`` while ignoring its ``geotransform``, ``bounds`` and ``crs``.

**GDAL is a subprocess, never an import.** Raster metadata comes from
``gdalinfo -json`` via the existing ``sheet_clips.raster_info``. The Python
bindings must match the system ``libgdal`` exactly and be built against an
installed numpy, or they import and then fail with ``no module named
_gdal_array``; ``README.md`` documents the recovery. Shelling out sidesteps
that entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from archeo_topia.datasets.sheet_clips import bounds as _bounds
from archeo_topia.datasets.sheet_clips import clip_geotransform, raster_info

#: The CRS every sheet in this project declares.
DEFAULT_CRS = "EPSG:25835"


@dataclass(frozen=True)
class Clip:
    """One clip's position inside its parent sheet.

    Attributes:
        file: Clip file name, for example ``K-35-21-G-a_3.png``.
        offset_xy: The clip's top-left corner in parent-sheet pixels.
        size: Clip ``(width, height)`` in pixels.
        geotransform: The clip's own GDAL geotransform, or None.
    """

    file: str
    offset_xy: tuple[int, int]
    size: tuple[int, int]
    geotransform: tuple[float, ...] | None = None

    def contains(self, x: float, y: float) -> bool:
        """Return whether a clip-space pixel falls inside this clip.

        Args:
            x: Clip x.
            y: Clip y.

        Returns:
            True if inside.
        """
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]


@dataclass(frozen=True)
class SheetReference:
    """A sheet's geometry: its clips, its geotransform and its CRS.

    Attributes:
        sheet_id: Sheet identifier, for example ``K-35-21-G-a``.
        size: Parent sheet ``(width, height)`` in pixels.
        geotransform: Parent GDAL geotransform ``[ox, px, rx, oy, ry, py]``,
            or None when the source carries no georeferencing.
        crs: CRS string such as ``EPSG:25835``, or None.
        clips: The clips this sheet was cut into, possibly empty.
    """

    sheet_id: str
    size: tuple[int, int]
    geotransform: tuple[float, ...] | None = None
    crs: str | None = None
    clips: tuple[Clip, ...] = field(default_factory=tuple)

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_clips_json(cls, path: str | Path) -> SheetReference:
        """Build a reference from a ``clips.json`` manifest.

        Args:
            path: The manifest, or the directory holding it.

        Returns:
            The reference.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "clips.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        clips = tuple(
            Clip(
                file=entry["file"],
                offset_xy=tuple(entry["offset_xy"]),
                size=tuple(entry["size"]),
                geotransform=tuple(entry["geotransform"]) if entry.get("geotransform") else None,
            )
            for entry in manifest.get("clips", [])
        )
        parent_gt = manifest.get("parent_geotransform")
        return cls(
            sheet_id=manifest["sheet_id"],
            size=tuple(manifest["parent_size"]),
            geotransform=tuple(parent_gt) if parent_gt else None,
            crs=manifest.get("crs"),
            clips=clips,
        )

    @classmethod
    def from_raster(cls, path: str | Path) -> SheetReference:
        """Build a reference from a GeoTIFF, with no clips.

        Reads through ``gdalinfo -json``; see the module docstring for why the
        Python bindings are avoided.

        Args:
            path: The raster.

        Returns:
            The reference.
        """
        path = Path(path)
        info = raster_info(path)
        gt = info.get("geotransform")
        return cls(
            sheet_id=path.stem.replace("_clipped", ""),
            size=tuple(info["size"]),
            geotransform=tuple(gt) if gt else None,
            crs=info.get("crs"),
        )

    # ---- properties ---------------------------------------------------------

    @property
    def georeferenced(self) -> bool:
        """Whether this sheet can produce ground coordinates at all."""
        return self.geotransform is not None

    @property
    def epsg(self) -> int | None:
        """The CRS as an EPSG code, or None."""
        if not self.crs or not self.crs.upper().startswith("EPSG:"):
            return None
        return int(self.crs.split(":", 1)[1])

    @property
    def bounds(self) -> list[float] | None:
        """The sheet's ground extent as ``[minx, miny, maxx, maxy]``, or None."""
        if not self.geotransform:
            return None
        return _bounds(list(self.geotransform), self.size[0], self.size[1])

    def clip(self, name: str) -> Clip:
        """Return one clip by file name.

        Args:
            name: Clip file name.

        Returns:
            The clip.

        Raises:
            KeyError: If no clip has that name.
        """
        for clip in self.clips:
            if clip.file == name:
                return clip
        raise KeyError(f"no clip {name!r} on {self.sheet_id}; have {[c.file for c in self.clips]}")

    # ---- the three hops -----------------------------------------------------

    def clip_to_sheet(self, clip: str, x: float, y: float) -> tuple[float, float]:
        """Map a clip pixel to a parent-sheet pixel.

        Args:
            clip: Clip file name.
            x: Clip x.
            y: Clip y.

        Returns:
            Sheet ``(x, y)``.
        """
        x0, y0 = self.clip(clip).offset_xy
        return (x + x0, y + y0)

    def sheet_to_clip(self, x: float, y: float) -> tuple[str, float, float] | None:
        """Map a parent-sheet pixel into whichever clip contains it.

        Args:
            x: Sheet x.
            y: Sheet y.

        Returns:
            ``(clip file, x, y)``, or None when no clip covers the point.
        """
        for clip in self.clips:
            local_x, local_y = x - clip.offset_xy[0], y - clip.offset_xy[1]
            if clip.contains(local_x, local_y):
                return (clip.file, local_x, local_y)
        return None

    def to_ground(self, x: float, y: float) -> tuple[float, float] | None:
        """Map a parent-sheet pixel to ground coordinates.

        Args:
            x: Sheet x.
            y: Sheet y.

        Returns:
            ``(easting, northing)``, or None when the sheet is not
            georeferenced.
        """
        if not self.geotransform:
            return None
        ox, px, rx, oy, ry, py = self.geotransform
        return (ox + x * px + y * rx, oy + x * ry + y * py)

    def from_ground(self, easting: float, northing: float) -> tuple[float, float] | None:
        """Map ground coordinates back to a parent-sheet pixel.

        Inverts the full six-term affine, so it stays correct for a rotated
        geotransform rather than only for a north-up one.

        Args:
            easting: Easting in the sheet's CRS.
            northing: Northing in the sheet's CRS.

        Returns:
            Sheet ``(x, y)``, or None when the sheet is not georeferenced.

        Raises:
            ValueError: If the geotransform is degenerate and cannot be
                inverted.
        """
        if not self.geotransform:
            return None
        ox, px, rx, oy, ry, py = self.geotransform
        determinant = px * py - rx * ry
        if determinant == 0:
            raise ValueError(f"{self.sheet_id}: geotransform is not invertible")
        dx, dy = easting - ox, northing - oy
        return ((dx * py - dy * rx) / determinant, (dy * px - dx * ry) / determinant)

    def clip_to_ground(self, clip: str, x: float, y: float) -> tuple[float, float] | None:
        """Map a clip pixel straight to ground coordinates.

        Uses the clip's own geotransform when ``clips.json`` recorded one, so a
        clip that has been separated from its parent still resolves. Falls back
        to the parent transform otherwise.

        Args:
            clip: Clip file name.
            x: Clip x.
            y: Clip y.

        Returns:
            ``(easting, northing)``, or None when neither transform exists.
        """
        entry = self.clip(clip)
        if entry.geotransform:
            ox, px, rx, oy, ry, py = entry.geotransform
            return (ox + x * px + y * rx, oy + x * ry + y * py)
        sheet_x, sheet_y = self.clip_to_sheet(clip, x, y)
        return self.to_ground(sheet_x, sheet_y)

    def clip_geotransform_for(self, clip: str) -> list[float] | None:
        """Return a clip's geotransform, deriving it from the parent if absent.

        Args:
            clip: Clip file name.

        Returns:
            The six-term geotransform, or None.
        """
        entry = self.clip(clip)
        if entry.geotransform:
            return list(entry.geotransform)
        if not self.geotransform:
            return None
        return clip_geotransform(list(self.geotransform), *entry.offset_xy)

    def as_dict(self) -> dict[str, Any]:
        """Return the reference as a plain dict, for recording in outputs.

        Returns:
            A JSON-serialisable summary.
        """
        return {
            "sheet_id": self.sheet_id,
            "size": list(self.size),
            "geotransform": list(self.geotransform) if self.geotransform else None,
            "crs": self.crs,
            "bounds": self.bounds,
            "clips": [c.file for c in self.clips],
        }
