#!/usr/bin/env python3
"""Cut a georeferenced map sheet into clips, and record how to get back.

The annotation convention on this project is four clips per 1:25k sheet, stored
as RGBA PNGs named ``<sheet>_1.png`` … ``<sheet>_4.png``. PNG carries no CRS, so
a clip on its own cannot be mapped back to the ground — which matters most for
the frozen test set, whose results are the ones worth showing on a map.

Both subcommands therefore write a ``clips.json`` next to the clips recording
the parent raster, each clip's pixel offset within it, and each clip's own
geotransform. With that file a detection at clip pixel ``(x, y)`` converts to
EPSG:25835 without the parent being present.

``split``
    Cut a GeoTIFF into an even grid of clips. Offsets are exact by
    construction.

``index``
    Write ``clips.json`` for clips that already exist, recovering each offset by
    template-matching a patch against the parent. The historical clips on this
    project were cut in GIS rather than programmatically, so their boundaries
    are irregular by a few pixels and cannot be reconstructed by arithmetic —
    but they are exact integer translations of parent pixels, so matching finds
    them reliably.

Usage:
    python -m archeo_topia.datasets.sheet_clips split \\
        --source <sheet>_clipped.tif --output-dir <dir>/<sheet> --grid 2x2

    python -m archeo_topia.datasets.sheet_clips index \\
        --clips-dir <dir>/<sheet> --parent <sheet>_clipped.tif
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Sheets are ~23 Mpx; well above Pillow's decompression-bomb default.
Image.MAX_IMAGE_PIXELS = None

#: Side of the square patch used to locate an existing clip in its parent.
MATCH_PATCH = 150


def raster_info(path: Path) -> dict[str, Any]:
    """Read size, CRS and geotransform from a raster via ``gdalinfo -json``.

    The Python ``osgeo`` bindings are not installed in this project's
    virtualenv, but the ``gdalinfo`` binary is, so shell out rather than import.

    Args:
        path: Raster path.

    Returns:
        Dict with ``size``, ``geotransform`` and ``crs``.

    Raises:
        RuntimeError: If gdalinfo fails.
    """
    result = subprocess.run(
        ["gdalinfo", "-json", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"gdalinfo failed on {path}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    crs = None
    wkt = info.get("coordinateSystem", {}).get("wkt", "")
    if "EPSG" in wkt:
        crs = "EPSG:" + wkt.rsplit('"EPSG",', 1)[-1].split("]")[0].strip()
    return {
        "size": info["size"],
        "geotransform": info.get("geoTransform"),
        "crs": crs,
    }


def clip_geotransform(parent_gt: list[float], x0: int, y0: int) -> list[float]:
    """Shift a parent geotransform to a clip's origin.

    Args:
        parent_gt: GDAL geotransform of the parent, ``[ox, px, rx, oy, ry, py]``.
        x0: Clip left edge in parent pixels.
        y0: Clip top edge in parent pixels.

    Returns:
        The clip's geotransform.
    """
    ox, px, rx, oy, ry, py = parent_gt
    return [ox + x0 * px + y0 * rx, px, rx, oy + x0 * ry + y0 * py, ry, py]


def bounds(gt: list[float], width: int, height: int) -> list[float]:
    """Return ``[minx, miny, maxx, maxy]`` for a raster of a given size.

    Args:
        gt: Geotransform.
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        Bounding box in the geotransform's CRS.
    """
    xs, ys = [], []
    for cx, cy in ((0, 0), (width, 0), (0, height), (width, height)):
        xs.append(gt[0] + cx * gt[1] + cy * gt[2])
        ys.append(gt[3] + cx * gt[4] + cy * gt[5])
    return [min(xs), min(ys), max(xs), max(ys)]


def _record(
    name: str, x0: int, y0: int, width: int, height: int, parent_gt: list[float] | None
) -> dict[str, Any]:
    """Build one clips.json entry.

    Args:
        name: Clip file name.
        x0: Left edge in parent pixels.
        y0: Top edge in parent pixels.
        width: Clip width.
        height: Clip height.
        parent_gt: Parent geotransform, or None if the parent is not
            georeferenced.

    Returns:
        The entry.
    """
    entry: dict[str, Any] = {
        "file": name,
        "offset_xy": [x0, y0],
        "size": [width, height],
    }
    if parent_gt:
        gt = clip_geotransform(parent_gt, x0, y0)
        entry["geotransform"] = gt
        entry["bounds"] = bounds(gt, width, height)
    return entry


def split(source: Path, output_dir: Path, cols: int, rows: int) -> dict[str, Any]:
    """Cut a raster into an even ``cols`` × ``rows`` grid of PNG clips.

    The split is exact and deterministic: column and row boundaries are evenly
    spaced, with the remainder distributed to the leading cells so no pixel is
    lost or duplicated. Clips are numbered row-major from 1.

    Args:
        source: Georeferenced source raster.
        output_dir: Destination directory, created if absent.
        cols: Number of columns.
        rows: Number of rows.

    Returns:
        The clips.json manifest.
    """
    info = raster_info(source)
    image = Image.open(source)
    width, height = image.size
    sheet = source.stem.replace("_clipped", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    def edges(extent: int, parts: int) -> list[int]:
        """Evenly spaced cut points covering the full extent."""
        step, rem = divmod(extent, parts)
        out, pos = [0], 0
        for i in range(parts):
            pos += step + (1 if i < rem else 0)
            out.append(pos)
        return out

    xs = edges(width, cols)
    ys = edges(height, rows)

    clips = []
    index = 1
    for r in range(rows):
        for c in range(cols):
            x0, x1, y0, y1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
            name = f"{sheet}_{index}.png"
            image.crop((x0, y0, x1, y1)).save(output_dir / name)
            clips.append(_record(name, x0, y0, x1 - x0, y1 - y0, info["geotransform"]))
            logger.info("wrote %s  %dx%d at (%d, %d)", name, x1 - x0, y1 - y0, x0, y0)
            index += 1

    return _manifest(sheet, source, info, clips, f"programmatic {cols}x{rows} grid")


def index_existing(clips_dir: Path, parent: Path) -> dict[str, Any]:
    """Recover offsets for clips that already exist and write their manifest.

    Args:
        clips_dir: Directory holding ``<sheet>_N.png`` clips.
        parent: The raster they were cut from.

    Returns:
        The clips.json manifest.

    Raises:
        RuntimeError: If a clip cannot be located in the parent.
    """
    import cv2

    info = raster_info(parent)
    reference = np.array(Image.open(parent).convert("RGB"))
    sheet = parent.stem.replace("_clipped", "")

    clips = []
    for path in sorted(clips_dir.glob("*.png"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        clip = np.array(Image.open(path).convert("RGB"))
        height, width = clip.shape[:2]
        # Take the patch from the clip's interior; edges may carry nodata.
        py = min(MATCH_PATCH, height // 4)
        px = min(MATCH_PATCH, width // 4)
        patch = clip[py : py + MATCH_PATCH, px : px + MATCH_PATCH]
        response = cv2.matchTemplate(reference, patch, cv2.TM_SQDIFF)
        score, _, location, _ = cv2.minMaxLoc(response)
        x0, y0 = location[0] - px, location[1] - py
        if x0 < 0 or y0 < 0 or x0 + width > reference.shape[1] or y0 + height > reference.shape[0]:
            raise RuntimeError(f"{path.name}: matched offset ({x0}, {y0}) does not fit the parent")
        entry = _record(path.name, x0, y0, width, height, info["geotransform"])
        entry["match_sqdiff"] = float(score)
        clips.append(entry)
        logger.info("%s -> offset (%d, %d), sqdiff %.0f", path.name, x0, y0, score)

    return _manifest(
        sheet,
        parent,
        info,
        clips,
        "gis_cut; offsets recovered by template match against the parent",
    )


def _manifest(
    sheet: str, parent: Path, info: dict[str, Any], clips: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    """Assemble a clips.json manifest.

    Args:
        sheet: Sheet id.
        parent: Parent raster path.
        info: Output of :func:`raster_info`.
        clips: Clip entries.
        method: How the clips were produced.

    Returns:
        The manifest.
    """
    return {
        "sheet_id": sheet,
        "parent": parent.name,
        "parent_path_at_write_time": str(parent),
        "parent_size": info["size"],
        "parent_geotransform": info["geotransform"],
        "crs": info["crs"],
        "method": method,
        "note": (
            "PNG clips carry no CRS. A pixel (x, y) in clip N maps to the parent "
            "at (x + offset_xy[0], y + offset_xy[1]), and to the ground via that "
            "clip's geotransform."
        ),
        "clips": clips,
    }


def main() -> None:
    """Parse arguments and run the requested subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    cut = sub.add_parser("split", help="cut a raster into a grid of PNG clips")
    cut.add_argument("--source", type=Path, required=True)
    cut.add_argument("--output-dir", type=Path, required=True)
    cut.add_argument("--grid", default="2x2", help="cols x rows, default 2x2")

    idx = sub.add_parser("index", help="write clips.json for clips that already exist")
    idx.add_argument("--clips-dir", type=Path, required=True)
    idx.add_argument("--parent", type=Path, required=True)

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    if args.command == "split":
        cols, rows = (int(v) for v in args.grid.lower().split("x"))
        manifest = split(args.source, args.output_dir, cols, rows)
        target = args.output_dir
    else:
        manifest = index_existing(args.clips_dir, args.parent)
        target = args.clips_dir

    (target / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{target / 'clips.json'}: {len(manifest['clips'])} clips, crs {manifest['crs']}")


if __name__ == "__main__":
    main()
