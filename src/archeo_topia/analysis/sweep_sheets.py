#!/usr/bin/env python3
"""Run the detector over whole map sheets that have no ground truth (v0.6 step 1).

Every detection path in the project so far has been bound to a *labelled*
dataset: ``train_mound_detector.predict_fold`` reads a fold's ``_val.txt`` and
``mapsam_end_to_end`` reads ``metadata/windows.jsonl`` plus a ground-truth COCO.
Sweeping 56 newly acquired sheets needs neither, so this module tiles a raster
directly and runs the detector over it.

**What this measures, and what it cannot.** The new sheets carry no
annotations, so nothing here is a false-positive rate. What comes back is
*candidate* density — per window and per megapixel — which is the quantity that
governs the annotation review budget, and which bounds the false-positive rate
from above. Calling it precision would be exactly the circular measurement the
blind test set exists to avoid.

**Tiling geometry is shared, not reimplemented.** Windows come from
``build_detection_windows.plan_windows`` and merging from
``detection_metrics.merge_detections``, so a candidate found here lands in the
same source-pixel space as an annotation and as a MapSAM prompt.

**Transparent windows are dropped.** The sheets are clipped to an irregular map
frame and carry an alpha band at around 91% valid, so a fixed tiling puts whole
windows outside the printed area. Counting those in the denominator would
understate candidates per window by roughly a tenth, in the flattering
direction.

Usage:
    python -m archeo_topia.analysis.sweep_sheets \\
        --source data_lake/raw/mound_test_20260915/01_maps_test \\
        --weights artifacts/detection/v0_5_yolo26s_gtfix/foldC/weights/last.pt \\
        --output-dir artifacts/detection/v0_6_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from archeo_topia.analysis.detection_metrics import Detection, merge_detections
from archeo_topia.datasets.build_detection_windows import Window, plan_windows
from archeo_topia.datasets.sheet_clips import raster_info

logger = logging.getLogger(__name__)

# Sheets are ~23 Mpx; well above Pillow's decompression-bomb default.
Image.MAX_IMAGE_PIXELS = None

#: Confidence operating points reported for every sheet.
THRESHOLDS = (0.05, 0.10, 0.25)

#: Windows per inference call. Matches ``predict_fold``.
BATCH = 32


class SheetSource:
    """A sheet to sweep, and the geometry needed to place its candidates.

    A sheet arrives either as a parent GeoTIFF or as a directory of clips with
    a ``clips.json``. Both cases reduce to the same thing: an array of pixels
    plus a geotransform mapping pixel to ground, so the sweep does not care
    which it was given.

    Attributes:
        sheet_id: Sheet identifier, for example ``K-34-10-A-g``.
        image_key: What candidate coordinates are relative to, and therefore
            what goes into ``Detection.image``. For a parent raster this is the
            sheet id; for an already-cut clip it is the clip file name, which is
            the key ``detection_metrics`` matches annotations on.
        path: The raster or clip it came from.
        image: Source pixels as ``HxWx4`` or ``HxWx3`` uint8.
        geotransform: GDAL geotransform of the full sheet, or None.
        crs: CRS string, or None.
    """

    def __init__(
        self,
        sheet_id: str,
        path: Path,
        image: np.ndarray,
        geotransform: list[float] | None,
        crs: str | None,
        image_key: str | None = None,
    ) -> None:
        self.sheet_id = sheet_id
        self.path = path
        self.image = image
        self.geotransform = geotransform
        self.crs = crs
        self.image_key = image_key or sheet_id

    @property
    def height(self) -> int:
        """Sheet height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Sheet width in pixels."""
        return int(self.image.shape[1])

    @property
    def megapixels(self) -> float:
        """Sheet area in megapixels."""
        return self.width * self.height / 1e6


def load_raster(path: Path) -> SheetSource:
    """Load a sheet raster, or an already-cut clip, and its georeferencing.

    Both are accepted because the regression check that makes this module
    trustworthy runs it over the three *annotated* sheets, which exist only as
    PNG clips. A PNG carries no geotransform, so ground coordinates are simply
    absent for those.

    Args:
        path: Path to a ``*_clipped.tif`` or to a clip ``.png``.

    Returns:
        The loaded sheet.
    """
    array = np.asarray(Image.open(path))
    if path.suffix.lower() == ".png":
        return SheetSource(path.stem, path, array, None, None, image_key=path.name)
    info = raster_info(path)
    sheet_id = path.stem.replace("_clipped", "")
    return SheetSource(sheet_id, path, array, info.get("geotransform"), info.get("crs"))


def discover(source: Path) -> list[Path]:
    """Find the sheet rasters under a path.

    Args:
        source: A raster, or a directory containing rasters at any depth.

    Returns:
        Raster and clip paths, sorted. A directory named ``_frozen`` is skipped: those
        sheets are reserved for blind annotation and must not have detector
        output produced for them.
    """
    if source.is_file():
        return [source]
    everything = sorted(p for p in source.rglob("*") if p.suffix.lower() in {".tif", ".png"})
    found = [p for p in everything if "_frozen" not in p.parts]
    skipped = len(everything) - len(found)
    if skipped:
        logger.warning("skipping %d raster(s) under _frozen/ — reserved for blind annotation", skipped)
    return found


def valid_fraction(crop: np.ndarray) -> float:
    """Fraction of a crop inside the printed map area.

    Args:
        crop: Window pixels, ``HxWxC``.

    Returns:
        Fraction of opaque pixels, or 1.0 when the source carries no alpha.
    """
    if crop.ndim != 3 or crop.shape[2] < 4:
        return 1.0
    return float((crop[:, :, 3] > 0).mean())


def to_ground(geotransform: list[float] | None, x: float, y: float) -> tuple[float, float] | None:
    """Map a sheet pixel to ground coordinates.

    Args:
        geotransform: GDAL geotransform, or None.
        x: Pixel x.
        y: Pixel y.

    Returns:
        ``(easting, northing)``, or None when the sheet is not georeferenced.
    """
    if not geotransform:
        return None
    ox, px, rx, oy, ry, py = geotransform
    return (ox + x * px + y * rx, oy + x * ry + y * py)


def sweep(
    sheet: SheetSource,
    model: Any,
    window: int,
    stride: int,
    imgsz: int,
    confidence: float,
    min_valid_fraction: float,
    merge_radius: float,
) -> tuple[list[Detection], list[Detection], dict[str, Any]]:
    """Tile a sheet, run the detector, and merge candidates across windows.

    Args:
        sheet: The loaded sheet.
        model: An Ultralytics ``YOLO``.
        window: Window side length in source pixels.
        stride: Step between window origins.
        imgsz: Inference size; must match training.
        confidence: Lowest confidence to emit.
        min_valid_fraction: Windows with less opaque area than this are skipped.
        merge_radius: Centre distance below which two candidates are one symbol.

    Returns:
        Merged candidates, the raw pre-merge detections, and the tiling
        statistics. The raw list is returned because the evaluation path merges
        once per confidence threshold rather than once overall, and the
        regression check against the v0.5 fold numbers has to do the same.
    """
    planned = plan_windows(sheet.width, sheet.height, window, stride)
    kept: list[Window] = []
    crops: list[np.ndarray] = []
    for win in planned:
        crop = sheet.image[win.y0 : win.y0 + win.size, win.x0 : win.x0 + win.size]
        if valid_fraction(crop) < min_valid_fraction:
            continue
        # Ultralytics reads image *files* with cv2, i.e. BGR. The training and
        # evaluation path passes file paths, so arrays must be handed over in
        # the same channel order or the detector sees different pixels here.
        kept.append(win)
        crops.append(np.ascontiguousarray(crop[:, :, 2::-1]))

    raw: list[Detection] = []
    for start in range(0, len(crops), BATCH):
        batch_windows = kept[start : start + BATCH]
        predictions = model.predict(
            crops[start : start + BATCH], imgsz=imgsz, conf=confidence, verbose=False
        )
        for win, result in zip(batch_windows, predictions, strict=True):
            boxes = result.boxes
            if boxes is None:
                continue
            for box, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
                bx0, by0, bx1, by1 = box
                raw.append(
                    Detection(
                        image=sheet.image_key,
                        x=(bx0 + bx1) / 2 + win.x0,
                        y=(by0 + by1) / 2 + win.y0,
                        score=float(score),
                        box=(bx0 + win.x0, by0 + win.y0, bx1 + win.x0, by1 + win.y0),
                    )
                )

    merged = merge_detections(raw, radius=merge_radius)
    stats = {
        "sheet_id": sheet.sheet_id,
        "source": str(sheet.path),
        "width": sheet.width,
        "height": sheet.height,
        "megapixels": round(sheet.megapixels, 3),
        "crs": sheet.crs,
        "windows_planned": len(planned),
        "windows_swept": len(kept),
        "windows_dropped_transparent": len(planned) - len(kept),
        "raw_candidates": len(raw),
        "merged_candidates": len(merged),
    }
    for threshold in THRESHOLDS:
        at = [d for d in merged if d.score >= threshold]
        key = f"{threshold:.2f}".replace(".", "_")
        stats[f"candidates_at_{key}"] = len(at)
        stats[f"per_window_at_{key}"] = round(len(at) / len(kept), 4) if kept else 0.0
        stats[f"per_megapixel_at_{key}"] = (
            round(len(at) / sheet.megapixels, 4) if sheet.megapixels else 0.0
        )
    scores = sorted((d.score for d in merged), reverse=True)
    if scores:
        stats["score_median"] = round(float(np.median(scores)), 4)
        stats["score_p90"] = round(float(np.percentile(scores, 90)), 4)
        stats["score_max"] = round(scores[0], 4)
    return merged, raw, stats


def write_candidates(
    path: Path, sheet: SheetSource, candidates: list[Detection]
) -> None:
    """Write one sheet's candidates as JSON lines.

    Args:
        path: Destination file.
        sheet: The sheet they came from.
        candidates: Merged candidates in sheet coordinates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for det in sorted(candidates, key=lambda d: -d.score):
            record: dict[str, Any] = {
                "sheet_id": sheet.sheet_id,
                "x": round(det.x, 2),
                "y": round(det.y, 2),
                "score": round(det.score, 4),
                "box": [round(v, 2) for v in det.box] if det.box else None,
            }
            ground = to_ground(sheet.geotransform, det.x, det.y)
            if ground:
                record["easting"] = round(ground[0], 3)
                record["northing"] = round(ground[1], 3)
                record["crs"] = sheet.crs
            handle.write(json.dumps(record) + "\n")


def render_crops(
    output_dir: Path, sheet: SheetSource, candidates: list[Detection], count: int, size: int = 128
) -> None:
    """Save crops around the highest-confidence candidates for visual review.

    A candidate count says whether the detector fires; only a picture says
    whether it fires on something that looks like a mound symbol.

    Args:
        output_dir: Destination directory.
        sheet: The sheet.
        candidates: Merged candidates.
        count: How many crops to save.
        size: Crop side length in source pixels.
    """
    if count <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    half = size // 2
    for rank, det in enumerate(sorted(candidates, key=lambda d: -d.score)[:count], start=1):
        cx, cy = int(round(det.x)), int(round(det.y))
        x0 = max(0, min(cx - half, sheet.width - size))
        y0 = max(0, min(cy - half, sheet.height - size))
        crop = sheet.image[y0 : y0 + size, x0 : x0 + size, :3]
        name = f"{rank:02d}_conf{det.score:.3f}_x{cx}_y{cy}.png"
        Image.fromarray(crop).save(output_dir / name)


def main() -> None:
    """Sweep every sheet under ``--source`` with every checkpoint in ``--weights``."""
    parser = argparse.ArgumentParser(description="Detector sweep over unlabelled map sheets")
    parser.add_argument("--source", required=True, help="Raster, or directory of rasters")
    parser.add_argument("--weights", required=True, nargs="+", help="One or more YOLO checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--min-valid-fraction", type=float, default=0.05)
    parser.add_argument("--merge-radius", type=float, default=10.0)
    parser.add_argument("--render-crops", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Sweep at most this many sheets")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    from ultralytics import YOLO

    rasters = discover(Path(args.source))
    if args.limit:
        rasters = rasters[: args.limit]
    output_root = Path(args.output_dir)
    logger.info("sweeping %d sheet(s) with %d checkpoint(s)", len(rasters), len(args.weights))

    for weights in args.weights:
        weights_path = Path(weights)
        # artifacts/detection/<run>/<fold>/weights/last.pt -> "<fold>_last"
        label = f"{weights_path.parents[1].name}_{weights_path.stem}"
        run_dir = output_root / label
        model = YOLO(str(weights_path))
        rows: list[dict[str, Any]] = []

        for index, raster in enumerate(rasters, start=1):
            sheet = load_raster(raster)
            candidates, _raw, stats = sweep(
                sheet,
                model,
                window=args.window,
                stride=args.stride,
                imgsz=args.imgsz,
                confidence=args.confidence,
                min_valid_fraction=args.min_valid_fraction,
                merge_radius=args.merge_radius,
            )
            stats["weights"] = str(weights_path)
            rows.append(stats)
            write_candidates(run_dir / sheet.sheet_id / "candidates.jsonl", sheet, candidates)
            render_crops(
                run_dir / sheet.sheet_id / "crops", sheet, candidates, args.render_crops
            )
            logger.info(
                "[%s] %d/%d %s: %d windows (%d dropped), %d candidates, %.2f/Mpx at 0.25",
                label,
                index,
                len(rasters),
                sheet.sheet_id,
                stats["windows_swept"],
                stats["windows_dropped_transparent"],
                stats["merged_candidates"],
                stats["per_megapixel_at_0_25"],
            )

        if rows:
            summary = run_dir / "sweep_summary.csv"
            summary.parent.mkdir(parents=True, exist_ok=True)
            fields = list(rows[0].keys())
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("wrote %s", summary)


if __name__ == "__main__":
    main()
