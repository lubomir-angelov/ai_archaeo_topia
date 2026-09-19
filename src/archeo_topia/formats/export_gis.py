#!/usr/bin/env python3
"""Turn a detector sweep into files the team can open in QGIS and ArcGIS.

Two audiences, one file. The team uses the output directly in field work, and
also as the surface on which they review the model's results before sending
their verdicts back.

**Two layers, and the distinction between them is not cosmetic.**

``mound_points``
    The archaeological location. What field work navigates to.
``mound_symbols``
    The printed cartographic symbol, joined on ``mound_id``.

``TARGET_PIPELINE.md`` states the reason plainly: *"The MapSAM polygon is the
printed cartographic mound symbol. It is not the physical footprint of the
archaeological mound."* Symbol component areas run 116-1037 px on a symbol
about 21 px across; that spread is drafting variation and scan condition, not
mound size. A consumer who reads ``symbol_area_px`` as ground extent is wrong
by an unbounded factor, so the fields carry a ``symbol_`` prefix to make the
misreading take effort.

**Formats.** A GeoPackage in the sheet's own EPSG:25835 is the working file: it
holds both layers, declares its CRS unambiguously in both tools, and keeps
full-length field names. A GeoJSON reprojected to WGS84 under RFC 7946 is the
portable copy. Shapefile is not offered -- it truncates field names at ten
characters, collapsing ``crossed_by_contour``, ``crossed_by_forestation_line``,
``crossed_by_grid``, ``crossed_by_powerline`` and ``crossed_by_road`` into a
run of ``crossed_by``, ``crossed__1`` and so on.

Usage:
    python -m archeo_topia.formats.export_gis \\
        --candidates artifacts/detection/v0_6_sweep/foldC_last \\
        --clips-root "$LAKE/cleaned/map_clips/dataset_03" \\
        --output "$LAKE/cleaned/gis/v0_6_foldC"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from archeo_topia.formats.features import FeatureCollection, point, polygon
from archeo_topia.formats.georeference import SheetReference
from archeo_topia.formats.labels import LabelSchema

logger = logging.getLogger(__name__)

#: The schema of record.
DEFAULT_SCHEMA = Path("annotation/cvat/labels.json")

#: Sheets held back from the main delivery, with the reason shown to reviewers.
#: K-34-47-G-v is central Sofia: it returned 957 / 140 / 418 candidates across
#: the three v0.5 checkpoints where no other sheet exceeded 110, with 11-25%
#: cross-checkpoint agreement, and the candidates land almost entirely inside
#: dense urban blocks. The annotated corpus is three rural sheets and contains
#: no urban fabric at all, so the detector has no basis for a decision there.
#: Left in the main delivery it would dominate the review queue with the least
#: representative sheet in the corpus.
SET_ASIDE = {
    "K-34-47-G-v": "dense urban fabric (central Sofia); candidate rate an order of magnitude high",
}


README_EN = """# Mound detections for review — {version}

{sheets} sheets, {points} proposed mound locations. Produced by the MapSAM
detector and segmenter. **Nothing in these files is verified.** Every feature is
a proposal until you say otherwise.

## Opening them

Open `sheets/<sheet>.gpkg` in QGIS or ArcGIS. Each file holds two layers:

| layer | what it is |
|---|---|
| `mound_points` | the location of the proposed mound — navigate to this in the field |
| `mound_symbols` | the outline of the **printed map symbol**, joined to the points by `mound_id` |

`mound_symbols` is the symbol drawn on the paper map, **not** the physical
extent of the mound. `symbol_area_px` is the size of that drawing and says
nothing about how large the mound is on the ground. Do not measure mounds with
it.

Coordinates are **EPSG:25835 (ETRS89 / UTM 35N)**, the same as the source map
sheets. There is also a `<sheet>.geojson` in WGS84 for tools that prefer it.

## What to fill in

For every feature, set **`review_status`**:

| value | meaning |
|---|---|
| `confirmed` | this is a mound, and the location is right |
| `corrected` | this is a mound, but I moved the point to the right place |
| `rejected` | this is not a mound |
| `added` | I found a mound the model missed and drew this myself |
| `unreviewed` | not yet looked at (the starting value) |

If you know them, also fill the descriptive attributes — `has_trig_point`,
`blurred_or_bad_print`, `water_line_crossing` and the rest. They are the same
fields as the annotation protocol and they feed directly into training.

## Two rules

**Never edit `mound_id`.** It is how a feature is matched back to the model's
output. Change it and the verdict is lost. New features you draw yourself can
leave it empty.

**Mark a wrong proposal `rejected` — do not delete it.** A deleted feature is
indistinguishable from one that was never sent, which makes it impossible to
work out how often the model was right.

## Sending it back

Return the `.gpkg` files. Keep the layer names and the CRS as they are; if your
GIS offers to reproject on save, decline.

## `set_aside/`

Sheets held back from the main batch, listed with a reason in
`export_report.json`. Review them last, or not at all.
"""

README_BG = """# Откривания на могили за преглед — {version}

{sheets} листа, {points} предложени местоположения на могили. Създадено от
детектора и сегментатора MapSAM. **Нищо в тези файлове не е проверено.** Всеки
обект е предложение, докато не кажете друго.

## Отваряне

Отворете `sheets/<лист>.gpkg` в QGIS или ArcGIS. Всеки файл съдържа два слоя:

| слой | какво е |
|---|---|
| `mound_points` | местоположението на предложената могила — това търсите на терен |
| `mound_symbols` | контурът на **печатния картен знак**, свързан с точките чрез `mound_id` |

`mound_symbols` е знакът, начертан върху хартиената карта, а **не** физическият
обхват на могилата. `symbol_area_px` е размерът на чертежа и не казва нищо за
големината на могилата на терен. Не измервайте могили с него.

Координатите са **EPSG:25835 (ETRS89 / UTM 35N)**, същите като изходните
картни листове. Има и `<лист>.geojson` в WGS84 за инструменти, които го
предпочитат.

## Какво да попълните

За всеки обект задайте **`review_status`**:

| стойност | значение |
|---|---|
| `confirmed` | това е могила и местоположението е вярно |
| `corrected` | това е могила, но преместих точката на правилното място |
| `rejected` | това не е могила |
| `added` | намерих могила, която моделът е пропуснал, и я начертах сам |
| `unreviewed` | още не е прегледано (началната стойност) |

Ако ги знаете, попълнете и описателните атрибути — `has_trig_point`,
`blurred_or_bad_print`, `water_line_crossing` и останалите. Това са същите
полета като в протокола за анотиране и влизат директно в обучението.

## Две правила

**Никога не променяйте `mound_id`.** Чрез него обектът се свързва обратно с
изхода на модела. Ако го промените, преценката се губи. При нови обекти, които
чертаете сами, може да остане празно.

**Отбележете грешното предложение като `rejected` — не го изтривайте.** Изтрит
обект не се различава от такъв, който никога не е бил изпратен, а това прави
невъзможно да се изчисли колко често моделът е бил прав.

## Връщане

Върнете файловете `.gpkg`. Запазете имената на слоевете и координатната
система; ако вашата ГИС предложи препроектиране при запис, откажете.

## `set_aside/`

Листове, отделени от основната партида, с причина в `export_report.json`.
Прегледайте ги последни или изобщо.
"""


def write_readme(output: Path, version: str, sheets: int, points: int) -> None:
    """Write the reviewer instructions beside the data, in both languages.

    Args:
        output: The delivery directory.
        version: A label for this delivery.
        sheets: How many sheets shipped.
        points: How many proposals shipped.
    """
    fields = {"version": version, "sheets": sheets, "points": points}
    (output / "README_EN.md").write_text(README_EN.format(**fields), encoding="utf-8")
    (output / "README_BG.md").write_text(README_BG.format(**fields), encoding="utf-8")


def load_candidates(path: Path, confidence: float) -> list[dict[str, Any]]:
    """Read a sheet's candidates above a confidence threshold.

    Args:
        path: The ``candidates.jsonl``.
        confidence: Lowest confidence to keep.

    Returns:
        Candidate records, highest confidence first.
    """
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in records if r["score"] >= confidence]


def build_layers(
    reference: SheetReference,
    candidates: list[dict[str, Any]],
    schema: LabelSchema,
    symbols: dict[int, list[float]] | None = None,
) -> tuple[FeatureCollection, FeatureCollection]:
    """Build the point and symbol layers for one sheet.

    Args:
        reference: The sheet's georeferencing.
        candidates: Sweep candidates in sheet-pixel coordinates.
        schema: Label schema supplying attribute defaults.
        symbols: Optional map from candidate index to a flat sheet-pixel ring.

    Returns:
        ``(mound_points, mound_symbols)``.
    """
    points = FeatureCollection(crs=reference.crs, name="mound_points")
    shapes = FeatureCollection(crs=reference.crs, name="mound_symbols")
    defaults = schema.defaults("mound")

    for index, candidate in enumerate(candidates, start=1):
        ground = reference.to_ground(candidate["x"], candidate["y"])
        if ground is None:
            continue
        mound_id = f"{reference.sheet_id}-{index:05d}"
        placed = reference.sheet_to_clip(candidate["x"], candidate["y"])

        properties = dict(defaults)
        properties.update(
            {
                "mound_id": mound_id,
                "sheet_id": reference.sheet_id,
                "source_image": placed[0] if placed else "",
                "clip_x": round(placed[1], 2) if placed else None,
                "clip_y": round(placed[2], 2) if placed else None,
                "sheet_x": candidate["x"],
                "sheet_y": candidate["y"],
                "detector_confidence": round(candidate["score"], 4),
                "annotation_provenance": "model_proposal_accepted",
                "review_status": "unreviewed",
            }
        )
        points.add(point(*ground), **properties)

        ring = (symbols or {}).get(index)
        if not ring:
            continue
        ground_ring = []
        for i in range(0, len(ring), 2):
            projected = reference.to_ground(ring[i], ring[i + 1])
            if projected is None:
                break
            ground_ring.append(projected)
        if len(ground_ring) < 3:
            continue
        centroid_x = sum(ring[0::2]) / (len(ring) / 2)
        centroid_y = sum(ring[1::2]) / (len(ring) / 2)
        shapes.add(
            polygon(ground_ring),
            mound_id=mound_id,
            sheet_id=reference.sheet_id,
            symbol_area_px=_ring_area(ring),
            # TARGET_PIPELINE stage 4 asks for this as a production monitor that
            # needs no ground truth: above about 5 px it predicts segmentation
            # degradation, and both quantities are already in hand here.
            detector_to_mask_centroid_distance_px=round(
                ((centroid_x - candidate["x"]) ** 2 + (centroid_y - candidate["y"]) ** 2) ** 0.5, 2
            ),
        )

    return points, shapes


def _ring_area(ring: list[float]) -> float:
    """Return a flat ring's area by the shoelace formula.

    Args:
        ring: Flat ``[x0, y0, x1, y1, ...]``.

    Returns:
        Area in square pixels.
    """
    xs, ys = ring[0::2], ring[1::2]
    total = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        total += xs[i] * ys[j] - xs[j] * ys[i]
    return round(abs(total) / 2.0, 2)


def export_sheet(
    sheet_id: str,
    candidates_path: Path,
    clips_dir: Path,
    output_dir: Path,
    schema: LabelSchema,
    confidence: float,
    symbols: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    """Write one sheet's GeoPackage and portable GeoJSON.

    Args:
        sheet_id: The sheet.
        candidates_path: Its ``candidates.jsonl``.
        clips_dir: Its clip directory, carrying ``clips.json``.
        output_dir: Destination directory.
        schema: Label schema.
        confidence: Lowest confidence to export.
        symbols: Optional symbol rings by candidate index.

    Returns:
        A summary row.
    """
    reference = SheetReference.from_clips_json(clips_dir)
    candidates = load_candidates(candidates_path, confidence)
    points, shapes = build_layers(reference, candidates, schema, symbols)

    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg = output_dir / f"{sheet_id}.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    points.to_geopackage(gpkg)
    if len(shapes):
        shapes.to_geopackage(gpkg, append=True)
    points.to_geojson_wgs84(output_dir / f"{sheet_id}.geojson")

    return {
        "sheet_id": sheet_id,
        "crs": reference.crs,
        "candidates": len(candidates),
        "points": len(points),
        "symbols": len(shapes),
        "set_aside": SET_ASIDE.get(sheet_id, ""),
    }


def main() -> None:
    """Export every swept sheet as GIS files."""
    parser = argparse.ArgumentParser(description="Detector sweep to GeoPackage and GeoJSON")
    parser.add_argument("--candidates", required=True, help="A sweep run directory")
    parser.add_argument("--clips-root", required=True, help="Directory of per-sheet clip folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--symbols", help="Optional JSON of decoded symbol rings per sheet")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    schema = LabelSchema.load(args.schema)
    sweep = Path(args.candidates)
    clips_root = Path(args.clips_root)
    output = Path(args.output)
    all_symbols = json.loads(Path(args.symbols).read_text()) if args.symbols else {}

    rows: list[dict[str, Any]] = []
    for sheet_dir in sorted(d for d in sweep.iterdir() if d.is_dir()):
        sheet_id = sheet_dir.name
        clips_dir = clips_root / sheet_id
        if not (clips_dir / "clips.json").exists():
            logger.warning("%s: no clips.json under %s; skipped", sheet_id, clips_dir)
            continue
        destination = output / ("set_aside" if sheet_id in SET_ASIDE else "sheets")
        rings = {int(k): v for k, v in all_symbols.get(sheet_id, {}).items()}
        row = export_sheet(
            sheet_id,
            sheet_dir / "candidates.jsonl",
            clips_dir,
            destination,
            schema,
            args.confidence,
            rings or None,
        )
        rows.append(row)
        logger.info(
            "%s: %d points, %d symbols%s",
            sheet_id,
            row["points"],
            row["symbols"],
            f"  [set aside: {row['set_aside']}]" if row["set_aside"] else "",
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "export_report.json").write_text(
        json.dumps(
            {
                "confidence": args.confidence,
                "sheets": len(rows),
                "points": sum(r["points"] for r in rows),
                "symbols": sum(r["symbols"] for r in rows),
                "set_aside": {k: v for k, v in SET_ASIDE.items()},
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_readme(
        output,
        Path(args.candidates).name,
        len(rows),
        sum(r["points"] for r in rows),
    )
    logger.info("wrote %s for %d sheets", output, len(rows))


if __name__ == "__main__":
    main()
