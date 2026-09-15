#!/usr/bin/env python3
"""Build a windowed mound-detection dataset from the CVAT COCO export.

MapSAM v0.5 stage 1. The segmentation pipeline consumes one prompt-centred
window per known mound; the detector has to consume the sheet, so this script
tiles each annotated clip into overlapping windows and writes YOLO-format
labels for the mounds whose centres fall inside each one.

Three design decisions are carried from ``docs/mapsam/v005/PLAN.md`` and are
worth stating where the code lives:

* **One class.** ``hard_negative_symbol`` is *not* a second class. The 530
  negatives teach the detector by appearing in windows that carry no target —
  270 of the 480 windows contain a hard negative and no mound, so the negatives
  arrive as background from tiling alone. Their boxes are still written to the
  window metadata so a false positive can be attributed to a ``negative_type``
  at evaluation time.
* **Targets are annotations, not mask components.** Eight pairs of touching
  mounds share a connected component, so the segmentation manifest holds 171
  samples for 180 annotations. Detection recall is measured against the
  annotations, so each one is its own box here, and the component grouping is
  recorded rather than applied.
* **``uncertain_ignore`` is excluded in both directions.** It is not a target,
  and a detection landing on one is neither a true nor a false positive. The
  regions are written to the window metadata so evaluation can neutralize them.

Windows are written once and shared between folds; each fold is a pair of
newline-delimited image lists plus an Ultralytics data YAML pointing at them.
That keeps three leave-one-sheet-out folds from triplicating the crops on disk.

Usage:
    python -m archeo_topia.datasets.build_detection_windows \\
        --coco-json annotation/cvat/v0.0.1/instances_default.json \\
        --images-root data/curated/datasets/mapsam_v02/images \\
        --output-dir data/curated/datasets/mapsam_det_v0 \\
        --window 512 --stride 384
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

MOUND = "mound"
HARD_NEGATIVE = "hard_negative_symbol"
UNCERTAIN = "uncertain_ignore"

#: Splits declared by 1:100k parent sheet. Written by v0.6; see the file's own
#: ``rationale`` for why the grouping is by parent and not by 1:25k sheet id.
SPLITS_FILE = Path(__file__).resolve().parents[3] / "configs" / "splits" / "v0_6_splits.json"

#: Preset name for the leave-one-sheet-out folds of v0.1-v0.5.
LOSO_PRESET = "loso_v0_5"


def load_folds(preset: str = LOSO_PRESET, splits_file: Path | None = None) -> dict[str, str]:
    """Load a named fold preset, mapping fold name to its held-out eval sheet.

    Folds were a hardcoded dict through v0.5, when three sheets made three
    leave-one-sheet-out folds the only option. v0.6 adds 60 sheets and a
    permanent train / validation / test split, so the definition moves into
    ``configs/splits/v0_6_splits.json`` -- but the v0.5 folds stay available by
    name, because every number in v0.1-v0.5 is reported against them and has to
    remain reproducible.

    Args:
        preset: Preset name under ``presets`` in the splits file.
        splits_file: Override for the splits file location.

    Returns:
        Mapping of fold name to held-out evaluation sheet.

    Raises:
        KeyError: If the preset is not declared.
    """
    path = splits_file or SPLITS_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    presets = document.get("presets", {})
    if preset not in presets:
        raise KeyError(f"no preset {preset!r} in {path}; have {sorted(presets)}")
    return dict(presets[preset]["folds"])


def load_splits(splits_file: Path | None = None) -> dict[str, list[str]]:
    """Load the permanent train / validation / test split.

    Returns:
        Mapping of split name to sheet ids.
    """
    path = splits_file or SPLITS_FILE
    return dict(json.loads(path.read_text(encoding="utf-8"))["splits"])


#: Leave-one-sheet-out folds, identical to the v0.4 MapSAM configs so detection
#: and segmentation numbers stay commensurable. Kept as a module-level name
#: because three other modules import it, and literal here so that an installed
#: copy without ``configs/`` alongside it still imports.
FOLDS = {
    "foldA": "K-35-51-B-a",
    "foldB": "K-35-8-G-a",
    "foldC": "K-34-35-B-g",
}


class Window(NamedTuple):
    """A square crop in source-image pixel coordinates.

    Attributes:
        x0: Left edge, inclusive.
        y0: Top edge, inclusive.
        size: Side length in source pixels.
    """

    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        """Right edge, exclusive."""
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        """Bottom edge, exclusive."""
        return self.y0 + self.size

    @property
    def name(self) -> str:
        """Filename-safe suffix identifying the window's position."""
        return f"x{self.x0:05d}_y{self.y0:05d}"


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------


def sheet_of(file_name: str) -> str:
    """Return the map-sheet id for a clip file name.

    Clips are named ``<sheet>_<index>.png``; the sheet id itself contains
    hyphens but no underscores, so the last underscore separates them.

    Args:
        file_name: Clip file name, for example ``K-35-51-B-a_3.png``.

    Returns:
        The sheet id, for example ``K-35-51-B-a``.
    """
    return Path(file_name).stem.rsplit("_", 1)[0]


def decode_mask(annotation: dict[str, Any], width: int, height: int) -> np.ndarray:
    """Rasterize a COCO annotation to a boolean mask.

    Handles the two geometry encodings CVAT emits for this dataset:
    uncompressed RLE (``segmentation`` as a dict, column-major run lengths) and
    a single polygon. Annotations with no geometry return an empty mask.

    Args:
        annotation: COCO annotation dict.
        width: Source image width.
        height: Source image height.

    Returns:
        Boolean array of shape ``(height, width)``.
    """
    segmentation = annotation.get("segmentation")
    mask = np.zeros((height, width), dtype=bool)

    if isinstance(segmentation, dict):
        mask_h, mask_w = segmentation["size"]
        flat = np.zeros(mask_h * mask_w, dtype=bool)
        index = 0
        filled = False
        for run in segmentation["counts"]:
            if filled:
                flat[index : index + run] = True
            index += run
            filled = not filled
        mask[:mask_h, :mask_w] = flat.reshape((mask_h, mask_w), order="F")
    elif isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        points = segmentation[0]
        if points:
            canvas = Image.new("L", (width, height), 0)
            ImageDraw.Draw(canvas).polygon(
                [(points[i], points[i + 1]) for i in range(0, len(points), 2)], fill=1
            )
            mask = np.array(canvas) > 0

    return mask


def component_groups(
    annotations: list[dict[str, Any]], width: int, height: int
) -> dict[int, list[int]]:
    """Group mound annotations by the connected component they share.

    Touching mound polygons merge into one component when rasterized, which is
    why 180 annotations yield 171 segmentation samples. Detection keeps them
    separate but records the grouping, because a point-based system will tend
    to collapse exactly these cases into a single candidate.

    Args:
        annotations: Mound annotations for one image.
        width: Source image width.
        height: Source image height.

    Returns:
        Mapping from component label to the annotation ids it contains.
        Annotations with no geometry are absent.
    """
    import cv2

    masks = {a["id"]: decode_mask(a, width, height) for a in annotations}
    union = np.zeros((height, width), dtype=np.uint8)
    for mask in masks.values():
        union |= mask.astype(np.uint8)

    count, labels = cv2.connectedComponents(union, connectivity=8)
    groups: dict[int, list[int]] = defaultdict(list)
    for ann_id, mask in masks.items():
        if not mask.any():
            continue
        for label in np.unique(labels[mask]):
            if label:
                groups[int(label)].append(ann_id)

    logger.debug("image has %d components over %d annotations", count - 1, len(annotations))
    return dict(groups)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


def plan_windows(width: int, height: int, size: int, stride: int) -> list[Window]:
    """Lay overlapping windows over an image, clamped at the right and bottom.

    The final row and column are pulled back to sit flush with the image edge
    rather than padded, so the detector never sees a synthetic border.

    Args:
        width: Source image width.
        height: Source image height.
        size: Window side length.
        stride: Step between window origins.

    Returns:
        Windows in row-major order.
    """

    def origins(extent: int) -> list[int]:
        if extent <= size:
            return [0]
        starts = list(range(0, extent - size + 1, stride))
        if starts[-1] + size < extent:
            starts.append(extent - size)
        return starts

    return [Window(x0, y0, size) for y0 in origins(height) for x0 in origins(width)]


def clip_box(bbox: list[float], window: Window) -> tuple[list[float], float]:
    """Clip a source-coordinate box to a window.

    Args:
        bbox: COCO ``[x, y, w, h]`` in source pixels.
        window: Target window.

    Returns:
        Tuple of the clipped box as ``[x0, y0, x1, y1]`` in *window*
        coordinates, and the fraction of the original box area still visible.
    """
    x, y, w, h = bbox
    x0 = max(x, window.x0)
    y0 = max(y, window.y0)
    x1 = min(x + w, window.x1)
    y1 = min(y + h, window.y1)
    visible = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(w * h, 1e-9)
    return [x0 - window.x0, y0 - window.y0, x1 - window.x0, y1 - window.y0], visible / area


def to_yolo(box: list[float], size: int) -> tuple[float, float, float, float]:
    """Convert a window-coordinate xyxy box to normalized YOLO cxcywh.

    Args:
        box: ``[x0, y0, x1, y1]`` in window pixels.
        size: Window side length.

    Returns:
        ``(cx, cy, w, h)``, each in ``[0, 1]``.
    """
    x0, y0, x1, y1 = box
    return (
        ((x0 + x1) / 2) / size,
        ((y0 + y1) / 2) / size,
        (x1 - x0) / size,
        (y1 - y0) / size,
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def find_image(images_root: Path, file_name: str) -> Path:
    """Locate a clip under any split subdirectory of the image root.

    Args:
        images_root: Directory holding ``train``/``val``/``test`` subdirectories.
        file_name: Clip file name from the COCO export.

    Returns:
        Path to the image.

    Raises:
        FileNotFoundError: If the clip is not present under any split.
    """
    direct = images_root / file_name
    if direct.exists():
        return direct
    for candidate in images_root.glob(f"*/{file_name}"):
        return candidate
    raise FileNotFoundError(f"{file_name} not found under {images_root}")


def build(
    coco_json: Path,
    images_root: Path,
    output_dir: Path,
    window_size: int,
    stride: int,
    write_images: bool = True,
) -> dict[str, Any]:
    """Tile every annotated clip and write a fold-shared detection dataset.

    Args:
        coco_json: CVAT COCO export.
        images_root: Root of the clip images.
        output_dir: Destination directory.
        window_size: Window side length in source pixels.
        stride: Step between window origins.
        write_images: Write the crops. Set false to refresh labels and
            metadata without re-encoding the PNGs.

    Returns:
        A build report dict, also written to ``metadata/build_report.json``.
    """
    with open(coco_json, encoding="utf-8") as handle:
        coco = json.load(handle)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}

    by_image: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        by_image[ann["image_id"]][categories[ann["category_id"]]].append(ann)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    window_records: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []
    by_sheet_windows: dict[str, list[str]] = defaultdict(list)
    counts = {
        "windows_total": 0,
        "windows_with_mound": 0,
        "windows_hard_negative_only": 0,
        "windows_empty": 0,
        "mound_annotations": 0,
        "mound_targets_written": 0,
        "mounds_without_full_window": 0,
        "mounds_without_geometry": 0,
        "merged_components": 0,
    }

    for image_id, image_meta in images.items():
        file_name = image_meta["file_name"]
        width, height = image_meta["width"], image_meta["height"]
        sheet = sheet_of(file_name)
        stem = Path(file_name).stem

        mounds = by_image[image_id][MOUND]
        negatives = by_image[image_id][HARD_NEGATIVE]
        ignores = by_image[image_id][UNCERTAIN]
        counts["mound_annotations"] += len(mounds)

        groups = component_groups(mounds, width, height)
        component_of = {
            ann_id: (label, len(members))
            for label, members in groups.items()
            for ann_id in members
        }
        counts["merged_components"] += sum(1 for m in groups.values() if len(m) > 1)

        source = Image.open(find_image(images_root, file_name)).convert("RGB")
        windows = plan_windows(width, height, window_size, stride)

        # Per annotation: which windows see it, and does any window hold it
        # clear of the edge? The second question is what decides whether a
        # miss can be blamed on tiling.
        seen_in: dict[int, list[str]] = defaultdict(list)
        fully_inside: dict[int, bool] = defaultdict(bool)

        for window in windows:
            window_stem = f"{stem}_{window.name}"
            targets: list[dict[str, Any]] = []

            for ann in mounds:
                x, y, w, h = ann["bbox"]
                cx, cy = x + w / 2, y + h / 2
                if not (window.x0 <= cx < window.x1 and window.y0 <= cy < window.y1):
                    continue
                box, visible = clip_box(ann["bbox"], window)
                seen_in[ann["id"]].append(window_stem)
                if visible >= 0.999:
                    fully_inside[ann["id"]] = True
                targets.append(
                    {
                        "annotation_id": ann["id"],
                        "box_window_xyxy": [round(v, 2) for v in box],
                        "visible_fraction": round(visible, 4),
                        "truncated": visible < 0.999,
                    }
                )

            negatives_in = [
                {
                    "annotation_id": n["id"],
                    "negative_type": (n.get("attributes") or {}).get("negative_type"),
                    "box_window_xyxy": [round(v, 2) for v in clip_box(n["bbox"], window)[0]],
                }
                for n in negatives
                if window.x0 <= n["bbox"][0] + n["bbox"][2] / 2 < window.x1
                and window.y0 <= n["bbox"][1] + n["bbox"][3] / 2 < window.y1
            ]
            ignores_in = [
                {
                    "annotation_id": g["id"],
                    "box_window_xyxy": [round(v, 2) for v in clip_box(g["bbox"], window)[0]],
                }
                for g in ignores
                if clip_box(g["bbox"], window)[1] > 0
            ]

            if write_images:
                crop = source.crop((window.x0, window.y0, window.x1, window.y1))
                crop.save(output_dir / "images" / f"{window_stem}.png")

            label_path = output_dir / "labels" / f"{window_stem}.txt"
            lines = [
                "0 " + " ".join(f"{v:.6f}" for v in to_yolo(t["box_window_xyxy"], window_size))
                for t in targets
            ]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            counts["windows_total"] += 1
            if targets:
                counts["windows_with_mound"] += 1
            elif negatives_in:
                counts["windows_hard_negative_only"] += 1
            else:
                counts["windows_empty"] += 1
            counts["mound_targets_written"] += len(targets)

            by_sheet_windows[sheet].append(window_stem)
            window_records.append(
                {
                    "window_id": window_stem,
                    "sheet_id": sheet,
                    "source_image": file_name,
                    # Inverse transform: source = window + [x0, y0].
                    "window_xyxy": [window.x0, window.y0, window.x1, window.y1],
                    "targets": targets,
                    "hard_negatives": negatives_in,
                    "uncertain_ignore": ignores_in,
                }
            )

        for ann in mounds:
            x, y, w, h = ann["bbox"]
            label, size = component_of.get(ann["id"], (None, 0))
            has_geometry = ann["id"] in component_of
            if not has_geometry:
                counts["mounds_without_geometry"] += 1
            if not fully_inside[ann["id"]]:
                counts["mounds_without_full_window"] += 1
            annotation_records.append(
                {
                    "annotation_id": ann["id"],
                    "sheet_id": sheet,
                    "source_image": file_name,
                    "center_source_xy": [round(x + w / 2, 2), round(y + h / 2, 2)],
                    "bbox_source_xywh": [round(v, 2) for v in ann["bbox"]],
                    "component_id": label,
                    "component_annotation_count": size,
                    "has_geometry": has_geometry,
                    "windows": seen_in[ann["id"]],
                    "has_untruncated_window": fully_inside[ann["id"]],
                    "attributes": ann.get("attributes") or {},
                }
            )

    _write_jsonl(output_dir / "metadata" / "windows.jsonl", window_records)
    _write_jsonl(output_dir / "metadata" / "annotations.jsonl", annotation_records)
    _write_folds(output_dir, by_sheet_windows)

    report = {
        "window_size": window_size,
        "stride": stride,
        "counts": counts,
        "windows_per_sheet": {s: len(v) for s, v in by_sheet_windows.items()},
        "mounds_per_sheet": _tally(annotation_records, "sheet_id"),
        "folds": {
            name: {"eval_sheet": sheet, "train_sheets": [s for s in FOLDS.values() if s != sheet]}
            for name, sheet in FOLDS.items()
        },
    }
    with open(output_dir / "metadata" / "build_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def _tally(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count records by a key.

    Args:
        records: Record dicts.
        key: Key to group on.

    Returns:
        Mapping from key value to count.
    """
    out: dict[str, int] = defaultdict(int)
    for record in records:
        out[record[key]] += 1
    return dict(out)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as newline-delimited JSON.

    Args:
        path: Destination file.
        records: Record dicts.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_folds(output_dir: Path, by_sheet: dict[str, list[str]]) -> None:
    """Write per-fold image lists and Ultralytics data YAMLs.

    The lists live at the dataset root rather than in a subdirectory:
    Ultralytics resolves a ``./`` path inside an image list against the list
    file's own parent, so a list in ``splits/`` would look for the crops under
    ``splits/images/`` and find no labels.

    Args:
        output_dir: Dataset root.
        by_sheet: Window ids grouped by sheet.
    """
    for fold, eval_sheet in FOLDS.items():
        for split, sheets in (
            ("train", [s for s in by_sheet if s != eval_sheet]),
            ("val", [eval_sheet]),
        ):
            lines = [
                f"./images/{window_id}.png" for sheet in sheets for window_id in by_sheet[sheet]
            ]
            (output_dir / f"{fold}_{split}.txt").write_text(
                "\n".join(sorted(lines)) + "\n", encoding="utf-8"
            )
        (output_dir / f"{fold}.yaml").write_text(
            "# Leave-one-sheet-out fold for MapSAM v0.5 mound detection.\n"
            f"# Evaluation sheet: {eval_sheet}\n"
            f"path: {output_dir.resolve()}\n"
            f"train: {fold}_train.txt\n"
            f"val: {fold}_val.txt\n"
            "names:\n"
            "  0: mound\n",
            encoding="utf-8",
        )


def main() -> None:
    """Parse arguments and build the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Refresh labels and metadata without re-encoding the crops",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    report = build(
        coco_json=args.coco_json,
        images_root=args.images_root,
        output_dir=args.output_dir,
        window_size=args.window,
        stride=args.stride,
        write_images=not args.labels_only,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
