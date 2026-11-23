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
  arrive as background from tiling alone. Their boxes are still written to
  ``metadata/regions.jsonl`` so a false positive can be attributed to a
  ``negative_type`` at evaluation time.
* **Targets are annotations, not mask components.** Eight pairs of touching
  mounds share a connected component, so the segmentation manifest holds 171
  samples for 180 annotations. Detection recall is measured against the
  annotations, so each one is its own box here, and the component grouping is
  recorded rather than applied.
* **``uncertain_ignore`` is excluded in both directions.** It is not a target,
  and a detection landing on one is neither a true nor a false positive. The
  regions are written to ``metadata/regions.jsonl`` so evaluation can
  neutralize them.

Regions are recorded once, from the COCO box, in source coordinates. Window
records carry no regions: a region that crosses a window edge would otherwise
be stored as several clipped fragments and reconstructed from whichever window
was read last.

Windows are written once and shared between listings. ``--listing`` declares
what the build is for and is required: ``loso`` writes the three
leave-one-sheet-out folds (image lists plus an Ultralytics data YAML each, so
the crops are not triplicated on disk); ``test`` writes only ``test.txt`` for
the frozen blind sheets and no training split at all. Every guard runs before
any file is written.

Usage:
    python -m archeo_topia.datasets.build_detection_windows \\
        --coco-json annotation/cvat/v0.0.2/instances_default.json \\
        --images-root data/curated/datasets/mapsam_v02/images \\
        --schema annotation/cvat/labels.json --listing loso \\
        --output-dir data/curated/datasets/mapsam_det_v1_c3 \\
        --window 512 --stride 384
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
from PIL import Image

from archeo_topia.analysis.runtime import new_directory
from archeo_topia.formats.coco import decode_mask as _decode_mask
from archeo_topia.formats.georeference import SheetReference
from archeo_topia.formats.labels import LabelSchema

logger = logging.getLogger(__name__)

MOUND = "mound"
HARD_NEGATIVE = "hard_negative_symbol"
UNCERTAIN = "uncertain_ignore"

#: Repository root. Anything recorded relative to it carries no host path.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Splits declared by 1:100k parent sheet. Written by v0.6; see the file's own
#: ``rationale`` for why the grouping is by parent and not by 1:25k sheet id.
SPLITS_FILE = REPO_ROOT / "configs" / "splits" / "v0_6_splits.json"

#: What a build is for. ``loso`` is the v0.1-v0.5 leave-one-sheet-out folds;
#: ``test`` is the frozen blind test sheets, scored and never trained on.
ListingPlan = Literal["loso", "test"]
LISTING_PLANS: tuple[str, ...] = get_args(ListingPlan)

#: Kinds of source-coordinate region written to ``regions.jsonl``.
RegionKind = Literal["hard_negative_symbol", "uncertain_ignore"]

#: Decimal places of every box written to the dataset metadata. Rounding exists
#: only to keep float noise out of the records and of the box-difference listing
#: of the C1 reproduction; scoring that must match v0.5 never sees it.
REGION_BOX_DECIMALS = 2

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


def load_split_document(splits_file: Path | None = None) -> dict[str, Any]:
    """Load the whole splits document.

    Args:
        splits_file: Override for the splits file location.

    Returns:
        The parsed document, with ``version``, ``splits`` and ``sheets``.
    """
    path = splits_file or SPLITS_FILE
    return json.loads(path.read_text(encoding="utf-8"))


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

    Delegates to ``archeo_topia.formats.coco.decode_mask``, which is the one
    decoder this project has. ``bbox_fallback=False`` preserves the behaviour
    v0.1 to v0.5 were computed with: the v0.0.1 export carries one mound with a
    box and no mask, and those runs counted it as having no geometry. See that
    function's docstring.

    Args:
        annotation: COCO annotation dict.
        width: Source image width.
        height: Source image height.

    Returns:
        Boolean array of shape ``(height, width)``.
    """
    return _decode_mask(annotation, width, height, bbox_fallback=False)


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


def crop_path(dataset: Path, window_id: str) -> Path:
    """Return where a window's crop lives inside a dataset.

    Args:
        dataset: Dataset root.
        window_id: Window id.

    Returns:
        ``<dataset>/images/<window_id>.png``.
    """
    return dataset / "images" / f"{window_id}.png"


def listing_entry(window_id: str) -> str:
    """Return the line a listing file holds for a window.

    Args:
        window_id: Window id.

    Returns:
        ``./images/<window_id>.png``.
    """
    return f"./images/{window_id}.png"


def repo_relative(path: Path) -> str:
    """Return a path relative to the repository when it lies inside it.

    Args:
        path: A path, absolute or relative.

    Returns:
        The repository-relative POSIX path, or the path as given when it is
        outside the repository.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def validate_export(
    coco: dict[str, Any],
    schema_path: Path,
    images_root: Path,
    clips_json_paths: Sequence[Path] | None,
) -> None:
    """Check a COCO export before anything is built from it.

    Every violation is collected and raised together, so one run shows the
    whole list.

    Args:
        coco: Parsed CVAT COCO export.
        schema_path: Annotation schema (``labels.json``) the categories must
            belong to.
        images_root: Root of the clip images.
        clips_json_paths: ``clips.json`` manifests that the export must match
            clip for clip, or None when there are none to match.

    Raises:
        ValueError: Listing every violation.
    """
    problems: list[str] = []

    declared = set(LabelSchema.load(schema_path).label_names)
    unknown = sorted({c["name"] for c in coco["categories"]} - declared)
    if unknown:
        problems.append(f"categories not in {schema_path}: {unknown}")
    category_ids = {c["id"] for c in coco["categories"]}

    for image in coco["images"]:
        try:
            path = find_image(images_root, image["file_name"])
        except FileNotFoundError as error:
            problems.append(str(error))
            continue
        with Image.open(path) as handle:
            size = handle.size
        if size != (image["width"], image["height"]):
            problems.append(
                f"{image['file_name']}: COCO says {image['width']}x{image['height']}, "
                f"the PNG is {size[0]}x{size[1]}"
            )

    names = {i["id"]: i["file_name"] for i in coco["images"]}
    seen: set[tuple[str, int]] = set()
    for ann in coco["annotations"]:
        if ann["image_id"] not in names:
            problems.append(f"annotation {ann['id']} names unknown image id {ann['image_id']}")
            continue
        if ann["category_id"] not in category_ids:
            problems.append(
                f"annotation {ann['id']} names unknown category id {ann['category_id']}"
            )
        identity = (names[ann["image_id"]], ann["id"])
        if identity in seen:
            problems.append(f"annotation identity {identity} is not unique")
        seen.add(identity)

    if clips_json_paths is not None:
        export_files = set(names.values())
        named_by_clips: set[str] = set()
        for path in clips_json_paths:
            clips = {clip.file for clip in SheetReference.from_clips_json(path).clips}
            if not clips:
                problems.append(f"{path} names no clips")
            absent = sorted(clips - export_files)
            if absent:
                problems.append(f"{path} names clips missing from the export: {absent}")
            named_by_clips |= clips
        unnamed = sorted(export_files - named_by_clips)
        if unnamed:
            problems.append(f"export images named by no clips.json: {unnamed}")

    if problems:
        raise ValueError("invalid COCO export:\n  " + "\n  ".join(problems))


def validate_listing_sheets(
    plan: str,
    sheets: set[str],
    splits_document: dict[str, Any],
    folds: dict[str, str] | None,
) -> None:
    """Check that the exported sheets are the ones the listing plan allows.

    Args:
        plan: Listing plan.
        sheets: Sheets present in the export.
        splits_document: Parsed splits file.
        folds: The ``loso`` fold preset (fold name to eval sheet) for the
            ``loso`` plan, None otherwise.

    Raises:
        ValueError: Listing every violation, or on an unknown plan.
    """
    problems: list[str] = []
    if plan == "loso":
        if folds is None:
            raise ValueError("the loso plan needs the fold preset")
        fold_sheets = set(folds.values())
        if sheets - fold_sheets:
            problems.append(
                f"exported sheets outside preset {LOSO_PRESET}: {sorted(sheets - fold_sheets)}"
            )
        if fold_sheets - sheets:
            problems.append(
                f"preset {LOSO_PRESET} sheets not exported: {sorted(fold_sheets - sheets)}"
            )
    elif plan == "test":
        test = set(splits_document["splits"]["test"])
        frozen = {
            sheet for sheet, meta in splits_document["sheets"].items() if meta["frozen_blind_test"]
        }
        if sheets - test:
            problems.append(f"exported sheets not in the test split: {sorted(sheets - test)}")
        if sheets - frozen:
            problems.append(f"exported sheets not frozen blind test: {sorted(sheets - frozen)}")
    else:
        raise ValueError(f"unknown listing plan {plan!r}; have {list(LISTING_PLANS)}")
    if problems:
        raise ValueError(f"listing plan {plan!r} refuses this export:\n  " + "\n  ".join(problems))


def build(
    coco_json: Path,
    images_root: Path,
    output_dir: Path,
    window_size: int,
    stride: int,
    schema_path: Path,
    plan: str,
    clips_json_paths: Sequence[Path] | None,
    splits_file: Path = SPLITS_FILE,
) -> dict[str, Any]:
    """Tile every annotated clip and write a listing-shared detection dataset.

    Args:
        coco_json: CVAT COCO export.
        images_root: Root of the clip images.
        output_dir: Destination directory.
        window_size: Window side length in source pixels.
        stride: Step between window origins.
        schema_path: Annotation schema the export's categories must belong to.
        plan: What the build is for, ``loso`` or ``test``.
        clips_json_paths: ``clips.json`` manifests. Required for ``test``,
            rejected for ``loso``.
        splits_file: Splits file declaring the sheets of each plan.

    Returns:
        A build report dict, also written to ``metadata/build_report.json``.

    Raises:
        ValueError: If the plan, the clips manifests or the export are
            inconsistent. Nothing is written in that case.
        FileExistsError: If ``output_dir`` already exists; a dataset is built
            once, so a stale listing can never be scored as a fresh one.
    """
    if plan not in LISTING_PLANS:
        raise ValueError(f"unknown listing plan {plan!r}; have {list(LISTING_PLANS)}")
    if plan == "test" and not clips_json_paths:
        raise ValueError("the test plan requires --clips-json")
    if plan == "loso" and clips_json_paths:
        raise ValueError("the loso plan takes no --clips-json")

    with open(coco_json, encoding="utf-8") as handle:
        coco = json.load(handle)

    validate_export(coco, schema_path, images_root, clips_json_paths)
    splits_document = load_split_document(splits_file)
    folds = load_folds(splits_file=splits_file) if plan == "loso" else None
    validate_listing_sheets(
        plan, {sheet_of(i["file_name"]) for i in coco["images"]}, splits_document, folds
    )

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}

    by_image: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        by_image[ann["image_id"]][categories[ann["category_id"]]].append(ann)

    new_directory(output_dir)
    for sub in ("images", "labels", "metadata"):
        (output_dir / sub).mkdir()

    window_records: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []
    region_records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
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

        # Regions are recorded once, whole, from the COCO box -- never per window.
        for kind, items in ((HARD_NEGATIVE, negatives), (UNCERTAIN, ignores)):
            for region in items:
                rx, ry, rw, rh = region["bbox"]
                region_records.append(
                    {
                        "kind": kind,
                        "annotation_id": region["id"],
                        "sheet_id": sheet,
                        "source_image": file_name,
                        "bbox_source_xyxy": [
                            round(v, REGION_BOX_DECIMALS) for v in (rx, ry, rx + rw, ry + rh)
                        ],
                        "negative_type": (
                            region["attributes"]["negative_type"]
                            if kind == HARD_NEGATIVE
                            else None
                        ),
                    }
                )

        groups = component_groups(mounds, width, height)
        component_of = {
            ann_id: (label, len(members))
            for label, members in groups.items()
            for ann_id in members
        }
        counts["merged_components"] += sum(1 for m in groups.values() if len(m) > 1)

        source_path = find_image(images_root, file_name)
        source_hashes[file_name] = sha256_file(source_path)
        source = Image.open(source_path).convert("RGB")
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
                        "box_window_xyxy": [round(v, REGION_BOX_DECIMALS) for v in box],
                        "visible_fraction": round(visible, 4),
                        "truncated": visible < 0.999,
                    }
                )

            has_negative = any(
                window.x0 <= n["bbox"][0] + n["bbox"][2] / 2 < window.x1
                and window.y0 <= n["bbox"][1] + n["bbox"][3] / 2 < window.y1
                for n in negatives
            )

            crop = source.crop((window.x0, window.y0, window.x1, window.y1))
            crop.save(crop_path(output_dir, window_stem))

            label_path = output_dir / "labels" / f"{window_stem}.txt"
            lines = [
                "0 " + " ".join(f"{v:.6f}" for v in to_yolo(t["box_window_xyxy"], window_size))
                for t in targets
            ]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            counts["windows_total"] += 1
            if targets:
                counts["windows_with_mound"] += 1
            elif has_negative:
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

    metadata = output_dir / "metadata"
    _write_jsonl(metadata / "windows.jsonl", window_records)
    _write_jsonl(metadata / "annotations.jsonl", annotation_records)
    _write_jsonl(metadata / "regions.jsonl", region_records)
    listing = _write_listings(output_dir, plan, by_sheet_windows, folds)

    outputs = {
        f"metadata/{name}.jsonl": sha256_file(metadata / f"{name}.jsonl")
        for name in ("windows", "annotations", "regions")
    }
    for entry in listing["listings"].values():
        outputs[entry["file"]] = sha256_file(output_dir / entry["file"])
    outputs["crops"] = crops_digest(output_dir, [w["window_id"] for w in window_records])

    report = {
        "window_size": window_size,
        "stride": stride,
        "counts": counts,
        "windows_per_sheet": {s: len(v) for s, v in by_sheet_windows.items()},
        "mounds_per_sheet": _tally(annotation_records, "sheet_id"),
        "inputs": {
            "coco_json": repo_relative(coco_json),
            "schema": repo_relative(schema_path),
            "source_images_sha256": source_hashes,
            "clips_json_sha256": {
                SheetReference.from_clips_json(p).sheet_id: sha256_file(p)
                for p in clips_json_paths or ()
            },
            "splits_file": repo_relative(splits_file),
            "splits_version": splits_document["version"],
        },
        "listing": listing,
        "regions": {
            kind: sum(1 for r in region_records if r["kind"] == kind)
            for kind in get_args(RegionKind)
        },
        "outputs_sha256": outputs,
    }
    with open(metadata / "build_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def sha256_file(path: Path) -> str:
    """Return the sha256 of a file, streamed.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def crops_digest(output_dir: Path, window_ids: Sequence[str]) -> str:
    """Return the sha256 of ``"<window id> <sha256 of its PNG>\\n"`` per window.

    Windows are taken in sorted id order, so the digest does not depend on how
    the build enumerated them.

    Args:
        output_dir: Dataset root.
        window_ids: Every window of the build.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    for window_id in sorted(window_ids):
        png = sha256_file(crop_path(output_dir, window_id))
        digest.update(f"{window_id} {png}\n".encode())
    return digest.hexdigest()


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


def _write_listing(
    output_dir: Path, name: str, sheets: list[str], by_sheet: dict[str, list[str]]
) -> dict[str, Any]:
    """Write one image list and describe it.

    Args:
        output_dir: Dataset root.
        name: Listing name; the file is ``<name>.txt``.
        sheets: Sheets the list covers.
        by_sheet: Window ids grouped by sheet.

    Returns:
        The listing's entry for the build report.
    """
    lines = [listing_entry(window_id) for sheet in sheets for window_id in by_sheet[sheet]]
    (output_dir / f"{name}.txt").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    return {"file": f"{name}.txt", "sheets": sheets, "windows": len(lines)}


def _write_listings(
    output_dir: Path,
    plan: str,
    by_sheet: dict[str, list[str]],
    folds: dict[str, str] | None,
) -> dict[str, Any]:
    """Write the image lists the plan calls for.

    The lists live at the dataset root rather than in a subdirectory:
    Ultralytics resolves a ``./`` path inside an image list against the list
    file's own parent, so a list in ``splits/`` would look for the crops under
    ``splits/images/`` and find no labels.

    ``loso`` writes per-fold train and val lists plus a data YAML each. ``test``
    writes ``test.txt`` only: no YAML, no train list, no fold files.

    Args:
        output_dir: Dataset root.
        plan: ``loso`` or ``test``.
        by_sheet: Window ids grouped by sheet.
        folds: Fold name to held-out sheet, for ``loso``.

    Returns:
        The ``listing`` block of the build report.
    """
    listings: dict[str, Any] = {}
    if plan == "loso":
        if folds is None:
            raise ValueError("the loso plan needs the fold preset")
        for fold, eval_sheet in folds.items():
            for split, sheets in (
                ("train", [s for s in by_sheet if s != eval_sheet]),
                ("val", [eval_sheet]),
            ):
                listings[f"{fold}_{split}"] = _write_listing(
                    output_dir, f"{fold}_{split}", sheets, by_sheet
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
        fold_block: dict[str, Any] | None = {
            name: {"eval_sheet": sheet, "train_sheets": [s for s in folds.values() if s != sheet]}
            for name, sheet in folds.items()
        }
    else:
        listings["test"] = _write_listing(output_dir, "test", sorted(by_sheet), by_sheet)
        fold_block = None
    return {"plan": plan, "listings": listings, "folds": fold_block}


def main() -> None:
    """Parse arguments and build the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--listing", choices=LISTING_PLANS, required=True)
    parser.add_argument("--clips-json", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    report = build(
        coco_json=args.coco_json,
        images_root=args.images_root,
        output_dir=args.output_dir,
        window_size=args.window,
        stride=args.stride,
        schema_path=args.schema,
        plan=args.listing,
        clips_json_paths=args.clips_json,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
