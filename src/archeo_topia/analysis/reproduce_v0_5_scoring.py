#!/usr/bin/env python3
"""Reproduce the v0.5 scores with the generalized scorer (protocol C1 check).

The generalized scorer must reproduce the v0.5 gtfix detection metrics and the
v0.5 end-to-end table exactly on ``mapsam_det_v1``; the one permitted
difference is the C3/C4 fix, and it is enumerated. This module runs the same
raw detections and the same masks through two region inputs:

``V0``, the v0.5-equivalent inputs
    Hard-negative and ignore regions rebuilt the v0.5 way from the **original**
    dataset's ``windows.jsonl`` (window-clipped boxes, last window wins), and no
    ignore neutralization in end to end. **V0 must equal v0.5 exactly**, with
    ``==`` and NaN equal to NaN and no tolerance. That proves the
    generalization changed nothing else.

``V1``, as fixed
    Regions from the rebuilt dataset's ``regions.jsonl`` (C3) and ignore
    neutralization in end to end (C4). **V1 minus V0 is enumerated**, leaf by
    leaf.

The v0.5 reconstruction is a deliberate replica of the defect and is private to
this module: it exists only to attribute differences. Nothing else may import
it (a test enforces that).

Before any inference the rebuilt dataset must be the original dataset apart
from the region records, or the run stops.

**The build input is restored, not tolerated.** ``mapsam_det_v1`` was built from
a CVAT export whose one hard-negative ``negative_type`` the repository copy
patched afterwards (``docs/mapsam/v005/RESULTS.md``, "Known data notes"). The
protocol allows exactly one difference, C3, so the rebuild must start from the
export the original was actually built from. ``restore-input`` recovers it from
``mapsam_det_v1`` itself and stops unless the recovered annotations are exactly
:data:`V05_REPOSITORY_PATCHED_ANNOTATIONS`; ``run`` re-proves that and records it.
No comparison anywhere has a tolerance.

Usage:
    python -m archeo_topia.analysis.reproduce_v0_5_scoring restore-input \\
        --original-dataset data/curated/datasets/mapsam_det_v1 \\
        --coco-json annotation/cvat/v0.0.2/instances_default.json \\
        --output artifacts/detection/v0_6_c1_build_input/instances_default.json

    python -m archeo_topia.analysis.reproduce_v0_5_scoring run \\
        --original-dataset data/curated/datasets/mapsam_det_v1 \\
        --rebuilt-dataset data/curated/datasets/mapsam_det_v1_c3 \\
        --detector-run artifacts/detection/v0_5_yolo26s_gtfix \\
        --v05-end-to-end artifacts/detection/v0_5_end_to_end/end_to_end.json \\
        --mapsam-root artifacts/models/mapsam --mapsam-prefix v0_4r_loso512 \\
        --sam-checkpoint models/checkpoints/sam/sam_vit_b_01ec64.pth \\
        --images-root data/curated/datasets/mapsam_v02/images \\
        --coco-json annotation/cvat/v0.0.2/instances_default.json \\
        --device cuda:0 --output-dir artifacts/detection/v0_6_c1_reproduction
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from archeo_topia.analysis.detection_metrics import (
    Detection,
    EvaluationSet,
    Region,
    confidence_key,
    load_evaluation_set,
    neutralize,
)
from archeo_topia.analysis.mapsam_end_to_end import (
    BOX_MODES,
    IOU_THRESHOLDS,
    decoder_checkpoint_path,
    load_mapsam,
    load_truth_masks,
    resolve_candidates,
    score_end_to_end,
    segment_candidates,
)
from archeo_topia.analysis.runtime import (
    environment_record,
    new_directory,
    resolve_device,
    write_record,
)
from archeo_topia.datasets.build_detection_windows import (
    FOLDS,
    HARD_NEGATIVE,
    REGION_BOX_DECIMALS,
    UNCERTAIN,
    find_image,
    repo_relative,
    sha256_file,
)
from archeo_topia.training.train_mound_detector import (
    DETECTOR_IMGSZ,
    HEADLINE_CONFIDENCE,
    MERGE_RADIUS,
    MIN_CONFIDENCE,
    THRESHOLDS,
    fold_weights_path,
    metrics_document,
    operating_curve,
    predict_windows,
    report_markdown,
    write_detections,
)

logger = logging.getLogger(__name__)

#: Window and clip size of the v0.5 end-to-end scoring (``RESULTS.md``).
WINDOW_PX = 512
IMAGE_SIZE = 1024

#: Region lists the v0.5 builder wrote into every window record.
LEGACY_REGION_KEYS = ("hard_negatives", "uncertain_ignore")

#: Region lists by kind, as the v0.5 window records name them.
_LEGACY_LISTS = {HARD_NEGATIVE: "hard_negatives", UNCERTAIN: "uncertain_ignore"}

#: Annotations whose ``negative_type`` the repository CVAT export carries
#: differently from the export ``mapsam_det_v1`` was built from. See
#: ``docs/mapsam/v005/RESULTS.md``, "Known data notes / The negative_type patch":
#: the repository copy was patched after the build, in this one field. This is the
#: only hand-declared fact; the values themselves are read from ``mapsam_det_v1``.
V05_REPOSITORY_PATCHED_ANNOTATIONS = frozenset({235})

#: Decimal places of the v0.5 per-annotation IoU values.
V05_IOU_DECIMALS = 4

RegionKey = tuple[str, str, int]


def legacy_window_clipped_regions(windows_jsonl: Path) -> dict[RegionKey, Region]:
    """Rebuild regions the v0.5 way: window-clipped boxes, the last window wins.

    A region crossing a window edge is recorded by every window it touches,
    clipped to that window. The v0.5 loader restored it from whichever window
    came last, so a crossing region came back as a fragment. This replicates
    that defect so its effect can be measured; it is not a loader.

    The sums are exactly the ones the v0.5 loader computed, unrounded, so V0
    scoring sees what v0.5 saw.

    Args:
        windows_jsonl: ``metadata/windows.jsonl`` of a dataset built before
            regions were written.

    Returns:
        Region by ``(kind, image, annotation id)``, in first-appearance order.
    """
    regions: dict[RegionKey, Region] = {}
    for line in windows_jsonl.read_text(encoding="utf-8").splitlines():
        window = json.loads(line)
        x0, y0, _, _ = window["window_xyxy"]
        for kind, list_key in _LEGACY_LISTS.items():
            for record in window[list_key]:
                bx0, by0, bx1, by1 = record["box_window_xyxy"]
                regions[(kind, window["source_image"], record["annotation_id"])] = Region(
                    image=window["source_image"],
                    x0=bx0 + x0,
                    y0=by0 + y0,
                    x1=bx1 + x0,
                    y1=by1 + y0,
                    sheet=window["sheet_id"],
                    negative_type=record["negative_type"] if kind == HARD_NEGATIVE else None,
                )
    return regions


def v05_input_differences(windows_jsonl: Path, coco: dict[str, Any]) -> dict[int, tuple[str, str]]:
    """Find where the repository export differs from the export v0.5 was built from.

    ``mapsam_det_v1``'s ``windows.jsonl`` recorded each hard negative's
    ``negative_type`` at build time, so it is the only surviving copy of that
    field in the export the dataset was built from.

    Args:
        windows_jsonl: ``metadata/windows.jsonl`` of the original dataset.
        coco: The repository CVAT COCO export.

    Returns:
        ``{annotation_id: (repo_value, built_value)}`` for every hard negative
        whose export value differs from the one the original build recorded.

    Raises:
        ValueError: If one annotation carries two values across windows, or a
            recorded annotation is absent from the export.
    """
    built: dict[int, str] = {}
    for line in windows_jsonl.read_text(encoding="utf-8").splitlines():
        for record in json.loads(line)["hard_negatives"]:
            annotation_id, value = record["annotation_id"], record["negative_type"]
            if built.setdefault(annotation_id, value) != value:
                raise ValueError(
                    f"annotation {annotation_id} carries negative_type {built[annotation_id]!r} "
                    f"and {value!r} across windows of {windows_jsonl}"
                )
    category = {c["id"]: c["name"] for c in coco["categories"]}
    exported = {
        ann["id"]: ann["attributes"]["negative_type"]
        for ann in coco["annotations"]
        if category[ann["category_id"]] == HARD_NEGATIVE
    }
    absent = sorted(built.keys() - exported.keys())
    if absent:
        raise ValueError(f"hard negatives in {windows_jsonl} absent from the export: {absent}")
    return {
        annotation_id: (exported[annotation_id], value)
        for annotation_id, value in built.items()
        if exported[annotation_id] != value
    }


def _require_declared_differences(differences_by_id: Mapping[int, tuple[str, str]]) -> None:
    """Stop unless the differing annotations are exactly the declared patched set."""
    found = set(differences_by_id)
    extra = sorted(found - V05_REPOSITORY_PATCHED_ANNOTATIONS)
    missing = sorted(V05_REPOSITORY_PATCHED_ANNOTATIONS - found)
    if extra or missing:
        raise ValueError(
            "the repository export differs from the v0.5 build input in annotations "
            f"{sorted(found)}; declared {sorted(V05_REPOSITORY_PATCHED_ANNOTATIONS)} "
            f"(unexpected: {extra}, not differing: {missing})"
        )


def _restored_entries(differences_by_id: Mapping[int, tuple[str, str]]) -> list[dict[str, Any]]:
    """Describe the restored annotations for a record."""
    return [
        {"annotation_id": annotation_id, "repo_value": repo_value, "built_value": built_value}
        for annotation_id, (repo_value, built_value) in sorted(differences_by_id.items())
    ]


def restore_v05_build_input(
    original_dataset: Path, coco_json: Path, output: Path
) -> dict[str, Any]:
    """Write the CVAT export ``mapsam_det_v1`` was built from.

    Args:
        original_dataset: The v0.5 dataset.
        coco_json: The repository CVAT export.
        output: Restored export to write; its parent directory is created and
            must not exist, so nothing is ever replaced.

    Returns:
        The source and output paths and hashes and the restored annotations.

    Raises:
        ValueError: Unless the differing annotations are exactly
            :data:`V05_REPOSITORY_PATCHED_ANNOTATIONS`.
        FileExistsError: If the output directory exists.
    """
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    differing = v05_input_differences(original_dataset / "metadata" / "windows.jsonl", coco)
    _require_declared_differences(differing)
    restored = copy.deepcopy(coco)
    for ann in restored["annotations"]:
        if ann["id"] in differing:
            ann["attributes"]["negative_type"] = differing[ann["id"]][1]
    new_directory(output.parent)
    write_record(output, json.dumps(restored, indent=2))
    return {
        "source": repo_relative(coco_json),
        "source_sha256": sha256_file(coco_json),
        "output": repo_relative(output),
        "output_sha256": sha256_file(output),
        "restored": _restored_entries(differing),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _box(region: Region) -> list[float]:
    """Return a region's box as a list."""
    return [region.x0, region.y0, region.x1, region.y1]


def region_differences(
    legacy: Mapping[RegionKey, Region], regions_jsonl: Path
) -> tuple[list[str], list[dict[str, Any]]]:
    """Compare the v0.5 reconstruction with the source-coordinate region records.

    Args:
        legacy: Output of :func:`legacy_window_clipped_regions`.
        regions_jsonl: ``metadata/regions.jsonl`` of the rebuilt dataset.

    The v0.5 box is compared after rounding to ``REGION_BOX_DECIMALS``, the
    precision the builder records, so float noise is not listed as a difference.
    That rounding exists only for this listing; V0 scoring never sees it, and
    ``v05_box`` is reported unrounded.

    Returns:
        Problems (a region lost or gained, a changed ``negative_type``) and the
        regions whose box differs, each with both boxes.
    """
    rebuilt = {
        (r["kind"], r["source_image"], r["annotation_id"]): r for r in _read_jsonl(regions_jsonl)
    }
    problems: list[str] = []
    lost = sorted(legacy.keys() - rebuilt.keys())
    gained = sorted(rebuilt.keys() - legacy.keys())
    if lost:
        problems.append(f"regions in the original dataset but not the rebuilt one: {lost}")
    if gained:
        problems.append(f"regions in the rebuilt dataset but not the original: {gained}")
    differing = []
    for key in sorted(legacy.keys() & rebuilt.keys()):
        record = rebuilt[key]
        if legacy[key].negative_type != record["negative_type"]:
            problems.append(
                f"{key}: negative_type {legacy[key].negative_type!r} became "
                f"{record['negative_type']!r}"
            )
        if [round(v, REGION_BOX_DECIMALS) for v in _box(legacy[key])] != record[
            "bbox_source_xyxy"
        ]:
            differing.append(
                {
                    "kind": key[0],
                    "image": key[1],
                    "annotation_id": key[2],
                    "v05_box": _box(legacy[key]),
                    "source_box": record["bbox_source_xyxy"],
                }
            )
    return problems, differing


def _files_differ(first: Path, second: Path, names: list[str]) -> list[str]:
    """Return the names whose bytes differ between two directories."""
    return [n for n in names if (first / n).read_bytes() != (second / n).read_bytes()]


def _name_set_problem(label: str, original: list[str], rebuilt: list[str]) -> str | None:
    """Describe a difference between two file-name lists, or None when equal."""
    if original == rebuilt:
        return None
    return (
        f"{label} differ: only in original {sorted(set(original) - set(rebuilt))}, "
        f"only in rebuilt {sorted(set(rebuilt) - set(original))}"
    )


def _crop_problems(original: Path, rebuilt: Path) -> list[str]:
    """Compare the crops pixel for pixel, which survives a PNG encoder change."""
    names = sorted(p.name for p in (original / "images").glob("*.png"))
    problem = _name_set_problem(
        "crops", names, sorted(p.name for p in (rebuilt / "images").glob("*.png"))
    )
    if problem:
        return [problem]
    differing = []
    for name in names:
        with (
            Image.open(original / "images" / name) as a,
            Image.open(rebuilt / "images" / name) as b,
        ):
            if a.mode != b.mode or not np.array_equal(np.asarray(a), np.asarray(b)):
                differing.append(name)
    return [f"crops differ in pixels: {differing}"] if differing else []


def _yaml_problems(original: Path, rebuilt: Path) -> list[str]:
    """Compare the fold YAMLs, whose ``path:`` line must name the rebuilt directory."""
    names = sorted(p.name for p in original.glob("fold*.yaml"))
    problem = _name_set_problem(
        "fold YAMLs", names, sorted(p.name for p in rebuilt.glob("fold*.yaml"))
    )
    if problem:
        return [problem]
    problems = []
    for name in names:
        old = (original / name).read_text(encoding="utf-8").splitlines()
        new = (rebuilt / name).read_text(encoding="utf-8").splitlines()
        if [ln for ln in old if not ln.startswith("path:")] != [
            ln for ln in new if not ln.startswith("path:")
        ]:
            problems.append(f"{name} differs beyond its path: line")
        if f"path: {rebuilt.resolve()}" not in new:
            problems.append(f"{name} does not name {rebuilt.resolve()} as its path")
    return problems


def dataset_equivalence(original: Path, rebuilt: Path) -> dict[str, Any]:
    """Check that a rebuilt dataset is the original apart from its region records.

    Args:
        original: The v0.5 dataset, built before regions were written.
        rebuilt: The same export rebuilt with ``build_detection_windows``.

    Returns:
        ``problems`` (empty when equivalent), the ``region_differences``
        (expected: only regions crossing a window edge) and region counts.
    """
    problems: list[str] = []
    if (original / "metadata" / "annotations.jsonl").read_bytes() != (
        rebuilt / "metadata" / "annotations.jsonl"
    ).read_bytes():
        problems.append("annotations.jsonl is not byte-identical")

    old_windows = [
        {k: v for k, v in record.items() if k not in LEGACY_REGION_KEYS}
        for record in _read_jsonl(original / "metadata" / "windows.jsonl")
    ]
    if old_windows != _read_jsonl(rebuilt / "metadata" / "windows.jsonl"):
        problems.append("windows.jsonl differs once the original's region lists are dropped")

    for pattern in ("fold*_train.txt", "fold*_val.txt", "labels/*.txt"):
        names = sorted(p.relative_to(original).as_posix() for p in original.glob(pattern))
        rebuilt_names = sorted(p.relative_to(rebuilt).as_posix() for p in rebuilt.glob(pattern))
        problem = _name_set_problem(pattern, names, rebuilt_names)
        if problem:
            problems.append(problem)
        else:
            differing = _files_differ(original, rebuilt, names)
            if differing:
                problems.append(f"{pattern} differ in bytes: {differing}")
    problems += _yaml_problems(original, rebuilt)
    problems += _crop_problems(original, rebuilt)

    old_report = json.loads((original / "metadata" / "build_report.json").read_text("utf-8"))
    new_report = json.loads((rebuilt / "metadata" / "build_report.json").read_text("utf-8"))
    for key in ("counts", "windows_per_sheet", "mounds_per_sheet"):
        if old_report[key] != new_report[key]:
            problems.append(f"build_report {key} differs: {old_report[key]} vs {new_report[key]}")

    legacy = legacy_window_clipped_regions(original / "metadata" / "windows.jsonl")
    region_problems, differing_boxes = region_differences(
        legacy, rebuilt / "metadata" / "regions.jsonl"
    )
    return {
        "problems": problems + region_problems,
        "region_differences": differing_boxes,
        "regions": {
            "v05_reconstruction": len(legacy),
            "rebuilt": len(_read_jsonl(rebuilt / "metadata" / "regions.jsonl")),
        },
    }


def v05_evaluation_set(
    rebuilt: EvaluationSet, legacy: Mapping[RegionKey, Region]
) -> EvaluationSet:
    """Return the evaluation set with regions as v0.5 saw them (V0).

    Args:
        rebuilt: Evaluation set over the rebuilt dataset.
        legacy: Output of :func:`legacy_window_clipped_regions`.

    Returns:
        The same set with the v0.5 reconstruction of the declared sheets' regions.
    """
    sheets = set(rebuilt.sheets)
    return dataclasses.replace(
        rebuilt,
        hard_negatives=tuple(
            r for (kind, _, _), r in legacy.items() if kind == HARD_NEGATIVE and r.sheet in sheets
        ),
        ignore_regions=tuple(
            r for (kind, _, _), r in legacy.items() if kind == UNCERTAIN and r.sheet in sheets
        ),
    )


def _same(expected: Any, actual: Any) -> bool:
    """Exact equality of two leaves, with NaN equal to NaN."""
    if isinstance(expected, float) and isinstance(actual, float):
        return expected == actual or (math.isnan(expected) and math.isnan(actual))
    return bool(expected == actual)


def differences(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    """Enumerate every leaf at which two nested documents differ.

    Equality is ``==`` with NaN equal to NaN. There is no tolerance: a one-ulp
    difference is a difference.

    Args:
        expected: The reference document.
        actual: The document compared with it.
        path: Location of these two values, for the report.

    Returns:
        One entry per difference: ``path`` and the two values, or ``path`` and
        which side holds a key the other lacks.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = [
            {"path": f"{path}.{key}", "only_in": "expected" if key in expected else "actual"}
            for key in sorted(expected.keys() ^ actual.keys())
        ]
        for key in sorted(expected.keys() & actual.keys()):
            found += differences(expected[key], actual[key], f"{path}.{key}")
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [{"path": f"{path}.length", "expected": len(expected), "actual": len(actual)}]
        return [
            d
            for i, (e, a) in enumerate(zip(expected, actual, strict=True))
            for d in differences(e, a, f"{path}[{i}]")
        ]
    if _same(expected, actual):
        return []
    return [{"path": path, "expected": expected, "actual": actual}]


def _normalized(document: Any) -> Any:
    """Round-trip through JSON so tuples and lists compare as the files do."""
    return json.loads(json.dumps(document))


def project_v05_detection(
    block: dict[str, Any], eval_sheet: str, confidence: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project a new pooled ``DetectionBlock`` onto the v0.5 ``metrics.json`` layout.

    The renamed or restructured keys map back as follows. ``precision.value``
    is ``precision_at_association_radius``. The hard-negative attribution and
    ``background`` form the flat ``by_negative_type``. The pooled ``recall`` of
    the one evaluated sheet is ``per_sheet[eval_sheet]``. Keys v0.5 never had
    (``precision`` interval, ``f1``, ``per_annotation_distance_px``) are dropped.

    Args:
        block: Pooled block from ``score_detections``.
        eval_sheet: The fold's held-out sheet.
        confidence: Confidence key of the block, for the report path.

    Returns:
        The projection and any mismatch the projection itself exposes: v0.5 had
        no ``untyped`` key, so a hard negative without a ``negative_type`` can
        not exist in a v0.5 comparison.
    """
    false_positives = block["false_positives"]
    near = false_positives["near_hard_negative"]
    problems = []
    if near["without_negative_type"] != 0:
        problems.append(
            {
                "path": f"{confidence}.false_positives.near_hard_negative.without_negative_type",
                "expected": 0,
                "actual": near["without_negative_type"],
            }
        )
    projected = {
        "detections": block["detections"],
        "detections_neutralized_by_ignore": block["detections_neutralized_by_ignore"],
        "annotations": block["annotations"],
        "association_radius": block["association_radius"],
        "recall": block["recall"],
        "localization_error_px": block["localization_error_px"],
        "false_positives": {
            "count": false_positives["count"],
            "per_window": false_positives["per_window"],
            "per_megapixel": false_positives["per_megapixel"],
            "by_negative_type": {
                **near["by_negative_type"],
                "background": false_positives["background"],
            },
        },
        "precision_at_association_radius": block["precision"]["value"],
        "subsets": block["subsets"],
        "per_sheet": {eval_sheet: block["recall"]},
    }
    return _normalized(projected), problems


def compare_detection_with_v05(
    v05: dict[str, Any], v0_document: dict[str, Any], eval_sheet: str
) -> list[dict[str, Any]]:
    """Compare V0 detection metrics with a v0.5 ``metrics.json``, leaf by leaf.

    Args:
        v05: The parsed v0.5 ``metrics.json``.
        v0_document: V0 ``metrics_document`` over the same detections.
        eval_sheet: The fold's held-out sheet.

    Returns:
        Every mismatch: the lowest-confidence block, the nine ``operating_points``
        fields at each threshold, and v0.5's absence of an untyped bucket.
    """
    lowest = confidence_key(MIN_CONFIDENCE)
    mismatches: list[dict[str, Any]] = []
    for key, block in v0_document["by_confidence"].items():
        projected, problems = project_v05_detection(block["pooled"], eval_sheet, key)
        mismatches += problems
        if key == lowest:
            expected = {k: v for k, v in v05.items() if k not in ("config", "operating_points")}
            mismatches += differences(expected, projected, lowest)
    mismatches += differences(
        v05["operating_points"], _normalized(operating_curve(v0_document)), "operating_points"
    )
    return mismatches


def project_v05_end_to_end(block: dict[str, Any]) -> dict[str, Any]:
    """Project a pooled ``E2EBlock`` onto one arm of the v0.5 ``end_to_end.json``.

    Per-annotation IoUs are rounded to the 4 dp v0.5 stored, and keyed by the
    annotation id alone as v0.5 keyed them.

    Args:
        block: Pooled block from ``score_end_to_end``.

    Returns:
        The projection.

    Raises:
        ValueError: If two annotations share an id, so the v0.5 keys would collide.
    """
    ious: dict[str, float] = {}
    for key, iou in block["per_annotation_iou"].items():
        annotation_id = key.rsplit("#", 1)[1]
        if annotation_id in ious:
            raise ValueError(f"annotation id {annotation_id} repeats; v0.5 keys would collide")
        ious[annotation_id] = round(iou, V05_IOU_DECIMALS)
    projected: dict[str, Any] = {
        "candidates": block["candidates"],
        "scorable_annotations": block["scorable_annotations"],
        "false_masks": block["false_masks"],
        "mean_iou": block["mean_iou"],
    }
    for threshold in IOU_THRESHOLDS:
        projected[f"rate_iou_{threshold:g}"] = block[f"iou_{threshold:g}"]["rate"]
        projected[f"hits_iou_{threshold:g}"] = block[f"iou_{threshold:g}"]["hits"]
    projected["missed_rate"] = block["missed"]["rate"]
    projected["per_annotation_iou"] = ious
    return _normalized(projected)


def compare_end_to_end_with_v05(
    v05_fold: dict[str, Any], variant: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compare V0 end-to-end arms with one fold of the v0.5 ``end_to_end.json``.

    Args:
        v05_fold: The v0.5 record of one fold.
        variant: V0 arms by name, each a ``score_end_to_end`` result.

    Returns:
        Every mismatch, with the arm in its path.
    """
    mismatches = []
    for arm, scored in variant.items():
        mismatches += differences(
            v05_fold["arms"][arm], project_v05_end_to_end(scored["pooled"]), arm
        )
    return mismatches


@dataclasses.dataclass(frozen=True)
class ReproductionInputs:
    """Where the reproduction reads from.

    Attributes:
        original_dataset: The v0.5 dataset, built before regions were written.
        rebuilt_dataset: The same export rebuilt with regions.
        detector_run: v0.5 detector run directory (``<fold>/weights``, ``<fold>/metrics.json``).
        v05_end_to_end: The v0.5 ``end_to_end.json``.
        mapsam_root: Directory of MapSAM decoder runs.
        mapsam_prefix: Decoder run-name prefix.
        sam_checkpoint: Base SAM checkpoint.
        images_root: Root of the source clips.
        coco_json: COCO export the dataset was built from, for truth masks.
    """

    original_dataset: Path
    rebuilt_dataset: Path
    detector_run: Path
    v05_end_to_end: Path
    mapsam_root: Path
    mapsam_prefix: str
    sam_checkpoint: Path
    images_root: Path
    coco_json: Path


def _end_to_end_variants(
    raw: list[Detection],
    v0_set: EvaluationSet,
    v1_set: EvaluationSet,
    decoder: Path,
    inputs: ReproductionInputs,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Segment once per arm and score the masks under V0 and V1.

    V0 is v0.5's end to end, which dropped the ignore regions, so its set has
    none. V1 neutralizes them.

    Returns:
        The V0 and V1 records, each with candidate counts and one block per arm.
    """
    v0e_set = dataclasses.replace(v0_set, ignore_regions=())
    candidates = resolve_candidates(raw, HEADLINE_CONFIDENCE, MERGE_RADIUS)
    clip_paths = {name: find_image((inputs.images_root,), name) for name in v1_set.image_sheet}
    truth_masks = load_truth_masks(inputs.coco_json, v1_set)
    sam = load_mapsam(decoder, inputs.sam_checkpoint, device)

    records = [
        {
            "candidates": len(candidates),
            "candidates_neutralized_by_ignore": neutralize(candidates, s.ignore_regions)[1],
            "arms": {},
        }
        for s in (v0e_set, v1_set)
    ]
    for arm in BOX_MODES:
        masks = segment_candidates(sam, candidates, clip_paths, arm, WINDOW_PX, IMAGE_SIZE, device)
        for record, evaluation_set in zip(records, (v0e_set, v1_set), strict=True):
            record["arms"][arm] = score_end_to_end(candidates, masks, evaluation_set, truth_masks)
    return records[0], records[1]


def reproduce_fold(
    inputs: ReproductionInputs,
    fold: str,
    legacy: Mapping[RegionKey, Region],
    v05_end_to_end: dict[str, Any],
    device: torch.device,
    out: Path,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Run one fold's inference once and compare V0 and V1 with v0.5.

    Args:
        inputs: Where to read from.
        fold: Fold name.
        legacy: The v0.5 reconstruction of the original dataset's regions.
        v05_end_to_end: The v0.5 end-to-end record of this fold.
        device: Device inference runs on.
        out: Output root, which gains this fold's files.

    Returns:
        The fold's detection result, end-to-end result and the V0 found-only
        mean IoU of the ``fixed`` arm.
    """
    eval_sheet = FOLDS[fold]
    rebuilt_set = load_evaluation_set(
        inputs.rebuilt_dataset, inputs.rebuilt_dataset / f"{fold}_val.txt", [eval_sheet]
    )
    v0_set = v05_evaluation_set(rebuilt_set, legacy)
    weights = fold_weights_path(inputs.detector_run, fold)
    raw = predict_windows(weights, rebuilt_set, DETECTOR_IMGSZ, MIN_CONFIDENCE, device)
    write_detections(out / "detections" / f"{fold}.jsonl", raw)

    documents = {
        tag: metrics_document(
            raw, weights, evaluation_set, DETECTOR_IMGSZ, MERGE_RADIUS, THRESHOLDS, device, fold
        )
        for tag, evaluation_set in (("v05_inputs", v0_set), ("as_fixed", rebuilt_set))
    }
    for tag, document in documents.items():
        fold_dir = out / tag / "detector" / fold
        fold_dir.mkdir(parents=True)
        write_record(fold_dir / "metrics.json", json.dumps(document, indent=2))
    write_record(
        out / "as_fixed" / "detector" / fold / "report.md",
        report_markdown(documents["as_fixed"], fold),
    )
    v05_metrics = json.loads(
        (inputs.detector_run / fold / "metrics.json").read_text(encoding="utf-8")
    )
    detection = {
        "v05_mismatches": compare_detection_with_v05(
            v05_metrics, documents["v05_inputs"], eval_sheet
        ),
        "c3_differences": differences(
            _normalized(documents["v05_inputs"]), _normalized(documents["as_fixed"])
        ),
    }

    v0e, v1e = _end_to_end_variants(
        raw,
        v0_set,
        rebuilt_set,
        decoder_checkpoint_path(inputs.mapsam_root, inputs.mapsam_prefix, fold),
        inputs,
        device,
    )
    for tag, record in (("v05_inputs", v0e), ("as_fixed", v1e)):
        fold_dir = out / tag / "end_to_end" / fold
        fold_dir.mkdir(parents=True)
        write_record(fold_dir / "end_to_end.json", json.dumps(record, indent=2))
    end_to_end = {
        "v05_mismatches": compare_end_to_end_with_v05(v05_end_to_end, v0e["arms"]),
        "c3_c4_differences": differences(_normalized(v0e), _normalized(v1e)),
    }
    found_only = v0e["arms"]["fixed"]["pooled"]["found"]["mean_iou"]
    return detection, end_to_end, found_only


def format_reproduction_report(report: Mapping[str, Any]) -> str:
    """Render the reproduction report for a human reviewer.

    Args:
        report: The ``reproduction_report.json`` document.

    Returns:
        Markdown text.
    """

    def listed(items: list[dict[str, Any]]) -> list[str]:
        return [f"- `{json.dumps(item)}`" for item in items] or ["- none"]

    lines = [
        "# v0.5 reproduction check",
        "",
        f"**passed: {report['passed']}** (V0 equals v0.5 exactly on every fold).",
        "",
        "## Restored build input",
        "",
        "The rebuild started from the CVAT export `mapsam_det_v1` was built from, not the "
        "repository copy. These annotations differ in `negative_type` between the two "
        "(the documented v0.5 repository patch). This is an input restoration, separate "
        "from the C3/C4 differences below.",
        "",
        *listed(report["restored_build_input"]["restored"]),
        "",
        "## Dataset equivalence",
        "",
        f"Regions: {report['dataset_equivalence']['regions']}. "
        "The boxes below differ between the v0.5 window-clipped reconstruction and the "
        "source-coordinate record (the C3 fix).",
        "",
        *listed(report["region_differences"]),
    ]
    for section, c_key in (("detection", "c3_differences"), ("end_to_end", "c3_c4_differences")):
        lines += ["", f"## {section}"]
        for fold, result in report[section].items():
            lines += [
                "",
                f"### {fold}: V0 against v0.5 ({len(result['v05_mismatches'])} mismatches)",
                "",
                *listed(result["v05_mismatches"]),
                "",
                f"### {fold}: V1 minus V0 ({len(result[c_key])} differences)",
                "",
                *listed(result[c_key]),
            ]
    lines += [
        "",
        "## Found-only mean IoU, fixed arm (reviewer comparison, not machine-checked)",
        "",
        "Compare with `docs/mapsam/v005/RESULTS.md` (end-to-end table, mean IoU, found only).",
        "",
        "| fold | computed (V0) |",
        "|---|---:|",
        *[f"| {fold} | {value:.4f} |" for fold, value in report["found_only_mean_iou"].items()],
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    """Check the datasets, then run and compare every fold.

    Args:
        args: Parsed ``run`` arguments.

    Raises:
        ValueError: If the repository export differs from the v0.5 build input
            in anything but the declared annotations, or the rebuilt dataset is
            not the original apart from its regions.
    """
    inputs = ReproductionInputs(
        original_dataset=args.original_dataset,
        rebuilt_dataset=args.rebuilt_dataset,
        detector_run=args.detector_run,
        v05_end_to_end=args.v05_end_to_end,
        mapsam_root=args.mapsam_root,
        mapsam_prefix=args.mapsam_prefix,
        sam_checkpoint=args.sam_checkpoint,
        images_root=args.images_root,
        coco_json=args.coco_json,
    )
    device = resolve_device(args.device)
    out = args.output_dir
    restored = v05_input_differences(
        inputs.original_dataset / "metadata" / "windows.jsonl",
        json.loads(inputs.coco_json.read_text(encoding="utf-8")),
    )
    _require_declared_differences(restored)
    equivalence = dataset_equivalence(inputs.original_dataset, inputs.rebuilt_dataset)
    if equivalence["problems"]:
        raise ValueError(
            "the rebuilt dataset is not the original apart from its regions:\n  "
            + "\n  ".join(equivalence["problems"])
        )
    legacy = legacy_window_clipped_regions(inputs.original_dataset / "metadata" / "windows.jsonl")
    v05_end_to_end = {
        record["fold"]: record
        for record in json.loads(inputs.v05_end_to_end.read_text(encoding="utf-8"))
    }

    new_directory(out)
    new_directory(out / "detections")
    detection: dict[str, Any] = {}
    end_to_end: dict[str, Any] = {}
    found_only: dict[str, float] = {}
    for fold in FOLDS:
        detection[fold], end_to_end[fold], found_only[fold] = reproduce_fold(
            inputs, fold, legacy, v05_end_to_end[fold], device, out
        )

    passed = all(not r["v05_mismatches"] for r in (*detection.values(), *end_to_end.values()))
    report = {
        "environment": environment_record(device),
        "inputs": {
            k: repo_relative(v) if isinstance(v, Path) else v
            for k, v in dataclasses.asdict(inputs).items()
        },
        "restored_build_input": {
            "source": repo_relative(inputs.coco_json),
            "restored": _restored_entries(restored),
        },
        "dataset_equivalence": {"regions": equivalence["regions"]},
        "region_differences": equivalence["region_differences"],
        "detection": detection,
        "end_to_end": end_to_end,
        "found_only_mean_iou": found_only,
        "passed": passed,
    }
    write_record(out / "reproduction_report.json", json.dumps(report, indent=2))
    markdown = format_reproduction_report(report)
    write_record(out / "reproduction_report.md", markdown)
    print(markdown)
    if not passed:
        sys.exit(1)


def main() -> None:
    """Parse arguments and run the ``restore-input`` or ``run`` subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    restore = commands.add_parser("restore-input", help="Write the export v0.5 was built from")
    restore.add_argument("--original-dataset", type=Path, required=True)
    restore.add_argument("--coco-json", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    restore.add_argument("--log-level", default="INFO")

    check = commands.add_parser("run", help="Reproduce v0.5 with the generalized scorer")
    check.add_argument("--original-dataset", type=Path, required=True)
    check.add_argument("--rebuilt-dataset", type=Path, required=True)
    check.add_argument("--detector-run", type=Path, required=True)
    check.add_argument("--v05-end-to-end", type=Path, required=True)
    check.add_argument("--mapsam-root", type=Path, required=True)
    check.add_argument("--mapsam-prefix", required=True)
    check.add_argument("--sam-checkpoint", type=Path, required=True)
    check.add_argument("--images-root", type=Path, required=True)
    check.add_argument("--coco-json", type=Path, required=True)
    check.add_argument("--device", required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    check.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    if args.command == "restore-input":
        record = restore_v05_build_input(args.original_dataset, args.coco_json, args.output)
        print(json.dumps(record, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
