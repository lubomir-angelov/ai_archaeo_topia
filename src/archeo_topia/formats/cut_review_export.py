#!/usr/bin/env python3
"""Cut the v0.0.5 training export from the v0.7 review returns (refit prerequisite).

Each reviewed sheet comes back as its own COCO file whose image and annotation
ids restart at 1, so the files cannot simply be concatenated. This merge
renumbers both ids globally and changes nothing else: ``segmentation``,
``bbox``, ``area``, ``iscrowd`` and ``attributes`` are carried byte for byte as
parsed, so the cut is a faithful union and not a re-export.

Every sheet of the ``test`` split is left out of the export and recorded as
excluded: a training export never carries a test sheet, and dropping a whole
sheet changes no label.

Every check runs before the output directory exists. The merged document is
serialized and read back, each record is mapped through the inverse id maps and
compared with its source, and every source file is hashed again, so a source
that moved during the cut stops it before anything is written. ``PROVENANCE.json``
is written last; ``require_cut`` is the one check that a directory is a v0.0.5,
and every consumer of the export calls it.

Usage:
    python -m archeo_topia.formats.cut_review_export \\
        --review-root artifacts/review/v0_7 \\
        --output-dir annotation/cvat/v0.0.5
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from archeo_topia.analysis.runtime import new_directory, write_record
from archeo_topia.datasets.build_detection_windows import (
    SPLITS_FILE,
    load_split_document,
    repo_relative,
    sha256_file,
    sheet_of,
)
from archeo_topia.formats.labels import REVIEW_STATUS

logger = logging.getLogger(__name__)

#: The split whose sheets a training export never carries.
EXCLUDED_SPLIT = "test"

#: Why a sheet is left out of the export.
Exclusion = Literal["test_split"]

#: The two files a review return holds, and nothing else.
COCO_FILE = "instances_default.json"
REPORT_FILE = "review_report.json"
PROVENANCE_FILE = "PROVENANCE.json"

#: Name the export is recorded under.
EXPORT_NAME = "v0.0.5"

#: Per-sheet original-id to merged-id maps.
IdMap = dict[int, int]


def _sheet_problems(
    sheet_dir: Path, coco: dict[str, Any], report: dict[str, Any], splits: dict[str, Any]
) -> list[str]:
    """Return what is wrong with one sheet's return."""
    sheet = sheet_dir.name
    problems: list[str] = []
    if report["sheet_id"] != sheet:
        problems.append(f"{sheet}: {REPORT_FILE} names sheet {report['sheet_id']!r}")
    if sheet not in splits["sheets"]:
        problems.append(f"{sheet}: not a sheet of the splits file")
    foreign = sorted({i["file_name"] for i in coco["images"] if sheet_of(i["file_name"]) != sheet})
    if foreign:
        problems.append(f"{sheet}: images of another sheet: {foreign}")
    image_ids = [i["id"] for i in coco["images"]]
    if len(set(image_ids)) != len(image_ids):
        problems.append(f"{sheet}: image ids are not unique")
    annotation_ids = [a["id"] for a in coco["annotations"]]
    if len(set(annotation_ids)) != len(annotation_ids):
        problems.append(f"{sheet}: annotation ids are not unique")
    unknown = sorted({a["image_id"] for a in coco["annotations"]} - set(image_ids))
    if unknown:
        problems.append(f"{sheet}: annotations name unknown image ids {unknown}")
    return problems


def _load_returns(
    review_root: Path, splits: dict[str, Any]
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[str, dict[str, str]]]:
    """Read every sheet return and stop on every inconsistency at once.

    Each file is read once; the hash is of the bytes that were parsed.

    Returns:
        Parsed COCO and review report, and the sha256 of both files, by sheet, in
        sorted sheet order.
    """
    entries = sorted(review_root.iterdir())
    if not entries:
        raise ValueError(f"{review_root} holds no sheet directory")
    problems = [f"{p}: not a sheet directory" for p in entries if not p.is_dir()]
    returns: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for sheet_dir in (p for p in entries if p.is_dir()):
        names = sorted(p.name for p in sheet_dir.iterdir())
        if names != sorted([COCO_FILE, REPORT_FILE]):
            problems.append(
                f"{sheet_dir.name}: holds {names}, expected {[COCO_FILE, REPORT_FILE]}"
            )
            continue
        coco_bytes = (sheet_dir / COCO_FILE).read_bytes()
        report_bytes = (sheet_dir / REPORT_FILE).read_bytes()
        coco = json.loads(coco_bytes)
        report = json.loads(report_bytes)
        problems += _sheet_problems(sheet_dir, coco, report, splits)
        returns[sheet_dir.name] = (coco, report)
        hashes[sheet_dir.name] = {
            "instances_default_sha256": hashlib.sha256(coco_bytes).hexdigest(),
            "review_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        }

    first_categories = next(iter(returns.values()))[0]["categories"] if returns else None
    seen_files: dict[str, str] = {}
    for sheet, (coco, _) in returns.items():
        if coco["categories"] != first_categories:
            problems.append(f"{sheet}: categories differ from the first sheet's")
        for image in coco["images"]:
            owner = seen_files.setdefault(image["file_name"], sheet)
            if owner != sheet:
                problems.append(f"{image['file_name']}: in both {owner} and {sheet}")
    if problems:
        raise ValueError(f"{review_root} cannot be merged:\n  " + "\n  ".join(problems))
    return returns, hashes


def _merge(
    returns: dict[str, tuple[dict[str, Any], dict[str, Any]]], included: list[str]
) -> tuple[dict[str, Any], dict[str, IdMap], dict[str, IdMap]]:
    """Concatenate the included sheets, renumbering image and annotation ids from 1.

    Returns:
        The merged COCO document and, by sheet, the image and annotation id maps.
    """
    first = returns[included[0]][0]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_maps: dict[str, IdMap] = {}
    annotation_maps: dict[str, IdMap] = {}
    for sheet in included:
        coco = returns[sheet][0]
        image_maps[sheet] = {}
        annotation_maps[sheet] = {}
        for image in coco["images"]:
            image_maps[sheet][image["id"]] = len(images) + 1
            images.append({**image, "id": len(images) + 1})
        for annotation in coco["annotations"]:
            annotation_maps[sheet][annotation["id"]] = len(annotations) + 1
            merged = copy.deepcopy(annotation)
            merged["id"] = len(annotations) + 1
            merged["image_id"] = image_maps[sheet][annotation["image_id"]]
            annotations.append(merged)
    merged_coco = {
        "licenses": first["licenses"],
        "info": {
            **first["info"],
            "description": f"{EXPORT_NAME}: merged v0.7 review returns, {len(included)} sheets",
        },
        "categories": first["categories"],
        "images": images,
        "annotations": annotations,
    }
    return merged_coco, image_maps, annotation_maps


def _round_trip_problems(
    written: dict[str, Any],
    returns: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    image_maps: dict[str, IdMap],
    annotation_maps: dict[str, IdMap],
) -> list[str]:
    """Map every merged record back through the inverse ids and compare with its source."""
    problems: list[str] = []
    images = {i["id"]: i for i in written["images"]}
    annotations = {a["id"]: a for a in written["annotations"]}
    expected_images = sum(len(m) for m in image_maps.values())
    expected_annotations = sum(len(m) for m in annotation_maps.values())
    if len(images) != expected_images or len(annotations) != expected_annotations:
        problems.append(
            f"merged file holds {len(images)} images and {len(annotations)} annotations, "
            f"expected {expected_images} and {expected_annotations}"
        )
    for sheet, image_map in image_maps.items():
        coco = returns[sheet][0]
        inverse_images = {new: old for old, new in image_map.items()}
        for source in coco["images"]:
            if {**images[image_map[source["id"]]], "id": source["id"]} != source:
                problems.append(f"{sheet}: image {source['id']} differs after the merge")
        for source in coco["annotations"]:
            merged = annotations[annotation_maps[sheet][source["id"]]]
            restored = {
                **merged,
                "id": source["id"],
                "image_id": inverse_images[merged["image_id"]],
            }
            if restored != source:
                problems.append(f"{sheet}: annotation {source['id']} differs after the merge")
    return problems


def _source_hashes(review_root: Path, sheets: list[str]) -> dict[str, dict[str, str]]:
    """Hash both files of every sheet return, to detect a change after they were parsed."""
    return {
        sheet: {
            "instances_default_sha256": sha256_file(review_root / sheet / COCO_FILE),
            "review_report_sha256": sha256_file(review_root / sheet / REPORT_FILE),
        }
        for sheet in sheets
    }


def cut(review_root: Path, output_dir: Path, splits_file: Path = SPLITS_FILE) -> dict[str, Any]:
    """Merge the review returns of every non-test sheet into one COCO export.

    Args:
        review_root: Directory holding one ``<sheet>/`` return per sheet.
        output_dir: Export directory; must not exist.
        splits_file: Splits file declaring each sheet's split.

    Returns:
        The ``PROVENANCE.json`` document, also written beside the export.

    Raises:
        ValueError: Listing every inconsistency of the returns, or if the merge
            does not round-trip, or a source changed during the cut; raised before
            ``output_dir`` exists.
        FileExistsError: If ``output_dir`` exists.
    """
    splits = load_split_document(splits_file)
    returns, source_hashes = _load_returns(review_root, splits)
    sheets = list(returns)

    excluded: dict[str, Exclusion] = {
        sheet: "test_split"
        for sheet in sheets
        if splits["sheets"][sheet]["split"] == EXCLUDED_SPLIT
    }
    included = [sheet for sheet in sheets if sheet not in excluded]
    if not included:
        raise ValueError(f"every sheet under {review_root} is excluded: {sorted(excluded)}")
    merged, image_maps, annotation_maps = _merge(returns, included)

    serialized = json.dumps(merged)
    problems = _round_trip_problems(json.loads(serialized), returns, image_maps, annotation_maps)
    if _source_hashes(review_root, sheets) != source_hashes:
        problems.append("a review return changed while the cut ran")
    if problems:
        raise ValueError(f"the cut for {output_dir} is not faithful:\n  " + "\n  ".join(problems))

    category_names = {c["id"]: c["name"] for c in merged["categories"]}
    counts: Counter[str] = Counter()
    statuses: dict[str, Counter[str]] = defaultdict(Counter)
    for annotation in merged["annotations"]:
        name = category_names[annotation["category_id"]]
        counts[name] += 1
        statuses[name][annotation["attributes"][REVIEW_STATUS]] += 1
    provenance = {
        "export": EXPORT_NAME,
        "source_root": repo_relative(review_root),
        "splits_file": repo_relative(splits_file),
        "splits_sha256": sha256_file(splits_file),
        "sheets": included,
        "excluded_sheets": excluded,
        "sources": {
            sheet: {
                **source_hashes[sheet],
                "image_id_map": ({} if sheet in excluded else _stringify(image_maps[sheet])),
                "annotation_id_map": (
                    {} if sheet in excluded else _stringify(annotation_maps[sheet])
                ),
            }
            for sheet in sheets
        },
        "counts_by_category": {name: counts[name] for name in category_names.values()},
        "review_status_by_category": {name: dict(c) for name, c in statuses.items()},
        "output_sha256": {COCO_FILE: hashlib.sha256(serialized.encode("utf-8")).hexdigest()},
    }

    new_directory(output_dir)
    write_record(output_dir / COCO_FILE, serialized)
    written_sha256 = sha256_file(output_dir / COCO_FILE)
    if written_sha256 != provenance["output_sha256"][COCO_FILE]:
        raise ValueError(
            f"{output_dir / COCO_FILE} hashes to {written_sha256}, not the serialized merge"
        )
    write_record(output_dir / PROVENANCE_FILE, json.dumps(provenance, indent=2) + "\n")
    logger.info("cut %d sheets (%d excluded) into %s", len(included), len(excluded), output_dir)
    return provenance


def require_cut(coco_json: Path, coco_sha256: str) -> None:
    """Refuse a COCO file that is not the one a completed cut wrote.

    The single definition of "is a v0.0.5": ``PROVENANCE.json`` is written last,
    so its presence and its recorded hash prove the cut finished and the file is
    unchanged since.

    Args:
        coco_json: The export's COCO file.
        coco_sha256: sha256 of the bytes the caller parsed from ``coco_json``.

    Raises:
        FileNotFoundError: If ``PROVENANCE.json`` is not beside ``coco_json``.
        ValueError: Listing every mismatch of name, export or hash.
    """
    provenance_path = coco_json.parent / PROVENANCE_FILE
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"{provenance_path} is missing: {coco_json} is not a completed cut"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if coco_json.name != COCO_FILE:
        problems.append(f"{coco_json.name} is not {COCO_FILE}")
    if provenance["export"] != EXPORT_NAME:
        problems.append(f"{provenance_path} records export {provenance['export']!r}")
    if provenance["output_sha256"][COCO_FILE] != coco_sha256:
        problems.append(f"{coco_json} is not the file the cut wrote (sha256 differs)")
    if problems:
        raise ValueError(f"{coco_json} is not a {EXPORT_NAME} cut:\n  " + "\n  ".join(problems))


def _stringify(id_map: IdMap) -> dict[str, int]:
    """Return an id map with string keys, as JSON stores it."""
    return {str(k): v for k, v in id_map.items()}


def main() -> None:
    """Parse arguments and cut the export."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    provenance = cut(args.review_root, args.output_dir)
    print(json.dumps(provenance["counts_by_category"], indent=2))


if __name__ == "__main__":
    main()
