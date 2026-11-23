#!/usr/bin/env python3
"""The one entry point that scores checkpoints on the frozen blind test (protocol C1).

An opening is the first time model output meets the frozen ground truth. This
runner has no flag that changes a measurement: the thresholds, arms, radii and
headline are code constants, so nothing is left to be chosen after the number
exists.

Order, and what each step guarantees:

1. The requested device must exist, or the run stops (no CPU substitution).
2. The manifest is verified and every mismatch stops the run (protocol C6).
3. An existing output directory is refused, so an opening can never overwrite
   another.
4. The environment record and the manifest hash are computed before any
   inference, so a late failure of those cannot follow the GPU work. All
   inference and scoring happens in a staging directory beside the output
   directory, ``run_record.json`` is written last, and the staging directory is
   promoted with a single rename. The output directory therefore exists only
   when it is complete. A failed run keeps its staging directory for
   inspection and deletes nothing; it holds no ``run_record.json`` and is not
   an opening output.

Each detector is scored once at every threshold (``evaluate_checkpoint``), and
its raw detections go through its paired decoder at the headline confidence,
both prompt arms, so detection and end-to-end numbers come from one inference.
Opening 1 pairs fold X's detector with fold X's decoder. Opening 2 (the refit)
has one detector, scored once, and runs its raw detections through each of the
three frozen decoders; it needs a manifest extended with ``frozen_manifest
extend`` and checks that the extension's selection record is the detector's.

The code commit is recorded by the human in ``TEST_RESULTS.md``; this runner
does not shell out to git.

Usage:
    python -m archeo_topia.analysis.frozen_test_opening \\
        --opening v0_6_test_baseline \\
        --manifest docs/mapsam/v006/analysis/frozen_manifest.json \\
        --lake-root "$LAKE" --device cuda:0 \\
        --output-dir artifacts/detection/v0_6_test_baseline

    python -m archeo_topia.analysis.frozen_test_opening \\
        --opening v0_6_refit \\
        --manifest docs/mapsam/v006/analysis/frozen_manifest.json \\
        --lake-root "$LAKE" --device cuda:0 \\
        --output-dir artifacts/detection/v0_6_refit/opening
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args

import torch

from archeo_topia.analysis.detection_metrics import (
    EvaluationSet,
    confidence_key,
    load_evaluation_set,
    macro_inputs,
    macro_summary,
)
from archeo_topia.analysis.frozen_manifest import verify_manifest
from archeo_topia.analysis.mapsam_end_to_end import BOX_MODES, format_report, run_pair
from archeo_topia.analysis.runtime import environment_record, resolve_device, write_record
from archeo_topia.datasets.build_detection_windows import REPO_ROOT, repo_relative, sha256_file
from archeo_topia.training.train_mound_detector import (
    DETECTOR_IMGSZ,
    HEADLINE_CONFIDENCE,
    MERGE_RADIUS,
    THRESHOLDS,
    evaluate_checkpoint,
    require_selection,
)

logger = logging.getLogger(__name__)

#: Which opening is being run.
Opening = Literal["v0_6_test_baseline", "v0_6_refit"]

#: The label the refit detector's outputs go under.
REFIT_DETECTOR_LABEL = "refit"

#: Measurement constants of every opening. They are not flags: see the module docstring.
WINDOW_PX = 512
IMAGE_SIZE = 1024


@dataclass(frozen=True)
class Pair:
    """A detector checkpoint and the decoder its candidates go through.

    Attributes:
        fold: Fold the decoder was trained under, the label of the pair's
            end-to-end outputs.
        detector_label: Label of the detector's outputs. Pairs sharing a label
            share one detector and one inference.
        detector: Repo-relative detector weights.
        decoder: Repo-relative decoder checkpoint.
    """

    fold: str
    detector_label: str
    detector: Path
    decoder: Path


@dataclass(frozen=True)
class Inputs:
    """Everything an opening reads, resolved from a verified manifest.

    Attributes:
        dataset: Frozen windows dataset root.
        sheets: Frozen sheets, as the dataset's build report declares them.
        ground_truth: Ground-truth COCO JSON.
        sam_checkpoint: Base SAM checkpoint.
        clip_paths: Clip path by file name.
    """

    dataset: Path
    sheets: tuple[str, ...]
    ground_truth: Path
    sam_checkpoint: Path
    clip_paths: dict[str, Path]


def _entries(manifest: Mapping[str, Any], role: str) -> list[dict[str, str]]:
    """Return the manifest entries of one role."""
    return [e for e in manifest["entries"] if e["role"] == role]


def pairs_for(opening: Opening, manifest: Mapping[str, Any]) -> list[Pair]:
    """Return the (detector, decoder) pairs an opening scores.

    Args:
        opening: Which opening.
        manifest: A verified manifest.

    Returns:
        The pairs. The baseline pairs fold X's detector with fold X's decoder;
        the refit pairs its one detector with every decoder.

    Raises:
        ValueError: If the detectors and decoders do not cover the same folds,
            the refit detector is absent or repeated (the manifest is not
            extended), or the opening is unknown.
    """
    if opening not in get_args(Opening):
        raise ValueError(f"unknown opening {opening!r}; have {list(get_args(Opening))}")
    decoders = {e["fold"]: Path(e["path"]) for e in _entries(manifest, "decoder")}
    if opening == "v0_6_refit":
        refit = _entries(manifest, "refit_detector")
        if len(refit) != 1:
            raise ValueError(
                f"the manifest has {len(refit)} refit_detector entries, expected 1: "
                "extend it with frozen_manifest extend first"
            )
        detector = Path(refit[0]["path"])
        return [Pair(fold, REFIT_DETECTOR_LABEL, detector, decoders[fold]) for fold in decoders]
    detectors = {e["fold"]: Path(e["path"]) for e in _entries(manifest, "detector")}
    if detectors.keys() != decoders.keys():
        raise ValueError(
            f"detector folds {sorted(detectors)} and decoder folds {sorted(decoders)} differ"
        )
    return [Pair(fold, fold, detectors[fold], decoders[fold]) for fold in detectors]


def check_refit_selection(manifest: Mapping[str, Any], repo_root: Path) -> dict[str, str]:
    """Check that the extension's selection record is the refit detector's.

    Args:
        manifest: A verified, extended manifest.
        repo_root: Root of ``repo`` entries.

    Returns:
        The selection record's repo-relative path and sha256, for the run record.

    Raises:
        ValueError: If the selection is absent or repeated, names another
            detector, or is not what the pre-registered rule yields.
    """
    selections = _entries(manifest, "refit_selection")
    (detector,) = _entries(manifest, "refit_detector")
    if len(selections) != 1:
        raise ValueError(f"the manifest has {len(selections)} refit_selection entries, expected 1")
    selection = json.loads((repo_root / selections[0]["path"]).read_text(encoding="utf-8"))
    if selection["chosen"]["path"] != detector["path"]:
        raise ValueError(
            f"the selection chose {selection['chosen']['path']}, the manifest pins {detector['path']}"
        )
    require_selection(selection, repo_root)
    return {"path": selections[0]["path"], "sha256": selections[0]["sha256"]}


def inputs_from(manifest: Mapping[str, Any], repo_root: Path, lake_root: Path) -> Inputs:
    """Resolve a verified manifest into the paths an opening reads.

    Args:
        manifest: A verified manifest.
        repo_root: Root of ``repo`` entries.
        lake_root: Root of ``lake`` entries.

    Returns:
        The inputs.

    Raises:
        ValueError: If the dataset's declared sheets and the manifest's clip
            sheets differ.
    """
    (report_entry,) = _entries(manifest, "windows_build_report")
    dataset = repo_root / Path(report_entry["path"]).parent.parent
    report = json.loads((repo_root / report_entry["path"]).read_text(encoding="utf-8"))
    sheets = tuple(report["listing"]["listings"]["test"]["sheets"])
    clip_sheets = {e["sheet"] for e in _entries(manifest, "clip")}
    if set(sheets) != clip_sheets:
        raise ValueError(
            f"the dataset's test sheets {sorted(sheets)} differ from the manifest's clip "
            f"sheets {sorted(clip_sheets)}"
        )
    (ground_truth,) = _entries(manifest, "ground_truth")
    (sam,) = _entries(manifest, "sam_base")
    return Inputs(
        dataset=dataset,
        sheets=sheets,
        ground_truth=repo_root / ground_truth["path"],
        sam_checkpoint=repo_root / sam["path"],
        clip_paths={e["file"]: lake_root / e["path"] for e in _entries(manifest, "clip")},
    )


def across_checkpoints(metrics_by_fold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the pooled detector metrics across the checkpoints, per confidence.

    Descriptive only: the checkpoints are not ensembled and none is selected.

    Args:
        metrics_by_fold: ``metrics.json`` document by fold.

    Returns:
        Per confidence and quantity, the value of each fold's checkpoint and
        their mean, min and max.
    """
    folds = list(metrics_by_fold)
    keys = [confidence_key(t) for t in THRESHOLDS]
    summary: dict[str, Any] = {}
    for key in keys:
        values = {
            fold: macro_inputs(metrics_by_fold[fold]["by_confidence"][key]["pooled"])
            for fold in folds
        }
        summary[key] = {
            name: {
                "by_fold": {fold: values[fold][name] for fold in folds},
                **macro_summary([values[fold][name] for fold in folds]),
            }
            for name in values[folds[0]]
        }
    return summary


def _score(
    pairs: list[Pair],
    inputs: Inputs,
    evaluation_set: EvaluationSet,
    repo_root: Path,
    device: torch.device,
    staging: Path,
) -> dict[str, Any]:
    """Run every pair and write its outputs into the staging directory.

    Each distinct detector label is inferred and scored once; every pair of that
    label reuses its raw detections.

    Returns:
        The detector ``metrics.json`` document by detector label.
    """
    raw_by_label: dict[str, Any] = {}
    metrics_by_label: dict[str, Any] = {}
    detector_by_label: dict[str, Path] = {}
    for pair in pairs:
        label = pair.detector_label
        if detector_by_label.setdefault(label, pair.detector) != pair.detector:
            raise ValueError(f"detector label {label!r} names two detectors")
        weights = repo_root / pair.detector
        if label not in metrics_by_label:
            raw, metrics = evaluate_checkpoint(
                weights=weights,
                evaluation_set=evaluation_set,
                imgsz=DETECTOR_IMGSZ,
                merge_radius=MERGE_RADIUS,
                thresholds=THRESHOLDS,
                device=device,
                output_dir=staging / "detector",
                label=label,
            )
            raw_by_label[label] = raw
            metrics_by_label[label] = metrics
            logger.info("%s: %d raw detections", label, len(raw))
        result = run_pair(
            raw=raw_by_label[label],
            detector_weights=weights,
            decoder_checkpoint=repo_root / pair.decoder,
            sam_checkpoint=inputs.sam_checkpoint,
            evaluation_set=evaluation_set,
            coco_json=inputs.ground_truth,
            clip_paths=inputs.clip_paths,
            confidence=HEADLINE_CONFIDENCE,
            box_modes=BOX_MODES,
            window_px=WINDOW_PX,
            image_size=IMAGE_SIZE,
            merge_radius=MERGE_RADIUS,
            device=device,
        )
        fold_dir = staging / "end_to_end" / pair.fold
        fold_dir.mkdir(parents=True)
        write_record(fold_dir / "end_to_end.json", json.dumps(result, indent=2))
        write_record(fold_dir / "end_to_end.md", format_report({pair.fold: result}))
    return metrics_by_label


def run_opening(
    opening: Opening,
    manifest_path: Path,
    lake_root: Path,
    device_name: str,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Open the frozen test set once.

    Args:
        opening: Which opening. ``v0_6_refit`` needs an extended manifest.
        manifest_path: The frozen manifest.
        lake_root: Root of ``lake`` entries.
        device_name: Requested torch device.
        output_dir: Where the outputs go; must not exist.
        repo_root: Root of ``repo`` entries.

    Returns:
        The ``run_record.json`` document.

    Raises:
        RuntimeError: If the device is unavailable.
        ValueError: If the manifest does not match its inputs.
        FileExistsError: If ``output_dir`` exists.
    """
    device = resolve_device(device_name)
    manifest = verify_manifest(manifest_path, lake_root, repo_root)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; an opening never overwrites one")
    pairs = pairs_for(opening, manifest)
    selection = check_refit_selection(manifest, repo_root) if opening == "v0_6_refit" else None
    inputs = inputs_from(manifest, repo_root, lake_root)
    evaluation_set = load_evaluation_set(
        inputs.dataset, inputs.dataset / "test.txt", inputs.sheets
    )

    environment = environment_record(device)
    manifest_record = {"path": repo_relative(manifest_path), "sha256": sha256_file(manifest_path)}
    if selection is not None:
        manifest_record["base_sha256"] = manifest["base"]["sha256"]
    started = datetime.now(UTC).isoformat()

    # The parent is not a record: it only has to exist for the staging directory.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"))
    try:
        metrics_by_label = _score(pairs, inputs, evaluation_set, repo_root, device, staging)
        # Spread across checkpoints is the baseline's S; one detector has none.
        if opening == "v0_6_test_baseline":
            write_record(
                staging / "detector" / "across_checkpoints.json",
                json.dumps(across_checkpoints(metrics_by_label), indent=2),
            )
        record: dict[str, Any] = {
            "opening": opening,
            "manifest": manifest_record,
            "environment": environment,
            "pairs": [
                {
                    "fold": p.fold,
                    "detector_label": p.detector_label,
                    "detector": str(p.detector),
                    "decoder": str(p.decoder),
                }
                for p in pairs
            ],
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
        }
        if selection is not None:
            record["selection"] = selection
        write_record(staging / "run_record.json", json.dumps(record, indent=2))
        # rename(2) silently replaces an empty directory, so re-check right before it.
        if output_dir.exists():
            raise FileExistsError(
                f"{output_dir} appeared during the run; an opening never overwrites one"
            )
        staging.rename(output_dir)
    except BaseException:
        logger.error(
            "opening failed; partial outputs kept for inspection at %s "
            "(no run_record.json, not an opening output)",
            staging,
        )
        raise
    return record


def main() -> None:
    """Parse arguments and run the opening."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--opening", choices=get_args(Opening), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lake-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    run_opening(args.opening, args.manifest, args.lake_root, args.device, args.output_dir)


if __name__ == "__main__":
    main()
