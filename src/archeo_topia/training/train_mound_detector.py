#!/usr/bin/env python3
"""Train and evaluate the YOLO26 mound detector (MapSAM v0.5 step 4).

One class, 512 px windows, leave-one-sheet-out folds matching the v0.4 MapSAM
configs. The model is ``yolo26s`` rather than a larger scale: with 180 mound
annotations, extra capacity is extra variance rather than extra information.

**The leave-one-sheet-out folds select no checkpoint.** Ultralytics tracks a
``best.pt`` against the validation split, but the validation split there *is*
the held-out sheet, so selecting on it would leak the evaluation into the model.
The ``loso`` command trains a fixed number of epochs and evaluates ``last.pt``.
Any number produced from ``best.pt`` would be optimistic by an unknown amount.

**The refit selects among six checkpoints, never by ``best.pt``.** Its
validation sheets are disjoint from its training sheets, so ``refit-select``
scores the six candidates the pre-registered refit names (``refit_candidates``)
with the scorer the test uses, at the headline confidence, and picks by pooled
recall@5px, then fewer false positives, then the later epoch. ``best.pt`` is
mAP-selected and is never opened.

Evaluation is not Ultralytics' mAP. It is the localization-first hierarchy in
``archeo_topia.analysis.detection_metrics``: recall inside a source-pixel radius
of the annotated mound centre, with recall@5px primary and p90 centre error as
the acceptance figure, because that is what composes with v0.4's prompt-jitter
tolerance curve.

Licence note: Ultralytics is AGPL-3.0-or-commercial while this repository is
MIT. The decision to accept that, and its conditions, are recorded in
``docs/mapsam/v005/PLAN.md``. This module lives behind the ``detect`` extra and
nothing in the MIT library surface imports it.

Usage:
    python -m archeo_topia.training.train_mound_detector loso \\
        --dataset data/curated/datasets/mapsam_det_v0 \\
        --fold all --model yolo26s.pt --epochs 300 \\
        --output-dir artifacts/detection/v0_5_yolo26s

    python -m archeo_topia.training.train_mound_detector refit-train \\
        --config configs/refit/v0_6_refit.json \\
        --dataset data/curated/datasets/mapsam_det_refit_v001 \\
        --output-dir artifacts/detection/v0_6_refit/train --device cuda:0

    python -m archeo_topia.training.train_mound_detector refit-select \\
        --train-dir artifacts/detection/v0_6_refit/train \\
        --output-dir artifacts/detection/v0_6_refit/selection --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from archeo_topia.analysis.detection_metrics import (
    ASSOCIATION_RADIUS,
    ATTRIBUTION_RADIUS,
    RECALL_RADII,
    Detection,
    EvaluationSet,
    confidence_key,
    format_metrics_report,
    load_evaluation_set,
    score_detections,
)
from archeo_topia.analysis.runtime import (
    environment_record,
    new_directory,
    resolve_device,
    write_record,
)
from archeo_topia.datasets.build_detection_windows import (
    FOLDS,
    REPO_ROOT,
    RefitConfig,
    build_output_problems,
    crop_path,
    load_refit_config,
    refit_candidates,
    repo_relative,
    sha256_file,
)

logger = logging.getLogger(__name__)

WINDOW_RE = re.compile(r"^(?P<stem>.+)_x(?P<x0>\d+)_y(?P<y0>\d+)$")

#: Confidence thresholds reported as operating points. The detector is
#: recall-oriented; precision is the second stage's problem.
THRESHOLDS = (0.05, 0.10, 0.25, 0.50, 0.70)

#: The operating point a single headline number is read at.
HEADLINE_CONFIDENCE = 0.25

#: Inference image size of every scored run; training uses the same value.
DETECTOR_IMGSZ = 512

#: Cross-window merge radius in source pixels, shared by every scored run.
MERGE_RADIUS = 10.0

#: Ultralytics run name of the refit inside its training directory.
REFIT_RUN_NAME = "refit"

#: The lowest confidence the detector emits; every threshold is scored from one
#: inference at this confidence.
MIN_CONFIDENCE = 0.05

if not {HEADLINE_CONFIDENCE, MIN_CONFIDENCE} <= set(THRESHOLDS):
    raise ValueError("HEADLINE_CONFIDENCE and MIN_CONFIDENCE must both be members of THRESHOLDS")


def fold_weights_path(run_dir: Path, fold: str) -> Path:
    """Return where Ultralytics leaves a fold's final weights.

    Args:
        run_dir: Run root holding one subdirectory per fold.
        fold: Fold name.

    Returns:
        Path to ``last.pt``.
    """
    return run_dir / fold / "weights" / "last.pt"


def parse_window(window_id: str) -> tuple[str, int, int]:
    """Recover the source clip and window origin from a window id.

    Args:
        window_id: File stem written by ``build_detection_windows``, for
            example ``K-35-51-B-a_3_x00384_y01152``.

    Returns:
        Tuple of clip file name, window x origin and window y origin.

    Raises:
        ValueError: If the id does not carry a window origin.
    """
    match = WINDOW_RE.match(window_id)
    if not match:
        raise ValueError(f"not a window id: {window_id}")
    return f"{match['stem']}.png", int(match["x0"]), int(match["y0"])


def train_fold(
    data_yaml: Path,
    run_name: str,
    output_dir: Path,
    model_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    seed: int,
    save_period: int,
    device: torch.device,
) -> Path:
    """Train one run and return its final weights.

    Args:
        data_yaml: Ultralytics data YAML of the run.
        run_name: Run name; the run gets this subdirectory of ``output_dir``.
        output_dir: Run root.
        model_name: Ultralytics model or checkpoint name.
        imgsz: Training image size. 512 is native for these windows.
        epochs: Fixed epoch count; no early stopping, see the module docstring.
        batch: Batch size.
        seed: Random seed.
        save_period: Save a checkpoint every N epochs (``-1`` for none). The
            loso folds only *report* the trajectory; the refit selects among
            these saves on its own validation sheets.
        device: Device training runs on.

    Returns:
        Path to ``last.pt``.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml.resolve()),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        seed=seed,
        deterministic=True,
        device=str(device),
        project=str(output_dir.resolve()),
        name=run_name,
        exist_ok=True,
        # Disable early stopping: ``best.pt`` is mAP-selected and no run selects
        # on it, so stopping on the validation split would select on it.
        patience=epochs + 1,
        save_period=save_period,
        val=True,
        plots=True,
        verbose=False,
    )
    return fold_weights_path(output_dir, run_name)


def _refit_dataset_problems(dataset: Path, config: RefitConfig) -> list[str]:
    """Return why a windows dataset is not the one this refit config builds."""
    report = json.loads((dataset / "metadata" / "build_report.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    if report["listing"]["plan"] != "train_val":
        problems.append(f"listing plan is {report['listing']['plan']!r}, not 'train_val'")
    elif report["refit"]["config"]["sha256"] != config.sha256:
        problems.append(f"{dataset} was not built from {config.path} (config sha256 differs)")
    return problems + build_output_problems(dataset)


def train_refit(
    config: RefitConfig, dataset: Path, output_dir: Path, device: torch.device
) -> Path:
    """Train the refit detector on the ``train_val`` dataset.

    Every check runs before the output directory exists, and ``train_record.json``
    is written last, so a crashed training leaves no record and selection
    refuses it.

    Args:
        config: The refit configuration.
        dataset: ``train_val`` windows dataset built from that configuration.
        output_dir: Training directory; must not exist.
        device: Device training runs on.

    Returns:
        Path to ``last.pt``.

    Raises:
        ValueError: If the dataset is not the refit's, naming every problem.
        FileExistsError: If ``output_dir`` exists.
    """
    problems = _refit_dataset_problems(dataset, config)
    if problems:
        raise ValueError("the refit dataset is not usable:\n  " + "\n  ".join(problems))
    environment = environment_record(device)
    new_directory(output_dir)
    started = datetime.now(UTC).isoformat()
    weights = train_fold(
        data_yaml=dataset / "train_val.yaml",
        run_name=REFIT_RUN_NAME,
        output_dir=output_dir,
        model_name=config.model,
        imgsz=DETECTOR_IMGSZ,
        epochs=config.epochs,
        batch=config.batch,
        seed=config.seed,
        save_period=config.save_period,
        device=device,
    )
    record = {
        "config": {"path": repo_relative(config.path), "sha256": config.sha256},
        "dataset": {
            "path": repo_relative(dataset),
            "build_report_sha256": sha256_file(dataset / "metadata" / "build_report.json"),
        },
        "model": config.model,
        "imgsz": DETECTOR_IMGSZ,
        "epochs": config.epochs,
        "batch": config.batch,
        "seed": config.seed,
        "save_period": config.save_period,
        "environment": environment,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    write_record(output_dir / "train_record.json", json.dumps(record, indent=2))
    return weights


def predict_windows(
    weights: Path,
    evaluation_set: EvaluationSet,
    imgsz: int,
    min_confidence: float,
    device: torch.device,
) -> list[Detection]:
    """Run the detector over the windows of an explicit listing.

    Boxes come back in window coordinates and are mapped to source-tile
    coordinates before anything else happens, so every downstream number is in
    the same space as the annotations and as MapSAM's prompts.

    Args:
        weights: Model weights.
        evaluation_set: Supplies the dataset and the windows to run over, in
            listing order (which fixes batch composition and detection order).
        imgsz: Inference image size; must match training.
        min_confidence: Lowest confidence to emit.
        device: Device inference runs on.

    Returns:
        Detections in source coordinates, before merging.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    paths = [
        str(crop_path(evaluation_set.dataset, window_id).resolve())
        for window_id in evaluation_set.window_ids
    ]

    detections: list[Detection] = []
    for start in range(0, len(paths), 32):
        batch = paths[start : start + 32]
        predictions = model.predict(
            batch, imgsz=imgsz, conf=min_confidence, device=str(device), verbose=False
        )
        for path, result in zip(batch, predictions, strict=True):
            image_name, x0, y0 = parse_window(Path(path).stem)
            boxes = result.boxes
            if boxes is None:
                continue
            for box, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
                bx0, by0, bx1, by1 = box
                detections.append(
                    Detection(
                        image=image_name,
                        x=(bx0 + bx1) / 2 + x0,
                        y=(by0 + by1) / 2 + y0,
                        score=float(score),
                        box=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                    )
                )

    logger.info(
        "%s: %d raw detections over %d windows",
        evaluation_set.listing.name,
        len(detections),
        len(paths),
    )
    return detections


def metrics_document(
    raw: Sequence[Detection],
    weights: Path,
    evaluation_set: EvaluationSet,
    imgsz: int,
    merge_radius: float,
    thresholds: Sequence[float],
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    """Score raw detections and describe the run that produced them.

    Args:
        raw: Raw detections from one inference at ``MIN_CONFIDENCE``.
        weights: Weights the detections came from (recorded).
        evaluation_set: What to score against.
        imgsz: Inference image size (recorded).
        merge_radius: Centre distance for cross-window deduplication.
        thresholds: Confidence operating points.
        device: Device inference ran on (recorded).
        label: Name of the run, for example the fold.

    Returns:
        The ``metrics.json`` document: ``config`` and ``by_confidence``.
    """
    return {
        "config": {
            "label": label,
            "weights": repo_relative(weights),
            "dataset": repo_relative(evaluation_set.dataset),
            "listing": evaluation_set.listing.name,
            "sheets": list(evaluation_set.sheets),
            "imgsz": imgsz,
            "min_confidence": MIN_CONFIDENCE,
            "merge_radius": merge_radius,
            "association_radius": ASSOCIATION_RADIUS,
            "recall_radii": list(RECALL_RADII),
            "attribution_radius": ATTRIBUTION_RADIUS,
            "thresholds": list(thresholds),
            "headline_confidence": HEADLINE_CONFIDENCE,
            "device": str(device),
        },
        **score_detections(raw, evaluation_set, thresholds, merge_radius),
    }


def write_detections(path: Path, raw: Sequence[Detection]) -> None:
    """Write raw, pre-merge detections as newline-delimited JSON.

    Args:
        path: Destination file.
        raw: Detections to write.
    """
    lines = (
        json.dumps(
            {
                "image": det.image,
                "x": det.x,
                "y": det.y,
                "score": det.score,
                "box": list(det.box) if det.box is not None else None,
            }
        )
        + "\n"
        for det in raw
    )
    write_record(path, "".join(lines))


def report_markdown(metrics: dict[str, Any], label: str) -> str:
    """Render a metrics document and its operating curve as markdown.

    Args:
        metrics: A document from :func:`metrics_document`.
        label: Name of the run, for the heading.

    Returns:
        Markdown text.
    """
    return (
        format_metrics_report(metrics, f"YOLO26-s - {label}")
        + "\n"
        + curve_table(operating_curve(metrics))
    )


def evaluate_checkpoint(
    weights: Path,
    evaluation_set: EvaluationSet,
    imgsz: int,
    merge_radius: float,
    thresholds: Sequence[float],
    device: torch.device,
    output_dir: Path,
    label: str,
) -> tuple[list[Detection], dict[str, Any]]:
    """Run a checkpoint over an evaluation set, score it, and write the record.

    One inference per checkpoint, emitted from ``MIN_CONFIDENCE`` and scored at
    every threshold. The raw detections are returned so the end-to-end stage
    consumes the same inference.

    Writes ``output_dir/label/{detections.jsonl, metrics.json, report.md}`` and
    never overwrites an existing ``metrics.json``.

    Args:
        weights: Model weights.
        evaluation_set: What to score against.
        imgsz: Inference image size.
        merge_radius: Centre distance for cross-window deduplication.
        thresholds: Confidence operating points; none below ``MIN_CONFIDENCE``.
        device: Device inference runs on.
        output_dir: Run root.
        label: Subdirectory name, for example the fold.

    Returns:
        The raw detections and the metrics document.

    Raises:
        FileExistsError: If ``metrics.json`` already exists for this label.
        ValueError: If a threshold is below ``MIN_CONFIDENCE``.
    """
    label_dir = output_dir / label
    metrics_path = label_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"{metrics_path} already exists; scoring never overwrites a record")
    if min(thresholds) < MIN_CONFIDENCE:
        raise ValueError(f"thresholds {list(thresholds)} go below MIN_CONFIDENCE {MIN_CONFIDENCE}")

    raw = predict_windows(weights, evaluation_set, imgsz, MIN_CONFIDENCE, device)
    metrics = metrics_document(
        raw, weights, evaluation_set, imgsz, merge_radius, thresholds, device, label
    )

    label_dir.mkdir(parents=True, exist_ok=True)
    write_detections(label_dir / "detections.jsonl", raw)
    write_record(metrics_path, json.dumps(metrics, indent=2))
    write_record(label_dir / "report.md", report_markdown(metrics, label))
    return raw, metrics


def operating_curve(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the pooled operating curve out of a ``by_confidence`` document.

    Args:
        metrics: A metrics document from :func:`evaluate_checkpoint`.

    Returns:
        One row per confidence, ascending.
    """
    curve = []
    for key, block in sorted(metrics["by_confidence"].items(), key=lambda kv: float(kv[0])):
        pooled = block["pooled"]
        curve.append(
            {
                "confidence": float(key),
                "candidates": pooled["detections"],
                "recall_5px": pooled["recall"]["5"]["recall"],
                "recall_10px": pooled["recall"]["10"]["recall"],
                "recall_15px": pooled["recall"]["15"]["recall"],
                "recall_25px": pooled["recall"]["25"]["recall"],
                "p90_error_px": pooled["localization_error_px"]["p90"],
                "false_positives": pooled["false_positives"]["count"],
                "fp_per_window": pooled["false_positives"]["per_window"],
            }
        )
    return curve


def curve_table(curve: list[dict[str, Any]]) -> str:
    """Render a confidence sweep as a markdown table.

    Args:
        curve: Operating points from :func:`operating_curve`.

    Returns:
        Markdown text.
    """
    lines = [
        "### Operating curve",
        "",
        "| conf | candidates | recall@5px | recall@10px | recall@15px | p90 err | FP | FP/window |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in curve:
        lines.append(
            f"| {point['confidence']:.2f} | {point['candidates']} "
            f"| {point['recall_5px']:.3f} | {point['recall_10px']:.3f} "
            f"| {point['recall_15px']:.3f} | {point['p90_error_px']:.1f} "
            f"| {point['false_positives']} | {point['fp_per_window']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def epoch_trajectory(
    weights_dir: Path,
    evaluation_set: EvaluationSet,
    imgsz: int,
    merge_radius: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Score every periodic checkpoint at the headline operating point.

    v0.4 traced a late-training decision drift in the MapSAM decoder: fold A
    went from 2 zero-IoU samples at epoch 5 to 19 at epoch 50 with the peak
    logit still correctly placed. The detector has the same exposure and the
    same absence of a clean validation split, so the trajectory is recorded as
    evidence rather than used to pick a checkpoint.

    Args:
        weights_dir: Ultralytics ``weights`` directory.
        evaluation_set: What to score against.
        imgsz: Inference image size.
        merge_radius: Centre distance for deduplication.
        device: Device inference runs on.

    Returns:
        One record per checkpoint, epoch-ordered. Empty when no periodic
        checkpoints were written.
    """
    checkpoints = sorted(
        weights_dir.glob("epoch*.pt"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or 0),
    )
    if not checkpoints:
        return []

    key = confidence_key(HEADLINE_CONFIDENCE)
    trajectory = []
    for checkpoint in [*checkpoints, weights_dir / "last.pt"]:
        detections = predict_windows(
            checkpoint, evaluation_set, imgsz, HEADLINE_CONFIDENCE, device
        )
        scored = score_detections(detections, evaluation_set, (HEADLINE_CONFIDENCE,), merge_radius)
        pooled = scored["by_confidence"][key]["pooled"]
        trajectory.append(
            {
                "checkpoint": checkpoint.stem,
                "recall_5px": pooled["recall"]["5"]["recall"],
                "recall_15px": pooled["recall"]["15"]["recall"],
                "p90_error_px": pooled["localization_error_px"]["p90"],
                "false_positives": pooled["false_positives"]["count"],
            }
        )
    return trajectory


def trajectory_table(trajectory: list[dict[str, Any]]) -> str:
    """Render an epoch trajectory as a markdown table.

    Args:
        trajectory: Records from :func:`epoch_trajectory`.

    Returns:
        Markdown text.
    """
    lines = [
        "### Epoch trajectory (conf 0.25; reported, never selected on)",
        "",
        "| checkpoint | recall@5px | recall@15px | p90 err | FP |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in trajectory:
        lines.append(
            f"| {point['checkpoint']} | {point['recall_5px']:.3f} "
            f"| {point['recall_15px']:.3f} | {point['p90_error_px']:.1f} "
            f"| {point['false_positives']} |"
        )
    return "\n".join(lines) + "\n"


#: The ordered criteria of checkpoint selection, as ``choose_checkpoint`` applies them.
SELECTION_RULE = ("max_hits_recall_5px", "min_false_positives", "later_epoch")

#: ``epoch<k>.pt`` names a periodic save by the 0-based epoch index.
EPOCH_FILE_RE = re.compile(r"^epoch(?P<index>\d+)\.pt$")


def choose_checkpoint(scores: Sequence[dict[str, Any]]) -> str:
    """Pick the candidate the pre-registered rule selects.

    Highest pooled recall@5px hits, then fewest false positives, then the later
    candidate. All candidates are scored on one evaluation set, so equal window
    counts make comparing false-positive counts the same as comparing them per
    window, without float equality.

    Args:
        scores: One row per candidate in epoch order, each with ``file``,
            ``hits``, ``false_positives`` and ``windows``.

    Returns:
        The chosen candidate's file name.

    Raises:
        ValueError: If there are no rows or the rows' window counts differ.
    """
    if not scores:
        raise ValueError("no candidate to choose from")
    if len({row["windows"] for row in scores}) != 1:
        raise ValueError(
            f"candidates were scored over different window counts: {[r['windows'] for r in scores]}"
        )
    best = max(
        enumerate(scores), key=lambda item: (item[1]["hits"], -item[1]["false_positives"], item[0])
    )
    return best[1]["file"]


def _headline_pooled(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the pooled block at the headline confidence."""
    return metrics["by_confidence"][confidence_key(HEADLINE_CONFIDENCE)]["pooled"]


def _headline_score(metrics: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return ``(hits, total, false_positives)`` as a candidate row records them."""
    pooled = _headline_pooled(metrics)
    return (
        pooled["recall"][f"{RECALL_RADII[0]:g}"]["hits"],
        pooled["annotations"],
        pooled["false_positives"]["count"],
    )


def _score_row(
    candidate: str,
    weights: Path,
    metrics: dict[str, Any],
    metrics_path: Path,
    windows: int,
    epochs: int,
) -> dict[str, Any]:
    """Describe one candidate's score at the headline confidence, pooled."""
    pooled = _headline_pooled(metrics)
    recall = pooled["recall"][f"{RECALL_RADII[0]:g}"]
    periodic = EPOCH_FILE_RE.match(candidate)
    index = int(periodic["index"]) if periodic else epochs - 1
    return {
        "file": candidate,
        "path": repo_relative(weights),
        "sha256": sha256_file(weights),
        "epoch_index_0_based": index,
        "epochs_completed": index + 1,
        "hits": recall["hits"],
        "total": pooled["annotations"],
        "recall": recall["recall"],
        "false_positives": pooled["false_positives"]["count"],
        "windows": windows,
        "fp_per_window": pooled["false_positives"]["per_window"],
        "metrics": repo_relative(metrics_path),
    }


def _selection_table(rows: Sequence[dict[str, Any]], chosen: str) -> str:
    """Render the candidate scores as markdown."""
    lines = [
        "| candidate | epochs completed | hits | recall@5px | FP | FP/window | chosen |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['file']} | {row['epochs_completed']} | {row['hits']}/{row['total']} "
            f"| {row['recall']:.3f} | {row['false_positives']} | {row['fp_per_window']:.2f} "
            f"| {'yes' if row['file'] == chosen else ''} |"
        )
    return "\n".join(lines) + "\n"


def select_checkpoint(train_dir: Path, device: torch.device, output_dir: Path) -> dict[str, Any]:
    """Score the six refit candidates on the selection sheets and choose one.

    Every check runs before the output directory exists. ``best.pt`` and any
    other file in the weights directory are recorded as ignored and never
    opened.

    Args:
        train_dir: Output directory of :func:`train_refit`.
        device: Device inference runs on.
        output_dir: Selection directory; must not exist.

    Returns:
        The ``selection.json`` document.

    Raises:
        FileNotFoundError: If ``train_record.json`` is absent.
        ValueError: If the training record no longer matches its config or
            dataset, or a candidate is missing.
        FileExistsError: If ``output_dir`` exists.
    """
    record_path = train_dir / "train_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    config = load_refit_config(REPO_ROOT / record["config"]["path"])
    dataset = REPO_ROOT / record["dataset"]["path"]
    report_path = dataset / "metadata" / "build_report.json"
    problems = []
    if config.sha256 != record["config"]["sha256"]:
        problems.append(f"{config.path} changed since training (sha256 differs)")
    if sha256_file(report_path) != record["dataset"]["build_report_sha256"]:
        problems.append(f"{report_path} changed since training (sha256 differs)")
    problems += build_output_problems(dataset)
    weights_dir = train_dir / REFIT_RUN_NAME / "weights"
    candidates = refit_candidates(config)
    missing = [c for c in candidates if not (weights_dir / c).is_file()]
    if missing:
        problems.append(f"candidates missing from {weights_dir}: {missing}")
    if problems:
        raise ValueError("the refit cannot be selected:\n  " + "\n  ".join(problems))
    ignored = sorted(p.name for p in weights_dir.iterdir() if p.name not in candidates)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    val_sheets = report["listing"]["listings"]["val"]["sheets"]
    evaluation_set = load_evaluation_set(dataset, dataset / "val.txt", val_sheets)
    environment = environment_record(device)

    new_directory(output_dir)
    rows = []
    for candidate in candidates:
        weights = weights_dir / candidate
        _, metrics = evaluate_checkpoint(
            weights=weights,
            evaluation_set=evaluation_set,
            imgsz=DETECTOR_IMGSZ,
            merge_radius=MERGE_RADIUS,
            thresholds=THRESHOLDS,
            device=device,
            output_dir=output_dir,
            label=weights.stem,
        )
        rows.append(
            _score_row(
                candidate,
                weights,
                metrics,
                output_dir / weights.stem / "metrics.json",
                len(evaluation_set.window_ids),
                config.epochs,
            )
        )
    chosen = choose_checkpoint(rows)
    chosen_row = next(row for row in rows if row["file"] == chosen)
    selection = {
        "train_record": {"path": repo_relative(record_path), "sha256": sha256_file(record_path)},
        "dataset": record["dataset"],
        "listing": "val.txt",
        "sheets": list(val_sheets),
        "confidence": confidence_key(HEADLINE_CONFIDENCE),
        "imgsz": DETECTOR_IMGSZ,
        "merge_radius": MERGE_RADIUS,
        "rule": list(SELECTION_RULE),
        "candidates": rows,
        "ignored": ignored,
        "chosen": {key: chosen_row[key] for key in ("file", "path", "sha256")},
        "note_epoch_naming": (
            "Ultralytics names periodic saves by the 0-based epoch index; "
            "epochN.pt holds the state after N+1 epochs."
        ),
        "environment": environment,
    }
    write_record(output_dir / "selection.json", json.dumps(selection, indent=2))
    write_record(output_dir / "selection.md", _selection_table(rows, chosen))
    return selection


def require_selection(selection: Mapping[str, Any], repo_root: Path) -> None:
    """Refuse a selection that is not what the pre-registered rule yields.

    The single definition of a valid refit selection: it re-derives the claim
    from the hashed training record, config and per-candidate metrics, so a
    hand-edited ``selection.json`` cannot steer the number a test opening pins.

    Args:
        selection: A parsed ``selection.json``.
        repo_root: Root the selection's repo-relative paths are relative to.

    Raises:
        ValueError: Listing every divergence from the rule, the training record,
            the config or the candidates' metrics.
    """
    problems: list[str] = []
    train_record = selection["train_record"]
    record_path = repo_root / train_record["path"]
    if sha256_file(record_path) != train_record["sha256"]:
        problems.append(f"{record_path} changed since the selection (sha256 differs)")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    config = load_refit_config(repo_root / record["config"]["path"])
    if config.sha256 != record["config"]["sha256"]:
        problems.append(f"{config.path} changed since training (sha256 differs)")
    if selection["dataset"] != record["dataset"]:
        problems.append("the selection's dataset is not the training record's")
    files = [row["file"] for row in selection["candidates"]]
    if files != list(refit_candidates(config)):
        problems.append(
            f"candidates {files} are not the config's {list(refit_candidates(config))}"
        )
    if selection["rule"] != list(SELECTION_RULE):
        problems.append(f"rule {selection['rule']} is not {list(SELECTION_RULE)}")
    if selection["confidence"] != confidence_key(HEADLINE_CONFIDENCE):
        problems.append(f"confidence {selection['confidence']} is not the headline confidence")
    if selection["imgsz"] != DETECTOR_IMGSZ or selection["merge_radius"] != MERGE_RADIUS:
        problems.append(
            f"scored at imgsz {selection['imgsz']} and merge radius {selection['merge_radius']}, "
            f"the openings use {DETECTOR_IMGSZ} and {MERGE_RADIUS}"
        )
    for row in selection["candidates"]:
        metrics = json.loads((repo_root / row["metrics"]).read_text(encoding="utf-8"))
        recorded = (row["hits"], row["total"], row["false_positives"])
        if _headline_score(metrics) != recorded:
            problems.append(
                f"{row['file']}: row says (hits, total, FP) {recorded}, "
                f"{row['metrics']} says {_headline_score(metrics)}"
            )
    chosen = selection["chosen"]
    if choose_checkpoint(selection["candidates"]) != chosen["file"]:
        problems.append(
            f"the rule chooses {choose_checkpoint(selection['candidates'])}, not {chosen['file']}"
        )
    chosen_rows = [row for row in selection["candidates"] if row["file"] == chosen["file"]]
    if len(chosen_rows) != 1 or any(
        chosen_rows[0][key] != chosen[key] for key in ("path", "sha256")
    ):
        problems.append(f"chosen {chosen['file']} does not match its candidate row")
    if sha256_file(repo_root / chosen["path"]) != chosen["sha256"]:
        problems.append(f"{chosen['path']} no longer hashes to the sha256 the selection chose")
    if problems:
        raise ValueError("the refit selection is not valid:\n  " + "\n  ".join(problems))


def run_loso(args: argparse.Namespace) -> None:
    """Train and evaluate the requested leave-one-sheet-out folds."""
    device = resolve_device(args.device)
    folds = list(FOLDS) if args.fold == "all" else [args.fold]
    summary = {}

    for fold in folds:
        weights = fold_weights_path(args.output_dir, fold)
        if not args.skip_train:
            weights = train_fold(
                data_yaml=args.dataset / f"{fold}.yaml",
                run_name=fold,
                output_dir=args.output_dir,
                model_name=args.model,
                imgsz=DETECTOR_IMGSZ,
                epochs=args.epochs,
                batch=args.batch,
                seed=args.seed,
                save_period=args.save_period,
                device=device,
            )
        evaluation_set = load_evaluation_set(
            args.dataset, args.dataset / f"{fold}_val.txt", [FOLDS[fold]]
        )
        _, metrics = evaluate_checkpoint(
            weights=weights,
            evaluation_set=evaluation_set,
            imgsz=DETECTOR_IMGSZ,
            merge_radius=MERGE_RADIUS,
            thresholds=THRESHOLDS,
            device=device,
            output_dir=args.output_dir,
            label=fold,
        )
        curve = operating_curve(metrics)
        summary[fold] = {"operating_points": curve}
        print(f"\n## {fold} (eval sheet {FOLDS[fold]})\n")
        print(curve_table(curve))

        trajectory = epoch_trajectory(
            weights_dir=weights.parent,
            evaluation_set=evaluation_set,
            imgsz=DETECTOR_IMGSZ,
            merge_radius=MERGE_RADIUS,
            device=device,
        )
        if trajectory:
            summary[fold]["epoch_trajectory"] = trajectory
            print(trajectory_table(trajectory))

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Parse arguments and run ``loso``, ``refit-train`` or ``refit-select``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    loso = commands.add_parser(
        "loso", parents=[common], help="train and evaluate the leave-one-sheet-out folds"
    )
    loso.add_argument("--dataset", type=Path, required=True)
    loso.add_argument("--output-dir", type=Path, required=True)
    loso.add_argument("--fold", default="all", choices=["all", *FOLDS])
    loso.add_argument("--model", default="yolo26s.pt")
    loso.add_argument("--epochs", type=int, default=300)
    loso.add_argument("--batch", type=int, default=16)
    loso.add_argument("--seed", type=int, default=42)
    loso.add_argument("--skip-train", action="store_true")
    loso.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Checkpoint every N epochs and report the trajectory (never select on it)",
    )
    loso.add_argument("--device", default="cuda")

    train = commands.add_parser("refit-train", parents=[common], help="train the refit detector")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", required=True)

    select = commands.add_parser(
        "refit-select", parents=[common], help="choose a refit checkpoint on validation"
    )
    select.add_argument("--train-dir", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--device", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    if args.command == "loso":
        run_loso(args)
    elif args.command == "refit-train":
        train_refit(
            load_refit_config(args.config),
            args.dataset,
            args.output_dir,
            resolve_device(args.device),
        )
    else:
        selection = select_checkpoint(args.train_dir, resolve_device(args.device), args.output_dir)
        print(f"chosen: {selection['chosen']['file']}")


if __name__ == "__main__":
    main()
