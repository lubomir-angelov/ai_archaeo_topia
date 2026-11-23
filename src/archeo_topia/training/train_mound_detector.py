#!/usr/bin/env python3
"""Train and evaluate the YOLO26 mound detector (MapSAM v0.5 step 4).

One class, 512 px windows, leave-one-sheet-out folds matching the v0.4 MapSAM
configs. The model is ``yolo26s`` rather than a larger scale: with 180 mound
annotations, extra capacity is extra variance rather than extra information.

**Checkpoint selection is deliberately not done.** Ultralytics tracks a
``best.pt`` against the validation split, but the validation split here *is* the
held-out sheet, so selecting on it would leak the evaluation into the model.
There is no leak-free validation split until more sheets land — that is step 8
of the plan — so this trains a fixed number of epochs and evaluates ``last.pt``.
Any number produced from ``best.pt`` would be optimistic by an unknown amount.

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
    python -m archeo_topia.training.train_mound_detector \\
        --dataset data/curated/datasets/mapsam_det_v0 \\
        --fold all --model yolo26s.pt --imgsz 512 --epochs 300 \\
        --output-dir artifacts/detection/v0_5_yolo26s
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
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
from archeo_topia.analysis.runtime import resolve_device, write_record
from archeo_topia.datasets.build_detection_windows import FOLDS, crop_path, repo_relative

logger = logging.getLogger(__name__)

WINDOW_RE = re.compile(r"^(?P<stem>.+)_x(?P<x0>\d+)_y(?P<y0>\d+)$")

#: Confidence thresholds reported as operating points. The detector is
#: recall-oriented; precision is the second stage's problem.
THRESHOLDS = (0.05, 0.10, 0.25, 0.50, 0.70)

#: The operating point a single headline number is read at.
HEADLINE_CONFIDENCE = 0.25

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
    dataset: Path,
    fold: str,
    output_dir: Path,
    model_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    seed: int,
    save_period: int = -1,
) -> Path:
    """Train one leave-one-sheet-out fold and return the final weights.

    Args:
        dataset: Detection dataset root.
        fold: Fold name.
        output_dir: Run root; the fold gets a subdirectory.
        model_name: Ultralytics model or checkpoint name.
        imgsz: Training image size. 512 is native for these windows.
        epochs: Fixed epoch count; no early stopping, see the module docstring.
        batch: Batch size.
        seed: Random seed.
        save_period: Save a checkpoint every N epochs, so the epoch trajectory
            can be *reported*. Reporting it is not selecting on it; see step 8
            of the plan for why no leak-free selection exists yet.

    Returns:
        Path to ``last.pt``.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data=str((dataset / f"{fold}.yaml").resolve()),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        seed=seed,
        deterministic=True,
        project=str(output_dir.resolve()),
        name=fold,
        exist_ok=True,
        # Disable early stopping: the only validation split available is the
        # held-out sheet, so stopping against it would select on the evaluation.
        patience=epochs + 1,
        save_period=save_period,
        val=True,
        plots=True,
        verbose=False,
    )
    return fold_weights_path(output_dir, fold)


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


def main() -> None:
    """Parse arguments, then train and evaluate the requested folds."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", default="all", choices=["all", *FOLDS])
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-radius", type=float, default=10.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Checkpoint every N epochs and report the trajectory (never select on it)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    device = resolve_device(args.device)
    folds = list(FOLDS) if args.fold == "all" else [args.fold]
    summary = {}

    for fold in folds:
        weights = fold_weights_path(args.output_dir, fold)
        if not args.skip_train:
            weights = train_fold(
                dataset=args.dataset,
                fold=fold,
                output_dir=args.output_dir,
                model_name=args.model,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                seed=args.seed,
                save_period=args.save_period,
            )
        evaluation_set = load_evaluation_set(
            args.dataset, args.dataset / f"{fold}_val.txt", [FOLDS[fold]]
        )
        _, metrics = evaluate_checkpoint(
            weights=weights,
            evaluation_set=evaluation_set,
            imgsz=args.imgsz,
            merge_radius=args.merge_radius,
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
            imgsz=args.imgsz,
            merge_radius=args.merge_radius,
            device=device,
        )
        if trajectory:
            summary[fold]["epoch_trajectory"] = trajectory
            print(trajectory_table(trajectory))

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
