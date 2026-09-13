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
from pathlib import Path
from typing import Any

from archeo_topia.analysis.detection_metrics import (
    Detection,
    evaluate,
    filter_by_sheet,
    format_report,
    load_annotations,
    merge_detections,
)
from archeo_topia.datasets.build_detection_windows import FOLDS

logger = logging.getLogger(__name__)

WINDOW_RE = re.compile(r"^(?P<stem>.+)_x(?P<x0>\d+)_y(?P<y0>\d+)$")

#: Confidence thresholds reported as operating points. The detector is
#: recall-oriented; precision is the second stage's problem.
THRESHOLDS = (0.05, 0.10, 0.25, 0.50, 0.70)


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
    return output_dir / fold / "weights" / "last.pt"


def predict_fold(
    weights: Path,
    dataset: Path,
    fold: str,
    imgsz: int,
    min_confidence: float,
) -> list[Detection]:
    """Run the detector over a fold's evaluation windows.

    Boxes come back in window coordinates and are mapped to source-tile
    coordinates before anything else happens, so every downstream number is in
    the same space as the annotations and as MapSAM's prompts.

    Args:
        weights: Model weights.
        dataset: Detection dataset root.
        fold: Fold name.
        imgsz: Inference image size; must match training.
        min_confidence: Lowest confidence to emit.

    Returns:
        Detections in source coordinates, before merging.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    listing = (dataset / f"{fold}_val.txt").read_text(encoding="utf-8").split()
    paths = [str((dataset / entry.lstrip("./")).resolve()) for entry in listing]

    detections: list[Detection] = []
    for start in range(0, len(paths), 32):
        batch = paths[start : start + 32]
        predictions = model.predict(batch, imgsz=imgsz, conf=min_confidence, verbose=False)
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
                    )
                )

    logger.info("%s: %d raw detections over %d windows", fold, len(detections), len(paths))
    return detections


def evaluate_fold(
    detections: list[Detection],
    dataset: Path,
    fold: str,
    output_dir: Path,
    merge_radius: float,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, Any]:
    """Score a fold's detections and write its metrics and report.

    Args:
        detections: Source-coordinate detections, unmerged.
        dataset: Detection dataset root.
        fold: Fold name.
        output_dir: Run root.
        merge_radius: Centre distance for cross-window deduplication.
        thresholds: Confidence operating points to report.

    Returns:
        Metrics at the lowest threshold, with the curve under
        ``operating_points``.
    """
    eval_sheet = FOLDS[fold]
    annotations, negatives, ignores = load_annotations(dataset / "metadata")
    eval_annotations = filter_by_sheet(annotations, eval_sheet)

    windows = [
        json.loads(line)
        for line in (dataset / "metadata" / "windows.jsonl").read_text().splitlines()
    ]
    eval_windows = [w for w in windows if w["sheet_id"] == eval_sheet]
    # Clip extent is the furthest window edge, since the final row and column
    # are clamped flush with the image.
    extents: dict[str, tuple[int, int]] = {}
    for window in eval_windows:
        _, _, x1, y1 = window["window_xyxy"]
        width, height = extents.get(window["source_image"], (0, 0))
        extents[window["source_image"]] = (max(width, x1), max(height, y1))
    megapixels = sum(w * h for w, h in extents.values()) / 1e6

    common = {
        "annotations": eval_annotations,
        "hard_negatives": filter_by_sheet(negatives, eval_sheet),
        "ignore_regions": filter_by_sheet(ignores, eval_sheet),
        "window_count": len(eval_windows),
        "megapixels": megapixels,
    }

    curve = []
    best: dict[str, Any] = {}
    for threshold in sorted(thresholds):
        kept = merge_detections(
            [d for d in detections if d.score >= threshold], radius=merge_radius
        )
        metrics = evaluate(detections=kept, **common)
        curve.append(
            {
                "confidence": threshold,
                "candidates": len(kept),
                "recall_5px": metrics["recall"]["5"]["recall"],
                "recall_10px": metrics["recall"]["10"]["recall"],
                "recall_15px": metrics["recall"]["15"]["recall"],
                "recall_25px": metrics["recall"]["25"]["recall"],
                "p90_error_px": metrics["localization_error_px"]["p90"],
                "false_positives": metrics["false_positives"]["count"],
                "fp_per_window": metrics["false_positives"]["per_window"],
            }
        )
        if not best:
            best = metrics
            best["config"] = {
                "fold": fold,
                "eval_sheet": eval_sheet,
                "merge_radius": merge_radius,
                "checkpoint": "last.pt (no selection; see module docstring)",
            }

    best["operating_points"] = curve
    fold_dir = output_dir / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (fold_dir / "report.md").write_text(
        format_report(best, f"YOLO26-s — {fold} (eval sheet {eval_sheet})")
        + "\n"
        + curve_table(curve),
        encoding="utf-8",
    )
    return best


def curve_table(curve: list[dict[str, Any]]) -> str:
    """Render a confidence sweep as a markdown table.

    Args:
        curve: Operating points.

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
    dataset: Path,
    fold: str,
    imgsz: int,
    min_confidence: float,
    merge_radius: float,
    output_dir: Path,
    confidence: float = 0.25,
) -> list[dict[str, Any]]:
    """Score every periodic checkpoint at one operating point.

    v0.4 traced a late-training decision drift in the MapSAM decoder: fold A
    went from 2 zero-IoU samples at epoch 5 to 19 at epoch 50 with the peak
    logit still correctly placed. The detector has the same exposure and the
    same absence of a clean validation split, so the trajectory is recorded as
    evidence rather than used to pick a checkpoint.

    Args:
        weights_dir: Ultralytics ``weights`` directory.
        dataset: Detection dataset root.
        fold: Fold name.
        imgsz: Inference image size.
        min_confidence: Lowest confidence to emit during prediction.
        merge_radius: Centre distance for deduplication.
        output_dir: Run root, for locating metadata.
        confidence: Operating point at which the trajectory is reported.

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

    eval_sheet = FOLDS[fold]
    annotations, negatives, ignores = load_annotations(dataset / "metadata")
    eval_annotations = filter_by_sheet(annotations, eval_sheet)
    window_count = sum(
        1
        for line in (dataset / "metadata" / "windows.jsonl").read_text().splitlines()
        if json.loads(line)["sheet_id"] == eval_sheet
    )

    trajectory = []
    for checkpoint in [*checkpoints, weights_dir / "last.pt"]:
        detections = predict_fold(checkpoint, dataset, fold, imgsz, min_confidence)
        kept = merge_detections(
            [d for d in detections if d.score >= confidence], radius=merge_radius
        )
        metrics = evaluate(
            detections=kept,
            annotations=eval_annotations,
            hard_negatives=filter_by_sheet(negatives, eval_sheet),
            ignore_regions=filter_by_sheet(ignores, eval_sheet),
            window_count=window_count,
        )
        trajectory.append(
            {
                "checkpoint": checkpoint.stem,
                "recall_5px": metrics["recall"]["5"]["recall"],
                "recall_15px": metrics["recall"]["15"]["recall"],
                "p90_error_px": metrics["localization_error_px"]["p90"],
                "false_positives": metrics["false_positives"]["count"],
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
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Checkpoint every N epochs and report the trajectory (never select on it)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    folds = list(FOLDS) if args.fold == "all" else [args.fold]
    summary = {}

    for fold in folds:
        weights = args.output_dir / fold / "weights" / "last.pt"
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
        detections = predict_fold(
            weights=weights,
            dataset=args.dataset,
            fold=fold,
            imgsz=args.imgsz,
            min_confidence=args.min_confidence,
        )
        metrics = evaluate_fold(
            detections=detections,
            dataset=args.dataset,
            fold=fold,
            output_dir=args.output_dir,
            merge_radius=args.merge_radius,
        )
        summary[fold] = {"operating_points": metrics["operating_points"]}
        print(f"\n## {fold} (eval sheet {FOLDS[fold]})\n")
        print(curve_table(metrics["operating_points"]))

        trajectory = epoch_trajectory(
            weights_dir=weights.parent,
            dataset=args.dataset,
            fold=fold,
            imgsz=args.imgsz,
            min_confidence=args.min_confidence,
            merge_radius=args.merge_radius,
            output_dir=args.output_dir,
        )
        if trajectory:
            summary[fold]["epoch_trajectory"] = trajectory
            print(trajectory_table(trajectory))

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
