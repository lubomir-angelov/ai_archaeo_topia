#!/usr/bin/env python3
"""Normalized cross-correlation baseline for mound detection (MapSAM v0.5 step 3).

Run *before* the learned detector, not alongside it. These are printed
cartographic symbols from a single Soviet 1:50k series, so a handful of
representative templates and normalized cross-correlation is a fair question to
ask first: how much of the detection problem is available without learning
anything?

The answer is informative in both directions. High cross-sheet recall means the
learned detector has a real bar to clear and the project has a fallback with no
licence, no training and no GPU. A collapse on print quality, contour crossings
and overlapping symbology is a measured statement of what the learned detector
buys.

Templates are k-means centroids over intensity-normalized crops taken from the
*training* sheets of a fold only, so the baseline obeys the same leave-one-
sheet-out discipline as everything else in v0.5. Matching uses
``cv2.TM_CCOEFF_NORMED``, which is the normalized cross-correlation this
baseline is named for; OpenCV is already a project dependency, so this adds
nothing to the environment.

Usage:
    python -m archeo_topia.analysis.detect_template_baseline \\
        --dataset data/curated/datasets/mapsam_det_v0 \\
        --images-root data/curated/datasets/mapsam_v02/images \\
        --fold foldA --output-dir artifacts/detection/v0_5_template
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from archeo_topia.analysis.detection_metrics import (
    Detection,
    evaluate,
    filter_by_sheet,
    format_report,
    load_annotations,
    merge_detections,
)
from archeo_topia.datasets.build_detection_windows import FOLDS, find_image

logger = logging.getLogger(__name__)

#: Side length of a symbol template, in source pixels. Mound boxes run 8-33 px
#: with a 25x23 median, so 32 holds the whole symbol with a little context.
TEMPLATE_SIZE = 32


def normalize(patch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance a patch so templates ignore print density.

    Args:
        patch: Grayscale patch.

    Returns:
        Normalized float32 patch.
    """
    patch = patch.astype(np.float32)
    std = float(patch.std())
    return (patch - patch.mean()) / (std if std > 1e-6 else 1.0)


def extract_templates(
    annotations: list[Any],
    images_root: Path,
    count: int,
    size: int = TEMPLATE_SIZE,
    seed: int = 42,
) -> np.ndarray:
    """Build k-means template centroids from training-sheet mound crops.

    Args:
        annotations: Mound annotations from the training sheets only.
        images_root: Root of the clip images.
        count: Number of templates, i.e. k.
        size: Template side length in source pixels.
        seed: Seed for k-means initialization.

    Returns:
        Array of shape ``(k, size, size)``, float32 and normalized.
    """
    half = size // 2
    cache: dict[str, np.ndarray] = {}
    crops = []

    for ann in annotations:
        if ann.image not in cache:
            path = find_image(images_root, ann.image)
            cache[ann.image] = cv2.cvtColor(
                np.array(cv2.imread(str(path), cv2.IMREAD_COLOR)), cv2.COLOR_BGR2GRAY
            )
        gray = cache[ann.image]
        x, y = int(round(ann.x)), int(round(ann.y))
        if x - half < 0 or y - half < 0 or x + half > gray.shape[1] or y + half > gray.shape[0]:
            continue
        crops.append(normalize(gray[y - half : y + half, x - half : x + half]))

    if not crops:
        raise ValueError("no usable mound crops for template extraction")

    stack = np.stack(crops).reshape(len(crops), -1).astype(np.float32)
    count = min(count, len(stack))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    cv2.setRNGSeed(seed)
    _, _, centres = cv2.kmeans(stack, count, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    logger.info("built %d templates from %d crops", count, len(crops))
    return centres.reshape(count, TEMPLATE_SIZE, TEMPLATE_SIZE)


def detect_on_image(
    gray: np.ndarray,
    templates: np.ndarray,
    image_name: str,
    threshold: float,
) -> list[Detection]:
    """Score every position against every template and keep the local peaks.

    Args:
        gray: Grayscale source clip.
        templates: Template stack from :func:`extract_templates`.
        image_name: Clip file name, carried into the detections.
        threshold: Minimum correlation to emit a candidate.

    Returns:
        Candidate detections in source coordinates, before merging.
    """
    size = templates.shape[1]
    half = size // 2
    field = np.full(gray.shape, -1.0, dtype=np.float32)

    for template in templates:
        response = cv2.matchTemplate(gray.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
        # matchTemplate indexes by the template's top-left corner; shift to centre.
        padded = np.full(gray.shape, -1.0, dtype=np.float32)
        padded[half : half + response.shape[0], half : half + response.shape[1]] = response
        np.maximum(field, padded, out=field)

    # Local maxima over a symbol-sized neighbourhood, thresholded.
    peaks = cv2.dilate(field, np.ones((size // 2 + 1, size // 2 + 1), np.uint8))
    ys, xs = np.nonzero((field >= threshold) & (field >= peaks))
    return [Detection(image_name, float(x), float(y), float(field[y, x])) for y, x in zip(ys, xs, strict=True)]


def run(
    dataset: Path,
    images_root: Path,
    fold: str,
    output_dir: Path,
    templates: int = 8,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    merge_radius: float = 10.0,
) -> dict[str, Any]:
    """Fit templates on the fold's training sheets and score its eval sheet.

    Candidates are generated once at the lowest threshold and filtered by score
    for the rest, so the whole operating curve costs one correlation pass. A
    proposal generator is not usefully summarized by a single operating point:
    what matters is the recall available at a false-positive rate a downstream
    filter or a reviewer could absorb.

    Args:
        dataset: Detection dataset root from ``build_detection_windows``.
        images_root: Root of the clip images.
        fold: Fold name, one of ``foldA``/``foldB``/``foldC``.
        output_dir: Where to write metrics, the report and the templates.
        templates: Number of k-means template centroids.
        thresholds: Normalized-correlation operating points to report.
        merge_radius: Centre distance for cross-position deduplication.

    Returns:
        The metrics dict at the lowest threshold, with the full curve under
        ``operating_points``.
    """
    eval_sheet = FOLDS[fold]
    annotations, negatives, ignores = load_annotations(dataset / "metadata")

    train_annotations = [a for a in annotations if a.sheet != eval_sheet]
    eval_annotations = filter_by_sheet(annotations, eval_sheet)
    bank = extract_templates(train_annotations, images_root, templates)

    detections: list[Detection] = []
    pixels = 0
    images = sorted({a.image for a in eval_annotations})
    # Include eval-sheet clips that hold no mound at all; they still produce
    # false positives and dropping them would flatter the precision figure.
    for path in sorted(images_root.glob(f"*/{eval_sheet}_*.png")):
        if path.name not in images:
            images.append(path.name)

    floor = min(thresholds)
    for image_name in images:
        gray = cv2.imread(str(find_image(images_root, image_name)), cv2.IMREAD_GRAYSCALE)
        pixels += gray.size
        detections += detect_on_image(gray, bank, image_name, floor)

    window_count = sum(
        1
        for line in (dataset / "metadata" / "windows.jsonl").read_text().splitlines()
        if json.loads(line)["sheet_id"] == eval_sheet
    )
    common = {
        "annotations": eval_annotations,
        "hard_negatives": filter_by_sheet(negatives, eval_sheet),
        "ignore_regions": filter_by_sheet(ignores, eval_sheet),
        "window_count": window_count,
        "megapixels": pixels / 1e6,
    }

    curve = []
    best: dict[str, Any] = {}
    for threshold in sorted(thresholds):
        kept = merge_detections(
            [d for d in detections if d.score >= threshold], radius=merge_radius
        )
        metrics = evaluate(detections=kept, **common)
        metrics["config"] = {
            "fold": fold,
            "eval_sheet": eval_sheet,
            "templates": int(bank.shape[0]),
            "template_size": TEMPLATE_SIZE,
            "threshold": threshold,
            "merge_radius": merge_radius,
            "candidates": len(kept),
        }
        curve.append(
            {
                "threshold": threshold,
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
        logger.info(
            "%s t=%.2f: %d candidates, recall@5px %.3f, FP/window %.1f",
            fold,
            threshold,
            len(kept),
            curve[-1]["recall_5px"],
            curve[-1]["fp_per_window"],
        )

    best["operating_points"] = curve
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{fold}_metrics.json").write_text(json.dumps(best, indent=2), "utf-8")
    (output_dir / f"{fold}_report.md").write_text(
        format_report(best, f"Template baseline — {fold} (eval sheet {eval_sheet})")
        + "\n"
        + _curve_table(curve),
        "utf-8",
    )
    np.save(output_dir / f"{fold}_templates.npy", bank)
    return best


def _curve_table(curve: list[dict[str, Any]]) -> str:
    """Render the threshold sweep as a markdown table.

    Args:
        curve: Operating points from :func:`run`.

    Returns:
        Markdown text.
    """
    lines = [
        "### Operating curve",
        "",
        "| threshold | candidates | recall@5px | recall@15px | p90 err | FP | FP/window |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in curve:
        lines.append(
            f"| {point['threshold']:.2f} | {point['candidates']} "
            f"| {point['recall_5px']:.3f} | {point['recall_15px']:.3f} "
            f"| {point['p90_error_px']:.1f} | {point['false_positives']} "
            f"| {point['fp_per_window']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments and run the baseline over one or all folds."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--fold", default="all", choices=["all", *FOLDS])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--templates", type=int, default=8)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--merge-radius", type=float, default=10.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    folds = list(FOLDS) if args.fold == "all" else [args.fold]
    for fold in folds:
        metrics = run(
            dataset=args.dataset,
            images_root=args.images_root,
            fold=fold,
            output_dir=args.output_dir,
            templates=args.templates,
            thresholds=tuple(args.thresholds),
            merge_radius=args.merge_radius,
        )
        print(_curve_table(metrics["operating_points"]))


if __name__ == "__main__":
    main()
