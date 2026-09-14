#!/usr/bin/env python3
"""End-to-end evaluation: detector candidates through the MapSAM decoder (v0.5 step 5).

This is the first number in the project that answers the actual question. Every
result before it was conditional on already knowing where the mound was.

The pipeline per candidate is exactly v0.4's representation, with the detector
supplying the prompt instead of the ground truth:

    candidate centre
        -> 512 px source window, centred on the candidate, clamped in the tile
        -> point and box prompts mapped into window coordinates
        -> resize to 1024, MapSAM (SAM ViT-B, frozen encoders, tuned decoder)
        -> binary mask, mapped back to source coordinates
        -> IoU against the annotation's own mask

**Two prompt arms, because they separate two error sources.** v0.4's jitter
sweep translated the ground-truth box rigidly: it modelled a detector that
sizes correctly and centres wrongly, and explicitly left box *scale* error
untested. So this runs both:

``fixed``
    A box of the dataset's median mound size centred on the candidate. Only
    localization error reaches the segmenter, so the result should be
    predictable from the jitter curve and the measured centre-error
    distribution. If it is not, something outside localization is wrong.

``detector``
    The detector's own box. This is what deployment would do, and the gap
    between the arms is the cost of box-scale error — the quantity v0.4 could
    not measure.

The decoder is frozen at its v0.4 fold checkpoint and is not retrained here.
Changing both ends at once would make the number uninterpretable.

Usage:
    python -m archeo_topia.analysis.mapsam_end_to_end \\
        --dataset data/curated/datasets/mapsam_det_v0 \\
        --detector-run artifacts/detection/v0_5_yolo26s \\
        --mapsam-root artifacts/models/mapsam \\
        --images-root data/curated/datasets/mapsam_v02/images \\
        --coco-json annotation/cvat/v0.0.1/instances_default.json \\
        --fold all --confidence 0.25 --output-dir artifacts/detection/v0_5_end_to_end
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from archeo_topia.analysis.detection_metrics import (
    ASSOCIATION_RADIUS,
    Detection,
    filter_by_sheet,
    load_annotations,
    merge_detections,
)
from archeo_topia.datasets.build_detection_windows import (
    FOLDS,
    MOUND,
    decode_mask,
    find_image,
)
from archeo_topia.datasets.mapsam_window import compute_window

logger = logging.getLogger(__name__)

#: Median mound bounding-box size in source pixels, from the 180 annotations.
MEDIAN_BOX_WH = (25.0, 23.0)

#: Padding applied to prompt boxes, matching generate_mapsam_prompts.
BOX_PADDING = 4.0

#: IoU thresholds the end-to-end rate is reported at.
IOU_THRESHOLDS = (0.5, 0.75)


def load_mapsam(
    fold: str,
    mapsam_root: Path,
    sam_checkpoint: Path,
    device: torch.device,
    run_prefix: str = "v0_4r_loso512",
) -> Any:
    """Load the v0.4 512-window decoder for one fold.

    Args:
        fold: Fold name.
        mapsam_root: Root holding the decoder run directories.
        sam_checkpoint: Base SAM ViT-B checkpoint the decoder was trained on.
        device: Target device.
        run_prefix: Run-directory prefix. Defaults to the v0.4 revision trained
            on the corrected annotations; pass ``v0_4_loso512`` to use the
            original checkpoints instead.

    Returns:
        The SAM model in eval mode.

    Raises:
        FileNotFoundError: If the fold checkpoint is missing.
    """
    from archeo_topia.training.train_mapsam_v0 import build_sam_model, load_checkpoint

    checkpoint = mapsam_root / f"{run_prefix}_{fold}_pw20" / "checkpoints" / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"MapSAM checkpoint not found: {checkpoint}")

    sam = build_sam_model(
        {
            "model": {
                "model_type": "vit_b",
                "sam_checkpoint": str(sam_checkpoint),
                "freeze_image_encoder": True,
                "train_prompt_encoder": False,
                "train_mask_decoder": True,
                "use_cached_embeddings": False,
            }
        }
    )
    load_checkpoint(checkpoint, sam)
    return sam.to(device).eval()


def prompt_box(detection: Detection, mode: str) -> tuple[float, float, float, float]:
    """Build the box prompt for a candidate.

    Args:
        detection: Candidate with a centre and, optionally, its own box.
        mode: ``fixed`` for a median-sized box on the candidate centre,
            ``detector`` for the candidate's own box.

    Returns:
        Source-coordinate ``(x0, y0, x1, y1)``.

    Raises:
        ValueError: If ``detector`` is requested for a boxless candidate.
    """
    if mode == "detector":
        if detection.box is None:
            raise ValueError("detector prompt mode needs candidates carrying boxes")
        x0, y0, x1, y1 = detection.box
        return (x0 - BOX_PADDING, y0 - BOX_PADDING, x1 + BOX_PADDING, y1 + BOX_PADDING)

    half_w = MEDIAN_BOX_WH[0] / 2 + BOX_PADDING
    half_h = MEDIAN_BOX_WH[1] / 2 + BOX_PADDING
    return (detection.x - half_w, detection.y - half_h, detection.x + half_w, detection.y + half_h)


def segment(
    sam: Any,
    image: torch.Tensor,
    detection: Detection,
    box_mode: str,
    window_px: int,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Segment one candidate and return its mask in window coordinates.

    Args:
        sam: Loaded MapSAM model.
        image: Source clip as ``(3, H, W)`` in [0, 1].
        detection: The candidate.
        box_mode: ``fixed`` or ``detector``.
        window_px: Window side length in source pixels.
        image_size: Encoder input size.
        device: Target device.

    Returns:
        Tuple of the binary mask at window resolution and the window
        ``(x0, y0, size)``.
    """
    from archeo_topia.training.train_mapsam_v0 import forward_sam

    _, height, width = image.shape
    window = compute_window((detection.x, detection.y), (height, width), window_px)
    crop = image[:, window.y0 : window.y0 + window.size, window.x0 : window.x0 + window.size]

    scale = image_size / window.size
    point = torch.tensor(
        [(detection.x - window.x0) * scale, (detection.y - window.y0) * scale],
        dtype=torch.float32,
    )
    bx0, by0, bx1, by1 = prompt_box(detection, box_mode)
    box = torch.tensor(
        [
            (bx0 - window.x0) * scale,
            (by0 - window.y0) * scale,
            (bx1 - window.x0) * scale,
            (by1 - window.y0) * scale,
        ],
        dtype=torch.float32,
    )

    resized = F.interpolate(
        crop.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    logits = forward_sam(
        sam,
        resized,
        box,
        point,
        torch.tensor([1], dtype=torch.int64),
        device,
    )
    upsampled = F.interpolate(
        logits, size=(window.size, window.size), mode="bilinear", align_corners=False
    )
    return (upsampled[0, 0] > 0).cpu().numpy(), (window.x0, window.y0, window.size)


def mask_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    Args:
        predicted: Predicted mask.
        truth: Ground-truth mask.

    Returns:
        IoU, zero when both are empty.
    """
    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    return float(intersection / union) if union else 0.0


def run_fold(
    fold: str,
    dataset: Path,
    detector_run: Path,
    mapsam_root: Path,
    images_root: Path,
    coco_json: Path,
    sam_checkpoint: Path,
    confidence: float,
    box_modes: tuple[str, ...],
    detector_imgsz: int,
    window_px: int,
    image_size: int,
    merge_radius: float,
    device: torch.device,
    mapsam_prefix: str = "v0_4r_loso512",
) -> dict[str, Any]:
    """Run the full pipeline for one fold and score it.

    Args:
        fold: Fold name.
        dataset: Detection dataset root.
        detector_run: Detector run root holding per-fold weights.
        mapsam_root: Root holding the v0.4 MapSAM runs.
        images_root: Root of the clip images.
        coco_json: CVAT COCO export, for per-annotation ground-truth masks.
        sam_checkpoint: Base SAM checkpoint.
        confidence: Detector confidence threshold.
        box_modes: Prompt arms to run.
        detector_imgsz: Detector inference size; must match its training size.
        window_px: Window side length.
        image_size: Encoder input size.
        merge_radius: Cross-window deduplication radius.
        device: Target device.
        mapsam_prefix: Decoder run-directory prefix.

    Returns:
        Metrics for the fold, one block per prompt arm.
    """
    from archeo_topia.training.train_mound_detector import predict_fold

    eval_sheet = FOLDS[fold]
    annotations, _, _ = load_annotations(dataset / "metadata")
    eval_annotations = filter_by_sheet(annotations, eval_sheet)
    scorable = [a for a in eval_annotations if a.has_geometry]

    weights = detector_run / fold / "weights" / "last.pt"
    raw = predict_fold(weights, dataset, fold, detector_imgsz, confidence)
    candidates = merge_detections(
        [d for d in raw if d.score >= confidence], radius=merge_radius
    )
    logger.info("%s: %d candidates at conf %.2f", fold, len(candidates), confidence)

    with open(coco_json, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    # Match the positive category by name. Category ids are assigned by the
    # exporter and are not stable across re-exports, so a hardcoded id here
    # would silently select the wrong class after a relabel.
    mound_ids = {c["id"] for c in coco["categories"] if c["name"] == MOUND}
    if not mound_ids:
        raise ValueError(f"no {MOUND!r} category in {coco_json}")
    truth_masks: dict[int, np.ndarray] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in mound_ids:
            continue
        meta = images[ann["image_id"]]
        if meta["file_name"].rsplit("_", 1)[0] != eval_sheet:
            continue
        mask = decode_mask(ann, meta["width"], meta["height"])
        if mask.any():
            truth_masks[ann["id"]] = mask

    sam = load_mapsam(fold, mapsam_root, sam_checkpoint, device, mapsam_prefix)

    clips: dict[str, torch.Tensor] = {}
    for name in sorted({c.image for c in candidates} | {a.image for a in scorable}):
        array = np.array(Image.open(find_image(images_root, name)).convert("RGB"))
        clips[name] = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    results: dict[str, Any] = {"fold": fold, "eval_sheet": eval_sheet, "arms": {}}

    for box_mode in box_modes:
        best_iou: dict[int, float] = defaultdict(float)
        claimed: dict[int, int] = {}
        false_masks = 0

        for index, candidate in enumerate(candidates):
            predicted, (wx0, wy0, size) = segment(
                sam, clips[candidate.image], candidate, box_mode, window_px, image_size, device
            )
            nearby = [
                a
                for a in scorable
                if a.image == candidate.image
                and abs(a.x - candidate.x) <= ASSOCIATION_RADIUS
                and abs(a.y - candidate.y) <= ASSOCIATION_RADIUS
            ]
            if not nearby:
                false_masks += 1
                continue
            for annotation in nearby:
                truth = truth_masks.get(annotation.annotation_id)
                if truth is None:
                    continue
                window_truth = truth[wy0 : wy0 + size, wx0 : wx0 + size]
                iou = mask_iou(predicted, window_truth)
                if iou > best_iou[annotation.annotation_id]:
                    best_iou[annotation.annotation_id] = iou
                    claimed[annotation.annotation_id] = index

        total = len(scorable)
        arm: dict[str, Any] = {
            "candidates": len(candidates),
            "scorable_annotations": total,
            "false_masks": false_masks,
            "mean_iou": float(np.mean([best_iou[a.annotation_id] for a in scorable])),
        }
        for threshold in IOU_THRESHOLDS:
            hits = sum(1 for a in scorable if best_iou[a.annotation_id] >= threshold)
            arm[f"rate_iou_{threshold:g}"] = hits / total if total else float("nan")
            arm[f"hits_iou_{threshold:g}"] = hits
        arm["missed_rate"] = sum(1 for a in scorable if best_iou[a.annotation_id] == 0.0) / total
        arm["per_annotation_iou"] = {
            str(a.annotation_id): round(best_iou[a.annotation_id], 4) for a in scorable
        }
        results["arms"][box_mode] = arm
        logger.info(
            "%s %s: IoU>=0.5 %.3f, IoU>=0.75 %.3f, mean %.4f",
            fold,
            box_mode,
            arm["rate_iou_0.5"],
            arm["rate_iou_0.75"],
            arm["mean_iou"],
        )

    return results


def format_report(results: list[dict[str, Any]]) -> str:
    """Render the end-to-end results as markdown.

    Args:
        results: Per-fold result dicts.

    Returns:
        Markdown text.
    """
    lines = [
        "## End-to-end: detector candidates through the frozen 512 decoder",
        "",
        "Denominator is annotations with geometry on the evaluation sheet, not",
        "segmentation samples and not all 180 annotations.",
        "",
        "| fold | sheet | arm | n | IoU>=0.5 | IoU>=0.75 | mean IoU | missed | false masks |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for arm_name, arm in result["arms"].items():
            lines.append(
                f"| {result['fold']} | {result['eval_sheet']} | `{arm_name}` "
                f"| {arm['scorable_annotations']} | {arm['rate_iou_0.5']:.3f} "
                f"| {arm['rate_iou_0.75']:.3f} | {arm['mean_iou']:.4f} "
                f"| {arm['missed_rate']:.3f} | {arm['false_masks']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments and run the end-to-end evaluation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--detector-run", type=Path, required=True)
    parser.add_argument("--mapsam-root", type=Path, default=Path("artifacts/models/mapsam"))
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument(
        "--sam-checkpoint", type=Path, default=Path("models/checkpoints/sam/sam_vit_b_01ec64.pth")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", default="all", choices=["all", *FOLDS])
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--box-modes", nargs="+", default=["fixed", "detector"])
    parser.add_argument("--detector-imgsz", type=int, default=512)
    parser.add_argument("--window-px", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--merge-radius", type=float, default=10.0)
    parser.add_argument("--mapsam-prefix", default="v0_4r_loso512")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    folds = list(FOLDS) if args.fold == "all" else [args.fold]

    results = []
    for fold in folds:
        results.append(
            run_fold(
                fold=fold,
                dataset=args.dataset,
                detector_run=args.detector_run,
                mapsam_root=args.mapsam_root,
                images_root=args.images_root,
                coco_json=args.coco_json,
                sam_checkpoint=args.sam_checkpoint,
                confidence=args.confidence,
                box_modes=tuple(args.box_modes),
                detector_imgsz=args.detector_imgsz,
                window_px=args.window_px,
                image_size=args.image_size,
                merge_radius=args.merge_radius,
                device=device,
                mapsam_prefix=args.mapsam_prefix,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "end_to_end.json").write_text(json.dumps(results, indent=2), "utf-8")
    report = format_report(results)
    (args.output_dir / "end_to_end.md").write_text(report, "utf-8")
    print(report)


if __name__ == "__main__":
    main()
