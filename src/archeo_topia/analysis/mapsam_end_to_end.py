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

The module is split along the inference / scoring seam. ``segment_candidates``
runs the model; ``score_end_to_end`` is pure. A detector and a decoder are
paired explicitly at every call site (``run_pair``), never by a fold-name
convention inside the scorer.

Neutralization follows the detection scorer: a candidate inside an
``uncertain_ignore`` region is segmented like every other, then excluded from
scoring through ``detection_metrics.neutralize`` and counted, not scored as a
false mask.

Usage:
    python -m archeo_topia.analysis.mapsam_end_to_end \\
        --dataset data/curated/datasets/mapsam_det_v0 \\
        --detector-run artifacts/detection/v0_5_yolo26s \\
        --mapsam-root artifacts/models/mapsam \\
        --images-root data/curated/datasets/mapsam_v02/images \\
        --coco-json annotation/cvat/v0.0.2/instances_default.json \\
        --fold all --confidence 0.25 --device cuda:0 \\
        --output-dir artifacts/detection/v0_5_end_to_end
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from archeo_topia.analysis.detection_metrics import (
    ASSOCIATION_RADIUS,
    Detection,
    EvaluationSet,
    annotation_key,
    load_evaluation_set,
    macro_summary,
    merge_detections,
    neutralize,
    wilson_interval,
)
from archeo_topia.analysis.runtime import new_directory, resolve_device, write_record
from archeo_topia.datasets.build_detection_windows import (
    FOLDS,
    MOUND,
    decode_mask,
    find_image,
    repo_relative,
)
from archeo_topia.datasets.mapsam_window import compute_window
from archeo_topia.training.train_mound_detector import (
    HEADLINE_CONFIDENCE,
    MIN_CONFIDENCE,
    fold_weights_path,
    predict_windows,
)

logger = logging.getLogger(__name__)

#: Median mound bounding-box size in source pixels, from the 180 annotations.
MEDIAN_BOX_WH = (25.0, 23.0)

#: Padding applied to prompt boxes, matching generate_mapsam_prompts.
BOX_PADDING = 4.0

#: IoU thresholds the end-to-end rate is reported at.
IOU_THRESHOLDS = (0.5, 0.75)

#: Prompt arms. ``fixed`` is a median-sized box on the candidate centre;
#: ``detector`` is the detector's own box.
BOX_MODES = ("fixed", "detector")


def decoder_checkpoint_path(mapsam_root: Path, prefix: str, fold: str) -> Path:
    """Return a LOSO fold's MapSAM decoder checkpoint.

    Args:
        mapsam_root: Directory of MapSAM training runs.
        prefix: Run-name prefix, for example ``v0_4r_loso512``.
        fold: Fold name.

    Returns:
        Path to ``final.pt``.
    """
    return mapsam_root / f"{prefix}_{fold}_pw20" / "checkpoints" / "final.pt"


def load_mapsam(checkpoint: Path, sam_checkpoint: Path, device: torch.device) -> Any:
    """Load a MapSAM decoder checkpoint on top of the base SAM.

    Args:
        checkpoint: Decoder checkpoint, for example a v0.4 fold ``final.pt``.
        sam_checkpoint: Base SAM ViT-B checkpoint the decoder was trained on.
        device: Target device.

    Returns:
        The SAM model in eval mode.

    Raises:
        FileNotFoundError: If the decoder checkpoint is missing.
    """
    from archeo_topia.training.train_mapsam_v0 import build_sam_model, load_checkpoint

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


def resolve_candidates(
    raw: Sequence[Detection], confidence: float, merge_radius: float
) -> list[Detection]:
    """Filter raw detections to an operating point and merge duplicates.

    Args:
        raw: Raw source-coordinate detections, before merging.
        confidence: Detector confidence threshold.
        merge_radius: Cross-window deduplication radius.

    Returns:
        The candidates the segmenter is run on.
    """
    return merge_detections([d for d in raw if d.score >= confidence], radius=merge_radius)


def segment_candidates(
    sam: Any,
    candidates: Sequence[Detection],
    clip_paths: Mapping[str, Path],
    box_mode: str,
    window_px: int,
    image_size: int,
    device: torch.device,
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Segment every candidate under one prompt arm.

    Args:
        sam: Loaded MapSAM model.
        candidates: Candidates to segment.
        clip_paths: Source clip path by file name; must cover every candidate.
        box_mode: ``fixed`` or ``detector``.
        window_px: Window side length in source pixels.
        image_size: Encoder input size.
        device: Target device.

    Returns:
        One ``(mask, (window x0, window y0, size))`` per candidate, in order.

    Raises:
        ValueError: On an unknown arm or a candidate whose clip has no path.
    """
    if box_mode not in BOX_MODES:
        raise ValueError(f"unknown box mode {box_mode!r}; have {list(BOX_MODES)}")
    clips: dict[str, torch.Tensor] = {}
    segmented = []
    for candidate in candidates:
        if candidate.image not in clips:
            if candidate.image not in clip_paths:
                raise ValueError(f"no clip path for candidate image {candidate.image}")
            array = np.array(Image.open(clip_paths[candidate.image]).convert("RGB"))
            clips[candidate.image] = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        segmented.append(
            segment(
                sam, clips[candidate.image], candidate, box_mode, window_px, image_size, device
            )
        )
    return segmented


def load_truth_masks(
    coco_json: Path, evaluation_set: EvaluationSet
) -> dict[tuple[str, int], np.ndarray]:
    """Rasterize the ground-truth mound masks of the evaluation set's images.

    Args:
        coco_json: CVAT COCO export.
        evaluation_set: Selects the images whose masks are needed.

    Returns:
        Mask by ``(image, annotation id)``, for annotations that have geometry.

    Raises:
        ValueError: If the export has no ``mound`` category.
    """
    with open(coco_json, encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {i["id"]: i for i in coco["images"]}
    # Match the positive category by name. Category ids are assigned by the
    # exporter and are not stable across re-exports, so a hardcoded id here
    # would silently select the wrong class after a relabel.
    mound_ids = {c["id"] for c in coco["categories"] if c["name"] == MOUND}
    if not mound_ids:
        raise ValueError(f"no {MOUND!r} category in {coco_json}")
    truth_masks: dict[tuple[str, int], np.ndarray] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in mound_ids:
            continue
        meta = images[ann["image_id"]]
        if meta["file_name"] not in evaluation_set.image_sheet:
            continue
        mask = decode_mask(ann, meta["width"], meta["height"])
        if mask.any():
            truth_masks[(meta["file_name"], ann["id"])] = mask
    return truth_masks


def _end_to_end_block(
    scorable: Sequence[Any],
    mounds_without_geometry: int,
    candidates: int,
    false_masks: int,
    best_iou: Mapping[str, float],
) -> dict[str, Any]:
    """Assemble one ``E2EBlock`` over a set of scorable annotations."""
    total = len(scorable)
    keys = [annotation_key(a.image, a.annotation_id) for a in scorable]
    ious = [best_iou[key] for key in keys]
    found_ious = [iou for iou in ious if iou > 0.0]
    block: dict[str, Any] = {
        "scorable_annotations": total,
        "mounds_without_geometry": mounds_without_geometry,
        "candidates": candidates,
        "false_masks": false_masks,
    }
    for threshold in IOU_THRESHOLDS:
        hits = sum(1 for iou in ious if iou >= threshold)
        low, high = wilson_interval(hits, total)
        block[f"iou_{threshold:g}"] = {
            "rate": hits / total if total else float("nan"),
            "hits": hits,
            "ci95": [low, high],
        }
    block["mean_iou"] = float(np.mean(ious)) if ious else float("nan")
    block["median_iou"] = float(np.median(ious)) if ious else float("nan")
    block["found"] = {
        "n": len(found_ious),
        "mean_iou": float(np.mean(found_ious)) if found_ious else float("nan"),
    }
    missed = total - len(found_ious)
    block["missed"] = {"count": missed, "rate": missed / total if total else float("nan")}
    block["per_annotation_iou"] = dict(zip(keys, ious, strict=True))
    return block


def _macro_inputs(block: dict[str, Any]) -> dict[str, float]:
    """Pick the per-sheet quantities that are summarized across sheets."""
    inputs = {f"rate_iou_{t:g}": block[f"iou_{t:g}"]["rate"] for t in IOU_THRESHOLDS}
    inputs["mean_iou"] = block["mean_iou"]
    inputs["median_iou"] = block["median_iou"]
    inputs["found_mean_iou"] = block["found"]["mean_iou"]
    inputs["missed_rate"] = block["missed"]["rate"]
    return inputs


def score_end_to_end(
    candidates: Sequence[Detection],
    masks: Sequence[tuple[np.ndarray, tuple[int, int, int]]],
    evaluation_set: EvaluationSet,
    truth_masks: Mapping[tuple[str, int], np.ndarray],
) -> dict[str, Any]:
    """Score one prompt arm's masks against the ground truth. Pure.

    Candidates inside an ``uncertain_ignore`` region are neutralized: excluded
    before association and never counted as false masks. An annotation is
    *found* when its best mask IoU over the live candidates is above zero, and
    *missed* when it is exactly zero (no candidate mask overlaps it); a missed
    annotation scores IoU 0 in the mean. The association radius still decides
    which candidates are scored against an annotation and which are false masks.

    Args:
        candidates: Candidates that were segmented.
        masks: ``segment_candidates`` output for the same candidates.
        evaluation_set: What to score against.
        truth_masks: ``load_truth_masks`` output.

    Returns:
        ``{"pooled", "per_sheet", "macro"}`` of ``E2EBlock`` s.

    Raises:
        ValueError: If masks and candidates disagree in length, a candidate lies
            outside the evaluation set, or a scorable annotation has no truth
            mask.
    """
    if len(masks) != len(candidates):
        raise ValueError(f"{len(masks)} masks for {len(candidates)} candidates")
    foreign = sorted({c.image for c in candidates} - set(evaluation_set.image_sheet))
    if foreign:
        raise ValueError(f"candidates on images outside the evaluation set: {foreign}")
    scorable = [a for a in evaluation_set.annotations if a.has_geometry]
    missing = sorted(
        annotation_key(a.image, a.annotation_id)
        for a in scorable
        if (a.image, a.annotation_id) not in truth_masks
    )
    if missing:
        raise ValueError(f"scorable annotations without a truth mask: {missing}")

    live, _ = neutralize(candidates, evaluation_set.ignore_regions)
    live_ids = {id(c) for c in live}

    best_iou: dict[str, float] = defaultdict(float)
    live_by_sheet: dict[str, int] = defaultdict(int)
    false_masks_by_sheet: dict[str, int] = defaultdict(int)
    for candidate, (predicted, (wx0, wy0, size)) in zip(candidates, masks, strict=True):
        if id(candidate) not in live_ids:
            continue
        sheet = evaluation_set.image_sheet[candidate.image]
        live_by_sheet[sheet] += 1
        nearby = [
            a
            for a in scorable
            if a.image == candidate.image
            and abs(a.x - candidate.x) <= ASSOCIATION_RADIUS
            and abs(a.y - candidate.y) <= ASSOCIATION_RADIUS
        ]
        if not nearby:
            false_masks_by_sheet[sheet] += 1
            continue
        for annotation in nearby:
            key = annotation_key(annotation.image, annotation.annotation_id)
            window_truth = truth_masks[(annotation.image, annotation.annotation_id)][
                wy0 : wy0 + size, wx0 : wx0 + size
            ]
            iou = mask_iou(predicted, window_truth)
            if iou > best_iou[key]:
                best_iou[key] = iou

    def block_for(sheets: Sequence[str]) -> dict[str, Any]:
        in_sheets = set(sheets)
        members = [a for a in evaluation_set.annotations if a.sheet in in_sheets]
        return _end_to_end_block(
            [a for a in members if a.has_geometry],
            sum(1 for a in members if not a.has_geometry),
            sum(live_by_sheet[s] for s in sheets),
            sum(false_masks_by_sheet[s] for s in sheets),
            best_iou,
        )

    per_sheet = {sheet: block_for([sheet]) for sheet in evaluation_set.sheets}
    inputs = [_macro_inputs(per_sheet[sheet]) for sheet in evaluation_set.sheets]
    return {
        "pooled": block_for(evaluation_set.sheets),
        "per_sheet": per_sheet,
        "macro": {name: macro_summary([row[name] for row in inputs]) for name in inputs[0]},
    }


def run_pair(
    raw: Sequence[Detection],
    detector_weights: Path,
    decoder_checkpoint: Path,
    sam_checkpoint: Path,
    evaluation_set: EvaluationSet,
    coco_json: Path,
    clip_paths: Mapping[str, Path],
    confidence: float,
    box_modes: Sequence[str],
    window_px: int,
    image_size: int,
    merge_radius: float,
    device: torch.device,
) -> dict[str, Any]:
    """Run one detector's candidates through one decoder and score every arm.

    The detector and the decoder are named explicitly. ``raw`` is the
    detector's inference over the evaluation set, so one inference serves the
    detection metrics and this stage.

    Args:
        raw: Raw detections from ``detector_weights`` over the evaluation set.
        detector_weights: Detector the raw detections came from (recorded).
        decoder_checkpoint: MapSAM decoder checkpoint.
        sam_checkpoint: Base SAM checkpoint.
        evaluation_set: What to score against.
        coco_json: CVAT COCO export, for ground-truth masks.
        clip_paths: Source clip path by file name.
        confidence: Detector confidence threshold.
        box_modes: Prompt arms to run.
        window_px: Window side length.
        image_size: Encoder input size.
        merge_radius: Cross-window deduplication radius.
        device: Target device.

    Returns:
        The pair record: ``config``, candidate counts and one block per arm.
    """
    candidates = resolve_candidates(raw, confidence, merge_radius)
    _, neutralized = neutralize(candidates, evaluation_set.ignore_regions)
    logger.info(
        "%s: %d candidates at conf %.2f (%d neutralized by ignore)",
        decoder_checkpoint,
        len(candidates),
        confidence,
        neutralized,
    )

    truth_masks = load_truth_masks(coco_json, evaluation_set)
    sam = load_mapsam(decoder_checkpoint, sam_checkpoint, device)

    arms: dict[str, Any] = {}
    for box_mode in box_modes:
        masks = segment_candidates(
            sam, candidates, clip_paths, box_mode, window_px, image_size, device
        )
        arms[box_mode] = score_end_to_end(candidates, masks, evaluation_set, truth_masks)
        pooled = arms[box_mode]["pooled"]
        logger.info(
            "%s: IoU>=0.5 %.3f, IoU>=0.75 %.3f, mean %.4f",
            box_mode,
            pooled["iou_0.5"]["rate"],
            pooled["iou_0.75"]["rate"],
            pooled["mean_iou"],
        )

    return {
        "config": {
            "detector_weights": repo_relative(detector_weights),
            "decoder_checkpoint": repo_relative(decoder_checkpoint),
            "sam_checkpoint": repo_relative(sam_checkpoint),
            "confidence": confidence,
            "box_modes": list(box_modes),
            "window_px": window_px,
            "image_size": image_size,
            "merge_radius": merge_radius,
            "sheets": list(evaluation_set.sheets),
            "device": str(device),
        },
        "candidates": len(candidates),
        "candidates_neutralized_by_ignore": neutralized,
        "arms": arms,
    }


def format_report(results: Mapping[str, dict[str, Any]]) -> str:
    """Render the end-to-end results as markdown.

    Args:
        results: Pair record by label, for example the fold.

    Returns:
        Markdown text.
    """
    lines = [
        "## End-to-end: detector candidates through the frozen 512 decoder",
        "",
        "Denominator is annotations with geometry, not segmentation samples and",
        "not all annotations. Candidates inside an ignore region are neutralized.",
        "",
        "| run | scope | arm | n | IoU>=0.5 | IoU>=0.75 | mean IoU | missed | false masks |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in results.items():
        for arm_name, arm in result["arms"].items():
            scopes = {"pooled": arm["pooled"], **arm["per_sheet"]}
            for scope, block in scopes.items():
                lines.append(
                    f"| {label} | {scope} | `{arm_name}` "
                    f"| {block['scorable_annotations']} | {block['iou_0.5']['rate']:.3f} "
                    f"| {block['iou_0.75']['rate']:.3f} | {block['mean_iou']:.4f} "
                    f"| {block['missed']['rate']:.3f} | {block['false_masks']} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments and run the end-to-end evaluation, one pair per fold."""
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
    parser.add_argument("--confidence", type=float, default=HEADLINE_CONFIDENCE)
    parser.add_argument("--box-modes", nargs="+", default=list(BOX_MODES), choices=BOX_MODES)
    parser.add_argument("--detector-imgsz", type=int, default=512)
    parser.add_argument("--window-px", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--merge-radius", type=float, default=10.0)
    parser.add_argument("--mapsam-prefix", default="v0_4r_loso512")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    device = resolve_device(args.device)
    folds = list(FOLDS) if args.fold == "all" else [args.fold]
    # Before any inference: a rerun into a published output directory stops here.
    new_directory(args.output_dir)

    results: dict[str, dict[str, Any]] = {}
    for fold in folds:
        evaluation_set = load_evaluation_set(
            args.dataset, args.dataset / f"{fold}_val.txt", [FOLDS[fold]]
        )
        weights = fold_weights_path(args.detector_run, fold)
        raw = predict_windows(weights, evaluation_set, args.detector_imgsz, MIN_CONFIDENCE, device)
        clip_paths = {
            name: find_image((args.images_root,), name) for name in evaluation_set.image_sheet
        }
        results[fold] = run_pair(
            raw=raw,
            detector_weights=weights,
            decoder_checkpoint=decoder_checkpoint_path(args.mapsam_root, args.mapsam_prefix, fold),
            sam_checkpoint=args.sam_checkpoint,
            evaluation_set=evaluation_set,
            coco_json=args.coco_json,
            clip_paths=clip_paths,
            confidence=args.confidence,
            box_modes=args.box_modes,
            window_px=args.window_px,
            image_size=args.image_size,
            merge_radius=args.merge_radius,
            device=device,
        )

    write_record(args.output_dir / "end_to_end.json", json.dumps(results, indent=2))
    report = format_report(results)
    write_record(args.output_dir / "end_to_end.md", report)
    print(report)


if __name__ == "__main__":
    main()
