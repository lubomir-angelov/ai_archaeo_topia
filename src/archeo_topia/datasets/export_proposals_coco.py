#!/usr/bin/env python3
"""Turn detector candidates into a CVAT-importable COCO file (MapSAM v0.6 step 3).

This is the stage where the pipeline stops being a thing that is measured and
starts being a thing that saves someone time. The reviewer adjusts an outline
instead of drawing one, and deletes a wrong proposal instead of hunting for a
missed symbol.

Three choices make it pay, all carried from ``docs/mapsam/v006/PLAN.md``:

**A low confidence threshold.** Rejecting a bad proposal takes seconds; finding
a missed mound takes minutes. The asymmetry favours recall heavily, so
precision is the reviewer's job here and not the model's.

**Masks, not boxes.** Accepted candidates go through the v0.4r decoder, which
clears mask IoU 0.5 on 0.958 of instances end to end. That rate is the only
number in the project that converts directly into saved human time.

**Low-confidence candidates arrive pre-labelled ``hard_negative_symbol``.** They
are confusable symbols by construction. The class holds 542 hand-picked
confusables today; the detector supplies them in bulk at no annotation cost, and
the reviewer only has to delete the ones that are actually mounds.

**Every annotation carries its provenance.** ``model_proposal_accepted``,
``model_proposal_corrected`` and ``human_added``. Everything written here starts
as ``model_proposal_accepted`` and the reviewer moves it. The ``human_added``
rate on assisted sheets, compared against the blind test set, *is* the estimate
of the bias this stage introduces; without the field the question is
unanswerable afterwards.

Geometry is written as polygons rather than RLE. The existing annotations are
polygons, CVAT round-trips them natively, and ``src/export_sam2_to_coco.py``
already records that COCO imports into CVAT more reliably than CVAT XML.

Usage:
    python -m archeo_topia.datasets.export_proposals_coco \\
        --candidates artifacts/detection/v0_6_sweep/foldC_last/K-35-21-G-a/candidates.jsonl \\
        --clips-dir data_lake/cleaned/map_clips/dataset_03/K-35-21-G-a \\
        --output annotation/proposals/K-35-21-G-a
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from archeo_topia.analysis.detection_metrics import Detection
from archeo_topia.formats.coco import CATEGORIES as SCHEMA_CATEGORIES
from archeo_topia.formats.coco import HARD_NEGATIVE, MOUND
from archeo_topia.formats.labels import LabelSchema

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

#: COCO category ids, from the one place they are declared.
CATEGORIES = [dict(category) for category in SCHEMA_CATEGORIES]

#: Smallest polygon, in points, CVAT will accept as a shape.
MIN_POLYGON_POINTS = 3

#: The schema of record. Attribute defaults come from this file rather than
#: from a literal list here: the two used to be maintained in parallel, and a
#: test existed for no purpose other than noticing when they drifted apart.
DEFAULT_SCHEMA = Path("annotation/cvat/labels.json")


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Read a sweep's ``candidates.jsonl``.

    Args:
        path: The file.

    Returns:
        Candidate records in sheet coordinates.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assign_to_clips(
    candidates: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Map sheet-coordinate candidates into the clip that contains each one.

    The sweep runs over the parent sheet once; CVAT annotates clips. The offsets
    in ``clips.json`` are exact integer translations of parent pixels, so this is
    a subtraction and not an approximation.

    Args:
        candidates: Records carrying sheet-space ``x`` and ``y``.
        manifest: A parsed ``clips.json``.

    Returns:
        Mapping of clip file name to its candidates, in clip coordinates.
    """
    out: dict[str, list[dict[str, Any]]] = {clip["file"]: [] for clip in manifest["clips"]}
    for candidate in candidates:
        for clip in manifest["clips"]:
            x0, y0 = clip["offset_xy"]
            width, height = clip["size"]
            x, y = candidate["x"] - x0, candidate["y"] - y0
            if 0 <= x < width and 0 <= y < height:
                local = dict(candidate)
                local["x"], local["y"] = x, y
                if candidate.get("box"):
                    bx0, by0, bx1, by1 = candidate["box"]
                    local["box"] = [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0]
                out[clip["file"]].append(local)
                break
    return out


def mask_to_polygon(mask: np.ndarray, offset_xy: tuple[int, int]) -> list[float] | None:
    """Convert a binary mask to a flat COCO polygon in clip coordinates.

    Args:
        mask: Binary mask at window resolution.
        offset_xy: The window's origin in clip coordinates.

    Returns:
        Flat ``[x0, y0, x1, y1, ...]``, or None when the mask is empty or the
        largest contour is too small to be a shape.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(largest) < MIN_POLYGON_POINTS:
        return None
    x0, y0 = offset_xy
    return [float(v) for point in largest for v in (point[0] + x0, point[1] + y0)]


def annotation_record(
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox: list[float],
    polygon: list[float] | None,
    score: float,
    schema: LabelSchema | None = None,
) -> dict[str, Any]:
    """Build one COCO annotation with the project's attribute defaults.

    Defaults come from the schema of record, so a new attribute added there
    appears here without an edit, and cannot be forgotten.

    Args:
        annotation_id: Unique id.
        image_id: Owning image id.
        category_id: COCO category.
        bbox: ``[x, y, w, h]`` in clip coordinates.
        polygon: Flat polygon, or None for a box-only shape.
        score: Detector confidence, carried so a reviewer can sort by it.
        schema: Label schema; loaded from ``DEFAULT_SCHEMA`` when omitted.

    Returns:
        The annotation.
    """
    schema = schema or LabelSchema.load(DEFAULT_SCHEMA)
    label = MOUND if category_id == 1 else HARD_NEGATIVE
    attributes = schema.defaults(label)
    attributes["water_line_crossing"] = "unreviewed"
    attributes["annotation_provenance"] = "model_proposal_accepted"
    attributes["detector_confidence"] = round(score, 4)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [polygon] if polygon else [],
        "area": float(bbox[2] * bbox[3]),
        "bbox": [float(v) for v in bbox],
        "iscrowd": 0,
        "attributes": attributes,
    }


def export(
    clips_dir: Path,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    sam: Any,
    device: torch.device,
    mound_confidence: float,
    negative_confidence: float,
    window_px: int,
    image_size: int,
) -> dict[str, Any]:
    """Segment candidates and write a CVAT-importable COCO document.

    Args:
        clips_dir: Directory of clip PNGs carrying a ``clips.json``.
        candidates: Sheet-coordinate candidates from the sweep.
        output_dir: Destination.
        sam: Loaded MapSAM decoder, or None to write boxes only.
        device: Torch device.
        mound_confidence: At or above this, a candidate becomes a ``mound``
            proposal and is segmented.
        negative_confidence: Between this and ``mound_confidence``, a candidate
            becomes a pre-labelled ``hard_negative_symbol``. Below it, dropped.
        window_px: Decoder window side length in source pixels.
        image_size: Decoder input size.

    Returns:
        A summary of what was written.
    """
    from archeo_topia.analysis.mapsam_end_to_end import segment

    manifest = json.loads((clips_dir / "clips.json").read_text(encoding="utf-8"))
    by_clip = assign_to_clips(candidates, manifest)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    counts = {"mound": 0, "hard_negative_symbol": 0, "unsegmented": 0}

    for image_id, clip in enumerate(manifest["clips"], start=1):
        name = clip["file"]
        width, height = clip["size"]
        # The optional COCO image fields are carried because CVAT's own exports
        # carry them, and matching the shape it emits is the cheapest way to
        # stay importable by the tool that will read this back.
        images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        tensor = None
        for candidate in sorted(by_clip[name], key=lambda c: -c["score"]):
            score = candidate["score"]
            if score < negative_confidence:
                continue
            is_mound = score >= mound_confidence
            detection = Detection(
                image=name,
                x=candidate["x"],
                y=candidate["y"],
                score=score,
                box=tuple(candidate["box"]) if candidate.get("box") else None,
            )

            polygon = None
            if is_mound and sam is not None:
                if tensor is None:
                    array = np.array(Image.open(clips_dir / name).convert("RGB"))
                    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
                mask, (wx0, wy0, _) = segment(
                    sam, tensor, detection, "fixed", window_px, image_size, device
                )
                polygon = mask_to_polygon(mask, (wx0, wy0))
                if polygon is None:
                    counts["unsegmented"] += 1

            if polygon:
                xs, ys = polygon[0::2], polygon[1::2]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            elif detection.box:
                bx0, by0, bx1, by1 = detection.box
                bbox = [bx0, by0, bx1 - bx0, by1 - by0]
            else:
                bbox = [detection.x - 12.5, detection.y - 11.5, 25.0, 23.0]

            annotations.append(
                annotation_record(
                    len(annotations) + 1,
                    image_id,
                    1 if is_mound else 2,
                    bbox,
                    polygon,
                    score,
                )
            )
            counts["mound" if is_mound else "hard_negative_symbol"] += 1

    document = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": (
                f"Model proposals for {manifest['sheet_id']}. Every annotation is "
                "annotation_provenance=model_proposal_accepted until a reviewer says "
                "otherwise. Nothing here is ground truth."
            ),
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "instances_default.json").write_text(json.dumps(document), encoding="utf-8")
    summary = {
        "sheet_id": manifest["sheet_id"],
        "clips": len(images),
        "candidates_in": len(candidates),
        **counts,
        "mound_confidence": mound_confidence,
        "negative_confidence": negative_confidence,
    }
    (output_dir / "export_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Export one sheet's candidates as a CVAT-importable COCO file."""
    parser = argparse.ArgumentParser(description="Detector proposals to CVAT COCO")
    parser.add_argument("--candidates", required=True, help="A sweep's candidates.jsonl")
    parser.add_argument("--clips-dir", required=True, help="Clip directory carrying clips.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", default="foldC", help="Decoder fold checkpoint to prompt with")
    parser.add_argument("--mapsam-root", default="artifacts/models/mapsam")
    parser.add_argument("--mapsam-prefix", default="v0_4r_loso512")
    parser.add_argument(
        "--sam-checkpoint", default="models/checkpoints/sam/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("--mound-confidence", type=float, default=0.25)
    parser.add_argument("--negative-confidence", type=float, default=0.05)
    parser.add_argument("--window-px", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--boxes-only", action="store_true", help="Skip the decoder")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device)
    sam = None
    if not args.boxes_only:
        from archeo_topia.analysis.mapsam_end_to_end import load_mapsam

        sam = load_mapsam(
            args.fold, Path(args.mapsam_root), Path(args.sam_checkpoint), device, args.mapsam_prefix
        )

    summary = export(
        clips_dir=Path(args.clips_dir),
        candidates=load_candidates(Path(args.candidates)),
        output_dir=Path(args.output),
        sam=sam,
        device=device,
        mound_confidence=args.mound_confidence,
        negative_confidence=args.negative_confidence,
        window_px=args.window_px,
        image_size=args.image_size,
    )
    logger.info("%s", json.dumps(summary))


if __name__ == "__main__":
    main()
