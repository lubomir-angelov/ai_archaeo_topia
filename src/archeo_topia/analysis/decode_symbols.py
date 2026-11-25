#!/usr/bin/env python3
"""Decode symbol polygons for sweep candidates (MapSAM v0.6, GIS delivery).

``candidates.jsonl`` carries a point and a box per detection but no mask, so
the symbol layer of the GIS deliverable needs a decoder pass. This module runs
the v0.4r decoder over a sweep's candidates and writes the resulting outlines
in **sheet-pixel** coordinates, leaving the projection to
``archeo_topia.formats.export_gis``.

That separation is deliberate: ``AGENTS.md`` requires vector export logic to be
kept out of detection and segmentation logic, so this module never sees a CRS
and the exporter never runs a model.

Prompting uses a canonical median box on the candidate point rather than the
detector's own box. v0.5 measured the ``fixed`` arm beating the ``detector``
arm at IoU>=0.75 on three folds of three; it also says plainly that the
magnitudes are not strong evidence, so this is a cheap default and not a
settled result.

Usage:
    python -m archeo_topia.analysis.decode_symbols \\
        --candidates artifacts/detection/v0_6_sweep/foldC_last \\
        --clips-root "$LAKE/cleaned/map_clips/dataset_03" \\
        --output artifacts/detection/v0_6_symbols/foldC.json
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
from archeo_topia.formats.georeference import SheetReference

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

#: Smallest ring, in positions, worth keeping.
MIN_POLYGON_POINTS = 3


def ring_from_mask(mask: np.ndarray, offset_xy: tuple[int, int]) -> list[float] | None:
    """Trace a binary mask's largest outline as a flat ring.

    Args:
        mask: Binary mask at window resolution.
        offset_xy: The window's origin, in the coordinate space to return.

    Returns:
        Flat ``[x0, y0, x1, y1, ...]``, or None when there is nothing to trace.
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
    return [float(v) for p in largest for v in (p[0] + x0, p[1] + y0)]


def decode_sheet(
    reference: SheetReference,
    clips_dir: Path,
    candidates: list[dict[str, Any]],
    sam: Any,
    device: torch.device,
    window_px: int = 512,
    image_size: int = 1024,
) -> dict[int, list[float]]:
    """Decode every candidate on one sheet into a sheet-pixel ring.

    Clips are segmented one at a time rather than the whole sheet: a 23 Mpx
    sheet as a float tensor is about 280 MB, where a clip is a quarter of that,
    and the clips have to exist anyway for CVAT.

    Args:
        reference: The sheet's clip geometry.
        clips_dir: Directory holding the clip PNGs.
        candidates: Sweep candidates in sheet-pixel coordinates.
        sam: A loaded MapSAM decoder.
        device: Torch device.
        window_px: Decoder window side length.
        image_size: Decoder input size.

    Returns:
        Candidate index (1-based, matching the exporter's ordering) to ring.
    """
    from archeo_topia.analysis.mapsam_end_to_end import segment

    by_clip: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, candidate in enumerate(candidates, start=1):
        placed = reference.sheet_to_clip(candidate["x"], candidate["y"])
        if placed is None:
            continue
        by_clip.setdefault(placed[0], []).append((index, candidate))

    rings: dict[int, list[float]] = {}
    for clip_name, entries in by_clip.items():
        array = np.array(Image.open(clips_dir / clip_name).convert("RGB"))
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        offset = reference.clip(clip_name).offset_xy

        for index, candidate in entries:
            local_x = candidate["x"] - offset[0]
            local_y = candidate["y"] - offset[1]
            detection = Detection(
                image=clip_name, x=local_x, y=local_y, score=candidate["score"], box=None
            )
            mask, (wx0, wy0, _) = segment(
                sam, tensor, detection, "fixed", window_px, image_size, device
            )
            # Trace in clip space, then shift to sheet space in one step.
            ring = ring_from_mask(mask, (wx0 + offset[0], wy0 + offset[1]))
            if ring:
                rings[index] = [round(v, 2) for v in ring]

    return rings


def main() -> None:
    """Decode symbol rings for every sheet in a sweep run."""
    parser = argparse.ArgumentParser(description="Decode symbol polygons for sweep candidates")
    parser.add_argument("--candidates", required=True, help="A sweep run directory")
    parser.add_argument("--clips-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", default="foldC", help="Decoder fold checkpoint")
    parser.add_argument("--mapsam-root", default="artifacts/models/mapsam")
    parser.add_argument("--mapsam-prefix", default="v0_4r_loso512")
    parser.add_argument("--sam-checkpoint", default="models/checkpoints/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--window-px", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    from archeo_topia.analysis.mapsam_end_to_end import load_mapsam

    device = torch.device(args.device)
    sam = load_mapsam(
        args.fold, Path(args.mapsam_root), Path(args.sam_checkpoint), device, args.mapsam_prefix
    )

    sweep = Path(args.candidates)
    clips_root = Path(args.clips_root)
    out: dict[str, dict[str, list[float]]] = {}

    sheet_dirs = sorted(d for d in sweep.iterdir() if d.is_dir())
    for position, sheet_dir in enumerate(sheet_dirs, start=1):
        sheet_id = sheet_dir.name
        clips_dir = clips_root / sheet_id
        if not (clips_dir / "clips.json").exists():
            logger.warning("%s: no clips.json; skipped", sheet_id)
            continue
        candidates = [
            json.loads(line)
            for line in (sheet_dir / "candidates.jsonl").read_text().splitlines()
            if line
        ]
        candidates = [c for c in candidates if c["score"] >= args.confidence]
        reference = SheetReference.from_clips_json(clips_dir)
        rings = decode_sheet(
            reference,
            clips_dir,
            candidates,
            sam,
            device,
            window_px=args.window_px,
            image_size=args.image_size,
        )
        out[sheet_id] = {str(k): v for k, v in rings.items()}
        logger.info(
            "[%d/%d] %s: %d of %d candidates decoded",
            position,
            len(sheet_dirs),
            sheet_id,
            len(rings),
            len(candidates),
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out), encoding="utf-8")
    logger.info(
        "wrote %s: %d sheets, %d rings",
        output,
        len(out),
        sum(len(v) for v in out.values()),
    )


if __name__ == "__main__":
    main()
