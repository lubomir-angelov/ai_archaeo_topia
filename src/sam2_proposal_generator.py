#!/usr/bin/env python3
"""Generate SAM2 bounding-box proposals for extracted clips.

Calls the Nuclio SAM2 function to generate candidate masks for each
extracted clip image, then converts masks to bounding boxes and writes
them alongside the review images and CSV.

Requires: requests, Pillow (both already installed).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

NUCLIO_URL = "http://127.0.0.1:8070"
SAM2_FUNCTION = "pth-facebookresearch-sam-vit-h"
IMAGE_DIR = Path(
    "/home/naim/repos/ai_archaeo_topia/artifacts/annotation_runs/cvat_flat"
)
OUTPUT_DIR = Path(
    "/home/naim/repos/ai_archaeo_topia/artifacts/annotation_runs/sam2_proposals"
)


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str
    confidence: float
    source: str


def mask_to_bbox(mask_data: bytes, w: int, h: int) -> BBox | None:
    """Convert a SAM2 binary mask to a bounding box.

    mask_data: raw bytes of a binary mask (one byte per pixel, 0 or 255)
    """
    try:
        mask_img = Image.frombytes("L", (w, h), mask_data)
    except Exception:
        return None

    # Find non-zero pixels
    pixels = list(mask_img.getdata())
    if not pixels or all(p == 0 for p in pixels):
        return None

    # Get bounding box of non-zero pixels
    coords = [i for i, p in enumerate(pixels) if p > 0]
    if not coords:
        return None

    xs = [i % w for i in coords]
    ys = [i // w for i in coords]

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    # Convert to normalized coordinates
    return BBox(
        xmin=xmin / w,
        ymin=ymin / h,
        xmax=(xmax + 1) / w,
        ymax=(ymax + 1) / h,
        label="mound",  # Default; refined by metadata later
        confidence=0.7,
        source="sam2",
    )


def generate_sam2_proposals(image_path: Path, w: int, h: int) -> list[BBox]:
    """Call the Nuclio SAM2 function to generate candidates for one image.

    SAM2 expects a POST with the image as multipart form data.
    Returns a list of BBox objects.
    """
    boxes: list[BBox] = []

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # SAM2 Nuclio function endpoint
        url = f"{NUCLIO_URL}/api/v1/functions/{SAM2_FUNCTION}/invocations"

        # SAM2 expects the image as a form field
        resp = requests.post(
            url,
            files={"image": (image_path.name, img_bytes, "image/png")},
            timeout=60,
        )

        if resp.status_code != 200:
            logger.warning(
                "SAM2 failed for %s: %d %s",
                image_path.name,
                resp.status_code,
                resp.text[:200],
            )
            return boxes

        result = resp.json()

        # SAM2 returns a list of masks with scores
        masks = result.get("masks", result.get("candidates", []))
        if isinstance(masks, list):
            for i, mask_entry in enumerate(masks[:10]):  # Top 10 masks
                mask_data = mask_entry.get("mask", mask_entry.get("binary_mask", b""))
                if isinstance(mask_data, str):
                    # Base64 encoded
                    import base64

                    try:
                        mask_data = base64.b64decode(mask_data)
                    except Exception:
                        continue

                if isinstance(mask_data, (bytes, bytearray)) and len(mask_data) > 0:
                    bbox = mask_to_bbox(mask_data, w, h)
                    if bbox:
                        score = mask_entry.get("score", mask_entry.get("confidence", 0.5))
                        bbox.confidence = float(score)
                        boxes.append(bbox)

    except requests.ConnectionError:
        logger.warning("SAM2/Nuclio unreachable at %s", url)
    except Exception as exc:
        logger.warning("SAM2 error for %s: %s", image_path.name, exc)

    return boxes


def run_sam2_proposals() -> dict:
    """Run SAM2 on all extracted clip images.

    Returns a summary dict.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.png")) + sorted(IMAGE_DIR.glob("*.jpg"))
    logger.info("Found %d images for SAM2 processing", len(image_files))

    all_proposals: dict[str, list[BBox]] = {}
    success_count = 0
    fail_count = 0
    total_boxes = 0

    for img_path in image_files:
        try:
            pil = Image.open(img_path)
            w, h = pil.size
        except Exception as e:
            logger.warning("Cannot open %s: %s", img_path.name, e)
            fail_count += 1
            continue

        boxes = generate_sam2_proposals(img_path, w, h)

        if boxes:
            all_proposals[img_path.name] = boxes
            success_count += 1
            total_boxes += len(boxes)
            logger.info(
                "%s: %d boxes (conf: %.2f-%.2f)",
                img_path.name,
                len(boxes),
                min(b.confidence for b in boxes),
                max(b.confidence for b in boxes),
            )
        else:
            fail_count += 1

    # Save proposals JSON
    proposals_path = OUTPUT_DIR / "sam2_proposals.json"
    serializable = {}
    for fname, boxes in all_proposals.items():
        serializable[fname] = [
            {
                "xmin": round(b.xmin, 4),
                "ymin": round(b.ymin, 4),
                "xmax": round(b.xmax, 4),
                "ymax": round(b.ymax, 4),
                "label": b.label,
                "confidence": round(b.confidence, 4),
                "source": b.source,
            }
            for b in boxes
        ]
    proposals_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Proposals saved to %s", proposals_path)

    summary = {
        "total_images": len(image_files),
        "successful": success_count,
        "failed": fail_count,
        "total_boxes": total_boxes,
        "images_with_boxes": len(all_proposals),
        "proposals_path": str(proposals_path),
    }
    return summary


if __name__ == "__main__":
    summary = run_sam2_proposals()
    print("\n=== SAM2 Proposal Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
