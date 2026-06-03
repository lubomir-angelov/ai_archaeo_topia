"""Proposal generation module.

Phase 1: deterministic grid-based proposal generation.
Phase 2: basic image heuristic proposals using Pillow + numpy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .errors import ValidationError
from .masks import (
    create_bbox_mask,
    mask_array_to_polygon,
    save_mask,
    validate_image_path,
)
from .settings import get_settings

logger = logging.getLogger(__name__)


def generate_proposals(
    image_path: str,
    output_dir: Path,
    label_hint: str = "",
    max_proposals: int = 50,
    mode: str = "mock",
) -> list[dict]:
    """Generate candidate segmentation proposals for an image.

    Uses a deterministic grid-based approach to propose bounding boxes
    across the image, then creates masks for each proposal.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save mask files.
        label_hint: Optional label hint.
        max_proposals: Maximum number of proposals.
        mode: Backend mode (mock or sam2).

    Returns:
        List of proposal item dicts.

    Raises:
        ValidationError: If parameters are invalid.
    """
    settings = get_settings()
    hard_cap = settings.max_proposals_hard_cap
    capped = min(max_proposals, hard_cap)

    if capped <= 0:
        raise ValidationError("max_proposals must be positive")

    w, h = validate_image_path(image_path, settings.max_image_pixels)

    if mode == "mock":
        return _mock_proposals(image_path, w, h, output_dir, label_hint, capped)

    return _heuristic_proposals(image_path, w, h, output_dir, label_hint, capped)


def _mock_proposals(
    image_path: str,
    width: int,
    height: int,
    output_dir: Path,
    label_hint: str,
    max_count: int,
) -> list[dict]:
    """Generate deterministic grid-based mock proposals.

    Divides the image into a grid and creates proposals for each cell.

    Args:
        image_path: Source image path.
        width: Image width.
        height: Image height.
        output_dir: Output directory.
        label_hint: Label hint.
        max_count: Maximum proposals.

    Returns:
        List of proposal item dicts.
    """
    n_rows = int(np.ceil(np.sqrt(max_count * height / max(width, 1))))
    n_cols = int(np.ceil(max_count / max(n_rows, 1)))
    n_rows = max(1, min(n_rows, 10))
    n_cols = max(1, min(n_cols, 10))

    cell_w = width / n_cols
    cell_h = height / n_rows

    proposals: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()

    for row in range(n_rows):
        for col in range(n_cols):
            if len(proposals) >= max_count:
                break

            xmin = int(col * cell_w)
            ymin = int(row * cell_h)
            xmax = int((col + 1) * cell_w)
            ymax = int((row + 1) * cell_h)

            key = (xmin, ymin, xmax, ymax)
            if key in seen:
                continue
            seen.add(key)

            margin_x = max(1, int((xmax - xmin) * 0.1))
            margin_y = max(1, int((ymax - ymin) * 0.1))
            bx = xmin + margin_x
            by = ymin + margin_y
            ba_x = xmax - margin_x
            ba_y = ymax - margin_y

            if ba_x <= bx + 2 or ba_y <= by + 2:
                continue

            bbox = [float(bx), float(by), float(ba_x), float(ba_y)]
            mask = create_bbox_mask(width, height, bbox)
            conf = 0.7 - (len(proposals) * 0.02)
            conf = round(max(conf, 0.1), 4)

            from .masks import make_mask_filename, prompt_hash

            ph = prompt_hash(f"proposal_{row}_{col}")
            fname = make_mask_filename(image_path, ph)
            mask_path = save_mask(mask, fname, output_dir)

            polygon = mask_array_to_polygon(mask)

            proposals.append(
                {
                    "image_path": image_path,
                    "label": label_hint or f"proposal_{len(proposals)}",
                    "bbox": [round(v, 4) for v in bbox],
                    "polygon": [[round(v, 4) for v in pt] for pt in polygon],
                    "mask_path": mask_path,
                    "confidence": conf,
                    "metadata": {
                        "grid_row": row,
                        "grid_col": col,
                        "polygon_quality": "bbox_fallback",
                    },
                }
            )

        if len(proposals) >= max_count:
            break

    logger.info("Generated %d mock proposals for %s", len(proposals), image_path)
    return proposals


def _heuristic_proposals(
    image_path: str,
    width: int,
    height: int,
    output_dir: Path,
    label_hint: str,
    max_count: int,
) -> list[dict]:
    """Generate proposals using basic image heuristics.

    Uses simple edge detection via Pillow/numpy to find regions of
    interest, then creates proposals around them.

    Args:
        image_path: Source image path.
        width: Image width.
        height: Image height.
        output_dir: Output directory.
        label_hint: Label hint.
        max_count: Maximum proposals.

    Returns:
        List of proposal item dicts.
    """
    from PIL import Image

    try:
        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float64)
    except Exception as exc:
        logger.warning("Cannot load image for heuristics, falling back to mock: %s", exc)
        return _mock_proposals(image_path, width, height, output_dir, label_hint, max_count)

    # Simple gradient-based edge detection
    gy, gx = np.gradient(arr)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Threshold to find edge regions
    threshold = np.percentile(magnitude, 85)
    edge_map = (magnitude > threshold).astype(np.uint8)

    # Find connected components via simple flood fill
    regions = _find_regions(edge_map, min_size=50)

    proposals: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()

    for region_bbox in regions[:max_count]:
        xmin, ymin, xmax, ymax = region_bbox
        key = (xmin, ymin, xmax, ymax)
        if key in seen:
            continue
        seen.add(key)

        margin_x = max(1, int((xmax - xmin) * 0.05))
        margin_y = max(1, int((ymax - ymin) * 0.05))
        bx = max(0, xmin - margin_x)
        by = max(0, ymin - margin_y)
        ba_x = min(width, xmax + margin_x)
        ba_y = min(height, ymax + margin_y)

        bbox = [float(bx), float(by), float(ba_x), float(ba_y)]
        mask = create_bbox_mask(width, height, bbox)
        conf = round(min(0.9, 0.5 + (xmax - xmin) * (ymax - ymin) / (width * height)), 4)

        from .masks import make_mask_filename, prompt_hash

        ph = prompt_hash(f"heuristic_{xmin}_{ymin}")
        fname = make_mask_filename(image_path, ph)
        mask_path = save_mask(mask, fname, output_dir)

        polygon = mask_array_to_polygon(mask)

        proposals.append(
            {
                "image_path": image_path,
                "label": label_hint or f"heuristic_{len(proposals)}",
                "bbox": [round(v, 4) for v in bbox],
                "polygon": [[round(v, 4) for v in pt] for pt in polygon],
                "mask_path": mask_path,
                "confidence": conf,
                "metadata": {
                    "method": "heuristic_edge",
                    "polygon_quality": "bbox_fallback",
                },
            }
        )

        if len(proposals) >= max_count:
            break

    # Fill remaining with mock proposals if needed
    if len(proposals) < max_count:
        remaining = max_count - len(proposals)
        mock = _mock_proposals(image_path, width, height, output_dir, label_hint, remaining)
        existing_keys = {(tuple(int(v) for v in p["bbox"])) for p in proposals}
        for m in mock:
            if len(proposals) >= max_count:
                break
            key = tuple(int(v) for v in m["bbox"])
            if key not in existing_keys:
                proposals.append(m)
                existing_keys.add(key)

    logger.info("Generated %d heuristic proposals for %s", len(proposals), image_path)
    return proposals


def _find_regions(
    binary_map: np.ndarray,
    min_size: int = 50,
) -> list[tuple[int, int, int, int]]:
    """Find connected regions in a binary map.

    Uses simple flood-fill to find connected components and returns
    their bounding boxes sorted by area (largest first).

    Args:
        binary_map: Binary array (0 and 1).
        min_size: Minimum pixel count for a valid region.

    Returns:
        List of (xmin, ymin, xmax, ymax) tuples.
    """
    h, w = binary_map.shape
    visited = np.zeros_like(binary_map, dtype=bool)
    regions: list[tuple[int, int, int, int, int]] = []

    for row in range(h):
        for col in range(w):
            if binary_map[row, col] == 0 or visited[row, col]:
                continue

            # Flood fill
            stack = [(row, col)]
            visited[row, col] = True
            pixels: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                pixels.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and binary_map[nr, nc] == 1
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if len(pixels) < min_size:
                continue

            rs = [p[0] for p in pixels]
            cs = [p[1] for p in pixels]
            regions.append((min(cs), min(rs), max(cs) + 1, max(rs) + 1, len(pixels)))

    regions.sort(key=lambda r: r[4], reverse=True)
    return [(r[0], r[1], r[2], r[3]) for r in regions]
