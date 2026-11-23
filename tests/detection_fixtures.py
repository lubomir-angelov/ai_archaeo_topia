"""Shared synthetic detection-dataset fixtures for the scoring tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def window(window_id: str, sheet: str, image: str, x0: int, y0: int) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "sheet_id": sheet,
        "source_image": image,
        "window_xyxy": [x0, y0, x0 + 512, y0 + 512],
        "targets": [],
    }


def annotation_record(annotation_id: int, sheet: str, image: str, x: float, y: float) -> dict:
    return {
        "annotation_id": annotation_id,
        "sheet_id": sheet,
        "source_image": image,
        "center_source_xy": [x, y],
        "component_annotation_count": 1,
        "has_geometry": True,
        "attributes": {},
    }


def region_record(
    kind: str,
    annotation_id: int,
    sheet: str,
    image: str,
    box: list[float],
    negative_type: str | None,
) -> dict:
    return {
        "kind": kind,
        "annotation_id": annotation_id,
        "sheet_id": sheet,
        "source_image": image,
        "bbox_source_xyxy": box,
        "negative_type": negative_type,
    }


def write_dataset(
    root: Path,
    *,
    regions: list[dict] | None = None,
    listing: list[str] | None = None,
) -> Path:
    """Write a two-sheet dataset: S-1 has two windows on one clip, S-2 has one."""
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    windows = [
        window("S-1_1_x00000_y00000", "S-1", "S-1_1.png", 0, 0),
        window("S-1_1_x00384_y00000", "S-1", "S-1_1.png", 384, 0),
        window("S-2_1_x00000_y00000", "S-2", "S-2_1.png", 0, 0),
    ]
    annotations = [
        annotation_record(1, "S-1", "S-1_1.png", 100.0, 100.0),
        annotation_record(2, "S-2", "S-2_1.png", 200.0, 200.0),
    ]
    if regions is None:
        regions = [
            region_record(
                "hard_negative_symbol", 11, "S-1", "S-1_1.png", [290, 290, 310, 310], "trig_point"
            ),
            region_record("uncertain_ignore", 12, "S-2", "S-2_1.png", [480, 480, 520, 520], None),
        ]
    for name, records in (
        ("windows", windows),
        ("annotations", annotations),
        ("regions", regions),
    ):
        (metadata / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
    lines = listing if listing is not None else [w["window_id"] for w in windows]
    (root / "all.txt").write_text(
        "".join(f"./images/{window_id}.png\n" for window_id in lines), encoding="utf-8"
    )
    return root
