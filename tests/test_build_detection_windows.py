"""Tests for the MapSAM v0.5 detection-window builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from archeo_topia.datasets.build_detection_windows import (
    Window,
    build,
    clip_box,
    plan_windows,
    sheet_of,
    to_yolo,
)


def test_sheet_of_keeps_hyphenated_sheet_id() -> None:
    """Only the trailing clip index is stripped."""
    assert sheet_of("K-35-51-B-a_3.png") == "K-35-51-B-a"
    assert sheet_of("K-34-35-B-g_12.png") == "K-34-35-B-g"


def test_plan_windows_clamps_final_row_and_column() -> None:
    """The last window sits flush with the edge rather than being padded."""
    windows = plan_windows(1000, 512, size=512, stride=384)
    assert windows[0] == Window(0, 0, 512)
    assert max(w.x1 for w in windows) == 1000
    assert all(w.x1 <= 1000 and w.y1 <= 512 for w in windows)


def test_plan_windows_handles_image_smaller_than_window() -> None:
    """A single window covers an image narrower than the window size."""
    assert plan_windows(300, 300, size=512, stride=384) == [Window(0, 0, 512)]


def test_clip_box_reports_visible_fraction() -> None:
    """A box straddling the left edge keeps half its area."""
    window = Window(100, 100, 512)
    box, visible = clip_box([90, 200, 20, 10], window)
    assert box == [0, 100, 10, 110]
    assert visible == pytest.approx(0.5)


def test_to_yolo_normalizes_to_window() -> None:
    """Window-pixel xyxy becomes normalized cxcywh."""
    assert to_yolo([0, 0, 512, 512], 512) == (0.5, 0.5, 1.0, 1.0)
    assert to_yolo([246, 246, 266, 266], 512) == pytest.approx((0.5, 0.5, 20 / 512, 20 / 512))


def _coco(tmp_path: Path) -> Path:
    """Write a two-clip COCO export with one mound, one negative, one ignore."""
    images = [
        {"id": 1, "file_name": "S-1_1.png", "width": 900, "height": 700},
        {"id": 2, "file_name": "S-2_1.png", "width": 900, "height": 700},
    ]
    annotations = [
        # Comfortably inside the first window of clip 1.
        {"id": 10, "image_id": 1, "category_id": 1, "bbox": [200, 200, 24, 22], "segmentation": []},
        {"id": 11, "image_id": 1, "category_id": 2, "bbox": [600, 600, 18, 18], "segmentation": []},
        {"id": 12, "image_id": 1, "category_id": 3, "bbox": [300, 300, 20, 20], "segmentation": []},
        {"id": 20, "image_id": 2, "category_id": 1, "bbox": [400, 400, 24, 22], "segmentation": []},
    ]
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "mound"},
            {"id": 2, "name": "hard_negative_symbol"},
            {"id": 3, "name": "uncertain_ignore"},
        ],
    }
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    image_root = tmp_path / "images" / "train"
    image_root.mkdir(parents=True)
    for image in images:
        Image.new("RGBA", (image["width"], image["height"]), (255, 255, 255, 255)).save(
            image_root / image["file_name"]
        )
    return path


def test_build_writes_one_class_labels_and_shared_windows(tmp_path: Path) -> None:
    """Mounds become class-0 boxes; negatives and ignores stay in metadata."""
    coco = _coco(tmp_path)
    out = tmp_path / "det"
    report = build(coco, tmp_path / "images", out, window_size=512, stride=384)

    assert report["counts"]["mound_annotations"] == 2
    assert report["counts"]["windows_total"] == len(list((out / "images").glob("*.png")))
    assert len(list((out / "labels").glob("*.txt"))) == report["counts"]["windows_total"]

    windows = [json.loads(line) for line in (out / "metadata" / "windows.jsonl").read_text().splitlines()]
    targets = [t for w in windows for t in w["targets"]]
    assert {t["annotation_id"] for t in targets} == {10, 20}

    negatives = [n for w in windows for n in w["hard_negatives"]]
    ignores = [g for w in windows for g in w["uncertain_ignore"]]
    assert negatives and all(n["annotation_id"] == 11 for n in negatives)
    assert ignores and all(g["annotation_id"] == 12 for g in ignores)

    # Every written label line is class 0.
    for label in (out / "labels").glob("*.txt"):
        for line in label.read_text().splitlines():
            assert line.startswith("0 ")


def test_build_targets_round_trip_to_source_coordinates(tmp_path: Path) -> None:
    """window_xyxy is a usable inverse transform for untruncated targets."""
    coco = _coco(tmp_path)
    out = tmp_path / "det"
    build(coco, tmp_path / "images", out, window_size=512, stride=384)

    centres = {10: (212.0, 211.0), 20: (412.0, 411.0)}
    windows = [json.loads(line) for line in (out / "metadata" / "windows.jsonl").read_text().splitlines()]
    checked = 0
    for window in windows:
        x0, y0, _, _ = window["window_xyxy"]
        for target in window["targets"]:
            if target["truncated"]:
                continue
            bx0, by0, bx1, by1 = target["box_window_xyxy"]
            source = ((bx0 + bx1) / 2 + x0, (by0 + by1) / 2 + y0)
            assert source == pytest.approx(centres[target["annotation_id"]])
            checked += 1
    assert checked > 0


def test_build_folds_are_grouped_by_sheet(tmp_path: Path) -> None:
    """No window appears in both halves of a fold."""
    coco = _coco(tmp_path)
    out = tmp_path / "det"
    build(coco, tmp_path / "images", out, window_size=512, stride=384)

    for fold in ("foldA", "foldB", "foldC"):
        train = set((out / f"{fold}_train.txt").read_text().split())
        val = set((out / f"{fold}_val.txt").read_text().split())
        assert not train & val
        assert (out / f"{fold}.yaml").exists()


def test_build_labels_only_preserves_images(tmp_path: Path) -> None:
    """A labels-only rebuild leaves the crops untouched."""
    coco = _coco(tmp_path)
    out = tmp_path / "det"
    build(coco, tmp_path / "images", out, window_size=512, stride=384)
    sample = next((out / "images").glob("*.png"))
    stamp = sample.stat().st_mtime_ns

    build(coco, tmp_path / "images", out, window_size=512, stride=384, write_images=False)
    assert sample.stat().st_mtime_ns == stamp
