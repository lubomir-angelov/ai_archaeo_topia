"""Tests for the MapSAM v0.5 detection-window builder."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from archeo_topia.datasets.build_detection_windows import (
    REPO_ROOT,
    Window,
    build,
    clip_box,
    crop_path,
    listing_entry,
    plan_windows,
    sheet_of,
    to_yolo,
)

SCHEMA = REPO_ROOT / "annotation" / "cvat" / "labels.json"


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


def _coco(
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None] | None = None,
    png_size: tuple[int, int] = (900, 700),
) -> Path:
    """Write a two-clip COCO export with one mound, one negative, two ignores."""
    images = [
        {"id": 1, "file_name": "S-1_1.png", "width": 900, "height": 700},
        {"id": 2, "file_name": "S-2_1.png", "width": 900, "height": 700},
    ]
    annotations = [
        # Comfortably inside the first window of clip 1.
        {
            "id": 10,
            "image_id": 1,
            "category_id": 1,
            "bbox": [200, 200, 24, 22],
            "segmentation": [],
        },
        {
            "id": 11,
            "image_id": 1,
            "category_id": 2,
            "bbox": [600, 600, 18, 18],
            "segmentation": [],
            "attributes": {"negative_type": "decorative_symbol"},
        },
        {
            "id": 12,
            "image_id": 1,
            "category_id": 3,
            "bbox": [300, 300, 20, 20],
            "segmentation": [],
        },
        # Crosses the right edge of the first window (x1 = 512) and is seen by three.
        {
            "id": 13,
            "image_id": 1,
            "category_id": 3,
            "bbox": [500, 100, 30, 20],
            "segmentation": [],
        },
        {
            "id": 20,
            "image_id": 2,
            "category_id": 1,
            "bbox": [400, 400, 24, 22],
            "segmentation": [],
        },
    ]
    payload: dict[str, Any] = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "mound"},
            {"id": 2, "name": "hard_negative_symbol"},
            {"id": 3, "name": "uncertain_ignore"},
        ],
    }
    if edit is not None:
        edit(payload)
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    image_root = tmp_path / "images" / "train"
    image_root.mkdir(parents=True, exist_ok=True)
    for image in images:
        Image.new("RGBA", png_size, (255, 255, 255, 255)).save(image_root / image["file_name"])
    return path


def _splits(
    tmp_path: Path,
    test: tuple[str, ...] = ("S-1", "S-2"),
    frozen: tuple[str, ...] = ("S-1", "S-2"),
) -> Path:
    """Write a splits file with a two-fold preset and a declared test split."""
    document = {
        "version": "vtest",
        "presets": {"loso_v0_5": {"folds": {"foldA": "S-1", "foldB": "S-2"}}},
        "splits": {"train": [], "val": [], "test": list(test)},
        "sheets": {s: {"frozen_blind_test": s in frozen} for s in ("S-1", "S-2", "S-3")},
    }
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _clips(tmp_path: Path, by_sheet: dict[str, list[str]]) -> list[Path]:
    """Write one clips.json per sheet."""
    paths = []
    for sheet, files in by_sheet.items():
        path = tmp_path / f"{sheet}_clips.json"
        path.write_text(
            json.dumps(
                {
                    "sheet_id": sheet,
                    "parent_size": [900, 700],
                    "clips": [{"file": f, "offset_xy": [0, 0], "size": [900, 700]} for f in files],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _build(
    tmp_path: Path,
    plan: str = "loso",
    coco: Path | None = None,
    clips: list[Path] | None = None,
    splits: Path | None = None,
    name: str = "det",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a dataset under ``tmp_path / name``."""
    return build(
        coco or _coco(tmp_path),
        tmp_path / "images",
        tmp_path / name,
        window_size=512,
        stride=384,
        schema_path=SCHEMA,
        plan=plan,
        clips_json_paths=clips,
        splits_file=splits or _splits(tmp_path),
        **kwargs,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _test_clips(tmp_path: Path) -> list[Path]:
    return _clips(tmp_path, {"S-1": ["S-1_1.png"], "S-2": ["S-2_1.png"]})


def test_build_writes_one_class_labels_and_shared_windows(tmp_path: Path) -> None:
    """Mounds become class-0 boxes; negatives and ignores go to regions.jsonl."""
    report = _build(tmp_path)
    out = tmp_path / "det"

    assert report["counts"]["mound_annotations"] == 2
    assert report["counts"]["windows_total"] == len(list((out / "images").glob("*.png")))
    assert len(list((out / "labels").glob("*.txt"))) == report["counts"]["windows_total"]

    windows = _jsonl(out / "metadata" / "windows.jsonl")
    targets = [t for w in windows for t in w["targets"]]
    assert {t["annotation_id"] for t in targets} == {10, 20}

    # Window records carry no regions: they live in regions.jsonl only.
    assert all(
        set(w) == {"window_id", "sheet_id", "source_image", "window_xyxy", "targets"}
        for w in windows
    )

    regions = _jsonl(out / "metadata" / "regions.jsonl")
    assert {(r["kind"], r["annotation_id"]) for r in regions} == {
        ("hard_negative_symbol", 11),
        ("uncertain_ignore", 12),
        ("uncertain_ignore", 13),
    }

    # Every written label line is class 0.
    for label in (out / "labels").glob("*.txt"):
        for line in label.read_text().splitlines():
            assert line.startswith("0 ")


def test_regions_cross_window_edges_once_with_the_full_source_box(tmp_path: Path) -> None:
    """An ignore crossing a window edge is one record, not clipped fragments."""
    _build(tmp_path)
    regions = _jsonl(tmp_path / "det" / "metadata" / "regions.jsonl")

    crossing = [r for r in regions if r["annotation_id"] == 13]
    assert len(crossing) == 1
    assert crossing[0]["bbox_source_xyxy"] == [500, 100, 530, 120]
    assert crossing[0]["negative_type"] is None
    assert crossing[0]["sheet_id"] == "S-1"
    assert crossing[0]["source_image"] == "S-1_1.png"

    negative = next(r for r in regions if r["annotation_id"] == 11)
    assert negative["negative_type"] == "decorative_symbol"
    assert negative["bbox_source_xyxy"] == [600, 600, 618, 618]


def test_build_targets_round_trip_to_source_coordinates(tmp_path: Path) -> None:
    """window_xyxy is a usable inverse transform for untruncated targets."""
    _build(tmp_path)
    out = tmp_path / "det"

    centres = {10: (212.0, 211.0), 20: (412.0, 411.0)}
    checked = 0
    for window in _jsonl(out / "metadata" / "windows.jsonl"):
        x0, y0, _, _ = window["window_xyxy"]
        for target in window["targets"]:
            if target["truncated"]:
                continue
            bx0, by0, bx1, by1 = target["box_window_xyxy"]
            source = ((bx0 + bx1) / 2 + x0, (by0 + by1) / 2 + y0)
            assert source == pytest.approx(centres[target["annotation_id"]])
            checked += 1
    assert checked > 0


def test_loso_writes_fold_lists_and_yamls_grouped_by_sheet(tmp_path: Path) -> None:
    """No window appears in both halves of a fold; each fold has a YAML."""
    _build(tmp_path, plan="loso")
    out = tmp_path / "det"

    for fold in ("foldA", "foldB"):
        train = set((out / f"{fold}_train.txt").read_text().split())
        val = set((out / f"{fold}_val.txt").read_text().split())
        assert train and val
        assert not train & val
        assert (out / f"{fold}.yaml").exists()
    assert not (out / "test.txt").exists()


def test_test_plan_writes_only_test_txt(tmp_path: Path) -> None:
    """A test build has no training split, no fold files and no YAML."""
    report = _build(tmp_path, plan="test", clips=_test_clips(tmp_path))
    out = tmp_path / "det"

    listed = (out / "test.txt").read_text().split()
    assert len(listed) == report["counts"]["windows_total"]
    assert not list(out.glob("*.yaml"))
    assert not list(out.glob("*_train.txt"))
    assert not list(out.glob("fold*"))
    assert report["listing"]["plan"] == "test"
    assert report["listing"]["folds"] is None
    assert set(report["listing"]["listings"]) == {"test"}


def test_test_plan_stops_on_a_sheet_outside_the_test_split(tmp_path: Path) -> None:
    """Nothing is written when an exported sheet is not a test sheet."""
    with pytest.raises(ValueError, match="not in the test split"):
        _build(
            tmp_path,
            plan="test",
            clips=_test_clips(tmp_path),
            splits=_splits(tmp_path, test=("S-1",), frozen=("S-1", "S-2")),
        )
    assert not (tmp_path / "det").exists()


def test_test_plan_stops_on_a_sheet_that_is_not_frozen(tmp_path: Path) -> None:
    """A test-split sheet that is not frozen blind test is refused."""
    with pytest.raises(ValueError, match="not frozen blind test"):
        _build(
            tmp_path,
            plan="test",
            clips=_test_clips(tmp_path),
            splits=_splits(tmp_path, frozen=("S-1",)),
        )
    assert not (tmp_path / "det").exists()


def test_loso_stops_on_a_sheet_outside_the_preset(tmp_path: Path) -> None:
    """A loso build must export exactly the preset's sheets."""
    document = json.loads(_splits(tmp_path).read_text())
    document["presets"]["loso_v0_5"]["folds"] = {"foldA": "S-1", "foldB": "S-3"}
    splits = tmp_path / "other_splits.json"
    splits.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="outside preset"):
        _build(tmp_path, splits=splits)
    assert not (tmp_path / "det").exists()


def test_test_plan_requires_clips_json_and_loso_rejects_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --clips-json"):
        _build(tmp_path, plan="test")
    with pytest.raises(ValueError, match="takes no --clips-json"):
        _build(tmp_path, plan="loso", clips=_test_clips(tmp_path))
    assert not (tmp_path / "det").exists()


def _rename_category(coco: dict[str, Any]) -> None:
    coco["categories"][0]["name"] = "tumulus"


def _duplicate_annotation(coco: dict[str, Any]) -> None:
    coco["annotations"].append(dict(coco["annotations"][0]))


@pytest.mark.parametrize(
    ("edit", "png_size", "message"),
    [
        (_rename_category, (900, 700), "categories not in"),
        (None, (800, 700), "the PNG is 800x700"),
        (_duplicate_annotation, (900, 700), "is not unique"),
    ],
)
def test_export_violations_stop_before_anything_is_written(
    tmp_path: Path,
    edit: Callable[[dict[str, Any]], None] | None,
    png_size: tuple[int, int],
    message: str,
) -> None:
    coco = _coco(tmp_path, edit=edit, png_size=png_size)
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, coco=coco)
    assert not (tmp_path / "det").exists()


def test_clip_completeness_violations_stop_before_anything_is_written(tmp_path: Path) -> None:
    """A clip missing from the export, an unnamed image and an empty manifest."""
    coco = _coco(tmp_path)
    with pytest.raises(ValueError) as caught:
        _build(
            tmp_path,
            plan="test",
            coco=coco,
            clips=_clips(tmp_path, {"S-1": ["S-1_1.png", "S-1_9.png"], "S-3": []}),
        )
    message = str(caught.value)
    assert "missing from the export: ['S-1_9.png']" in message
    assert "names no clips" in message
    assert "named by no clips.json: ['S-2_1.png']" in message
    assert not (tmp_path / "det").exists()


def test_report_records_inputs_listing_regions_and_verifiable_hashes(tmp_path: Path) -> None:
    """outputs_sha256 matches the files, and a test build claims no folds."""
    clips = _test_clips(tmp_path)
    report = _build(tmp_path, plan="test", clips=clips)
    out = tmp_path / "det"

    assert report == json.loads((out / "metadata" / "build_report.json").read_text())
    assert "folds" not in report
    assert report["inputs"]["splits_version"] == "vtest"
    assert "images_root" not in report["inputs"]
    assert "clips_json" not in report["inputs"]
    assert report["inputs"]["clips_json_sha256"] == {
        "S-1": hashlib.sha256(clips[0].read_bytes()).hexdigest(),
        "S-2": hashlib.sha256(clips[1].read_bytes()).hexdigest(),
    }
    assert report["inputs"]["source_images_sha256"] == {
        name: hashlib.sha256((tmp_path / "images" / "train" / name).read_bytes()).hexdigest()
        for name in ("S-1_1.png", "S-2_1.png")
    }
    assert report["regions"] == {"hard_negative_symbol": 1, "uncertain_ignore": 2}

    sha = report["outputs_sha256"]
    for name in ("windows", "annotations", "regions"):
        path = out / "metadata" / f"{name}.jsonl"
        assert sha[f"metadata/{name}.jsonl"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha["test.txt"] == hashlib.sha256((out / "test.txt").read_bytes()).hexdigest()

    digest = hashlib.sha256()
    for window_id in sorted(w["window_id"] for w in _jsonl(out / "metadata" / "windows.jsonl")):
        png = hashlib.sha256((out / "images" / f"{window_id}.png").read_bytes()).hexdigest()
        digest.update(f"{window_id} {png}\n".encode())
    assert sha["crops"] == digest.hexdigest()


def test_loso_report_lists_the_folds(tmp_path: Path) -> None:
    report = _build(tmp_path, plan="loso")
    assert report["listing"]["folds"] == {
        "foldA": {"eval_sheet": "S-1", "train_sheets": ["S-2"]},
        "foldB": {"eval_sheet": "S-2", "train_sheets": ["S-1"]},
    }
    assert "foldA_val.txt" in report["outputs_sha256"]


def test_rebuild_from_identical_inputs_gives_an_identical_report(tmp_path: Path) -> None:
    first = _build(tmp_path)
    second = _build(tmp_path, name="det_again")
    assert first == second


def test_build_into_an_existing_directory_raises_and_writes_nothing(tmp_path: Path) -> None:
    """A dataset is built once, so a stale listing can never be scored as fresh."""
    existing = tmp_path / "det"
    existing.mkdir()
    (existing / "fold_stale_val.txt").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _build(tmp_path)
    assert sorted(p.name for p in existing.iterdir()) == ["fold_stale_val.txt"]


def test_crop_path_and_listing_entry_name_the_same_crop() -> None:
    assert crop_path(Path("ds"), "S-1_1_x00000_y00000") == Path(
        "ds/images/S-1_1_x00000_y00000.png"
    )
    assert listing_entry("S-1_1_x00000_y00000") == "./images/S-1_1_x00000_y00000.png"
