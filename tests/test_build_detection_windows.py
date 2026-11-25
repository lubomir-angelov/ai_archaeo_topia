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
    LabelSource,
    RefitConfig,
    Window,
    _human_added_share,
    _overlaps,
    build,
    clip_box,
    crop_path,
    find_image,
    listing_entry,
    load_refit_config,
    plan_windows,
    refit_candidates,
    sheet_of,
    to_yolo,
    validate_export,
)
from archeo_topia.datasets.build_detection_windows import (
    main as bdw_main,
)
from tests.refit_fixtures import (
    COMMITTED_CONFIG,
    SCHEMA,
    TRAIN_VAL_SPLITS,
    RefitWorld,
    write_clips,
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


def _build(
    tmp_path: Path,
    plan: str = "loso",
    coco: Path | None = None,
    clips: list[Path] | None = None,
    splits: Path | None = None,
    name: str = "det",
    refit_config: RefitConfig | None = None,
) -> dict[str, Any]:
    """Build a dataset under ``tmp_path / name``."""
    coco = coco or _coco(tmp_path)
    return build(
        [LabelSource(coco.parent.name, coco, "all")],
        [tmp_path / "images"],
        tmp_path / name,
        window_size=512,
        stride=384,
        schema_path=SCHEMA,
        plan=plan,
        clips_json_paths=clips,
        refit_config=refit_config,
        splits_file=splits or _splits(tmp_path),
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _test_clips(tmp_path: Path) -> list[Path]:
    return write_clips(tmp_path, {"S-1": ["S-1_1.png"], "S-2": ["S-2_1.png"]})


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
            clips=write_clips(tmp_path, {"S-1": ["S-1_1.png", "S-1_9.png"], "S-3": []}),
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
    assert "coco_json" not in report["inputs"]
    (source,) = report["inputs"]["label_sources"]
    assert source["name"] == tmp_path.name
    assert source["mound_rule"] == "all"
    assert (
        source["sha256"] == hashlib.sha256((tmp_path / "instances.json").read_bytes()).hexdigest()
    )
    assert report["refit"] is None
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


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)
    return path


def test_find_image_searches_every_root(tmp_path: Path) -> None:
    """A clip is found in whichever root holds it, directly or one level down."""
    first, second = tmp_path / "a", tmp_path / "b"
    in_first = _png(first / "train" / "A_1.png")
    in_second = _png(second / "S-9" / "S-9_1.png")

    assert find_image([first, second], "A_1.png") == in_first
    assert find_image([first, second], "S-9_1.png") == in_second


def test_find_image_stops_when_the_clip_is_in_no_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="A_1.png not found"):
        find_image([tmp_path / "a", tmp_path / "b"], "A_1.png")


def test_find_image_stops_on_a_clip_in_two_places(tmp_path: Path) -> None:
    """The same name in two roots, or directly and one level down, is ambiguous."""
    _png(tmp_path / "a" / "A_1.png")
    _png(tmp_path / "b" / "val" / "A_1.png")
    with pytest.raises(ValueError, match="found 2 times"):
        find_image([tmp_path / "a", tmp_path / "b"], "A_1.png")

    _png(tmp_path / "a" / "val" / "B_1.png")
    _png(tmp_path / "a" / "B_1.png")
    with pytest.raises(ValueError, match="found 2 times"):
        find_image([tmp_path / "a"], "B_1.png")


# ---------------------------------------------------------------------------
# The refit: configuration and the train_val plan
# ---------------------------------------------------------------------------

PROTOCOL_CANDIDATES = (
    "epoch25.pt",
    "epoch50.pt",
    "epoch75.pt",
    "epoch100.pt",
    "epoch125.pt",
    "last.pt",
)


def _config_document(**edits: Any) -> dict[str, Any]:
    document = json.loads(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    document.update(edits)
    return document


def _write_config(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "refit.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_committed_refit_config_loads_and_yields_the_protocols_six_candidates() -> None:
    """The protocol's prose names six files; the derivation must agree (drift check)."""
    config = load_refit_config(COMMITTED_CONFIG)

    assert refit_candidates(config) == PROTOCOL_CANDIDATES
    assert (config.window, config.stride, config.epochs, config.batch, config.seed) == (
        512,
        384,
        150,
        16,
        42,
    )
    assert [(s.name, s.mound_rule) for s in config.label_sources] == [
        ("v0.0.3", "all"),
        ("v0.0.5", "reviewed"),
    ]
    assert all(s.coco_json.is_absolute() for s in config.label_sources)
    assert config.sha256 == hashlib.sha256(COMMITTED_CONFIG.read_bytes()).hexdigest()


def _without(key: str) -> dict[str, Any]:
    document = _config_document()
    del document[key]
    return document


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (_without("seed"), "missing \\['seed'\\]"),
        (_config_document(extra=1), "unexpected \\['extra'\\]"),
        (_config_document(window=True), "window must be a positive int"),
        (_config_document(batch=0), "batch must be a positive int"),
        (_config_document(epochs=150, save_period=40), "not a multiple of save_period"),
        (_config_document(development_source="v0.0.5"), "development source must use"),
        (_config_document(development_source="nope"), "names no source"),
        (_config_document(label_sources=[]), "label_sources is empty"),
        (
            _config_document(
                label_sources=[
                    {"name": "a", "coco_json": "x.json", "mound_rule": "some"},
                ],
                development_source="a",
            ),
            "mound_rule 'some' is unknown",
        ),
        (
            _config_document(
                label_sources=[
                    {"name": "a", "coco_json": "/abs/x.json", "mound_rule": "all"},
                ],
                development_source="a",
            ),
            "repository-relative",
        ),
    ],
)
def test_a_bad_refit_config_stops(tmp_path: Path, document: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_refit_config(_write_config(tmp_path, document))


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_train_val_writes_one_listing_pair_and_one_yaml(tmp_path: Path) -> None:
    world = RefitWorld(tmp_path)
    report = world.build()
    out = tmp_path / "refit"

    assert sorted(p.name for p in out.glob("*.txt")) == ["train.txt", "val.txt"]
    assert [p.name for p in out.glob("*.yaml")] == ["train_val.yaml"]
    assert report["listing"]["plan"] == "train_val"
    assert report["listing"]["folds"] is None
    assert report["listing"]["listings"]["train"]["sheets"] == ["D-1", "T-1"]
    assert report["listing"]["listings"]["val"]["sheets"] == ["V-1"]
    assert "outputs_sha256" in report and "train.txt" in report["outputs_sha256"]
    yaml = (out / "train_val.yaml").read_text(encoding="utf-8")
    assert "train: train.txt" in yaml and "val: val.txt" in yaml
    assert [s["name"] for s in report["inputs"]["label_sources"]] == ["v0.0.3", "v0.0.5"]
    assert report["refit"]["config"]["name"] == "v0_6_refit"


def test_train_val_excludes_unreviewed_and_absent_sheets_whole(tmp_path: Path) -> None:
    report = RefitWorld(tmp_path).build()
    refit = report["refit"]

    assert refit["excluded_sheets"] == {
        "V-2": "unreviewed_mounds",
        "V-3": "absent_from_label_source",
    }
    assert refit["sheets_by_source"] == {"v0.0.3": ["D-1"], "v0.0.5": ["T-1", "V-1"]}
    assert not any(
        w["sheet_id"] == "V-2" for w in _jsonl(tmp_path / "refit/metadata/windows.jsonl")
    )


def test_a_rejected_mound_is_background_and_a_negative_is_a_region_whatever_its_status(
    tmp_path: Path,
) -> None:
    report = RefitWorld(tmp_path).build()
    out = tmp_path / "refit"

    t1 = {
        r["annotation_id"]
        for r in _jsonl(out / "metadata/annotations.jsonl")
        if r["sheet_id"] == "T-1"
    }
    assert t1 == {1}
    labelled = [
        w
        for w in _jsonl(out / "metadata/windows.jsonl")
        if w["sheet_id"] == "T-1" and w["targets"]
    ]
    assert {t["annotation_id"] for w in labelled for t in w["targets"]} == {1}
    assert report["refit"]["rejected_mounds_as_background"] == {"count": 1, "by_sheet": {"T-1": 1}}
    assert report["counts"]["rejected_mounds_as_background"] == 1
    negative = next(
        r for r in _jsonl(out / "metadata/regions.jsonl") if r["kind"] == "hard_negative_symbol"
    )
    assert (negative["sheet_id"], negative["negative_type"]) == ("T-1", "unreviewed")


def test_a_train_window_over_an_ignore_box_is_recorded_but_not_listed_for_training(
    tmp_path: Path,
) -> None:
    report = RefitWorld(tmp_path).build()
    out = tmp_path / "refit"
    dropped = ["D-1_1_x00000_y00000", "D-1_1_x00000_y00188"]

    assert report["refit"]["uncertain_dropped_train_windows"] == {
        "count": 2,
        "window_ids": dropped,
        "mound_targets_in_dropped": 2,
    }
    recorded = {w["window_id"] for w in _jsonl(out / "metadata/windows.jsonl")}
    assert set(dropped) <= recorded
    assert (out / "images" / f"{dropped[0]}.png").exists()
    train = _lines(out / "train.txt")
    assert not any(any(d in line for d in dropped) for line in train)
    assert sum("D-1_1_" in line for line in train) == 4


def test_a_val_window_over_an_ignore_box_stays_in_the_val_listing(tmp_path: Path) -> None:
    report = RefitWorld(tmp_path).build()
    out = tmp_path / "refit"

    val_windows = [w for w in _jsonl(out / "metadata/windows.jsonl") if w["sheet_id"] == "V-1"]
    assert report["listing"]["listings"]["val"]["windows"] == len(val_windows) == 6
    assert len(_lines(out / "val.txt")) == 6
    assert report["refit"]["uncertain_dropped_train_windows"]["count"] == 2


def test_overlap_needs_positive_area() -> None:
    """A box that only touches the window edge is not an overlap."""
    window = Window(0, 0, 512)
    assert not _overlaps(window, [512, 0, 10, 10])
    assert not _overlaps(window, [-10, 0, 10, 10])
    assert _overlaps(window, [511.5, 0, 10, 10])


def test_the_human_added_share_is_counted_on_the_selection_sheets(tmp_path: Path) -> None:
    report = RefitWorld(tmp_path).build()

    assert report["refit"]["selection_human_added"] == {
        "sheets": ["V-1"],
        "mound_targets": 2,
        "human_added": 1,
        "share": 0.5,
        "by_sheet": {"V-1": {"mound_targets": 2, "human_added": 1}},
    }
    assert _human_added_share([], ["V-1"])["share"] is None


def _put_in_both(world: RefitWorld) -> None:
    world.reviewed_images.append("D-1_1.png")
    world.clips_by_sheet["D-1"] = ["D-1_1.png"]


def _dev_sheet_missing(world: RefitWorld) -> None:
    world.splits = {
        **TRAIN_VAL_SPLITS,
        "presets": {"loso_v0_5": {"folds": {"foldA": "D-1", "foldB": "D-2"}}},
        "splits": {**TRAIN_VAL_SPLITS["splits"], "train": ["D-1", "D-2", "T-1"]},
    }


def _foreign_in_dev_source(world: RefitWorld) -> None:
    world.dev_images.append("T-9_1.png")
    world.splits = {
        **TRAIN_VAL_SPLITS,
        "splits": {**TRAIN_VAL_SPLITS["splits"], "train": ["D-1", "T-1", "T-9"]},
    }


def _test_sheet_appears(world: RefitWorld) -> None:
    world.reviewed_images.append("X-1_1.png")
    world.clips_by_sheet["X-1"] = ["X-1_1.png"]


def _sheet_in_no_split(world: RefitWorld) -> None:
    world.reviewed_images.append("Z-1_1.png")
    world.clips_by_sheet["Z-1"] = ["Z-1_1.png"]


def _dev_sheet_in_val(world: RefitWorld) -> None:
    world.splits = {
        **TRAIN_VAL_SPLITS,
        "splits": {
            **TRAIN_VAL_SPLITS["splits"],
            "train": ["T-1"],
            "val": ["D-1", "V-1", "V-2", "V-3"],
        },
    }


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_put_in_both, "sheet D-1 is in both v0.0.3 and v0.0.5"),
        (_dev_sheet_missing, "development sheets missing from v0.0.3: \\['D-2'\\]"),
        (_foreign_in_dev_source, "not development sheets: \\['T-9'\\]"),
        (_test_sheet_appears, "test-split sheets are never used: \\['X-1'\\]"),
        (_sheet_in_no_split, "sheets in no split: \\['Z-1'\\]"),
        (_dev_sheet_in_val, "development sheets outside the train split"),
        (
            lambda w: w.clips_by_sheet.pop("T-1"),
            "named by no clips.json: \\['T-1_1.png'\\]",
        ),
        (
            lambda w: w.reviewed_annotations[0].pop("attributes"),
            "without a review_status",
        ),
    ],
)
def test_train_val_stops_before_writing_anything(
    tmp_path: Path, edit: Callable[[RefitWorld], Any], message: str
) -> None:
    world = RefitWorld(tmp_path, edit)

    with pytest.raises(ValueError, match=message):
        world.build()
    assert not (tmp_path / "refit").exists()


def test_train_val_stops_when_no_selection_sheet_remains(tmp_path: Path) -> None:
    world = RefitWorld(
        tmp_path,
        lambda w: w.reviewed_annotations[3].update(attributes={"review_status": "unreviewed"}),
    )
    world.splits = {
        **TRAIN_VAL_SPLITS,
        "splits": {**TRAIN_VAL_SPLITS["splits"], "val": ["V-1", "V-2"]},
    }

    with pytest.raises(ValueError, match="no reviewed sheet of the val split remains"):
        world.build()
    assert not (tmp_path / "refit").exists()


def test_train_val_refuses_a_reviewed_source_with_no_provenance(tmp_path: Path) -> None:
    world = RefitWorld(tmp_path)
    reviewed = world.config().label_sources[1].coco_json
    (reviewed.parent / "PROVENANCE.json").unlink()

    with pytest.raises(FileNotFoundError, match="PROVENANCE.json"):
        world.build()
    assert not (tmp_path / "refit").exists()


def test_train_val_refuses_a_reviewed_source_edited_after_the_cut(tmp_path: Path) -> None:
    world = RefitWorld(tmp_path)
    reviewed = world.config().label_sources[1].coco_json
    reviewed.write_text(reviewed.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="not the file the cut wrote"):
        world.build()
    assert not (tmp_path / "refit").exists()


def test_validate_export_reports_an_ambiguous_clip_with_the_other_violations(
    tmp_path: Path,
) -> None:
    world = RefitWorld(tmp_path)
    coco = json.loads(world.config().label_sources[0].coco_json.read_text(encoding="utf-8"))
    second_root = tmp_path / "second"
    (second_root / "train").mkdir(parents=True)
    Image.new("RGBA", (900, 700)).save(second_root / "train" / "D-1_1.png")
    coco["annotations"][0]["category_id"] = 99

    with pytest.raises(ValueError, match="found 2 times") as error:
        validate_export(coco, SCHEMA, [tmp_path / "images", second_root], None)
    assert "unknown category id 99" in str(error.value)


def test_train_val_refuses_arguments_that_differ_from_the_config(tmp_path: Path) -> None:
    world = RefitWorld(tmp_path)
    config = world.config()
    cases = {
        "requires a refit config": {"refit_config": None},
        "window/stride 256/384": {"window_size": 256},
        "label sources differ": {"label_sources": config.label_sources[:1]},
        "requires --clips-json": {"clips_json_paths": None},
    }
    for message, overrides in cases.items():
        with pytest.raises(ValueError, match=message):
            build(
                **{
                    "label_sources": config.label_sources,
                    "images_roots": [tmp_path / "images"],
                    "output_dir": tmp_path / "refit",
                    "window_size": 512,
                    "stride": 384,
                    "schema_path": SCHEMA,
                    "plan": "train_val",
                    "clips_json_paths": [tmp_path / "unused.json"],
                    "refit_config": config,
                    "splits_file": world.splits_path(),
                    **overrides,
                }
            )
    assert not (tmp_path / "refit").exists()


def test_the_single_source_plans_refuse_a_refit_config(tmp_path: Path) -> None:
    world = RefitWorld(tmp_path)
    config = world.config()
    with pytest.raises(ValueError, match="takes no refit config"):
        _build(tmp_path, refit_config=config)


def test_cli_keeps_window_and_stride_out_of_train_val_and_required_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = [
        "--images-root",
        str(tmp_path),
        "--schema",
        str(SCHEMA),
        "--output-dir",
        str(tmp_path / "o"),
    ]
    for argv in (
        ["--listing", "train_val", "--refit-config", str(COMMITTED_CONFIG), "--window", "512"],
        ["--listing", "train_val"],
        ["--listing", "loso", "--coco-json", str(tmp_path / "c.json"), "--stride", "384"],
        ["--listing", "loso", "--refit-config", str(COMMITTED_CONFIG)],
    ):
        monkeypatch.setattr("sys.argv", ["build", *common, *argv])
        with pytest.raises(SystemExit):
            bdw_main()
    assert not (tmp_path / "o").exists()
