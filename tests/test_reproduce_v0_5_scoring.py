"""Tests for the v0.5 reproduction check, on synthetic datasets."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from archeo_topia.analysis import reproduce_v0_5_scoring as repro
from archeo_topia.analysis.detection_metrics import Detection, load_evaluation_set
from archeo_topia.datasets.build_detection_windows import HARD_NEGATIVE, UNCERTAIN
from archeo_topia.training.train_mound_detector import (
    THRESHOLDS,
    metrics_document,
    operating_curve,
)
from tests.detection_fixtures import region_record, write_dataset

REPO = Path(__file__).resolve().parents[1]
WINDOW_ONE = "S-1_1_x00000_y00000"
WINDOW_TWO = "S-1_1_x00384_y00000"


def _legacy_windows() -> list[dict[str, Any]]:
    """Two windows of one clip, both clipping the same negative and the same ignore."""
    first = {
        "window_id": WINDOW_ONE,
        "sheet_id": "S-1",
        "source_image": "S-1_1.png",
        "window_xyxy": [0, 0, 512, 512],
        "targets": [],
        "hard_negatives": [
            {
                "annotation_id": 7,
                "negative_type": "road",
                "box_window_xyxy": [490.0, 10.0, 512.0, 30.0],
            }
        ],
        "uncertain_ignore": [
            {"annotation_id": 9, "box_window_xyxy": [100.5, 100.25, 120.0, 120.0]}
        ],
    }
    second = {
        "window_id": WINDOW_TWO,
        "sheet_id": "S-1",
        "source_image": "S-1_1.png",
        "window_xyxy": [384, 0, 896, 512],
        "targets": [],
        "hard_negatives": [
            {
                "annotation_id": 7,
                "negative_type": "road",
                "box_window_xyxy": [106.0, 10.0, 130.0, 30.0],
            }
        ],
        "uncertain_ignore": [],
    }
    return [first, second]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _tiny_dataset(root: Path, *, legacy: bool) -> Path:
    """A dataset with every file the equivalence check reads; ``legacy`` keeps region lists."""
    windows = _legacy_windows()
    if not legacy:
        windows = [
            {k: v for k, v in w.items() if k not in repro.LEGACY_REGION_KEYS} for w in windows
        ]
        _write_jsonl(
            root / "metadata" / "regions.jsonl",
            [
                region_record(
                    HARD_NEGATIVE, 7, "S-1", "S-1_1.png", [490.0, 10.0, 514.0, 30.0], "road"
                ),
                region_record(
                    UNCERTAIN, 9, "S-1", "S-1_1.png", [100.5, 100.25, 120.0, 120.0], None
                ),
            ],
        )
    _write_jsonl(root / "metadata" / "windows.jsonl", windows)
    _write_jsonl(root / "metadata" / "annotations.jsonl", [{"annotation_id": 1}])
    (root / "metadata" / "build_report.json").write_text(
        json.dumps(
            {"counts": {"w": 2}, "windows_per_sheet": {"S-1": 2}, "mounds_per_sheet": {"S-1": 1}}
        ),
        encoding="utf-8",
    )
    (root / "foldA_train.txt").write_text("./images/a.png\n", encoding="utf-8")
    (root / "foldA_val.txt").write_text("./images/b.png\n", encoding="utf-8")
    (root / "foldA.yaml").write_text(
        f"# fold\npath: {root.resolve()}\ntrain: foldA_train.txt\n", encoding="utf-8"
    )
    (root / "labels").mkdir()
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (root / "images").mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "images" / "a.png")
    return root


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    return _tiny_dataset(tmp_path / "original", legacy=True), _tiny_dataset(
        tmp_path / "rebuilt", legacy=False
    )


def test_legacy_reconstruction_takes_the_last_window_s_fragment(tmp_path: Path) -> None:
    original = _tiny_dataset(tmp_path / "original", legacy=True)

    regions = repro.legacy_window_clipped_regions(original / "metadata" / "windows.jsonl")

    assert list(regions) == [
        (HARD_NEGATIVE, "S-1_1.png", 7),
        (UNCERTAIN, "S-1_1.png", 9),
    ]
    negative = regions[(HARD_NEGATIVE, "S-1_1.png", 7)]
    # The first window clipped the box at x = 512; the second, the last, saw it to x = 514.
    assert (negative.x0, negative.y0, negative.x1, negative.y1) == (490.0, 10.0, 514.0, 30.0)
    assert (negative.sheet, negative.negative_type) == ("S-1", "road")
    ignore = regions[(UNCERTAIN, "S-1_1.png", 9)]
    assert ignore.negative_type is None
    assert (ignore.x0, ignore.y0) == (100.5, 100.25)


def test_region_differences_lists_crossing_regions_and_flags_losses(
    pair: tuple[Path, Path],
) -> None:
    original, rebuilt = pair
    legacy = repro.legacy_window_clipped_regions(original / "metadata" / "windows.jsonl")

    problems, differing = repro.region_differences(legacy, rebuilt / "metadata" / "regions.jsonl")
    assert problems == []
    assert differing == []

    # The source box is wider than the last window's fragment: the C3 difference.
    records = [
        json.loads(line)
        for line in (rebuilt / "metadata" / "regions.jsonl").read_text().splitlines()
    ]
    records[0]["bbox_source_xyxy"] = [475.0, 10.0, 514.0, 30.0]
    records[1]["negative_type"] = "road"
    records.append(region_record(HARD_NEGATIVE, 99, "S-1", "S-1_1.png", [1, 2, 3, 4], None))
    _write_jsonl(rebuilt / "metadata" / "regions.jsonl", records)

    problems, differing = repro.region_differences(legacy, rebuilt / "metadata" / "regions.jsonl")
    assert differing == [
        {
            "kind": HARD_NEGATIVE,
            "image": "S-1_1.png",
            "annotation_id": 7,
            "v05_box": [490.0, 10.0, 514.0, 30.0],
            "source_box": [475.0, 10.0, 514.0, 30.0],
        }
    ]
    assert any("not the original" in p and "99" in p for p in problems)
    assert any("negative_type None became 'road'" in p for p in problems)


def test_equivalent_datasets_have_no_problems(pair: tuple[Path, Path]) -> None:
    result = repro.dataset_equivalence(*pair)

    assert result["problems"] == []
    assert result["regions"] == {"v05_reconstruction": 2, "rebuilt": 2}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: (r / "metadata" / "annotations.jsonl").write_text("{}\n"), "annotations.jsonl"),
        (lambda r: (r / "labels" / "a.txt").write_text("0 0 0 0 0\n"), "labels/\\*.txt differ"),
        (
            lambda r: (r / "foldA_val.txt").write_text("./images/c.png\n"),
            "fold\\*_val.txt differ in bytes",
        ),
        (
            lambda r: Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8)).save(
                r / "images" / "a.png"
            ),
            "crops differ in pixels",
        ),
        (
            lambda r: (r / "foldA.yaml").write_text("# fold\npath: x\ntrain: other\n"),
            "beyond its path",
        ),
        (
            lambda r: (r / "foldA.yaml").write_text("# fold\npath: x\ntrain: foldA_train.txt\n"),
            "does not name",
        ),
        (
            lambda r: (r / "metadata" / "build_report.json").write_text(
                json.dumps({"counts": {"w": 3}, "windows_per_sheet": {}, "mounds_per_sheet": {}})
            ),
            "build_report counts differs",
        ),
        (
            lambda r: _write_jsonl(r / "metadata" / "windows.jsonl", [{"window_id": "x"}]),
            "windows.jsonl differs",
        ),
    ],
)
def test_each_kind_of_dataset_difference_is_reported(
    pair: tuple[Path, Path], mutate: Any, message: str
) -> None:
    mutate(pair[1])

    problems = repro.dataset_equivalence(*pair)["problems"]

    assert any(re.search(message, p) for p in problems), problems


def test_a_crop_re_encoded_to_the_same_pixels_is_equivalent(pair: tuple[Path, Path]) -> None:
    original, rebuilt = pair
    Image.open(original / "images" / "a.png").save(rebuilt / "images" / "a.png", compress_level=0)

    assert repro.dataset_equivalence(original, rebuilt)["problems"] == []


def test_v05_evaluation_set_swaps_in_the_reconstructed_regions(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds", listing=[WINDOW_ONE, WINDOW_TWO])
    rebuilt = load_evaluation_set(dataset, dataset / "all.txt", ["S-1"])
    original = _tiny_dataset(tmp_path / "original", legacy=True)
    legacy = repro.legacy_window_clipped_regions(original / "metadata" / "windows.jsonl")

    v0 = repro.v05_evaluation_set(rebuilt, legacy)

    assert [r.negative_type for r in v0.hard_negatives] == ["road"]
    assert len(v0.ignore_regions) == 1
    assert v0.annotations == rebuilt.annotations
    assert v0.sheets == rebuilt.sheets


def test_differences_is_exact_and_treats_nan_as_equal() -> None:
    nan = float("nan")
    assert repro.differences({"a": [1, nan, {"b": 0.5}]}, {"a": [1, nan, {"b": 0.5}]}) == []

    found = repro.differences({"a": 1.0, "k": 1}, {"a": math.nextafter(1.0, 2.0), "n": 1}, "x")
    assert {d["path"] for d in found} == {"x.a", "x.k", "x.n"}
    assert {"path": "x.k", "only_in": "expected"} in found
    assert {"path": "x.n", "only_in": "actual"} in found
    assert repro.differences([1, 2], [1], "l") == [
        {"path": "l.length", "expected": 2, "actual": 1}
    ]
    assert repro.differences({"a": 1}, 1, "t") == [
        {"path": "t", "expected": {"a": 1}, "actual": 1}
    ]


@pytest.fixture
def v0_document(tmp_path: Path) -> dict[str, Any]:
    dataset = write_dataset(tmp_path / "ds", listing=[WINDOW_ONE, WINDOW_TWO])
    evaluation_set = load_evaluation_set(dataset, dataset / "all.txt", ["S-1"])
    # Nothing reaches 0.70, so the top threshold has no candidates and undefined precision.
    raw = [Detection("S-1_1.png", 100, 100, 0.6), Detection("S-1_1.png", 300, 300, 0.4)]
    return metrics_document(
        raw, Path("w.pt"), evaluation_set, 512, 10.0, THRESHOLDS, torch.device("cpu"), "foldA"
    )


def _as_v05(document: dict[str, Any]) -> dict[str, Any]:
    """Build the v0.5 ``metrics.json`` a V0 document would have produced."""
    projected, _ = repro.project_v05_detection(
        document["by_confidence"]["0.05"]["pooled"], "S-1", "0.05"
    )
    return {
        **projected,
        "config": {"fold": "foldA"},
        "operating_points": json.loads(json.dumps(operating_curve(document))),
    }


def test_project_v05_detection_maps_the_new_layout_onto_the_old(
    v0_document: dict[str, Any],
) -> None:
    pooled = v0_document["by_confidence"]["0.05"]["pooled"]

    projected, problems = repro.project_v05_detection(pooled, "S-1", "0.05")

    assert problems == []
    assert projected["precision_at_association_radius"] == pooled["precision"]["value"]
    assert projected["per_sheet"] == {"S-1": pooled["recall"]}
    by_type = projected["false_positives"]["by_negative_type"]
    assert by_type["background"] == pooled["false_positives"]["background"]
    assert set(projected) == {
        "detections",
        "detections_neutralized_by_ignore",
        "annotations",
        "association_radius",
        "recall",
        "localization_error_px",
        "false_positives",
        "precision_at_association_radius",
        "subsets",
        "per_sheet",
    }


def test_a_hard_negative_without_a_type_is_a_mismatch_against_v05(
    v0_document: dict[str, Any],
) -> None:
    pooled = json.loads(json.dumps(v0_document["by_confidence"]["0.05"]["pooled"]))
    pooled["false_positives"]["near_hard_negative"]["without_negative_type"] = 2

    _, problems = repro.project_v05_detection(pooled, "S-1", "0.05")

    assert [p["actual"] for p in problems] == [2]


def test_comparison_with_v05_passes_when_equal_and_catches_one_ulp(
    v0_document: dict[str, Any],
) -> None:
    v05 = _as_v05(v0_document)
    assert repro.compare_detection_with_v05(v05, v0_document, "S-1") == []

    v05["localization_error_px"]["p90"] = math.nextafter(v05["localization_error_px"]["p90"], 99.0)
    v05["operating_points"][2]["recall_5px"] += 0.001
    mismatches = repro.compare_detection_with_v05(v05, v0_document, "S-1")

    assert {m["path"] for m in mismatches} == {
        "0.05.localization_error_px.p90",
        "operating_points[2].recall_5px",
    }


def test_nan_in_both_documents_is_not_a_mismatch(v0_document: dict[str, Any]) -> None:
    # No candidate reaches 0.70, so precision is NaN in the V0 document and in its v0.5 image.
    pooled = v0_document["by_confidence"]["0.70"]["pooled"]
    assert math.isnan(pooled["precision"]["value"])
    assert repro.compare_detection_with_v05(_as_v05(v0_document), v0_document, "S-1") == []


def test_end_to_end_projection_keys_by_id_and_rounds_to_four_places() -> None:
    block = {
        "candidates": 3,
        "scorable_annotations": 2,
        "false_masks": 1,
        "mean_iou": 0.5,
        "iou_0.5": {"rate": 0.5, "hits": 1},
        "iou_0.75": {"rate": 0.0, "hits": 0},
        "missed": {"rate": 0.5},
        "per_annotation_iou": {"S-1_1.png#280": 0.87564, "S-2_1.png#294": 0.0},
    }

    projected = repro.project_v05_end_to_end(block)

    assert projected["per_annotation_iou"] == {"280": 0.8756, "294": 0.0}
    assert projected["hits_iou_0.5"] == 1
    assert projected["rate_iou_0.75"] == 0.0
    assert projected["missed_rate"] == 0.5

    block["per_annotation_iou"] = {"S-1_1.png#1": 0.1, "S-2_1.png#1": 0.2}
    with pytest.raises(ValueError, match="repeats"):
        repro.project_v05_end_to_end(block)


def test_end_to_end_comparison_names_the_arm(v0_document: dict[str, Any]) -> None:
    block = {
        "candidates": 1,
        "scorable_annotations": 1,
        "false_masks": 0,
        "mean_iou": 0.5,
        "iou_0.5": {"rate": 1.0, "hits": 1},
        "iou_0.75": {"rate": 0.0, "hits": 0},
        "missed": {"rate": 0.0},
        "per_annotation_iou": {"S-1_1.png#1": 0.5},
    }
    v05_fold = {"arms": {"fixed": repro.project_v05_end_to_end(block)}}
    assert repro.compare_end_to_end_with_v05(v05_fold, {"fixed": {"pooled": block}}) == []

    block["false_masks"] = 2
    mismatches = repro.compare_end_to_end_with_v05(v05_fold, {"fixed": {"pooled": block}})
    assert mismatches == [{"path": "fixed.false_masks", "expected": 0, "actual": 2}]


def test_the_legacy_reconstruction_is_private_to_the_reproduction() -> None:
    allowed = {
        REPO / "src/archeo_topia/analysis/reproduce_v0_5_scoring.py",
        REPO / "tests/test_reproduce_v0_5_scoring.py",
    }
    users = {
        path
        for root in ("src", "tests")
        for path in (REPO / root).rglob("*.py")
        if "legacy_window_clipped_regions" in path.read_text(encoding="utf-8")
    }
    assert users == allowed


ORIGINAL = REPO / "data/curated/datasets/mapsam_det_v1"
REBUILT = REPO / "data/curated/datasets/mapsam_det_v1_c3"


@pytest.mark.skipif(
    not (ORIGINAL.exists() and REBUILT.exists()),
    reason="needs mapsam_det_v1 and its C3 rebuild",
)
def test_real_rebuild_is_equivalent_to_the_original_apart_from_regions() -> None:
    result = repro.dataset_equivalence(ORIGINAL, REBUILT)

    assert result["problems"] == []


def test_a_fold_is_run_once_and_scored_under_both_region_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The source-coordinate regions: a trig point near the third detection, and an
    # ignore box that covers the second.
    regions = [
        region_record(HARD_NEGATIVE, 11, "S-1", "S-1_1.png", [290, 290, 310, 310], "trig_point"),
        region_record(UNCERTAIN, 12, "S-1", "S-1_1.png", [105, 105, 115, 115], None),
    ]
    rebuilt = write_dataset(
        tmp_path / "rebuilt", regions=regions, listing=[WINDOW_ONE, WINDOW_TWO]
    )
    (rebuilt / "foldA_val.txt").write_text(
        f"./images/{WINDOW_ONE}.png\n./images/{WINDOW_TWO}.png\n", encoding="utf-8"
    )
    original = _tiny_dataset(tmp_path / "original", legacy=True)
    legacy = repro.legacy_window_clipped_regions(original / "metadata" / "windows.jsonl")

    inference_calls: list[Path] = []
    segmented: list[str] = []
    detections = [
        Detection("S-1_1.png", 100, 100, 0.9, box=(90, 90, 110, 110)),
        # Inside the v0.5 ignore region: neutralized in V0, a false positive in V1.
        Detection("S-1_1.png", 110, 110, 0.8, box=(100, 100, 120, 120)),
        # Background to v0.5, which had no trig point there; attributed in V1.
        Detection("S-1_1.png", 300, 300, 0.7, box=(290, 290, 310, 310)),
    ]

    def predict(weights, evaluation_set, imgsz, min_confidence, device):
        inference_calls.append(evaluation_set.listing)
        return list(detections)

    def segment(sam, candidates, clip_paths, arm, window_px, image_size, device):
        segmented.append(arm)
        mask = np.zeros((512, 512), dtype=bool)
        mask[90:110, 90:110] = True
        return [(mask, (0, 0, 512)) for _ in candidates]

    truth = np.zeros((1000, 1000), dtype=bool)
    truth[90:110, 90:110] = True
    monkeypatch.setattr(repro, "FOLDS", {"foldA": "S-1"})
    monkeypatch.setattr(repro, "predict_windows", predict)
    monkeypatch.setattr(repro, "segment_candidates", segment)
    monkeypatch.setattr(repro, "load_mapsam", lambda *a: object())
    monkeypatch.setattr(repro, "find_image", lambda roots, name: roots[0] / name)
    monkeypatch.setattr(repro, "load_truth_masks", lambda coco, es: {("S-1_1.png", 1): truth})

    detector_run = tmp_path / "run"
    (detector_run / "foldA").mkdir(parents=True)
    (detector_run / "foldA" / "metrics.json").write_text(
        json.dumps({"operating_points": []}), encoding="utf-8"
    )
    inputs = repro.ReproductionInputs(
        original_dataset=original,
        rebuilt_dataset=rebuilt,
        detector_run=detector_run,
        v05_end_to_end=tmp_path / "unused.json",
        mapsam_root=tmp_path / "mapsam",
        mapsam_prefix="pre",
        sam_checkpoint=tmp_path / "sam.pth",
        images_root=tmp_path / "images",
        coco_json=tmp_path / "coco.json",
    )
    device = torch.device("cpu")

    first = tmp_path / "first"
    (first / "detections").mkdir(parents=True)
    detection, end_to_end, _ = repro.reproduce_fold(
        inputs, "foldA", legacy, {"arms": {"fixed": {}, "detector": {}}}, device, first
    )
    # Against empty v0.5 records everything mismatches, which is how a wrong v0.5 shows up.
    assert detection["v05_mismatches"]
    assert end_to_end["v05_mismatches"]

    # One inference per fold, one segmentation per arm, serving both region inputs.
    assert inference_calls == [rebuilt / "foldA_val.txt"]
    assert segmented == list(repro.BOX_MODES)

    # A v0.5 record that is exactly the V0 result reproduces with no mismatch.
    v0_metrics = json.loads(
        (first / "v05_inputs" / "detector" / "foldA" / "metrics.json").read_text(encoding="utf-8")
    )
    v0_e2e = json.loads(
        (first / "v05_inputs" / "end_to_end" / "foldA" / "end_to_end.json").read_text(
            encoding="utf-8"
        )
    )
    (detector_run / "foldA" / "metrics.json").write_text(
        json.dumps(_as_v05(v0_metrics)), encoding="utf-8"
    )
    v05_fold = {
        "arms": {
            arm: repro.project_v05_end_to_end(scored["pooled"])
            for arm, scored in v0_e2e["arms"].items()
        }
    }
    (tmp_path / "second" / "detections").mkdir(parents=True)
    detection, end_to_end, found_only = repro.reproduce_fold(
        inputs, "foldA", legacy, v05_fold, device, tmp_path / "second"
    )

    assert detection["v05_mismatches"] == []
    assert end_to_end["v05_mismatches"] == []
    assert found_only == v0_e2e["arms"]["fixed"]["pooled"]["found"]["mean_iou"]
    paths = {d["path"] for d in end_to_end["c3_c4_differences"]}
    assert ".candidates_neutralized_by_ignore" in paths
    assert any(".trig_point" in d["path"] for d in detection["c3_differences"])
    assert (tmp_path / "second" / "as_fixed" / "detector" / "foldA" / "report.md").is_file()
    assert (tmp_path / "second" / "detections" / "foldA.jsonl").is_file()


def _coco(negative_types: dict[int, str]) -> dict[str, Any]:
    """A COCO export holding one hard negative per entry and one mound."""
    return {
        "categories": [{"id": 1, "name": "mound"}, {"id": 2, "name": HARD_NEGATIVE}],
        "annotations": [
            {"id": 1, "category_id": 1, "attributes": {}},
            *[
                {"id": i, "category_id": 2, "attributes": {"negative_type": value, "x": i}}
                for i, value in negative_types.items()
            ],
        ],
    }


def _original(tmp_path: Path) -> Path:
    return _tiny_dataset(tmp_path / "original", legacy=True)


def _write_coco(tmp_path: Path, coco: dict[str, Any]) -> Path:
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco), encoding="utf-8")
    return path


def test_input_differences_name_the_annotations_the_export_changed(tmp_path: Path) -> None:
    original = _original(tmp_path)

    # The original build recorded annotation 7 as "road".
    assert repro.v05_input_differences(
        original / "metadata" / "windows.jsonl", _coco({7: "other"})
    ) == {7: ("other", "road")}
    assert (
        repro.v05_input_differences(
            original / "metadata" / "windows.jsonl", _coco({7: "road", 8: "other"})
        )
        == {}
    )


def test_input_differences_raise_on_inconsistent_windows_and_absent_annotations(
    tmp_path: Path,
) -> None:
    original = _original(tmp_path)
    windows = _legacy_windows()
    windows[1]["hard_negatives"][0]["negative_type"] = "trig_point"
    _write_jsonl(tmp_path / "inconsistent.jsonl", windows)

    with pytest.raises(ValueError, match="across windows"):
        repro.v05_input_differences(tmp_path / "inconsistent.jsonl", _coco({7: "road"}))
    with pytest.raises(ValueError, match="absent from the export"):
        repro.v05_input_differences(original / "metadata" / "windows.jsonl", _coco({8: "road"}))


@pytest.fixture
def patched_seven(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repro, "V05_REPOSITORY_PATCHED_ANNOTATIONS", frozenset({7}))


def test_restoration_changes_only_the_declared_annotation(
    tmp_path: Path, patched_seven: None
) -> None:
    original = _original(tmp_path)
    coco = _coco({7: "other", 8: "decorative_symbol"})
    source = _write_coco(tmp_path, coco)
    output = tmp_path / "restored" / "instances_default.json"

    record = repro.restore_v05_build_input(original, source, output)

    restored = json.loads(output.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(coco))
    expected["annotations"][1]["attributes"]["negative_type"] = "road"
    assert restored == expected
    assert record["restored"] == [
        {"annotation_id": 7, "repo_value": "other", "built_value": "road"}
    ]
    assert record["source_sha256"] != record["output_sha256"]
    assert json.loads(source.read_text(encoding="utf-8")) == coco


def test_restoration_stops_when_another_annotation_differs(
    tmp_path: Path, patched_seven: None
) -> None:
    original = _original(tmp_path)
    windows = _legacy_windows()
    windows[0]["hard_negatives"].append(
        {"annotation_id": 8, "negative_type": "road", "box_window_xyxy": [1, 1, 2, 2]}
    )
    _write_jsonl(original / "metadata" / "windows.jsonl", windows)
    source = _write_coco(tmp_path, _coco({7: "other", 8: "other"}))

    with pytest.raises(ValueError, match=r"unexpected: \[8\]"):
        repro.restore_v05_build_input(original, source, tmp_path / "restored" / "i.json")
    assert not (tmp_path / "restored").exists()


def test_restoration_stops_when_the_declared_annotation_does_not_differ(
    tmp_path: Path, patched_seven: None
) -> None:
    source = _write_coco(tmp_path, _coco({7: "road"}))

    with pytest.raises(ValueError, match=r"not differing: \[7\]"):
        repro.restore_v05_build_input(
            _original(tmp_path), source, tmp_path / "restored" / "i.json"
        )
    assert not (tmp_path / "restored").exists()


def test_restoration_never_replaces_an_existing_output(
    tmp_path: Path, patched_seven: None
) -> None:
    source = _write_coco(tmp_path, _coco({7: "other"}))
    existing = tmp_path / "restored"
    existing.mkdir()
    (existing / "i.json").write_text("published", encoding="utf-8")

    with pytest.raises(FileExistsError, match="written once"):
        repro.restore_v05_build_input(_original(tmp_path), source, existing / "i.json")
    assert (existing / "i.json").read_text(encoding="utf-8") == "published"


def test_float_noise_in_the_v05_sum_is_not_a_box_difference(tmp_path: Path) -> None:
    """V0 keeps the unrounded v0.5 sum; only the listing compares at record precision."""
    noisy = 32.09 + 384
    assert noisy != round(noisy, 2)
    windows = _legacy_windows()
    windows[1]["hard_negatives"][0]["box_window_xyxy"] = [32.09, 10.0, 130.0, 30.0]
    _write_jsonl(tmp_path / "windows.jsonl", windows)
    legacy = repro.legacy_window_clipped_regions(tmp_path / "windows.jsonl")
    assert legacy[(HARD_NEGATIVE, "S-1_1.png", 7)].x0 == noisy

    _write_jsonl(
        tmp_path / "regions.jsonl",
        [
            region_record(
                HARD_NEGATIVE, 7, "S-1", "S-1_1.png", [round(noisy, 2), 10.0, 514.0, 30.0], "road"
            ),
            region_record(UNCERTAIN, 9, "S-1", "S-1_1.png", [100.5, 100.25, 120.0, 120.0], None),
        ],
    )
    problems, differing = repro.region_differences(legacy, tmp_path / "regions.jsonl")
    assert (problems, differing) == ([], [])

    dataset = write_dataset(tmp_path / "ds", listing=[WINDOW_ONE, WINDOW_TWO])
    rebuilt = load_evaluation_set(dataset, dataset / "all.txt", ["S-1"])
    v0 = repro.v05_evaluation_set(rebuilt, legacy)
    assert [r.x0 for r in v0.hard_negatives] == [noisy]


REPO_EXPORT = REPO / "annotation/cvat/v0.0.2/instances_default.json"


@pytest.mark.skipif(
    not (ORIGINAL.exists() and REPO_EXPORT.exists()),
    reason="needs mapsam_det_v1 and the repository export",
)
def test_real_repository_export_differs_from_the_v05_build_input_exactly_as_declared() -> None:
    differing = repro.v05_input_differences(
        ORIGINAL / "metadata" / "windows.jsonl", json.loads(REPO_EXPORT.read_text("utf-8"))
    )

    assert differing.keys() == repro.V05_REPOSITORY_PATCHED_ANNOTATIONS
