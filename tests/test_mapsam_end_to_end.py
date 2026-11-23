"""Tests for the end-to-end scorer, with segmentation replaced by fixed masks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from archeo_topia.analysis import mapsam_end_to_end as e2e
from archeo_topia.analysis.detection_metrics import Detection, annotation_key, load_evaluation_set
from tests.detection_fixtures import write_dataset

WINDOW = (0, 0, 512)


def _truth(cx: int, cy: int, half: int = 10) -> np.ndarray:
    """A square truth mask on a 1000 x 1000 clip."""
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[cy - half : cy + half, cx - half : cx + half] = True
    return mask


def _predicted(cx: int, cy: int, half: int = 10) -> tuple[np.ndarray, tuple[int, int, int]]:
    """A square predicted mask in window coordinates, window at the origin."""
    mask = np.zeros((512, 512), dtype=bool)
    mask[cy - half : cy + half, cx - half : cx + half] = True
    return mask, WINDOW


TRUTH = {("S-1_1.png", 1): _truth(100, 100), ("S-2_1.png", 2): _truth(200, 200)}


def _set(tmp_path: Path, **kwargs: Any):
    dataset = write_dataset(tmp_path / "ds", **kwargs)
    return load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])


def test_a_perfect_candidate_scores_iou_one_and_a_missing_one_scores_zero(tmp_path: Path) -> None:
    candidates = [Detection("S-1_1.png", 100, 100, 0.9)]
    scored = e2e.score_end_to_end(candidates, [_predicted(100, 100)], _set(tmp_path), TRUTH)

    pooled = scored["pooled"]
    assert pooled["scorable_annotations"] == 2
    assert pooled["candidates"] == 1
    assert pooled["false_masks"] == 0
    assert pooled["per_annotation_iou"] == {
        annotation_key("S-1_1.png", 1): 1.0,
        annotation_key("S-2_1.png", 2): 0.0,
    }
    assert pooled["iou_0.5"]["hits"] == 1
    assert pooled["iou_0.5"]["rate"] == 0.5
    low, high = pooled["iou_0.5"]["ci95"]
    assert 0.0 < low < 0.5 < high < 1.0
    assert pooled["iou_0.75"]["hits"] == 1
    # A miss counts as zero in the mean, and only found annotations count in the found mean.
    assert pooled["mean_iou"] == 0.5
    assert pooled["median_iou"] == 0.5
    assert pooled["found"] == {"n": 1, "mean_iou": 1.0}
    assert pooled["missed"] == {"count": 1, "rate": 0.5}


def test_missed_means_no_candidate_mask_overlaps_the_mound(tmp_path: Path) -> None:
    es = _set(tmp_path)
    far = e2e.score_end_to_end(
        [Detection("S-1_1.png", 130, 100, 0.9)], [_predicted(130, 100)], es, TRUTH
    )["pooled"]
    # 30 px away: the candidate is a false mask and the annotation is missed.
    assert far["false_masks"] == 1
    assert far["missed"]["count"] == 2

    near = e2e.score_end_to_end(
        [Detection("S-1_1.png", 120, 100, 0.9)], [_predicted(300, 300)], es, TRUTH
    )["pooled"]
    # 20 px away but the mask overlaps nothing: not a false mask, yet the mound is missed.
    assert near["false_masks"] == 0
    assert near["found"]["n"] == 0
    assert math.isnan(near["found"]["mean_iou"])
    assert near["missed"]["count"] == 2


def test_a_candidate_near_two_mounds_finds_only_the_one_its_mask_overlaps(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds")
    lines = (dataset / "metadata" / "annotations.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    records[1]["sheet_id"] = "S-1"
    records[1]["source_image"] = "S-1_1.png"
    records[1]["center_source_xy"] = [110.0, 100.0]
    (dataset / "metadata" / "annotations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    es = load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])
    truth = {
        ("S-1_1.png", 1): _truth(100, 100),
        ("S-1_1.png", 2): _truth(110, 100, half=2),
    }
    scored = e2e.score_end_to_end(
        [Detection("S-1_1.png", 100, 100, 0.9)],
        [_predicted(100, 100, half=5)],
        es,
        truth,
    )["pooled"]
    # Both mounds are within 25 px of the candidate; the mask reaches only the first.
    assert scored["false_masks"] == 0
    assert scored["found"]["n"] == 1
    assert scored["missed"]["count"] == 1


def test_a_candidate_in_an_ignore_region_is_neutralized_not_a_false_mask(tmp_path: Path) -> None:
    candidates = [Detection("S-2_1.png", 500, 500, 0.9)]
    masks = [_predicted(500, 500)]

    ignored = e2e.score_end_to_end(candidates, masks, _set(tmp_path / "a"), TRUTH)["pooled"]
    assert ignored["false_masks"] == 0
    assert ignored["candidates"] == 0

    no_regions = e2e.score_end_to_end(candidates, masks, _set(tmp_path / "b", regions=[]), TRUTH)
    assert no_regions["pooled"]["false_masks"] == 1
    assert no_regions["pooled"]["candidates"] == 1


def test_per_sheet_blocks_and_macro(tmp_path: Path) -> None:
    candidates = [Detection("S-1_1.png", 100, 100, 0.9)]
    scored = e2e.score_end_to_end(candidates, [_predicted(100, 100)], _set(tmp_path), TRUTH)

    assert list(scored["per_sheet"]) == ["S-1", "S-2"]
    assert scored["per_sheet"]["S-1"]["mean_iou"] == 1.0
    assert scored["per_sheet"]["S-2"]["mean_iou"] == 0.0
    assert scored["macro"]["mean_iou"] == {"mean": 0.5, "min": 0.0, "max": 1.0}
    assert scored["macro"]["rate_iou_0.5"] == {"mean": 0.5, "min": 0.0, "max": 1.0}
    # S-2 has no found annotation, so its found mean is undefined and so is the macro.
    assert all(math.isnan(v) for v in scored["macro"]["found_mean_iou"].values())
    assert scored["macro"]["missed_rate"] == {"mean": 0.5, "min": 0.0, "max": 1.0}
    assert set(scored["macro"]) == {
        "rate_iou_0.5",
        "rate_iou_0.75",
        "mean_iou",
        "median_iou",
        "found_mean_iou",
        "missed_rate",
    }


def test_mounds_without_geometry_are_counted_per_sheet_and_not_scored(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "ds")
    lines = (dataset / "metadata" / "annotations.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    records[1]["has_geometry"] = False
    (dataset / "metadata" / "annotations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    es = load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])

    scored = e2e.score_end_to_end([], [], es, {("S-1_1.png", 1): TRUTH[("S-1_1.png", 1)]})

    assert scored["per_sheet"]["S-1"]["mounds_without_geometry"] == 0
    assert scored["per_sheet"]["S-2"]["mounds_without_geometry"] == 1
    assert scored["per_sheet"]["S-2"]["scorable_annotations"] == 0
    assert scored["pooled"]["mounds_without_geometry"] == 1
    assert scored["pooled"]["scorable_annotations"] == 1


def test_a_missing_truth_mask_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without a truth mask"):
        e2e.score_end_to_end([], [], _set(tmp_path), {("S-1_1.png", 1): TRUTH[("S-1_1.png", 1)]})


def test_iou_is_stored_unrounded(tmp_path: Path) -> None:
    truth = np.zeros((1000, 1000), dtype=bool)
    truth[100, 99:102] = True
    predicted = np.zeros((512, 512), dtype=bool)
    predicted[100, 100] = True
    scored = e2e.score_end_to_end(
        [Detection("S-1_1.png", 100, 100, 0.9)],
        [(predicted, WINDOW)],
        _set(tmp_path),
        {("S-1_1.png", 1): truth, ("S-2_1.png", 2): TRUTH[("S-2_1.png", 2)]},
    )
    assert scored["pooled"]["per_annotation_iou"][annotation_key("S-1_1.png", 1)] == 1 / 3


def test_mask_and_candidate_counts_must_agree_and_candidates_must_be_in_the_set(
    tmp_path: Path,
) -> None:
    es = _set(tmp_path)
    with pytest.raises(ValueError, match="0 masks for 1 candidates"):
        e2e.score_end_to_end([Detection("S-1_1.png", 1, 1, 0.9)], [], es, TRUTH)
    with pytest.raises(ValueError, match="outside the evaluation set"):
        e2e.score_end_to_end(
            [Detection("S-9_1.png", 1, 1, 0.9)], [_predicted(100, 100)], es, TRUTH
        )


def test_resolve_candidates_filters_then_merges() -> None:
    raw = [
        Detection("S-1_1.png", 100, 100, 0.9),
        Detection("S-1_1.png", 104, 100, 0.6),
        Detection("S-1_1.png", 300, 300, 0.2),
        Detection("S-1_1.png", 500, 500, 0.3),
    ]
    candidates = e2e.resolve_candidates(raw, 0.25, 10.0)
    assert [(c.x, c.y) for c in candidates] == [(100, 100), (500, 500)]


def test_box_modes_are_the_one_definition_of_the_arms() -> None:
    assert e2e.BOX_MODES == ("fixed", "detector")
    with pytest.raises(ValueError, match="unknown box mode"):
        e2e.segment_candidates(None, [], {}, "other", 512, 1024, torch.device("cpu"))
    with pytest.raises(ValueError, match="no clip path"):
        e2e.segment_candidates(
            None, [Detection("S-1_1.png", 1, 1, 0.9)], {}, "fixed", 512, 1024, torch.device("cpu")
        )


def test_load_truth_masks_selects_mounds_by_name_on_the_listed_images(tmp_path: Path) -> None:
    square = [[10, 10, 30, 10, 30, 30, 10, 30]]
    coco = {
        "images": [
            {"id": 1, "file_name": "S-1_1.png", "width": 100, "height": 100},
            {"id": 2, "file_name": "S-9_1.png", "width": 100, "height": 100},
        ],
        "categories": [{"id": 7, "name": "mound"}, {"id": 8, "name": "hard_negative_symbol"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 10, 20, 20],
                "segmentation": square,
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 8,
                "bbox": [10, 10, 20, 20],
                "segmentation": square,
            },
            {
                "id": 4,
                "image_id": 2,
                "category_id": 7,
                "bbox": [10, 10, 20, 20],
                "segmentation": square,
            },
        ],
    }
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco))

    masks = e2e.load_truth_masks(path, _set(tmp_path))
    assert list(masks) == [("S-1_1.png", 1)]
    assert masks[("S-1_1.png", 1)].shape == (100, 100)
    assert masks[("S-1_1.png", 1)].any()

    coco["categories"] = [{"id": 8, "name": "hard_negative_symbol"}]
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="no 'mound' category"):
        e2e.load_truth_masks(path, _set(tmp_path / "again"))


def test_run_pair_pairs_explicitly_and_runs_every_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[Path] = []
    modes: list[str] = []
    monkeypatch.setattr(e2e, "load_truth_masks", lambda coco, es: TRUTH)
    monkeypatch.setattr(
        e2e, "load_mapsam", lambda checkpoint, sam, device: loaded.append(checkpoint)
    )

    def segment(sam, candidates, clip_paths, box_mode, window_px, image_size, device):
        modes.append(box_mode)
        return [_predicted(round(c.x), round(c.y)) for c in candidates]

    monkeypatch.setattr(e2e, "segment_candidates", segment)

    raw = [
        Detection("S-1_1.png", 100, 100, 0.9),
        Detection("S-2_1.png", 500, 500, 0.8),
        Detection("S-1_1.png", 700, 300, 0.1),
    ]
    result = e2e.run_pair(
        raw=raw,
        detector_weights=Path("det.pt"),
        decoder_checkpoint=Path("dec.pt"),
        sam_checkpoint=Path("sam.pth"),
        evaluation_set=_set(tmp_path),
        coco_json=Path("coco.json"),
        clip_paths={},
        confidence=0.25,
        box_modes=e2e.BOX_MODES,
        window_px=512,
        image_size=1024,
        merge_radius=10.0,
        device=torch.device("cpu"),
    )

    assert loaded == [Path("dec.pt")]
    assert modes == ["fixed", "detector"]
    assert result["candidates"] == 2
    assert result["candidates_neutralized_by_ignore"] == 1
    assert result["config"]["detector_weights"] == "det.pt"
    assert result["config"]["decoder_checkpoint"] == "dec.pt"
    assert result["config"]["device"] == "cpu"
    assert result["config"]["sheets"] == ["S-1", "S-2"]
    assert list(result["arms"]) == ["fixed", "detector"]
    assert result["arms"]["fixed"]["pooled"]["found"]["n"] == 1

    report = e2e.format_report({"foldA": result})
    assert "| foldA | pooled | `fixed` |" in report
    assert "| foldA | S-2 | `detector` |" in report


def test_main_refuses_an_existing_output_directory_before_any_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "end_to_end.json").write_text("published", encoding="utf-8")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("inference ran against an existing output")

    monkeypatch.setattr(e2e, "predict_windows", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mapsam_end_to_end",
            "--dataset", str(tmp_path),
            "--detector-run", str(tmp_path),
            "--images-root", str(tmp_path),
            "--coco-json", str(tmp_path / "coco.json"),
            "--output-dir", str(out),
            "--device", "cpu",
        ],
    )  # fmt: skip

    with pytest.raises(FileExistsError, match="written once"):
        e2e.main()
    assert (out / "end_to_end.json").read_text(encoding="utf-8") == "published"
