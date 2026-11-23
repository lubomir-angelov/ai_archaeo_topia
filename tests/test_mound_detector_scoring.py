"""Tests for the detector scoring entry points, with inference replaced."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from archeo_topia.analysis.detection_metrics import Detection, load_evaluation_set
from archeo_topia.training import train_mound_detector as tmd
from tests.detection_fixtures import write_dataset

DETECTIONS = [
    Detection("S-1_1.png", 100, 100, 0.9, box=(90, 90, 110, 110)),
    Detection("S-2_1.png", 200, 200, 0.3),
    Detection("S-1_1.png", 800, 400, 0.1),
]


@pytest.fixture
def evaluation_set(tmp_path: Path):
    dataset = write_dataset(tmp_path / "ds")
    return load_evaluation_set(dataset, dataset / "all.txt", ["S-1", "S-2"])


@pytest.fixture
def fake_inference(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``predict_windows``; the list records the confidences it was asked for."""
    asked: list[float] = []

    def predict(weights, evaluation_set, imgsz, min_confidence, device):
        asked.append(min_confidence)
        images = {tmd.parse_window(window_id)[0] for window_id in evaluation_set.window_ids}
        return [d for d in DETECTIONS if d.score >= min_confidence and d.image in images]

    monkeypatch.setattr(tmd, "predict_windows", predict)
    return asked


def test_headline_and_minimum_confidence_are_operating_points() -> None:
    assert tmd.HEADLINE_CONFIDENCE in tmd.THRESHOLDS
    assert tmd.MIN_CONFIDENCE in tmd.THRESHOLDS


def test_evaluate_checkpoint_writes_the_record(
    tmp_path: Path, evaluation_set, fake_inference: list[float]
) -> None:
    raw, metrics = tmd.evaluate_checkpoint(
        weights=Path("w.pt"),
        evaluation_set=evaluation_set,
        imgsz=512,
        merge_radius=10.0,
        thresholds=tmd.THRESHOLDS,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        label="foldA",
    )

    label_dir = tmp_path / "run" / "foldA"
    written = json.loads((label_dir / "metrics.json").read_text())
    assert written.keys() == {"config", "by_confidence"}
    assert list(written["by_confidence"]) == ["0.05", "0.10", "0.25", "0.50", "0.70"]
    assert written["config"]["headline_confidence"] == 0.25
    assert written["config"]["device"] == "cpu"
    assert written["config"]["listing"] == "all.txt"
    assert written["config"]["sheets"] == ["S-1", "S-2"]
    assert metrics.keys() == written.keys()

    # One inference, from the lowest confidence, serves every threshold.
    assert fake_inference == [tmd.MIN_CONFIDENCE]
    assert len(raw) == 3

    lines = [
        json.loads(line) for line in (label_dir / "detections.jsonl").read_text().splitlines()
    ]
    assert [(d["image"], d["score"]) for d in lines] == [(d.image, d.score) for d in DETECTIONS]
    assert lines[0]["box"] == [90, 90, 110, 110]
    assert lines[1]["box"] is None

    report = (label_dir / "report.md").read_text()
    assert "confidence 0.25" in report
    assert "understate the uncertainty" in report
    assert "### Operating curve" in report


def test_evaluate_checkpoint_never_overwrites_a_record(
    tmp_path: Path, evaluation_set, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "run" / "foldA"
    existing.mkdir(parents=True)
    (existing / "metrics.json").write_text("published")

    def forbidden(*args, **kwargs):
        raise AssertionError("inference ran against an existing record")

    monkeypatch.setattr(tmd, "predict_windows", forbidden)
    with pytest.raises(FileExistsError, match="never overwrites"):
        tmd.evaluate_checkpoint(
            Path("w.pt"),
            evaluation_set,
            512,
            10.0,
            tmd.THRESHOLDS,
            torch.device("cpu"),
            tmp_path / "run",
            "foldA",
        )
    assert (existing / "metrics.json").read_text() == "published"


def test_evaluate_checkpoint_rejects_a_threshold_below_the_inference_floor(
    tmp_path: Path, evaluation_set, fake_inference: list[float]
) -> None:
    with pytest.raises(ValueError, match="below MIN_CONFIDENCE"):
        tmd.evaluate_checkpoint(
            Path("w.pt"),
            evaluation_set,
            512,
            10.0,
            (0.01, 0.25),
            torch.device("cpu"),
            tmp_path / "run",
            "x",
        )
    assert fake_inference == []


def test_operating_curve_is_read_from_by_confidence(
    tmp_path: Path, evaluation_set, fake_inference: list[float]
) -> None:
    _, metrics = tmd.evaluate_checkpoint(
        Path("w.pt"),
        evaluation_set,
        512,
        10.0,
        tmd.THRESHOLDS,
        torch.device("cpu"),
        tmp_path / "run",
        "x",
    )
    curve = tmd.operating_curve(metrics)

    assert [row["confidence"] for row in curve] == list(tmd.THRESHOLDS)
    pooled = metrics["by_confidence"]["0.25"]["pooled"]
    row = next(r for r in curve if r["confidence"] == 0.25)
    assert row["recall_5px"] == pooled["recall"]["5"]["recall"]
    assert row["false_positives"] == pooled["false_positives"]["count"]
    assert "| 0.25 |" in tmd.curve_table(curve)


def test_loso_main_scores_the_fold_val_listing_of_the_fold_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_inference: list[float]
) -> None:
    dataset = write_dataset(tmp_path / "ds")
    (dataset / "foldA_val.txt").write_text(
        "./images/S-1_1_x00000_y00000.png\n./images/S-1_1_x00384_y00000.png\n"
    )
    monkeypatch.setattr(tmd, "FOLDS", {"foldA": "S-1"})
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_mound_detector",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(out),
            "--skip-train",
            "--device",
            "cpu",
        ],
    )

    tmd.main()

    metrics = json.loads((out / "foldA" / "metrics.json").read_text())
    assert metrics["config"]["sheets"] == ["S-1"]
    assert metrics["config"]["listing"] == "foldA_val.txt"
    assert metrics["by_confidence"]["0.25"]["pooled"]["annotations"] == 1
    summary = json.loads((out / "summary.json").read_text())
    assert [p["confidence"] for p in summary["foldA"]["operating_points"]] == list(tmd.THRESHOLDS)
    assert set(summary["foldA"]) == {"operating_points"}


def test_predict_windows_reads_the_evaluation_set_in_listing_order(
    tmp_path: Path, evaluation_set, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths come from the evaluation set's window ids, so batch order is the listing's."""
    seen: list[list[str]] = []

    class FakeResult:
        boxes = None

    class FakeYolo:
        def __init__(self, weights: str) -> None:
            pass

        def predict(self, batch: list[str], **kwargs: object) -> list[FakeResult]:
            seen.append(batch)
            return [FakeResult() for _ in batch]

    monkeypatch.setattr("ultralytics.YOLO", FakeYolo)
    tmd.predict_windows(Path("w.pt"), evaluation_set, 512, 0.05, torch.device("cpu"))

    expected = [
        str(tmd.crop_path(evaluation_set.dataset, window_id).resolve())
        for window_id in evaluation_set.window_ids
    ]
    assert seen == [expected]


def test_write_detections_refuses_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "detections.jsonl"
    path.write_text("published")
    with pytest.raises(FileExistsError, match="written once"):
        tmd.write_detections(path, DETECTIONS)
    assert path.read_text() == "published"
