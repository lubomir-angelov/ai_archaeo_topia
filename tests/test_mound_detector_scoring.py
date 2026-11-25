"""Tests for the detector scoring entry points, with inference replaced."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from archeo_topia.analysis.detection_metrics import Detection, load_evaluation_set
from archeo_topia.training import train_mound_detector as tmd
from tests.detection_fixtures import write_dataset
from tests.refit_fixtures import RefitWorld

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
            "loso",
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


# ---------------------------------------------------------------------------
# The refit: training and checkpoint selection
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


def _row(file: str, hits: int, false_positives: int, windows: int = 6) -> dict:
    return {"file": file, "hits": hits, "false_positives": false_positives, "windows": windows}


def test_choose_checkpoint_prefers_hits_then_fewer_false_positives_then_the_later_epoch() -> None:
    assert tmd.choose_checkpoint([_row("a", 1, 0), _row("b", 2, 9), _row("c", 1, 0)]) == "b"
    assert tmd.choose_checkpoint([_row("a", 2, 5), _row("b", 2, 3), _row("c", 2, 4)]) == "b"
    assert tmd.choose_checkpoint([_row("a", 2, 3), _row("b", 2, 3), _row("last.pt", 2, 3)]) == (
        "last.pt"
    )


def test_choose_checkpoint_stops_on_unequal_window_counts_or_no_rows() -> None:
    with pytest.raises(ValueError, match="different window counts"):
        tmd.choose_checkpoint([_row("a", 1, 0, windows=6), _row("b", 1, 0, windows=7)])
    with pytest.raises(ValueError, match="no candidate"):
        tmd.choose_checkpoint([])


class Trained:
    """A built refit dataset and a recorded training run, with training replaced."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.world = RefitWorld(tmp_path)
        self.world.build()
        self.config = self.world.config()
        self.dataset = tmp_path / "refit"
        self.train_dir = tmp_path / "train"
        self.trained: list[dict] = []

        def train_fold(**kwargs):
            self.trained.append(kwargs)
            return tmd.fold_weights_path(kwargs["output_dir"], kwargs["run_name"])

        monkeypatch.setattr(tmd, "train_fold", train_fold)
        tmd.train_refit(self.config, self.dataset, self.train_dir, CPU)

    def write_weights(self, names: tuple[str, ...]) -> None:
        weights_dir = self.train_dir / tmd.REFIT_RUN_NAME / "weights"
        weights_dir.mkdir(parents=True)
        for name in names:
            (weights_dir / name).write_bytes(name.encode())


ALL_WEIGHTS = (
    "epoch0.pt",
    "epoch25.pt",
    "epoch50.pt",
    "epoch75.pt",
    "epoch100.pt",
    "epoch125.pt",
    "last.pt",
    "best.pt",
)

#: V-1_1.png holds mound 4 at (212, 211) and mound 5 at (512, 311).
HIT_4 = Detection("V-1_1.png", 212, 211, 0.9, box=(200, 200, 224, 222))
HIT_5 = Detection("V-1_1.png", 512, 311, 0.9, box=(500, 300, 524, 322))
STRAY = Detection("V-1_1.png", 800, 600, 0.9, box=(790, 590, 810, 610))
DETECTIONS_BY_CANDIDATE = {
    "epoch25": [HIT_4],
    "epoch50": [HIT_4, HIT_5, STRAY],
    "epoch75": [HIT_4, HIT_5],
    "epoch100": [HIT_4, HIT_5],
    "epoch125": [HIT_5],
    "last": [HIT_4, HIT_5, STRAY],
}


def test_train_refit_trains_the_config_on_the_split_named_yaml_and_records_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Trained(tmp_path, monkeypatch)

    (call,) = run.trained
    assert call["data_yaml"] == run.dataset / "train_val.yaml"
    assert call["run_name"] == tmd.REFIT_RUN_NAME
    assert call["imgsz"] == tmd.DETECTOR_IMGSZ
    assert (call["epochs"], call["batch"], call["seed"], call["save_period"]) == (150, 16, 42, 25)
    assert call["model_name"] == "yolo26s.pt"
    assert call["device"] == CPU
    record = json.loads((run.train_dir / "train_record.json").read_text())
    assert record["config"]["sha256"] == run.config.sha256
    assert record["environment"]["device"] == "cpu"


def test_train_refit_stops_before_training_on_a_dataset_that_is_not_the_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Trained(tmp_path / "a", monkeypatch)
    run.trained.clear()
    report_path = run.dataset / "metadata" / "build_report.json"
    report = json.loads(report_path.read_text())

    with pytest.raises(ValueError, match="config sha256 differs"):
        tmd.train_refit(replace(run.config, sha256="0" * 64), run.dataset, tmp_path / "t2", CPU)

    report["listing"]["plan"] = "loso"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not 'train_val'"):
        tmd.train_refit(run.config, run.dataset, tmp_path / "t2", CPU)
    assert not (tmp_path / "t2").exists()
    assert run.trained == []


def test_train_refit_refuses_an_existing_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Trained(tmp_path, monkeypatch)
    run.trained.clear()

    with pytest.raises(FileExistsError):
        tmd.train_refit(run.config, run.dataset, run.train_dir, CPU)
    assert run.trained == []


@pytest.fixture
def scored_weights(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace inference; the list records every weights file it was asked to load."""
    asked: list[Path] = []

    def predict(weights, evaluation_set, imgsz, min_confidence, device):
        asked.append(weights)
        return list(DETECTIONS_BY_CANDIDATE[weights.stem])

    monkeypatch.setattr(tmd, "predict_windows", predict)
    return asked


def test_select_scores_six_candidates_and_never_opens_best_or_epoch0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scored_weights: list[Path]
) -> None:
    run = Trained(tmp_path, monkeypatch)
    run.write_weights(ALL_WEIGHTS)

    selection = tmd.select_checkpoint(run.train_dir, CPU, tmp_path / "selection")

    assert [p.name for p in scored_weights] == [
        "epoch25.pt",
        "epoch50.pt",
        "epoch75.pt",
        "epoch100.pt",
        "epoch125.pt",
        "last.pt",
    ]
    assert selection["ignored"] == ["best.pt", "epoch0.pt"]
    rows = selection["candidates"]
    assert [(r["file"], r["hits"], r["false_positives"]) for r in rows] == [
        ("epoch25.pt", 1, 0),
        ("epoch50.pt", 2, 1),
        ("epoch75.pt", 2, 0),
        ("epoch100.pt", 2, 0),
        ("epoch125.pt", 1, 0),
        ("last.pt", 2, 1),
    ]
    assert [r["epochs_completed"] for r in rows] == [26, 51, 76, 101, 126, 150]
    assert selection["chosen"]["file"] == "epoch100.pt"
    assert selection["chosen"]["sha256"] == rows[3]["sha256"]
    assert selection["sheets"] == ["V-1"]
    assert selection["confidence"] == "0.25"
    assert selection["rule"] == list(tmd.SELECTION_RULE)
    assert json.loads((tmp_path / "selection" / "selection.json").read_text()) == selection
    assert "| epoch100.pt | 101 |" in (tmp_path / "selection" / "selection.md").read_text()
    assert (tmp_path / "selection" / "epoch25" / "metrics.json").exists()


def _selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Trained, dict]:
    run = Trained(tmp_path, monkeypatch)
    run.write_weights(ALL_WEIGHTS)
    return run, tmd.select_checkpoint(run.train_dir, CPU, tmp_path / "selection")


def test_require_selection_accepts_a_real_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scored_weights: list[Path]
) -> None:
    _, selection = _selected(tmp_path, monkeypatch)

    tmd.require_selection(selection, tmd.REPO_ROOT)


def _chosen_best(selection: dict) -> None:
    best = tmd.REPO_ROOT / selection["chosen"]["path"]
    best = best.with_name("best.pt")
    selection["chosen"] = {"file": "best.pt", "path": str(best), "sha256": tmd.sha256_file(best)}


def _drop_candidate(selection: dict) -> None:
    del selection["candidates"][0]


def _edit_hits(selection: dict) -> None:
    selection["candidates"][0]["hits"] += 5


def _other_dataset(selection: dict) -> None:
    selection["dataset"] = {**selection["dataset"], "path": "elsewhere"}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_chosen_best, "rule chooses|does not match its candidate row"),
        (_drop_candidate, "are not the config's"),
        (_edit_hits, "row says"),
        (_other_dataset, "dataset is not the training record's"),
    ],
)
def test_require_selection_refuses_a_tampered_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scored_weights: list[Path],
    edit: Callable[[dict], None],
    message: str,
) -> None:
    _, selection = _selected(tmp_path, monkeypatch)
    edit(selection)

    with pytest.raises(ValueError, match=message):
        tmd.require_selection(selection, tmd.REPO_ROOT)


def test_require_selection_refuses_a_train_record_or_config_edited_after_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scored_weights: list[Path]
) -> None:
    run, selection = _selected(tmp_path, monkeypatch)
    record = run.train_dir / "train_record.json"
    record.write_text(record.read_text() + " ")
    with pytest.raises(ValueError, match="changed since the selection"):
        tmd.require_selection(selection, tmd.REPO_ROOT)

    run.config.path.write_text(run.config.path.read_text() + " ")
    with pytest.raises(ValueError, match="changed since"):
        tmd.require_selection(selection, tmd.REPO_ROOT)


def test_select_stops_before_scoring_when_a_candidate_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scored_weights: list[Path]
) -> None:
    run = Trained(tmp_path, monkeypatch)
    run.write_weights(tuple(w for w in ALL_WEIGHTS if w not in ("epoch75.pt", "last.pt")))

    with pytest.raises(ValueError, match=r"candidates missing .*\['epoch75.pt', 'last.pt'\]"):
        tmd.select_checkpoint(run.train_dir, CPU, tmp_path / "selection")
    assert not (tmp_path / "selection").exists()
    assert scored_weights == []


def test_select_stops_when_the_config_changed_since_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scored_weights: list[Path]
) -> None:
    run = Trained(tmp_path, monkeypatch)
    run.write_weights(ALL_WEIGHTS)
    run.config.path.write_text(run.config.path.read_text() + " ")

    with pytest.raises(ValueError, match="changed since training"):
        tmd.select_checkpoint(run.train_dir, CPU, tmp_path / "selection")
    assert not (tmp_path / "selection").exists()


def test_select_refuses_a_training_directory_without_a_record(
    tmp_path: Path, scored_weights: list[Path]
) -> None:
    (tmp_path / "train").mkdir()

    with pytest.raises(FileNotFoundError, match="train_record.json"):
        tmd.select_checkpoint(tmp_path / "train", CPU, tmp_path / "selection")
