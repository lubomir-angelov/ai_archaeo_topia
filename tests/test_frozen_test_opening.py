"""Tests for the opening runner, with inference and segmentation replaced."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from archeo_topia.analysis import frozen_test_opening as opening
from archeo_topia.analysis.detection_metrics import Detection
from archeo_topia.analysis.frozen_manifest import write_manifest
from archeo_topia.datasets.build_detection_windows import FOLDS
from archeo_topia.training import train_mound_detector as tmd
from tests.frozen_fixtures import World, make_world

DETECTIONS = [
    Detection("S-1_1.png", 212, 211, 0.9, box=(200, 200, 224, 222)),
    Detection("S-2_1.png", 412, 411, 0.8, box=(400, 400, 424, 422)),
]


def _block() -> dict[str, Any]:
    return {
        "scorable_annotations": 1,
        "iou_0.5": {"rate": 1.0},
        "iou_0.75": {"rate": 1.0},
        "mean_iou": 1.0,
        "missed": {"rate": 0.0},
        "false_masks": 0,
    }


class Spies:
    def __init__(self) -> None:
        self.predicted: list[Path] = []
        self.pairs: list[tuple[Path, Path]] = []


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> Spies:
    spies = Spies()

    def predict(weights, evaluation_set, imgsz, min_confidence, device):
        spies.predicted.append(weights)
        return list(DETECTIONS)

    def run_pair(**kwargs: Any) -> dict[str, Any]:
        spies.pairs.append((kwargs["detector_weights"], kwargs["decoder_checkpoint"]))
        assert kwargs["box_modes"] == opening.BOX_MODES
        assert kwargs["confidence"] == tmd.HEADLINE_CONFIDENCE
        assert set(kwargs["clip_paths"]) == {"S-1_1.png", "S-2_1.png"}
        return {"arms": {"fixed": {"pooled": _block(), "per_sheet": {"S-1": _block()}}}}

    monkeypatch.setattr(tmd, "predict_windows", predict)
    monkeypatch.setattr(opening, "run_pair", run_pair)
    return spies


@pytest.fixture
def world(tmp_path: Path) -> tuple[World, Path]:
    built = make_world(tmp_path)
    write_manifest(built.sources, built.output)
    return built, built.output


def _open(world_and_manifest: tuple[World, Path], output_dir: Path, device: str = "cpu") -> Any:
    built, manifest = world_and_manifest
    return opening.run_opening(
        "v0_6_test_baseline",
        manifest,
        built.sources.lake_root,
        device,
        output_dir,
        repo_root=built.sources.repo_root,
    )


def test_baseline_pairs_each_detector_with_the_decoder_of_its_fold(
    world: tuple[World, Path], spies: Spies
) -> None:
    record = _open(world, world[0].sources.repo_root.parent / "out")
    repo = world[0].sources.repo_root

    assert [p["fold"] for p in record["pairs"]] == list(FOLDS)
    assert spies.pairs == [
        (
            repo / f"runs/det/{fold}/weights/last.pt",
            repo / f"runs/mapsam/pre_{fold}_pw20/checkpoints/final.pt",
        )
        for fold in FOLDS
    ]


def test_outputs_have_the_documented_layout(world: tuple[World, Path], spies: Spies) -> None:
    out = world[0].sources.repo_root.parent / "out"
    record = _open(world, out)

    assert sorted(p.name for p in out.iterdir()) == ["detector", "end_to_end", "run_record.json"]
    for fold in FOLDS:
        assert {p.name for p in (out / "detector" / fold).iterdir()} == {
            "metrics.json",
            "report.md",
            "detections.jsonl",
        }
        assert (out / "end_to_end" / fold / "end_to_end.json").is_file()
        assert (out / "end_to_end" / fold / "end_to_end.md").is_file()
    written = json.loads((out / "run_record.json").read_text(encoding="utf-8"))
    assert written == record
    assert written["environment"]["device"] == "cpu"
    assert written["manifest"]["sha256"] == opening.sha256_file(world[1])
    assert written["started_utc"] <= written["finished_utc"]


def test_across_checkpoints_holds_each_fold_and_their_mean_min_max(
    world: tuple[World, Path], spies: Spies
) -> None:
    out = world[0].sources.repo_root.parent / "out"
    _open(world, out)

    across = json.loads((out / "detector" / "across_checkpoints.json").read_text(encoding="utf-8"))
    assert list(across) == ["0.05", "0.10", "0.25", "0.50", "0.70"]
    recall = across["0.25"]["recall_5px"]
    assert list(recall["by_fold"]) == list(FOLDS)
    values = list(recall["by_fold"].values())
    assert recall["mean"] == pytest.approx(sum(values) / len(values))
    assert (recall["min"], recall["max"]) == (min(values), max(values))
    assert not math.isnan(recall["mean"])


def test_a_manifest_mismatch_stops_before_the_output_directory_exists(
    world: tuple[World, Path], spies: Spies
) -> None:
    built, _ = world
    (built.sources.repo_root / "sam.pth").write_bytes(b"changed")
    out = built.sources.repo_root.parent / "out"

    with pytest.raises(ValueError, match="sam_base"):
        _open(world, out)
    assert not out.exists()
    assert spies.predicted == []


def test_an_existing_output_directory_is_refused(world: tuple[World, Path], spies: Spies) -> None:
    out = world[0].sources.repo_root.parent / "out"
    out.mkdir()

    with pytest.raises(FileExistsError, match="never overwrites"):
        _open(world, out)
    assert list(out.iterdir()) == []
    assert spies.predicted == []


def test_an_unavailable_device_stops_the_run(
    world: tuple[World, Path], spies: Spies, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = world[0].sources.repo_root.parent / "out"

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _open(world, out, device="cuda:0")
    assert not out.exists()


def test_a_crash_during_scoring_leaves_no_output_directory_and_keeps_the_staging(
    world: tuple[World, Path], spies: Spies, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def crash_on_second(weights, evaluation_set, imgsz, min_confidence, device):
        calls.append(weights)
        if len(calls) == 2:
            raise RuntimeError("inference crashed")
        return list(DETECTIONS)

    monkeypatch.setattr(tmd, "predict_windows", crash_on_second)
    parent = world[0].sources.repo_root.parent
    out = parent / "out"

    with pytest.raises(RuntimeError, match="inference crashed"):
        _open(world, out)
    assert not out.exists()
    (staging,) = parent.glob(".out.staging-*")
    assert not (staging / "run_record.json").exists()


def test_success_promotes_the_staging_leaving_nothing_behind(
    world: tuple[World, Path], spies: Spies
) -> None:
    parent = world[0].sources.repo_root.parent
    _open(world, parent / "out")

    assert list(parent.glob(".out.staging-*")) == []
    assert (parent / "out" / "run_record.json").is_file()


def test_the_environment_is_recorded_before_any_inference(
    world: tuple[World, Path], spies: Spies, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real_environment = opening.environment_record

    def environment(device):
        order.append("environment")
        return real_environment(device)

    def evaluate(**kwargs):
        order.append("evaluate")
        raise RuntimeError("stop after the first call")

    monkeypatch.setattr(opening, "environment_record", environment)
    monkeypatch.setattr(opening, "evaluate_checkpoint", evaluate)

    with pytest.raises(RuntimeError, match="stop after"):
        _open(world, world[0].sources.repo_root.parent / "out")
    assert order == ["environment", "evaluate"]


def test_pairs_for_stops_when_detectors_and_decoders_cover_different_folds() -> None:
    manifest = {
        "entries": [
            {"role": "detector", "fold": "foldA", "path": "a"},
            {"role": "decoder", "fold": "foldB", "path": "b"},
        ]
    }
    with pytest.raises(ValueError, match="differ"):
        opening.pairs_for("v0_6_test_baseline", manifest)
    with pytest.raises(ValueError, match="unknown opening"):
        opening.pairs_for("v0_9_other", manifest)  # type: ignore[arg-type]
