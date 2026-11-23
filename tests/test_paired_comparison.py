"""Tests for the paired comparison of two result files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from archeo_topia.analysis import paired_comparison as pc


def _metrics(distances: dict[str, float | None]) -> dict[str, Any]:
    return {"by_confidence": {"0.25": {"pooled": {"per_annotation_distance_px": distances}}}}


def _pair(ious: dict[str, float]) -> dict[str, Any]:
    return {"arms": {"fixed": {"pooled": {"per_annotation_iou": ious}}}}


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_detector_hit_is_a_match_within_the_primary_radius() -> None:
    hits = pc.detector_hits(_metrics({"a": 4.9, "b": 5.0, "c": 5.1, "d": None}))
    assert hits == {"a": True, "b": True, "c": False, "d": False}


def test_end_to_end_hit_is_an_iou_at_the_threshold() -> None:
    pair = _pair({"a": 0.5, "b": 0.49, "c": 0.75})
    assert pc.end_to_end_hits(pair, "fixed", "iou_0.5") == {"a": True, "b": False, "c": True}
    assert pc.end_to_end_hits(pair, "fixed", "iou_0.75") == {"a": False, "b": False, "c": True}
    with pytest.raises(ValueError, match="unknown arm"):
        pc.end_to_end_hits(pair, "other", "iou_0.5")
    with pytest.raises(ValueError, match="unknown metric"):
        pc.end_to_end_hits(pair, "fixed", "iou_0.9")


def test_compare_detector_files(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.json", _metrics({"x": 1.0, "y": None, "z": 2.0}))
    second = _write(tmp_path / "b.json", _metrics({"x": 9.0, "y": None, "z": 2.0}))

    result = pc.compare("detector", first, second, None, None)

    assert result["first_only"] == 1
    assert result["second_only"] == 0
    assert result["both"] == 1
    assert result["neither"] == 1
    assert result["p_value_exact_two_sided"] == 1.0
    assert result["confidence"] == "0.25"


def test_compare_end_to_end_files_and_mismatched_annotations(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.json", _pair({"x": 0.9, "y": 0.1}))
    second = _write(tmp_path / "b.json", _pair({"x": 0.1, "y": 0.1}))
    result = pc.compare("end_to_end", first, second, "fixed", "iou_0.5")
    assert (result["first_only"], result["second_only"], result["n"]) == (1, 0, 2)

    other = _write(tmp_path / "c.json", _pair({"x": 0.9, "w": 0.1}))
    with pytest.raises(ValueError, match="identical annotations"):
        pc.compare("end_to_end", first, other, "fixed", "iou_0.5")
    with pytest.raises(ValueError, match="needs an arm and a metric"):
        pc.compare("end_to_end", first, second, None, None)


def test_cli_writes_the_result_and_refuses_an_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _write(tmp_path / "a.json", _metrics({"x": 1.0}))
    second = _write(tmp_path / "b.json", _metrics({"x": 1.0}))
    output = tmp_path / "paired.json"
    argv = ["paired", "detector", "--first", str(first), "--second", str(second)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output)])

    pc.main()
    assert json.loads(output.read_text(encoding="utf-8"))["n"] == 1
    assert "p_value_exact_two_sided" in capsys.readouterr().out

    with pytest.raises(FileExistsError):
        pc.main()
