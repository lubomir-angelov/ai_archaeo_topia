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
    hits = pc.detector_hits(_metrics({"a": 4.9, "b": 5.0, "c": 5.1, "d": None}), 0.25)
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


# ---------------------------------------------------------------------------
# The refit claim and the comparison of the two openings
# ---------------------------------------------------------------------------

N = 100
FOLDS = ("foldA", "foldB", "foldC")


def _system(hits: int, *, low_hits: int | None = None) -> dict[str, Any]:
    """A detector ``metrics.json`` that hits annotations ``a0 .. a{hits-1}`` of ``N``."""

    def pooled(count: int) -> dict[str, Any]:
        return {
            "per_annotation_distance_px": {f"a{i}": 1.0 if i < count else 99.0 for i in range(N)},
            "recall": {"n": N, "5": {"hits": count, "recall": count / N}},
        }

    return {
        "by_confidence": {
            "0.25": {"pooled": pooled(hits)},
            "0.05": {"pooled": pooled(hits if low_hits is None else low_hits)},
        }
    }


def _claim(baseline_hits: tuple[int, int, int], refit_hits: int) -> dict[str, Any]:
    baseline = {fold: _system(h) for fold, h in zip(FOLDS, baseline_hits, strict=True)}
    return pc.refit_claim(baseline, _system(refit_hits))


@pytest.mark.parametrize(
    ("baseline_hits", "refit_hits", "case"),
    [
        # R is above every checkpoint, significant against all, and the margin 8 >= S 2.
        ((40, 41, 42), 50, "improves"),
        # The margin equals the spread exactly (46 -> 52, S = 46 - 40 = 6): still improves.
        ((40, 41, 46), 52, "improves"),
        # One paired p-value is 0.0625: no claim although the margin is large enough.
        ((40, 41, 42), 47, "no_detectable_difference"),
        # Significant against all three, but the margin 10 is below the spread 20.
        ((40, 41, 60), 70, "no_detectable_difference"),
        # Inside the baseline range.
        ((40, 41, 60), 50, "no_detectable_difference"),
        ((60, 61, 62), 50, "lowers"),
        # Below every checkpoint, but not significant against the nearest one.
        ((60, 61, 62), 58, "no_detectable_difference"),
    ],
)
def test_the_claim_follows_the_pre_registered_rule(
    baseline_hits: tuple[int, int, int], refit_hits: int, case: str
) -> None:
    claim = _claim(baseline_hits, refit_hits)

    assert claim["case"] == case
    assert claim["wording"] == pc.CLAIM_WORDING[case]
    assert claim["baseline_max_hits"] - claim["baseline_min_hits"] == claim["spread_hits"]
    assert claim["refit"]["hits"] == refit_hits
    assert set(claim["discordance"]) == set(FOLDS)


def test_the_claim_stops_when_the_systems_scored_different_annotation_counts() -> None:
    refit = _system(50)
    refit["by_confidence"]["0.25"]["pooled"]["recall"]["n"] = N + 1

    with pytest.raises(ValueError, match="different annotation counts"):
        pc.refit_claim({fold: _system(40) for fold in FOLDS}, refit)


def test_detector_hits_read_the_requested_operating_point() -> None:
    metrics = _system(10, low_hits=20)

    assert sum(pc.detector_hits(metrics, 0.25).values()) == 10
    assert sum(pc.detector_hits(metrics, 0.05).values()) == 20


def _opening_dir(
    root: Path,
    opening: str,
    manifest: dict[str, str],
    labels: tuple[str, ...],
    hits: tuple[int, ...],
    folds: tuple[str, ...] = FOLDS,
) -> Path:
    for label, count in zip(labels, hits, strict=True):
        _write_nested(
            root / "detector" / label / "metrics.json", _system(count, low_hits=count + 1)
        )
    for fold in folds:
        _write_nested(
            root / "end_to_end" / fold / "end_to_end.json",
            {
                "arms": {
                    arm: {"pooled": {"per_annotation_iou": {"x": 0.6, "y": 0.2}}}
                    for arm in pc.BOX_MODES
                }
            },
        )
    record = {
        "opening": opening,
        "manifest": manifest,
        "pairs": [{"fold": fold} for fold in folds],
    }
    _write_nested(root / "run_record.json", record)
    return root


def _write_nested(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _openings(tmp_path: Path, **refit_edits: Any) -> tuple[Path, Path]:
    baseline = _opening_dir(
        tmp_path / "baseline", "v0_6_test_baseline", {"sha256": "b" * 64}, FOLDS, (40, 41, 42)
    )
    refit_args: dict[str, Any] = {
        "opening": "v0_6_refit",
        "manifest": {"sha256": "c" * 64, "base_sha256": "b" * 64},
        "labels": ("refit",),
        "hits": (50,),
        **refit_edits,
    }
    refit = _opening_dir(
        tmp_path / "refit",
        refit_args["opening"],
        refit_args["manifest"],
        refit_args["labels"],
        refit_args["hits"],
        refit_args.get("folds", FOLDS),
    )
    return baseline, refit


def test_compare_openings_reports_the_claim_and_every_descriptive_pairing(
    tmp_path: Path,
) -> None:
    result = pc.compare_openings(*_openings(tmp_path))

    assert result["claim"]["case"] == "improves"
    assert list(result["detector_descriptive"]) == ["0.05"]
    assert set(result["detector_descriptive"]["0.05"]) == set(FOLDS)
    assert set(result["end_to_end"]) == set(FOLDS)
    for fold in FOLDS:
        assert set(result["end_to_end"][fold]) == set(pc.BOX_MODES)
        for arm in pc.BOX_MODES:
            assert set(result["end_to_end"][fold][arm]) == set(pc.IOU_METRICS)
    one = result["end_to_end"]["foldA"][pc.BOX_MODES[0]]["iou_0.5"]
    assert (one["first_only"], one["second_only"], one["both"], one["neither"]) == (0, 0, 1, 1)


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ({"manifest": {"sha256": "c" * 64, "base_sha256": "0" * 64}}, "does not extend"),
        ({"opening": "v0_6_test_baseline"}, "is opening 'v0_6_test_baseline'"),
        ({"folds": ("foldA", "foldB")}, "same folds"),
    ],
)
def test_compare_openings_stops_on_a_broken_chain_swapped_names_or_other_folds(
    tmp_path: Path, edits: dict[str, Any], message: str
) -> None:
    baseline, refit = _openings(tmp_path, **edits)

    with pytest.raises(ValueError, match=message):
        pc.compare_openings(baseline, refit)


def test_compare_openings_stops_when_the_directories_are_swapped(tmp_path: Path) -> None:
    baseline, refit = _openings(tmp_path)

    with pytest.raises(ValueError, match="not a comparable pair"):
        pc.compare_openings(refit, baseline)


def test_cli_openings_writes_the_comparison_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline, refit = _openings(tmp_path)
    output = tmp_path / "paired.json"
    argv = ["paired", "openings", "--baseline", str(baseline), "--refit", str(refit)]
    monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output)])

    pc.main()
    assert json.loads(output.read_text(encoding="utf-8"))["claim"]["case"] == "improves"
    assert "No difference" not in capsys.readouterr().out

    with pytest.raises(FileExistsError):
        pc.main()
