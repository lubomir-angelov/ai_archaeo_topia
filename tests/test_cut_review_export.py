"""Tests for the v0.0.5 cut tool, on synthetic review returns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from archeo_topia.formats import cut_review_export as cut_module
from archeo_topia.formats.cut_review_export import cut, require_cut

CATEGORIES = [
    {"id": 1, "name": "mound"},
    {"id": 2, "name": "hard_negative_symbol"},
    {"id": 3, "name": "uncertain_ignore"},
]
SPLITS = {"S-1": "train", "S-2": "val", "S-3": "test"}


def _annotation(ann_id: int, image_id: int, category_id: int, status: str, area: float) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [[1.5, 2.5, 3.5, 2.5, 3.5, 4.5]],
        "area": area,
        "bbox": [1.123456789012345, 2.5, 25.0, 23.0],
        "iscrowd": 0,
        "attributes": {
            "review_status": status,
            "annotation_provenance": "model_proposal_accepted",
        },
    }


def _return(root: Path, sheet: str, edit: Callable[[dict[str, Any]], None] | None = None) -> None:
    coco: dict[str, Any] = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"description": f"review of {sheet}", "year": ""},
        "categories": CATEGORIES,
        "images": [
            {"id": 1, "file_name": f"{sheet}_1.png", "width": 900, "height": 700},
            {"id": 2, "file_name": f"{sheet}_2.png", "width": 900, "height": 700},
        ],
        "annotations": [
            # Deliberately not width * height: the cut must not recompute it.
            _annotation(1, 1, 1, "confirmed", 123.0),
            _annotation(2, 2, 1, "added", 77.5),
            _annotation(3, 2, 2, "rejected", 10.0),
        ],
    }
    report = {"sheet_id": sheet}
    if edit is not None:
        edit(coco)
    (root / sheet).mkdir(parents=True)
    (root / sheet / "instances_default.json").write_text(json.dumps(coco), encoding="utf-8")
    (root / sheet / "review_report.json").write_text(json.dumps(report), encoding="utf-8")


def _splits(tmp_path: Path, sheets: dict[str, str] | None = None) -> Path:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"sheets": {s: {"split": v} for s, v in (sheets or SPLITS).items()}}),
        encoding="utf-8",
    )
    return path


def _root(tmp_path: Path, sheets: tuple[str, ...] = ("S-1", "S-2", "S-3")) -> Path:
    root = tmp_path / "review"
    root.mkdir()
    for sheet in sheets:
        _return(root, sheet)
    return root


def test_the_merge_renumbers_ids_and_changes_nothing_else(tmp_path: Path) -> None:
    root = _root(tmp_path)
    out = tmp_path / "out"
    provenance = cut(root, out, _splits(tmp_path))

    merged = json.loads((out / "instances_default.json").read_text(encoding="utf-8"))
    assert [i["id"] for i in merged["images"]] == [1, 2, 3, 4]
    assert [a["id"] for a in merged["annotations"]] == [1, 2, 3, 4, 5, 6]
    assert [a["image_id"] for a in merged["annotations"]] == [1, 2, 2, 3, 4, 4]
    assert merged["info"]["description"] == "v0.0.5: merged v0.7 review returns, 2 sheets"

    source = json.loads((root / "S-2" / "instances_default.json").read_text(encoding="utf-8"))
    for original, new in zip(source["annotations"], merged["annotations"][3:], strict=True):
        assert {k: v for k, v in new.items() if k not in ("id", "image_id")} == {
            k: v for k, v in original.items() if k not in ("id", "image_id")
        }
    assert merged["annotations"][4]["area"] == 77.5

    assert provenance["sheets"] == ["S-1", "S-2"]
    assert provenance["sources"]["S-2"]["annotation_id_map"] == {"1": 4, "2": 5, "3": 6}
    assert provenance["sources"]["S-2"]["image_id_map"] == {"1": 3, "2": 4}
    assert provenance["counts_by_category"] == {
        "mound": 4,
        "hard_negative_symbol": 2,
        "uncertain_ignore": 0,
    }
    assert provenance["review_status_by_category"] == {
        "mound": {"confirmed": 2, "added": 2},
        "hard_negative_symbol": {"rejected": 2},
    }
    assert json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8")) == provenance


def test_a_test_split_sheet_is_excluded_but_its_hashes_are_recorded(tmp_path: Path) -> None:
    root = _root(tmp_path)
    provenance = cut(root, tmp_path / "out", _splits(tmp_path))

    assert provenance["excluded_sheets"] == {"S-3": "test_split"}
    assert set(provenance["sources"]) == {"S-1", "S-2", "S-3"}
    assert provenance["sources"]["S-3"]["image_id_map"] == {}
    assert provenance["sources"]["S-3"]["annotation_id_map"] == {}
    merged = json.loads((tmp_path / "out" / "instances_default.json").read_text(encoding="utf-8"))
    assert not any(i["file_name"].startswith("S-3_") for i in merged["images"])


def _stops(tmp_path: Path, message: str, splits: Path | None = None) -> None:
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        cut(tmp_path / "review", out, splits or _splits(tmp_path))
    assert not out.exists()


def test_stops_on_a_category_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "review"
    root.mkdir()
    _return(root, "S-1")
    _return(root, "S-2", lambda coco: coco["categories"].reverse())
    _stops(tmp_path, "categories differ")


def test_stops_on_an_image_of_another_sheet(tmp_path: Path) -> None:
    root = tmp_path / "review"
    root.mkdir()
    _return(root, "S-1", lambda coco: coco["images"][0].update(file_name="S-2_1.png"))
    _stops(tmp_path, "images of another sheet")


def test_stops_on_a_file_name_in_two_sheets(tmp_path: Path) -> None:
    root = tmp_path / "review"
    root.mkdir()
    _return(root, "S-1")
    # sheet_of("S-1_1.png") is S-1, so the foreign-sheet check fires as well.
    _return(root, "S-2", lambda coco: coco["images"][0].update(file_name="S-1_1.png"))
    _stops(tmp_path, "in both S-1 and S-2")


def test_stops_on_an_extra_file_in_a_sheet_directory(tmp_path: Path) -> None:
    root = _root(tmp_path, ("S-1",))
    (root / "S-1" / "notes.txt").write_text("x", encoding="utf-8")
    _stops(tmp_path, "notes.txt")


def test_stops_on_a_non_directory_entry(tmp_path: Path) -> None:
    root = _root(tmp_path, ("S-1",))
    (root / "stray.json").write_text("{}", encoding="utf-8")
    _stops(tmp_path, "stray.json: not a sheet directory")


def test_stops_on_a_report_naming_another_sheet(tmp_path: Path) -> None:
    root = _root(tmp_path, ("S-1",))
    (root / "S-1" / "review_report.json").write_text(json.dumps({"sheet_id": "S-2"}))
    _stops(tmp_path, "names sheet 'S-2'")


def test_stops_on_a_sheet_unknown_to_the_splits(tmp_path: Path) -> None:
    _root(tmp_path, ("S-1", "S-2"))
    _stops(tmp_path, "not a sheet of the splits file", _splits(tmp_path, {"S-1": "train"}))


def test_stops_on_duplicate_ids_and_dangling_images(tmp_path: Path) -> None:
    root = tmp_path / "review"
    root.mkdir()

    def break_ids(coco: dict[str, Any]) -> None:
        coco["images"][1]["id"] = 1
        coco["annotations"][1]["id"] = 1
        coco["annotations"][2]["image_id"] = 99

    _return(root, "S-1", break_ids)
    _stops(tmp_path, "image ids are not unique")


def test_stops_when_every_sheet_is_excluded(tmp_path: Path) -> None:
    _root(tmp_path, ("S-3",))
    _stops(tmp_path, "every sheet .* is excluded")


def test_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        cut(root, tmp_path / "out", _splits(tmp_path))


def test_a_source_changing_during_the_cut_stops_it_and_leaves_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root(tmp_path)
    real = cut_module._source_hashes

    def drifting(review_root: Path, sheets: list[str]) -> dict[str, dict[str, str]]:
        hashes = real(review_root, sheets)
        hashes[sheets[0]]["instances_default_sha256"] = "0" * 64
        return hashes

    monkeypatch.setattr(cut_module, "_source_hashes", drifting)
    _stops(tmp_path, "changed while the cut ran")


def test_a_failed_round_trip_leaves_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root(tmp_path)
    monkeypatch.setattr(cut_module, "_round_trip_problems", lambda *args: ["forced"])
    _stops(tmp_path, "not faithful")


def test_a_missing_review_status_leaves_no_directory(tmp_path: Path) -> None:
    root = tmp_path / "review"
    root.mkdir()
    _return(root, "S-1", lambda coco: coco["annotations"][0]["attributes"].pop("review_status"))
    out = tmp_path / "out"
    with pytest.raises(KeyError):
        cut(root, out, _splits(tmp_path))
    assert not out.exists()


def test_require_cut_accepts_a_real_cut(tmp_path: Path) -> None:
    cut(_root(tmp_path), tmp_path / "out", _splits(tmp_path))
    coco = tmp_path / "out" / "instances_default.json"
    require_cut(coco, hashlib.sha256(coco.read_bytes()).hexdigest())


def test_require_cut_refuses_a_directory_without_provenance(tmp_path: Path) -> None:
    cut(_root(tmp_path), tmp_path / "out", _splits(tmp_path))
    coco = tmp_path / "out" / "instances_default.json"
    (tmp_path / "out" / "PROVENANCE.json").unlink()
    with pytest.raises(FileNotFoundError, match="PROVENANCE.json"):
        require_cut(coco, hashlib.sha256(coco.read_bytes()).hexdigest())


def test_require_cut_refuses_a_file_the_cut_did_not_write(tmp_path: Path) -> None:
    cut(_root(tmp_path), tmp_path / "out", _splits(tmp_path))
    coco = tmp_path / "out" / "instances_default.json"
    with pytest.raises(ValueError, match="not the file the cut wrote"):
        require_cut(coco, "0" * 64)


def test_require_cut_refuses_another_export_name(tmp_path: Path) -> None:
    cut(_root(tmp_path), tmp_path / "out", _splits(tmp_path))
    coco = tmp_path / "out" / "instances_default.json"
    provenance_path = tmp_path / "out" / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["export"] = "v0.0.3"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="records export 'v0.0.3'"):
        require_cut(coco, hashlib.sha256(coco.read_bytes()).hexdigest())
