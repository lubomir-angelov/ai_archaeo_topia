"""Tests for the frozen-input manifest, on synthetic files only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from archeo_topia.analysis.frozen_manifest import (
    extend_manifest,
    manifest_text,
    parse_source_hash,
    verify_manifest,
    write_manifest,
)
from archeo_topia.datasets.build_detection_windows import (
    FOLDS,
    build_output_problems,
    sha256_file,
)
from tests.frozen_fixtures import (
    CHOSEN,
    DATASET,
    SHEETS,
    World,
    hash_row,
    make_world,
    write_selection,
)


def test_parse_source_hash_reads_the_one_row(tmp_path: Path) -> None:
    source = tmp_path / "SOURCE.md"
    source.write_text(hash_row("a" * 64), encoding="utf-8")
    assert parse_source_hash(source) == "a" * 64


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("| Item | Value |\n|---|---|\n| COCO zip sha256 | `ab` |\n", "has 0 rows"),
        (hash_row("a" * 64) + hash_row("b" * 64), "has 2 rows"),
        (hash_row("not-a-digest"), "not a sha256 digest"),
    ],
)
def test_parse_source_hash_stops_unless_exactly_one_digest(
    tmp_path: Path, text: str, message: str
) -> None:
    source = tmp_path / "SOURCE.md"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parse_source_hash(source)


def test_write_pins_every_input_once(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    manifest = write_manifest(world.sources, world.output)

    roles = [e["role"] for e in manifest["entries"]]
    expected = {
        "ground_truth": 1,
        "clips_json": len(SHEETS),
        "clip": len(SHEETS),
        "detector": len(FOLDS),
        "decoder": len(FOLDS),
        "sam_base": 1,
        "windows_build_report": 1,
    }
    assert {role: roles.count(role) for role in expected} == expected
    assert len(roles) == sum(expected.values())
    assert json.loads(world.output.read_text(encoding="utf-8")) == manifest
    for entry in manifest["entries"]:
        assert entry["root"] in ("repo", "lake")
        assert not Path(entry["path"]).is_absolute()
    clip = next(e for e in manifest["entries"] if e["role"] == "clip")
    assert (clip["root"], clip["file"]) == ("lake", f"{clip['sheet']}_1.png")


def test_write_stops_on_a_source_md_hash_mismatch_before_creating_the_output(
    tmp_path: Path,
) -> None:
    world = make_world(tmp_path)
    (world.sources.repo_root / "SOURCE.md").write_text(hash_row("0" * 64), encoding="utf-8")

    with pytest.raises(ValueError, match="records " + "0" * 64):
        write_manifest(world.sources, world.output)
    assert not world.output.exists()


def test_write_refuses_an_existing_output(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    world.output.write_text("kept", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_manifest(world.sources, world.output)
    assert world.output.read_text(encoding="utf-8") == "kept"


def test_write_stops_when_the_dataset_was_built_from_another_ground_truth(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    other = replace(world.sources, ground_truth=Path("annotation/frozen/other.json"))
    (other.repo_root / other.ground_truth).write_bytes(b"{}")
    (other.repo_root / "SOURCE.md").write_text(
        hash_row(sha256_file(other.repo_root / other.ground_truth)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not the ground truth"):
        write_manifest(other, world.output)
    assert not world.output.exists()


def test_write_stops_when_the_dataset_is_not_a_test_build(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    report_path = world.sources.repo_root / DATASET / "metadata" / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["listing"]["plan"] = "loso"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="not 'test'"):
        write_manifest(world.sources, world.output)


def test_write_stops_on_absolute_repo_paths(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    absolute = replace(world.sources, sam_checkpoint=tmp_path / "sam.pth")
    with pytest.raises(ValueError, match="must be relative"):
        write_manifest(absolute, world.output)


def test_verify_returns_the_manifest_when_nothing_changed(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    written = write_manifest(world.sources, world.output)

    assert (
        verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root) == written
    )


def test_verify_reports_every_tampered_entry_at_once(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    write_manifest(world.sources, world.output)
    repo, lake = world.sources.repo_root, world.sources.lake_root
    (repo / "sam.pth").write_bytes(b"changed")
    (repo / "runs/det/foldA/weights/last.pt").unlink()
    (lake / "clips/S-1/S-1_1.png").write_bytes(b"changed")

    with pytest.raises(ValueError) as raised:
        verify_manifest(world.output, lake, repo)

    message = str(raised.value)
    assert "sam_base" in message
    assert "detector" in message and "is missing" in message
    assert "clip" in message and "S-1_1.png" in message


def test_verify_checks_the_build_reports_output_hashes(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    write_manifest(world.sources, world.output)
    dataset = world.sources.repo_root / DATASET
    assert build_output_problems(dataset) == []

    image = next((dataset / "images").glob("*.png"))
    Image.new("RGBA", (512, 512), (1, 2, 3, 255)).save(image)

    assert build_output_problems(dataset) == [
        f"{dataset}/images: crops digest differs from build_report.json"
    ]
    with pytest.raises(ValueError, match="crops digest differs"):
        verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root)


def test_verify_rejects_a_malformed_manifest(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    manifest: dict[str, Any] = write_manifest(world.sources, world.output)
    manifest["entries"][0]["root"] = "elsewhere"
    manifest["entries"][1]["path"] = "/etc/passwd"
    manifest["entries"].pop(-1)
    world.output.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root)

    message = str(raised.value)
    assert "unknown root" in message
    assert "not a relative path" in message
    assert "windows_build_report" in message


def test_write_stops_when_the_build_saw_other_pixels_than_the_pinned_clip(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    Image.new("RGBA", (900, 700), (1, 2, 3, 255)).save(
        world.sources.lake_root / "clips/S-1/S-1_1.png"
    )

    with pytest.raises(ValueError, match=r"source image hashes differ.*S-1_1\.png"):
        write_manifest(world.sources, world.output)
    assert not world.output.exists()


def test_write_stops_when_the_build_validated_against_another_clips_json(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    clips_json = world.sources.lake_root / "clips/S-1/clips.json"
    document = json.loads(clips_json.read_text(encoding="utf-8"))
    document["parent_size"] = [901, 700]
    clips_json.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"clips\.json hashes differ.*S-1"):
        write_manifest(world.sources, world.output)
    assert not world.output.exists()


def test_a_missing_key_is_reported_as_missing_not_as_an_empty_value(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    manifest: dict[str, Any] = write_manifest(world.sources, world.output)
    del manifest["entries"][0]["sha256"]
    world.output.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root)

    assert "entry 0: missing keys ['sha256']" in str(raised.value)
    assert "is not a digest" not in str(raised.value)


# ---------------------------------------------------------------------------
# extend: the refit detector joins the manifest, and the base stays provable
# ---------------------------------------------------------------------------


def _extend(world: World, selection: Path) -> dict[str, Any]:
    return extend_manifest(
        world.output, world.sources.lake_root, world.sources.repo_root, selection
    )


@pytest.fixture
def written(tmp_path: Path) -> tuple[World, dict[str, Any]]:
    world = make_world(tmp_path)
    return world, write_manifest(world.sources, world.output)


def test_extend_appends_two_entries_and_pins_the_base_by_hash(
    written: tuple[World, dict[str, Any]],
) -> None:
    world, base = written
    base_text = world.output.read_text(encoding="utf-8")
    extended = _extend(world, write_selection(world))

    assert extended["entries"][: len(base["entries"])] == base["entries"]
    assert [e["role"] for e in extended["entries"][len(base["entries"]) :]] == [
        "refit_detector",
        "refit_selection",
    ]
    assert extended["entries"][-2]["checkpoint"] == "epoch100.pt"
    assert extended["base"] == {
        "entries": len(base["entries"]),
        "sha256": hashlib.sha256(base_text.encode()).hexdigest(),
    }
    assert world.output.read_text(encoding="utf-8") == manifest_text(extended)
    assert verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root) == (
        extended
    )
    assert not world.output.with_name("manifest.json.extending").exists()


def test_a_base_entry_edited_after_the_extension_fails_the_base_proof(
    written: tuple[World, dict[str, Any]],
) -> None:
    world, _ = written
    extended = _extend(world, write_selection(world))
    extended["entries"][0]["path"] = "annotation/frozen/other.json"
    world.output.write_text(manifest_text(extended), encoding="utf-8")

    with pytest.raises(ValueError, match="do not hash to the recorded base sha256"):
        verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root)


def test_a_manifest_is_extended_once(written: tuple[World, dict[str, Any]]) -> None:
    world, _ = written
    selection = write_selection(world)
    _extend(world, selection)
    after_first = world.output.read_bytes()

    with pytest.raises(ValueError, match="already extended"):
        _extend(world, selection)
    assert world.output.read_bytes() == after_first


def test_extend_refuses_a_non_canonical_base(written: tuple[World, dict[str, Any]]) -> None:
    world, base = written
    world.output.write_text(json.dumps(base, indent=4) + "\n", encoding="utf-8")
    before = world.output.read_bytes()

    with pytest.raises(ValueError, match="not in canonical form"):
        _extend(world, write_selection(world))
    assert world.output.read_bytes() == before


def test_extend_refuses_a_selection_whose_hash_or_constants_differ(
    written: tuple[World, dict[str, Any]],
) -> None:
    world, _ = written
    before = world.output.read_bytes()
    bad_hash = write_selection(
        world, chosen={"file": "epoch100.pt", "path": CHOSEN.as_posix(), "sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="no longer hashes"):
        _extend(world, bad_hash)

    with pytest.raises(ValueError, match="imgsz 640"):
        _extend(world, write_selection(world, imgsz=640))
    assert world.output.read_bytes() == before
    assert not world.output.with_name("manifest.json.extending").exists()


def test_extend_refuses_a_selection_whose_chosen_is_not_what_the_rule_yields(
    written: tuple[World, dict[str, Any]],
) -> None:
    world, _ = written
    before = world.output.read_bytes()
    repo = world.sources.repo_root
    other = write_selection(world)
    document = json.loads((repo / other).read_text(encoding="utf-8"))
    row = next(r for r in document["candidates"] if r["file"] == "last.pt")
    document["chosen"] = {key: row[key] for key in ("file", "path", "sha256")}
    (repo / other).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="the rule chooses epoch100.pt, not last.pt"):
        _extend(world, other)
    assert world.output.read_bytes() == before
    assert not world.output.with_name("manifest.json.extending").exists()


def test_extend_stops_on_a_tampered_base_entry_and_leaves_the_file_alone(
    written: tuple[World, dict[str, Any]],
) -> None:
    world, _ = written
    selection = write_selection(world)
    (world.sources.repo_root / "sam.pth").write_bytes(b"changed")
    before = world.output.read_bytes()

    with pytest.raises(ValueError, match="sam_base"):
        _extend(world, selection)
    assert world.output.read_bytes() == before
    assert not world.output.with_name("manifest.json.extending").exists()


def test_a_leftover_staging_file_stops_the_extension(
    written: tuple[World, dict[str, Any]],
) -> None:
    world, _ = written
    staged = world.output.with_name("manifest.json.extending")
    staged.write_text("left over", encoding="utf-8")
    before = world.output.read_bytes()

    with pytest.raises(FileExistsError):
        _extend(world, write_selection(world))
    assert staged.read_text(encoding="utf-8") == "left over"
    assert world.output.read_bytes() == before


def test_a_refit_role_without_a_base_is_malformed(written: tuple[World, dict[str, Any]]) -> None:
    world, base = written
    base["entries"].append({**base["entries"][0], "role": "refit_detector"})
    world.output.write_text(manifest_text(base), encoding="utf-8")

    with pytest.raises(ValueError, match="needs a base"):
        verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root)


def test_verify_covers_the_refit_entries(written: tuple[World, dict[str, Any]]) -> None:
    world, _ = written
    _extend(world, write_selection(world))
    (world.sources.repo_root / CHOSEN).write_bytes(b"swapped")

    with pytest.raises(ValueError, match="refit_detector"):
        verify_manifest(world.output, world.sources.lake_root, world.sources.repo_root)
