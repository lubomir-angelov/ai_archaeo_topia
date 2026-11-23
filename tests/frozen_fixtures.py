"""A synthetic frozen-input world (repo, lake, dataset) shared by the manifest and opening tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from archeo_topia.analysis.frozen_manifest import ManifestSources
from archeo_topia.datasets.build_detection_windows import FOLDS, REPO_ROOT, build, sha256_file

SCHEMA = REPO_ROOT / "annotation" / "cvat" / "labels.json"
SHEETS = ("S-1", "S-2")
GROUND_TRUTH = Path("annotation/frozen/instances_Test.json")
DATASET = Path("data/ds")


@dataclass(frozen=True)
class World:
    sources: ManifestSources
    output: Path


def hash_row(digest: str) -> str:
    return (
        "| Item | Value |\n|---|---|\n"
        "| COCO zip sha256 | `deadbeef` |\n"
        f"| `instances_Test.json` sha256 | `{digest}` |\n"
    )


def make_world(tmp_path: Path) -> World:
    """A repo and a lake with two frozen sheets, one clip each, and a built dataset."""
    repo, lake = tmp_path / "repo", tmp_path / "lake"
    clips_dir = Path("clips")
    coco = {
        "images": [
            {"id": i, "file_name": f"{sheet}_1.png", "width": 900, "height": 700}
            for i, sheet in enumerate(SHEETS, start=1)
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [200, 200, 24, 22]},
            {"id": 20, "image_id": 2, "category_id": 1, "bbox": [400, 400, 24, 22]},
        ],
        "categories": [{"id": 1, "name": "mound"}],
    }
    (repo / GROUND_TRUTH).parent.mkdir(parents=True)
    (repo / GROUND_TRUTH).write_text(json.dumps(coco), encoding="utf-8")
    (repo / "SOURCE.md").write_text(hash_row(sha256_file(repo / GROUND_TRUTH)), encoding="utf-8")

    clips_json = []
    for sheet in SHEETS:
        sheet_dir = lake / clips_dir / sheet
        sheet_dir.mkdir(parents=True)
        Image.new("RGBA", (900, 700), (255, 255, 255, 255)).save(sheet_dir / f"{sheet}_1.png")
        path = sheet_dir / "clips.json"
        path.write_text(
            json.dumps(
                {
                    "sheet_id": sheet,
                    "parent_size": [900, 700],
                    "clips": [{"file": f"{sheet}_1.png", "offset_xy": [0, 0], "size": [900, 700]}],
                }
            ),
            encoding="utf-8",
        )
        clips_json.append(path)

    splits = repo / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "version": "vtest",
                "presets": {"loso_v0_5": {"folds": {"foldA": "S-1", "foldB": "S-2"}}},
                "splits": {"train": [], "val": [], "test": list(SHEETS)},
                "sheets": {s: {"frozen_blind_test": True} for s in SHEETS},
            }
        ),
        encoding="utf-8",
    )
    build(
        repo / GROUND_TRUTH,
        lake / clips_dir,
        repo / DATASET,
        window_size=512,
        stride=384,
        schema_path=SCHEMA,
        plan="test",
        clips_json_paths=clips_json,
        splits_file=splits,
    )

    for fold in FOLDS:
        for relative in (
            f"runs/det/{fold}/weights/last.pt",
            f"runs/mapsam/pre_{fold}_pw20/checkpoints/final.pt",
        ):
            (repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (repo / relative).write_bytes(f"{relative}".encode())
    (repo / "sam.pth").write_bytes(b"sam")

    sources = ManifestSources(
        repo_root=repo,
        lake_root=lake,
        clips_dir=clips_dir,
        ground_truth=GROUND_TRUTH,
        source_md=Path("SOURCE.md"),
        detector_run=Path("runs/det"),
        mapsam_root=Path("runs/mapsam"),
        mapsam_prefix="pre",
        sam_checkpoint=Path("sam.pth"),
        dataset=DATASET,
        splits_file=splits,
    )
    return World(sources, tmp_path / "manifest.json")
