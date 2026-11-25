"""A synthetic frozen-input world (repo, lake, dataset) shared by the manifest and opening tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from archeo_topia.analysis.detection_metrics import RECALL_RADII, confidence_key
from archeo_topia.analysis.frozen_manifest import ManifestSources
from archeo_topia.datasets.build_detection_windows import (
    FOLDS,
    REPO_ROOT,
    LabelSource,
    build,
    load_refit_config,
    refit_candidates,
    sha256_file,
)
from archeo_topia.training.train_mound_detector import (
    DETECTOR_IMGSZ,
    HEADLINE_CONFIDENCE,
    MERGE_RADIUS,
    SELECTION_RULE,
)

SCHEMA = REPO_ROOT / "annotation" / "cvat" / "labels.json"
COMMITTED_CONFIG = REPO_ROOT / "configs" / "refit" / "v0_6_refit.json"
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
        [LabelSource(GROUND_TRUTH.parent.name, repo / GROUND_TRUTH, "all")],
        [lake / clips_dir],
        repo / DATASET,
        window_size=512,
        stride=384,
        schema_path=SCHEMA,
        plan="test",
        clips_json_paths=clips_json,
        refit_config=None,
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


SELECTION = Path("refit/selection.json")
TRAIN_RECORD = Path("refit/train_record.json")
CONFIG = Path("configs/refit/v0_6_refit.json")
CHOSEN = Path("refit/weights/epoch100.pt")
WEIGHTS_DIR = Path("refit/weights")


def _metrics(hits: int, false_positives: int) -> dict[str, Any]:
    """The slice of a ``metrics.json`` the selection reads at the headline confidence."""
    return {
        "by_confidence": {
            confidence_key(HEADLINE_CONFIDENCE): {
                "pooled": {
                    "annotations": 4,
                    "recall": {f"{RECALL_RADII[0]:g}": {"hits": hits}},
                    "false_positives": {"count": false_positives},
                }
            }
        }
    }


def write_selection(world: World, **edits: Any) -> Path:
    """Write a consistent refit ``selection.json`` and everything it proves from.

    The training record, the config, every candidate's weights and metrics are
    written into the synthetic repo, so ``require_selection`` verifies the real
    chain; ``edits`` then override top-level keys of the selection document.
    """
    repo = world.sources.repo_root
    (repo / CONFIG).parent.mkdir(parents=True, exist_ok=True)
    (repo / TRAIN_RECORD).parent.mkdir(parents=True, exist_ok=True)
    (repo / CONFIG).write_bytes(COMMITTED_CONFIG.read_bytes())
    config = load_refit_config(repo / CONFIG)
    dataset = {"path": DATASET.as_posix(), "build_report_sha256": "0" * 64}
    (repo / TRAIN_RECORD).write_text(
        json.dumps(
            {
                "config": {"path": CONFIG.as_posix(), "sha256": config.sha256},
                "dataset": dataset,
            }
        ),
        encoding="utf-8",
    )

    rows = []
    for name in refit_candidates(config):
        weights = WEIGHTS_DIR / name
        metrics = Path("refit") / Path(name).stem / "metrics.json"
        (repo / weights).parent.mkdir(parents=True, exist_ok=True)
        (repo / metrics).parent.mkdir(parents=True, exist_ok=True)
        (repo / weights).write_bytes(f"refit weights {name}".encode())
        # epoch100 alone has the most hits, so the rule chooses it.
        hits = 3 if name == CHOSEN.name else 2
        (repo / metrics).write_text(json.dumps(_metrics(hits, 1)), encoding="utf-8")
        rows.append(
            {
                "file": name,
                "path": weights.as_posix(),
                "sha256": sha256_file(repo / weights),
                "hits": hits,
                "total": 4,
                "false_positives": 1,
                "windows": 6,
                "metrics": metrics.as_posix(),
            }
        )
    chosen = next(row for row in rows if row["file"] == CHOSEN.name)
    document = {
        "train_record": {
            "path": TRAIN_RECORD.as_posix(),
            "sha256": sha256_file(repo / TRAIN_RECORD),
        },
        "dataset": dataset,
        "confidence": confidence_key(HEADLINE_CONFIDENCE),
        "imgsz": DETECTOR_IMGSZ,
        "merge_radius": MERGE_RADIUS,
        "rule": list(SELECTION_RULE),
        "candidates": rows,
        "chosen": {key: chosen[key] for key in ("file", "path", "sha256")},
        **edits,
    }
    (repo / SELECTION).write_text(json.dumps(document), encoding="utf-8")
    return SELECTION
