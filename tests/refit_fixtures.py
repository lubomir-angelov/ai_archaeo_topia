"""Synthetic label sources for the refit (``train_val``) build, shared by the refit tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from archeo_topia.datasets.build_detection_windows import (
    REPO_ROOT,
    LabelSource,
    RefitConfig,
    build,
)
from archeo_topia.formats.cut_review_export import EXPORT_NAME, PROVENANCE_FILE

SCHEMA = REPO_ROOT / "annotation" / "cvat" / "labels.json"
COMMITTED_CONFIG = REPO_ROOT / "configs" / "refit" / "v0_6_refit.json"


def write_clips(tmp_path: Path, by_sheet: dict[str, list[str]]) -> list[Path]:
    """Write one clips.json per sheet."""
    paths = []
    for sheet, files in by_sheet.items():
        path = tmp_path / f"{sheet}_clips.json"
        path.write_text(
            json.dumps(
                {
                    "sheet_id": sheet,
                    "parent_size": [900, 700],
                    "clips": [{"file": f, "offset_xy": [0, 0], "size": [900, 700]} for f in files],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


TRAIN_VAL_SPLITS = {
    "version": "vtest",
    "presets": {"loso_v0_5": {"folds": {"foldA": "D-1"}}},
    "splits": {
        "train": ["D-1", "T-1"],
        "val": ["V-1", "V-2", "V-3"],
        "test": ["X-1"],
    },
    "sheets": {},
}

STATUS_PROVENANCE = {
    "confirmed": "model_proposal_accepted",
    "added": "human_added",
    "rejected": "model_proposal_rejected",
    "unreviewed": "model_proposal_accepted",
}


def annotation(
    ann_id: int, image_id: int, status: str | None, bbox: list[float], category_id: int = 1
) -> dict[str, Any]:
    annotation: dict[str, Any] = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": [],
    }
    if status is not None:
        annotation["attributes"] = {
            "review_status": status,
            "annotation_provenance": STATUS_PROVENANCE[status],
        }
        if category_id == 2:
            annotation["attributes"]["negative_type"] = "unreviewed"
    return annotation


def write_export(tmp_path: Path, name: str, images: list[str], annotations: list[dict]) -> Path:
    path = tmp_path / name / "instances_default.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "file_name": f, "width": 900, "height": 700}
                    for i, f in enumerate(images, start=1)
                ],
                "annotations": annotations,
                "categories": [
                    {"id": 1, "name": "mound"},
                    {"id": 2, "name": "hard_negative_symbol"},
                    {"id": 3, "name": "uncertain_ignore"},
                ],
            }
        ),
        encoding="utf-8",
    )
    image_root = tmp_path / "images" / "train"
    image_root.mkdir(parents=True, exist_ok=True)
    for file_name in images:
        Image.new("RGBA", (900, 700), (255, 255, 255, 255)).save(image_root / file_name)
    return path


def write_cut_provenance(coco_json: Path) -> None:
    """Write the ``PROVENANCE.json`` a completed cut leaves beside its export."""
    (coco_json.parent / PROVENANCE_FILE).write_text(
        json.dumps(
            {
                "export": EXPORT_NAME,
                "output_sha256": {
                    coco_json.name: hashlib.sha256(coco_json.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )


class RefitWorld:
    """Two label sources, a splits file, clips.json files and the refit config."""

    def __init__(self, tmp_path: Path, edit: Callable[[RefitWorld], None] | None = None) -> None:
        self.tmp_path = tmp_path
        self.splits = dict(TRAIN_VAL_SPLITS)
        self.dev_images = ["D-1_1.png"]
        self.dev_annotations = [
            annotation(1, 1, None, [200, 200, 24, 22]),
            annotation(2, 1, None, [300, 300, 20, 20], category_id=3),
        ]
        self.reviewed_images = ["T-1_1.png", "V-1_1.png", "V-2_1.png"]
        self.reviewed_annotations = [
            annotation(1, 1, "confirmed", [200, 200, 24, 22]),
            annotation(2, 1, "rejected", [600, 500, 18, 18], category_id=2),
            annotation(3, 1, "rejected", [100, 400, 24, 22]),
            annotation(4, 2, "confirmed", [200, 200, 24, 22]),
            annotation(5, 2, "added", [500, 300, 24, 22]),
            annotation(6, 2, None, [300, 100, 20, 20], category_id=3),
            annotation(7, 3, "unreviewed", [200, 200, 24, 22]),
        ]
        self.clips_by_sheet = {"T-1": ["T-1_1.png"], "V-1": ["V-1_1.png"], "V-2": ["V-2_1.png"]}
        self._config: RefitConfig | None = None
        if edit is not None:
            edit(self)

    def config(self) -> RefitConfig:
        """Write the exports and a valid config file, and return the matching config.

        The file names the committed sources (never read from disk by the
        loader); the returned object points at the synthetic exports instead.
        Written once, so every caller sees the same files.
        """
        if self._config is not None:
            return self._config
        dev = write_export(self.tmp_path, "v0.0.3", self.dev_images, self.dev_annotations)
        reviewed = write_export(
            self.tmp_path, "v0.0.5", self.reviewed_images, self.reviewed_annotations
        )
        write_cut_provenance(reviewed)
        document = json.loads(COMMITTED_CONFIG.read_text(encoding="utf-8"))
        config_path = self.tmp_path / "refit_config.json"
        config_path.write_text(json.dumps(document), encoding="utf-8")
        self._config = RefitConfig(
            name=document["name"],
            path=config_path,
            sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            label_sources=(
                LabelSource("v0.0.3", dev, "all"),
                LabelSource("v0.0.5", reviewed, "reviewed"),
            ),
            development_source=document["development_source"],
            window=document["window"],
            stride=document["stride"],
            model=document["model"],
            epochs=document["epochs"],
            batch=document["batch"],
            seed=document["seed"],
            save_period=document["save_period"],
        )
        return self._config

    def splits_path(self) -> Path:
        path = self.tmp_path / "splits.json"
        path.write_text(json.dumps(self.splits), encoding="utf-8")
        return path

    def build(self, **overrides: Any) -> dict[str, Any]:
        config = self.config()
        arguments: dict[str, Any] = {
            "label_sources": config.label_sources,
            "images_roots": [self.tmp_path / "images"],
            "output_dir": self.tmp_path / "refit",
            "window_size": 512,
            "stride": 384,
            "schema_path": SCHEMA,
            "plan": "train_val",
            "clips_json_paths": write_clips(self.tmp_path, self.clips_by_sheet),
            "refit_config": config,
            "splits_file": self.splits_path(),
        }
        arguments.update(overrides)
        return build(**arguments)
