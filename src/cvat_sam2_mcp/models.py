"""Pydantic models for MCP tool schemas and data exchange."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Protocol models ───────────────────────────────────────────────


class LabelType(StrEnum):
    mask = "mask"
    rectangle = "rectangle"
    polygon = "polygon"
    points = "points"


class ProtocolLabel(BaseModel):
    name: str
    type: LabelType = LabelType.mask
    color: str = "#ff0000"


class SamConfig(BaseModel):
    mode: str = "candidate_masks"
    max_masks_per_image: int = 200
    min_area_px: int = 20
    exclude_text_regions: bool = True


class CvatConfig(BaseModel):
    import_mode: str = "candidate_only"
    overwrite_verified: bool = False


class AnnotationProtocol(BaseModel):
    protocol_version: str = "archaeology_symbols_v1"
    labels: list[ProtocolLabel] = Field(default_factory=list)
    hard_negatives: list[str] = Field(default_factory=list)
    sam2: SamConfig = Field(default_factory=SamConfig)
    cvat: CvatConfig = Field(default_factory=CvatConfig)


# ── CVAT API response models ──────────────────────────────────────


class CvatLabel(BaseModel):
    name: str
    color: str = "#ff0000"
    attributes: list[Any] = Field(default_factory=list)


class CvatProjectInfo(BaseModel):
    id: int
    name: str
    labels: list[CvatLabel] = Field(default_factory=list)
    tasks: int = 0


class CvatTaskInfo(BaseModel):
    id: int
    name: str
    status: str
    project_id: int | None = None
    labels: list[CvatLabel] = Field(default_factory=list)
    images: int = 0
    subset: str = ""


# ── SAM2 / Nuclio models ─────────────────────────────────────────


class Sam2Candidate(BaseModel):
    image_name: str
    mask_count: int
    artifact_path: str


class Sam2GenerationResult(BaseModel):
    status: str
    total_images: int
    candidates: list[Sam2Candidate] = Field(default_factory=list)
    error: str | None = None


# ── Import / export models ────────────────────────────────────────


class ImportResult(BaseModel):
    status: str
    imported_count: int = 0
    skipped_count: int = 0
    error: str | None = None


# ── Run report models ─────────────────────────────────────────────


class RunStep(BaseModel):
    step: str
    status: str
    duration_seconds: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class AnnotationRunReport(BaseModel):
    run_id: str
    protocol_path: str
    input_dir: str
    project_id: int | None = None
    task_id: int | None = None
    started_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    completed_at: str | None = None
    steps: list[RunStep] = Field(default_factory=list)
    dry_run: bool = False
    status: str = "pending"
    next_steps: list[str] = Field(default_factory=list)

    def add_step(self, step: str, status: str, details: dict[str, Any] | None = None) -> None:
        self.steps.append(RunStep(step=step, status=status, details=details or {}))
