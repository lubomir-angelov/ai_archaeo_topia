"""Pydantic schemas for the annotation pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClipMetadata(BaseModel):
    """Metadata for one extracted PDF clip."""

    source_pdf: str
    pdf_page: int
    clip_index: int
    clip_image_path: str = ""
    annotator: str = ""
    sample_id: str = ""
    sheet_25k: str = ""
    sheet_5k: str = ""
    province: str = ""
    position_on_25k_sheet: str = ""
    original_description: str = ""
    target_present: bool = True
    target_count_claimed: str = ""
    contains_necropolis: bool = False
    contains_single_mound: bool = False
    contains_hard_negative: bool = False
    relief_original: str = ""
    relief_normalized: str = ""
    difficulty: str = "medium"
    uncertainty: bool = False
    notes: str = ""
    width: int = 0
    height: int = 0
    review_image_path: str = ""


class Sam2Proposal(BaseModel):
    """One proposal returned by SAM2 MCP."""

    bbox: list[float] = Field(description="Bounding box [x_min, y_min, x_max, y_max]")
    polygon: list[list[float]] = Field(default_factory=list)
    mask_path: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = ""


class AnnotationRow(BaseModel):
    """One row in the annotation CSV."""

    sample_id: str
    annotator: str = ""
    image_path: str = ""
    source_pdf: str = ""
    pdf_page: int = 0
    clip_index: int = 0
    clip_image_path: str = ""
    review_image_path: str = ""
    annotation_source: str = "sam2_mcp"
    model_backend: str = "mock"
    label: str = ""
    bbox_x_min: float = 0.0
    bbox_y_min: float = 0.0
    bbox_x_max: float = 0.0
    bbox_y_max: float = 0.0
    polygon: str = ""
    mask_path: str = ""
    confidence: float = 0.0
    review_status: str = "pending_review"
    sheet_25k: str = ""
    sheet_5k: str = ""
    province: str = ""
    position_on_25k_sheet: str = ""
    original_description: str = ""
    target_present: str = "true"
    target_count_claimed: str = ""
    contains_necropolis: str = "false"
    contains_single_mound: str = "false"
    contains_hard_negative: str = "false"
    relief_original: str = ""
    relief_normalized: str = ""
    difficulty: str = "medium"
    uncertainty: str = "false"
    annotation_status: str = "annotated"
    notes: str = ""


class RunReport(BaseModel):
    """Final report for one pipeline run."""

    run_id: str
    pdf_path: str
    output_root: str
    dry_run: bool = False
    sam2_mode: str = "mock"
    sam2_mcp_url: str = ""
    pages_processed: int = 0
    clips_extracted: int = 0
    proposals_generated: int = 0
    csv_rows_written: int = 0
    csv_path: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)
