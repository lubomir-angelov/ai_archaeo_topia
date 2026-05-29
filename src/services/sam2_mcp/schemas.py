"""Pydantic schemas for SAM2 MCP tool inputs and outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Input schemas ──────────────────────────────────────────────────


class SegmentBoxInput(BaseModel):
    """Input for sam2_segment_box tool."""

    image_path: str = Field(description="Path to the input image file")
    bbox: list[float] = Field(
        description="Bounding box as [x_min, y_min, x_max, y_max]",
        min_length=4,
        max_length=4,
    )
    label: str = Field(default="", description="Optional label for the segmentation")


class SegmentPointsInput(BaseModel):
    """Input for sam2_segment_points tool."""

    image_path: str = Field(description="Path to the input image file")
    points: list[list[float]] = Field(
        description="Point prompts as [[x, y], ...]",
    )
    point_labels: list[int] = Field(
        description="Label for each point: 1 = foreground, 0 = background",
    )
    label: str = Field(default="", description="Optional label for the segmentation")


class GenerateProposalsInput(BaseModel):
    """Input for sam2_generate_proposals tool."""

    image_path: str | None = Field(
        default=None,
        description="Path to a single input image",
    )
    image_dir: str | None = Field(
        default=None,
        description="Path to a directory of images",
    )
    label_hint: str = Field(
        default="",
        description="Optional label hint for proposals",
    )
    max_proposals: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of proposals to generate",
    )


# ── Output schemas ─────────────────────────────────────────────────


class SegmentationItem(BaseModel):
    """Single segmentation result."""

    image_path: str
    bbox: list[float] = Field(description="Bounding box [x_min, y_min, x_max, y_max]")
    polygon: list[list[float]] = Field(
        default_factory=list,
        description="Polygon vertices as [[x, y], ...]",
    )
    mask_path: str = Field(
        default="",
        description="Path to saved binary mask PNG",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score",
    )
    label: str = Field(default="", description="Label assigned to this segmentation")
    source: str = Field(default="sam2_mcp", description="Origin of the result")
    status: str = Field(
        default="pending_review",
        description="Review status of the annotation",
    )


class SegmentBoxOutput(BaseModel):
    """Output for sam2_segment_box tool."""

    image_path: str
    label: str = ""
    bbox: list[float]
    mask_path: str = ""
    polygon: list[list[float]] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "sam2_mcp"
    status: str = "pending_review"


class SegmentPointsOutput(BaseModel):
    """Output for sam2_segment_points tool."""

    image_path: str
    label: str = ""
    points: list[list[float]] = Field(default_factory=list)
    point_labels: list[int] = Field(default_factory=list)
    mask_path: str = ""
    polygon: list[list[float]] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "sam2_mcp"
    status: str = "pending_review"


class ProposalsOutput(BaseModel):
    """Output for sam2_generate_proposals tool."""

    items: list[SegmentationItem] = Field(default_factory=list)


class HealthOutput(BaseModel):
    """Output for sam2_health tool."""

    ok: bool
    service: str = "sam2_mcp"
    backend: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


# ── Error response helper ──────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    error: str
    detail: str = ""


# ── Backend HTTP contract schemas ─────────────────────────────────
# These schemas define the wire format between the MCP service and
# the live SAM2 inference backend.  They are NOT exposed to MCP
# clients; they govern the internal client.py layer.


class HealthResponse(BaseModel):
    """Expected response from GET /health on the SAM2 backend."""

    ok: bool
    service: str = "sam2_backend"
    model: str = "sam2"
    device: str = "unknown"
    details: dict[str, Any] = Field(default_factory=dict)


class SegmentBboxRequest(BaseModel):
    """POST /segment request for a bounding-box prompt."""

    image_path: str
    prompt_type: str = "bbox"
    bbox: list[float] = Field(min_length=4, max_length=4)
    label: str = ""
    output_dir: str = ""


class SegmentPointsRequest(BaseModel):
    """POST /segment request for a point prompt."""

    image_path: str
    prompt_type: str = "points"
    points: list[list[float]]
    point_labels: list[int]
    label: str = ""
    output_dir: str = ""


class SegmentResponse(BaseModel):
    """Expected response from POST /segment on the SAM2 backend."""

    image_path: str
    label: str = ""
    bbox: list[float] = Field(min_length=4, max_length=4)
    polygon: list[list[float]] = Field(default_factory=list)
    mask_path: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposalRequest(BaseModel):
    """POST /propose request."""

    image_path: str
    label_hint: str = ""
    max_proposals: int = Field(default=50, ge=1)
    output_dir: str = ""


class ProposalItem(BaseModel):
    """Single proposal item inside the POST /propose response."""

    image_path: str
    label: str = ""
    bbox: list[float] = Field(min_length=4, max_length=4)
    polygon: list[list[float]] = Field(default_factory=list)
    mask_path: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposalResponse(BaseModel):
    """Expected response from POST /propose on the SAM2 backend."""

    items: list[ProposalItem] = Field(default_factory=list)


class BackendErrorResponse(BaseModel):
    """Error response from the SAM2 backend."""

    error: str
    detail: str = ""
    code: int = 0
