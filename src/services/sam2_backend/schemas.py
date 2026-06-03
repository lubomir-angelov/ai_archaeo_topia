"""Pydantic schemas for the SAM2 backend HTTP contract.

SAM 2 backend is a **map-only segmentation service**.  It operates only
on image inputs that represent maps, map tiles, or cropped map regions.

These schemas match the wire format defined in
docs/automation/SAM2_BACKEND_CONTRACT.md and the client-side schemas
in src/services/sam2_mcp/schemas.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Request schemas ───────────────────────────────────────────────


class SegmentBboxRequest(BaseModel):
    """POST /segment request for a bounding-box prompt."""

    image_path: str = Field(description="Absolute path to the map image on the backend")
    prompt_type: str = Field(default="bbox", pattern="^(bbox|points)$")
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[x_min, y_min, x_max, y_max] in pixel coordinates",
    )
    label: str = Field(default="", description="Optional label for the segmentation")
    output_dir: str = Field(default="", description="Directory for generated mask files")
    input_kind: str = Field(
        default="map_image",
        description="Kind of map input: map_image, map_tile, map_crop",
    )
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    class_hint: str = Field(default="", description="Class hint from geospatial agent")
    tile_metadata: dict[str, Any] | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )


class SegmentPointsRequest(BaseModel):
    """POST /segment request for a point prompt."""

    image_path: str = Field(description="Absolute path to the map image on the backend")
    prompt_type: str = Field(default="points", pattern="^(bbox|points)$")
    points: list[list[float]] = Field(
        description="[[x, y], ...] in pixel coordinates",
    )
    point_labels: list[int] = Field(
        description="1 = foreground, 0 = background",
    )
    label: str = Field(default="", description="Optional label for the segmentation")
    output_dir: str = Field(default="", description="Directory for generated mask files")
    input_kind: str = Field(
        default="map_image",
        description="Kind of map input: map_image, map_tile, map_crop",
    )
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    class_hint: str = Field(default="", description="Class hint from geospatial agent")
    tile_metadata: dict[str, Any] | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )


class ProposalRequest(BaseModel):
    """POST /propose request."""

    image_path: str = Field(description="Absolute path to the map image on the backend")
    label_hint: str = Field(default="", description="Suggested label for proposals")
    max_proposals: int = Field(
        default=50,
        ge=1,
        description="Maximum number of proposals to generate",
    )
    output_dir: str = Field(default="", description="Directory for generated mask files")
    input_kind: str = Field(
        default="map_image",
        description="Kind of map input: map_image, map_tile, map_crop",
    )
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    class_hint: str = Field(default="", description="Class hint from geospatial agent")
    tile_metadata: dict[str, Any] | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )


# ── Response schemas ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response from GET /health."""

    ok: bool
    service: str = "sam2_backend"
    model: str = "sam2"
    mode: str = "mock"
    device: str = "unknown"
    model_loaded: bool = False
    checkpoint: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class SegmentResponse(BaseModel):
    """Response from POST /segment."""

    image_path: str
    label: str = ""
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[x_min, y_min, x_max, y_max] of result",
    )
    polygon: list[list[float]] = Field(
        default_factory=list,
        description="[[x, y], ...] polygon vertices",
    )
    mask_path: str = Field(
        default="",
        description="Absolute path to generated mask PNG",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Float in [0.0, 1.0]",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    coordinate_space: str = Field(
        default="image_pixel", description="Coordinate space of geometry"
    )
    tile_metadata: dict[str, Any] | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )


class ProposalItem(BaseModel):
    """Single proposal item inside the POST /propose response."""

    image_path: str
    label: str = ""
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[x_min, y_min, x_max, y_max]",
    )
    polygon: list[list[float]] = Field(
        default_factory=list,
        description="[[x, y], ...] polygon vertices",
    )
    mask_path: str = Field(
        default="",
        description="Absolute path to generated mask PNG",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Float in [0.0, 1.0]",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    coordinate_space: str = Field(
        default="image_pixel", description="Coordinate space of geometry"
    )
    tile_metadata: dict[str, Any] | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )


class ProposalResponse(BaseModel):
    """Response from POST /propose."""

    items: list[ProposalItem] = Field(default_factory=list)


# ── Error response ────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    error: str
    detail: str = ""
