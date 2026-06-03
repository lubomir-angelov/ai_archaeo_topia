"""Pydantic schemas for SAM2 MCP tool inputs and outputs.

SAM 2 MCP is a **map-only segmentation service**.  It operates only on
image inputs that represent maps, map tiles, or cropped map regions.
It does **not** handle PDFs, documents, OCR, legend interpretation,
or generic image analysis.  Those responsibilities belong to the
DeepSeek OCR MCP and the Qwen geospatial agent.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Enums ──────────────────────────────────────────────────────────


class InputKind(StrEnum):
    """Kind of map image being segmented."""

    MAP_IMAGE = "map_image"
    MAP_TILE = "map_tile"
    MAP_CROP = "map_crop"


class PromptType(StrEnum):
    """Type of segmentation prompt."""

    POINT = "point"
    BOX = "box"
    AUTO = "auto"


class CoordinateSpace(StrEnum):
    """Coordinate space for geometry output."""

    TILE_PIXEL = "tile_pixel"
    IMAGE_PIXEL = "image_pixel"
    MAP_PIXEL = "map_pixel"


# ── Tile metadata ──────────────────────────────────────────────────


class TileMetadata(BaseModel):
    """Tile-local to global coordinate mapping.

    Required when input_kind is map_tile so the geospatial agent can
    reconstruct full-map coordinates from tile-local results.
    """

    tile_id: str = Field(default="", description="Unique tile identifier")
    tile_offset_x: int = Field(default=0, description="Tile X offset in parent image")
    tile_offset_y: int = Field(default=0, description="Tile Y offset in parent image")
    tile_width: int = Field(default=0, description="Tile width in pixels")
    tile_height: int = Field(default=0, description="Tile height in pixels")
    parent_image_path: str = Field(
        default="", description="Path to the parent full-map image"
    )
    source_map_sheet: str = Field(
        default="", description="Source map sheet identifier"
    )


# ── Validation helpers ─────────────────────────────────────────────

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
INVALID_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".html", ".xml", ".json", ".csv"}


def validate_map_image_path(image_path: str) -> None:
    """Validate that the path points to a valid map image file.

    Rejects PDFs, documents, text files, and non-image artifacts.
    Raises ValidationError for invalid paths.
    """
    from .errors import ValidationError

    p = Path(image_path)
    suffix = p.suffix.lower()

    if suffix in INVALID_EXTENSIONS:
        raise ValidationError(
            f"Input {image_path!r} has extension {suffix!r} which is not a map image. "
            f"SAM 2 MCP only accepts map images (PNG, JPEG, TIFF, BMP, WebP). "
            f"For PDFs, use the DeepSeek OCR MCP or a document preprocessing step."
        )

    if suffix not in VALID_IMAGE_EXTENSIONS:
        raise ValidationError(
            f"Input {image_path!r} has extension {suffix!r} which is not a supported "
            f"map image format. Supported: {sorted(VALID_IMAGE_EXTENSIONS)}"
        )

    if not p.exists():
        raise ValidationError(f"Map image not found: {image_path!r}")

    if not p.is_file():
        raise ValidationError(f"Not a file: {image_path!r}")


def polygon_to_wkt(polygon: list[list[float]]) -> str:
    """Convert polygon vertices to WKT POLYGON string."""
    if not polygon:
        return ""
    coords = " ".join(f"{x} {y}" for x, y in polygon)
    return f"POLYGON(({coords}))"


def polygon_to_geojson(polygon: list[list[float]]) -> dict[str, Any]:
    """Convert polygon vertices to GeoJSON Feature."""
    if not polygon:
        return {}
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[x, y] for x, y in polygon] + [polygon[0]],
        },
        "properties": {},
    }


# ── Input schemas ──────────────────────────────────────────────────


class SegmentBoxInput(BaseModel):
    """Input for sam2_segment_box tool.

    Accepts only map images, map tiles, or map crops.  The geospatial
    agent supplies the bounding box derived from its own visual
    reasoning; SAM 2 returns the geometry artifacts.
    """

    image_path: str = Field(description="Path to the map image file")
    bbox: list[float] = Field(
        description="Bounding box as [x_min, y_min, x_max, y_max]",
        min_length=4,
        max_length=4,
    )
    label: str = Field(
        default="", description="Optional label for the segmentation"
    )
    input_kind: InputKind = Field(
        default=InputKind.MAP_IMAGE,
        description="Kind of map input: map_image, map_tile, or map_crop",
    )
    artifact_id: str = Field(
        default="", description="Unique identifier for the source artifact"
    )
    run_id: str = Field(
        default="", description="Run identifier for provenance tracking"
    )
    class_hint: str = Field(
        default="",
        description="Optional class hint from the geospatial agent "
        "(e.g. mound, road, contour)",
    )
    tile_metadata: TileMetadata | None = Field(
        default=None,
        description="Tile metadata when operating on a map tile",
    )
    output_dir: str = Field(
        default="", description="Optional output directory for mask artifacts"
    )

    @field_validator("image_path")
    @classmethod
    def _validate_image(cls, v: str) -> str:
        validate_map_image_path(v)
        return v


class SegmentPointsInput(BaseModel):
    """Input for sam2_segment_points tool.

    Accepts only map images, map tiles, or map crops.  Point prompts
    are supplied by the geospatial agent after it identifies candidate
    features in the map.
    """

    image_path: str = Field(description="Path to the map image file")
    points: list[list[float]] = Field(
        description="Point prompts as [[x, y], ...]",
    )
    point_labels: list[int] = Field(
        description="Label for each point: 1 = foreground, 0 = background",
    )
    label: str = Field(
        default="", description="Optional label for the segmentation"
    )
    input_kind: InputKind = Field(
        default=InputKind.MAP_IMAGE,
        description="Kind of map input: map_image, map_tile, or map_crop",
    )
    artifact_id: str = Field(
        default="", description="Unique identifier for the source artifact"
    )
    run_id: str = Field(
        default="", description="Run identifier for provenance tracking"
    )
    class_hint: str = Field(
        default="",
        description="Optional class hint from the geospatial agent",
    )
    tile_metadata: TileMetadata | None = Field(
        default=None,
        description="Tile metadata when operating on a map tile",
    )
    output_dir: str = Field(
        default="", description="Optional output directory for mask artifacts"
    )

    @field_validator("image_path")
    @classmethod
    def _validate_image(cls, v: str) -> str:
        validate_map_image_path(v)
        return v


class GenerateProposalsInput(BaseModel):
    """Input for sam2_generate_proposals tool.

    Generates candidate segmentation proposals for map images only.
    """

    image_path: str | None = Field(
        default=None,
        description="Path to a single map input image",
    )
    image_dir: str | None = Field(
        default=None,
        description="Path to a directory of map images",
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
    input_kind: InputKind = Field(
        default=InputKind.MAP_IMAGE,
        description="Kind of map input: map_image, map_tile, or map_crop",
    )
    artifact_id: str = Field(
        default="", description="Unique identifier for the source artifact"
    )
    run_id: str = Field(
        default="", description="Run identifier for provenance tracking"
    )
    class_hint: str = Field(
        default="",
        description="Optional class hint from the geospatial agent",
    )
    tile_metadata: TileMetadata | None = Field(
        default=None,
        description="Tile metadata when operating on map tiles",
    )
    output_dir: str = Field(
        default="", description="Optional output directory for mask artifacts"
    )

    @field_validator("image_path")
    @classmethod
    def _validate_image(cls, v: str | None) -> str | None:
        if v is not None:
            validate_map_image_path(v)
        return v


# ── Output schemas ─────────────────────────────────────────────────


class SegmentationItem(BaseModel):
    """Single segmentation result with provenance metadata."""

    image_path: str
    bbox: list[float] = Field(
        description="Bounding box [x_min, y_min, x_max, y_max]"
    )
    polygon: list[list[float]] = Field(
        default_factory=list,
        description="Polygon vertices as [[x, y], ...]",
    )
    polygon_wkt: str = Field(
        default="",
        description="WKT POLYGON string derived from polygon vertices",
    )
    polygon_geojson: str = Field(
        default="",
        description="GeoJSON Feature string derived from polygon vertices",
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
    artifact_id: str = Field(
        default="", description="Source artifact identifier for provenance"
    )
    run_id: str = Field(
        default="", description="Run identifier for provenance tracking"
    )
    coordinate_space: CoordinateSpace = Field(
        default=CoordinateSpace.IMAGE_PIXEL,
        description="Coordinate space of bbox and polygon",
    )
    tile_metadata: TileMetadata | None = Field(
        default=None,
        description="Tile metadata for tile-local to global mapping",
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings"
    )
    errors: list[str] = Field(default_factory=list, description="Error messages")

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Derive WKT and GeoJSON from polygon vertices."""
        if self.polygon and not self.polygon_wkt:
            self.polygon_wkt = polygon_to_wkt(self.polygon)
        if self.polygon and not self.polygon_geojson:
            gj = polygon_to_geojson(self.polygon)
            if gj:
                self.polygon_geojson = json.dumps(gj)


class SegmentBoxOutput(BaseModel):
    """Output for sam2_segment_box tool."""

    image_path: str
    label: str = ""
    bbox: list[float]
    polygon: list[list[float]] = Field(default_factory=list)
    polygon_wkt: str = Field(default="", description="WKT POLYGON string")
    polygon_geojson: str = Field(default="", description="GeoJSON Feature string")
    mask_path: str = ""
    confidence: float = 0.0
    source: str = "sam2_mcp"
    status: str = "pending_review"
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    coordinate_space: CoordinateSpace = Field(
        default=CoordinateSpace.IMAGE_PIXEL,
        description="Coordinate space of bbox and polygon",
    )
    tile_metadata: TileMetadata | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )
    class_hint: str = Field(default="", description="Class hint from geospatial agent")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if self.polygon and not self.polygon_wkt:
            self.polygon_wkt = polygon_to_wkt(self.polygon)
        if self.polygon and not self.polygon_geojson:
            gj = polygon_to_geojson(self.polygon)
            if gj:
                self.polygon_geojson = json.dumps(gj)


class SegmentPointsOutput(BaseModel):
    """Output for sam2_segment_points tool."""

    image_path: str
    label: str = ""
    points: list[list[float]] = Field(default_factory=list)
    point_labels: list[int] = Field(default_factory=list)
    mask_path: str = ""
    polygon: list[list[float]] = Field(default_factory=list)
    polygon_wkt: str = Field(default="", description="WKT POLYGON string")
    polygon_geojson: str = Field(default="", description="GeoJSON Feature string")
    confidence: float = 0.0
    source: str = "sam2_mcp"
    status: str = "pending_review"
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    coordinate_space: CoordinateSpace = Field(
        default=CoordinateSpace.IMAGE_PIXEL,
        description="Coordinate space of bbox and polygon",
    )
    tile_metadata: TileMetadata | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )
    class_hint: str = Field(default="", description="Class hint from geospatial agent")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if self.polygon and not self.polygon_wkt:
            self.polygon_wkt = polygon_to_wkt(self.polygon)
        if self.polygon and not self.polygon_geojson:
            gj = polygon_to_geojson(self.polygon)
            if gj:
                self.polygon_geojson = json.dumps(gj)


class ProposalsOutput(BaseModel):
    """Output for sam2_generate_proposals tool."""

    items: list[SegmentationItem] = Field(default_factory=list)
    artifact_id: str = Field(
        default="", description="Source artifact identifier"
    )
    run_id: str = Field(default="", description="Run identifier")
    input_kind: InputKind = Field(
        default=InputKind.MAP_IMAGE, description="Kind of map input"
    )
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


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

    image_path: str
    prompt_type: str = "points"
    points: list[list[float]]
    point_labels: list[int]
    label: str = ""
    output_dir: str = ""
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


class SegmentResponse(BaseModel):
    """Expected response from POST /segment on the SAM2 backend."""

    image_path: str
    label: str = ""
    bbox: list[float] = Field(min_length=4, max_length=4)
    polygon: list[list[float]] = Field(default_factory=list)
    mask_path: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str = Field(default="", description="Source artifact identifier")
    run_id: str = Field(default="", description="Run identifier")
    coordinate_space: str = Field(
        default="image_pixel", description="Coordinate space of geometry"
    )
    tile_metadata: dict[str, Any] | None = Field(
        default=None, description="Tile metadata for coordinate mapping"
    )


class ProposalRequest(BaseModel):
    """POST /propose request."""

    image_path: str
    label_hint: str = ""
    max_proposals: int = Field(default=50, ge=1)
    output_dir: str = ""
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


class ProposalItem(BaseModel):
    """Single proposal item inside the POST /propose response."""

    image_path: str
    label: str = ""
    bbox: list[float] = Field(min_length=4, max_length=4)
    polygon: list[list[float]] = Field(default_factory=list)
    mask_path: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
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
    """Expected response from POST /propose on the SAM2 backend."""

    items: list[ProposalItem] = Field(default_factory=list)


class BackendErrorResponse(BaseModel):
    """Error response from the SAM2 backend."""

    error: str
    detail: str = ""
    code: int = 0
