"""MCP server exposing SAM2 map-only segmentation tools via stdio transport.

SAM 2 MCP is a **map-only segmentation service**.  It operates only on
image inputs that represent maps, map tiles, or cropped map regions.
It does **not** handle PDFs, documents, OCR, legend interpretation,
or generic image analysis.

Usage:
    python -m services.sam2_mcp.server

Or as an entry point:
    sam2-mcp
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool

from .schemas import ErrorResponse
from .service import Sam2Service

logger = logging.getLogger("sam2_mcp.server")

# ── Tool definitions ──────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="sam2_health",
        description="Check SAM2 MCP service and backend availability.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="sam2_segment_box",
        description=(
            "Segment a **map image, map tile, or map crop** using a "
            "bounding-box prompt.  Returns mask path, polygon (vertices, "
            "WKT, GeoJSON), bbox, confidence, and provenance metadata. "
            "Accepts only map imagery (PNG, JPEG, TIFF, BMP, WebP). "
            "Rejects PDFs, documents, and non-map images. "
            "All outputs are marked pending_review."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the map image file.",
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Bounding box as [x_min, y_min, x_max, y_max].",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the segmentation.",
                },
                "input_kind": {
                    "type": "string",
                    "enum": ["map_image", "map_tile", "map_crop"],
                    "description": "Kind of map input. Default: map_image.",
                },
                "artifact_id": {
                    "type": "string",
                    "description": "Unique identifier for the source artifact.",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run identifier for provenance tracking.",
                },
                "class_hint": {
                    "type": "string",
                    "description": "Optional class hint from the geospatial agent.",
                },
                "tile_metadata": {
                    "type": ["object", "null"],
                    "description": "Tile metadata for coordinate mapping.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory for mask artifacts.",
                },
            },
            "required": ["image_path", "bbox"],
        },
    ),
    Tool(
        name="sam2_segment_points",
        description=(
            "Segment a **map image, map tile, or map crop** using point "
            "prompts.  Returns mask path, polygon (vertices, WKT, GeoJSON), "
            "confidence, and provenance metadata.  Accepts only map imagery "
            "(PNG, JPEG, TIFF, BMP, WebP).  Rejects PDFs, documents, and "
            "non-map images.  All outputs are marked pending_review."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the map image file.",
                },
                "points": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "description": "Point prompts as [[x, y], ...].",
                },
                "point_labels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Label for each point: 1 = foreground, 0 = background.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the segmentation.",
                },
                "input_kind": {
                    "type": "string",
                    "enum": ["map_image", "map_tile", "map_crop"],
                    "description": "Kind of map input. Default: map_image.",
                },
                "artifact_id": {
                    "type": "string",
                    "description": "Unique identifier for the source artifact.",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run identifier for provenance tracking.",
                },
                "class_hint": {
                    "type": "string",
                    "description": "Optional class hint from the geospatial agent.",
                },
                "tile_metadata": {
                    "type": ["object", "null"],
                    "description": "Tile metadata for coordinate mapping.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory for mask artifacts.",
                },
            },
            "required": ["image_path", "points", "point_labels"],
        },
    ),
    Tool(
        name="sam2_generate_proposals",
        description=(
            "Generate candidate segmentation proposals for **map images only**. "
            "Accepts a single map image path or a directory of map images. "
            "All outputs are machine-generated and marked pending_review. "
            "Rejects PDFs, documents, and non-map images."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": ["string", "null"],
                    "description": "Path to a single map input image.",
                },
                "image_dir": {
                    "type": ["string", "null"],
                    "description": "Path to a directory of map images.",
                },
                "label_hint": {
                    "type": "string",
                    "description": "Optional label hint for proposals.",
                },
                "max_proposals": {
                    "type": "integer",
                    "description": "Maximum number of proposals to generate.",
                    "default": 50,
                },
                "input_kind": {
                    "type": "string",
                    "enum": ["map_image", "map_tile", "map_crop"],
                    "description": "Kind of map input. Default: map_image.",
                },
                "artifact_id": {
                    "type": "string",
                    "description": "Unique identifier for the source artifact.",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run identifier for provenance tracking.",
                },
                "class_hint": {
                    "type": "string",
                    "description": "Optional class hint from the geospatial agent.",
                },
                "tile_metadata": {
                    "type": ["object", "null"],
                    "description": "Tile metadata for coordinate mapping.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory for mask artifacts.",
                },
            },
            "required": [],
        },
    ),
]


# ── Tool handlers ─────────────────────────────────────────────────


def _make_error(text: str) -> list[TextContent]:
    """Wrap an error message in a standard MCP response."""
    return [TextContent(type="text", text=json.dumps(ErrorResponse(error=text).model_dump()))]


def _make_result(data: Any) -> list[TextContent]:
    """Wrap a successful result in a JSON MCP response."""
    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    elif isinstance(data, dict):
        payload = data
    else:
        payload = {"result": str(data)}
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


async def handle_tools(
    service: Sam2Service,
    name: str,
    arguments: list[Any] | dict[str, Any] | None,
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Route tool calls to service methods."""
    args = arguments or {}

    try:
        if name == "sam2_health":
            return _make_result(service.health_check())

        if name == "sam2_segment_box":
            from .schemas import InputKind, SegmentBoxInput, TileMetadata

            tile_meta = None
            if args.get("tile_metadata"):
                tile_meta = TileMetadata(**args["tile_metadata"])

            inp = SegmentBoxInput(
                image_path=args["image_path"],
                bbox=args["bbox"],
                label=args.get("label", ""),
                input_kind=args.get("input_kind", InputKind.MAP_IMAGE),
                artifact_id=args.get("artifact_id", ""),
                run_id=args.get("run_id", ""),
                class_hint=args.get("class_hint", ""),
                tile_metadata=tile_meta,
                output_dir=args.get("output_dir", ""),
            )
            return _make_result(service.segment_box(inp))

        if name == "sam2_segment_points":
            from .schemas import InputKind, SegmentPointsInput, TileMetadata

            tile_meta = None
            if args.get("tile_metadata"):
                tile_meta = TileMetadata(**args["tile_metadata"])

            inp = SegmentPointsInput(
                image_path=args["image_path"],
                points=args["points"],
                point_labels=args["point_labels"],
                label=args.get("label", ""),
                input_kind=args.get("input_kind", InputKind.MAP_IMAGE),
                artifact_id=args.get("artifact_id", ""),
                run_id=args.get("run_id", ""),
                class_hint=args.get("class_hint", ""),
                tile_metadata=tile_meta,
                output_dir=args.get("output_dir", ""),
            )
            return _make_result(service.segment_points(inp))

        if name == "sam2_generate_proposals":
            from .schemas import GenerateProposalsInput, InputKind, TileMetadata

            tile_meta = None
            if args.get("tile_metadata"):
                tile_meta = TileMetadata(**args["tile_metadata"])

            inp = GenerateProposalsInput(
                image_path=args.get("image_path"),
                image_dir=args.get("image_dir"),
                label_hint=args.get("label_hint", ""),
                max_proposals=args.get("max_proposals", 50),
                input_kind=args.get("input_kind", InputKind.MAP_IMAGE),
                artifact_id=args.get("artifact_id", ""),
                run_id=args.get("run_id", ""),
                class_hint=args.get("class_hint", ""),
                tile_metadata=tile_meta,
                output_dir=args.get("output_dir", ""),
            )
            return _make_result(service.generate_proposals(inp))

        return _make_error(f"Unknown tool: {name}")

    except Exception as exc:
        logger.exception("Tool %s failed: %s", name, exc)
        return _make_error(f"{type(exc).__name__}: {exc}")


# ── Server factory ────────────────────────────────────────────────


def create_server() -> Server:
    """Create and configure the MCP server with SAM2 tools."""
    server = Server("sam2-mcp")
    service = Sam2Service()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(
        name: str, arguments: list[Any] | dict[str, Any] | None
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        return await handle_tools(service, name, arguments)

    return server


def main() -> None:
    """Entry point for the SAM2 MCP server."""
    from .settings import get_settings

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting SAM2 MCP server (backend=%s)", settings.backend_url)
    server = create_server()

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    import asyncio

    asyncio.run(run())


if __name__ == "__main__":
    main()
