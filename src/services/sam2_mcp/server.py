"""MCP server exposing SAM2 segmentation tools via stdio transport.

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
        description="Segment an image using a bounding-box prompt. "
        "Returns mask path, polygon, and confidence. "
        "All outputs are marked pending_review.",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image file.",
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
            },
            "required": ["image_path", "bbox"],
        },
    ),
    Tool(
        name="sam2_segment_points",
        description="Segment an image using point prompts. "
        "Returns mask path, polygon, and confidence. "
        "All outputs are marked pending_review.",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image file.",
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
            },
            "required": ["image_path", "points", "point_labels"],
        },
    ),
    Tool(
        name="sam2_generate_proposals",
        description="Generate candidate segmentation proposals for an image "
        "or directory. All outputs are machine-generated and pending_review.",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": ["string", "null"],
                    "description": "Path to a single input image.",
                },
                "image_dir": {
                    "type": ["string", "null"],
                    "description": "Path to a directory of images.",
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
            from .schemas import SegmentBoxInput

            inp = SegmentBoxInput(
                image_path=args["image_path"],
                bbox=args["bbox"],
                label=args.get("label", ""),
            )
            return _make_result(service.segment_box(inp))

        if name == "sam2_segment_points":
            from .schemas import SegmentPointsInput

            inp = SegmentPointsInput(
                image_path=args["image_path"],
                points=args["points"],
                point_labels=args["point_labels"],
                label=args.get("label", ""),
            )
            return _make_result(service.segment_points(inp))

        if name == "sam2_generate_proposals":
            from .schemas import GenerateProposalsInput

            inp = GenerateProposalsInput(
                image_path=args.get("image_path"),
                image_dir=args.get("image_dir"),
                label_hint=args.get("label_hint", ""),
                max_proposals=args.get("max_proposals", 50),
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
