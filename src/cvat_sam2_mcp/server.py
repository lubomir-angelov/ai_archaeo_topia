"""MCP server exposing CVAT + SAM2 tools to opencode.

Usage:
    python -m cvat_sam2_mcp.server

Or as an entry point:
    cvat-sam2-mcp
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool,
)

from .annotation_runner import get_report, start_run
from .cvat_client import CvatClient
from .sam2_client import Sam2Client
from .settings import get_settings

logger = logging.getLogger("cvat_sam2_mcp.server")

# ── Tool definitions ──────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="cvat_health",
        description="Check CVAT and SAM2/Nuclio availability and authentication.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="cvat_list_projects",
        description="List all CVAT projects with IDs, names, and labels.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="cvat_list_tasks",
        description="List CVAT tasks, optionally filtered by project_id.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": ["integer", "null"],
                    "description": "Filter tasks by project ID. Omit for all tasks.",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="cvat_create_task_from_directory",
        description="Create or reuse a CVAT task from images in a local directory. "
        "Validates that image_dir exists. Does not duplicate tasks if allow_reuse is true.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": ["integer", "null"],
                    "description": "CVAT project ID to attach the task to.",
                },
                "task_name": {
                    "type": "string",
                    "description": "Name for the new CVAT task.",
                },
                "image_dir": {
                    "type": "string",
                    "description": "Path to local directory containing images.",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of label dicts with 'name' and 'color' keys.",
                },
                "allow_reuse": {
                    "type": "boolean",
                    "description": "If true, reuse an existing task with the same name.",
                    "default": True,
                },
            },
            "required": ["task_name", "image_dir", "labels"],
        },
    ),
    Tool(
        name="cvat_export_task",
        description="Export annotations/images for a CVAT task. Writes to artifact root.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "CVAT task ID to export.",
                },
                "export_format": {
                    "type": "string",
                    "description": "Export format, e.g. 'CVAT for images 1.1'.",
                    "default": "CVAT for images 1.1",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="sam2_generate_candidates",
        description="Run SAM2 candidate mask generation for images in a task directory. "
        "Returns artifact paths and counts. Does not pass large masks through MCP responses.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "CVAT task ID (used for artifact naming).",
                },
                "image_dir": {
                    "type": ["string", "null"],
                    "description": "Path to image directory. If null, uses task's image dir.",
                },
                "protocol_path": {
                    "type": ["string", "null"],
                    "description": "Path to annotation protocol file.",
                },
                "max_masks_per_image": {
                    "type": "integer",
                    "description": "Maximum masks per image.",
                    "default": 200,
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="cvat_import_candidate_annotations",
        description="Import candidate annotations into a CVAT task. Additive only - "
        "never overwrites verified annotations. Adds source=sam2, status=candidate metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "CVAT task ID to import into.",
                },
                "annotations_path": {
                    "type": "string",
                    "description": "Path to annotations file (XML or JSON).",
                },
                "format_name": {
                    "type": "string",
                    "description": "Annotation format name.",
                    "default": "CVAT for images 1.1",
                },
            },
            "required": ["task_id", "annotations_path"],
        },
    ),
    Tool(
        name="annotation_start_run",
        description="Run the full annotation pipeline: parse protocol, create/reuse task, "
        "export, SAM2 generation, import candidates, write report. "
        "Set dry_run=true to preview without modifying CVAT.",
        inputSchema={
            "type": "object",
            "properties": {
                "protocol_path": {
                    "type": "string",
                    "description": "Path to annotation protocol YAML/JSON file.",
                },
                "input_dir": {
                    "type": "string",
                    "description": "Path to directory with images to annotate.",
                },
                "project_id": {
                    "type": ["integer", "null"],
                    "description": "CVAT project ID. Omit to create a standalone task.",
                },
                "task_name": {
                    "type": ["string", "null"],
                    "description": "Task name. Auto-generated if omitted.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, validate and preview without executing.",
                    "default": False,
                },
            },
            "required": ["protocol_path", "input_dir"],
        },
    ),
    Tool(
        name="annotation_get_report",
        description="Read and return a saved annotation run report by run_id.",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run ID from a previous annotation_start_run call.",
                },
            },
            "required": ["run_id"],
        },
    ),
]


async def _handle_tools(
    name: str, arguments: list[Any] | dict[str, Any] | None
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Route tool calls to implementation functions."""
    if name == "cvat_health":
        return _cvat_health()

    if name == "cvat_list_projects":
        return _cvat_list_projects()

    if name == "cvat_list_tasks":
        args = arguments or {}
        return _cvat_list_tasks(project_id=args.get("project_id"))

    if name == "cvat_create_task_from_directory":
        args = arguments or {}
        return _cvat_create_task(
            project_id=args.get("project_id"),
            task_name=args["task_name"],
            image_dir=args["image_dir"],
            labels=args["labels"],
            allow_reuse=args.get("allow_reuse", True),
        )

    if name == "cvat_export_task":
        args = arguments or {}
        return _cvat_export_task(
            task_id=args["task_id"], export_format=args.get("export_format", "CVAT for images 1.1")
        )

    if name == "sam2_generate_candidates":
        args = arguments or {}
        return _sam2_generate(
            task_id=args["task_id"],
            image_dir=args.get("image_dir"),
            protocol_path=args.get("protocol_path"),
            max_masks_per_image=args.get("max_masks_per_image", 200),
        )

    if name == "cvat_import_candidate_annotations":
        args = arguments or {}
        return _cvat_import(
            task_id=args["task_id"],
            annotations_path=args["annotations_path"],
            format_name=args.get("format_name", "CVAT for images 1.1"),
        )

    if name == "annotation_start_run":
        args = arguments or {}
        return _annotation_start_run(
            protocol_path=args["protocol_path"],
            input_dir=args["input_dir"],
            project_id=args.get("project_id"),
            task_name=args.get("task_name"),
            dry_run=args.get("dry_run", False),
        )

    if name == "annotation_get_report":
        args = arguments or {}
        return _annotation_get_report(run_id=args["run_id"])

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def _cvat_health() -> list[TextContent]:
    try:
        client = CvatClient()
        result = client.health()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _cvat_list_projects() -> list[TextContent]:
    try:
        client = CvatClient()
        projects = client.list_projects()
        summary = [
            {
                "id": p.id,
                "name": p.name,
                "labels": [lbl.name for lbl in p.labels],
                "tasks": p.tasks,
            }
            for p in projects
        ]
        return [TextContent(type="text", text=json.dumps(summary, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _cvat_list_tasks(project_id: int | None = None) -> list[TextContent]:
    try:
        client = CvatClient()
        tasks = client.list_tasks(project_id=project_id)
        summary = [
            {
                "id": t.id,
                "name": t.name,
                "status": t.status,
                "project_id": t.project_id,
                "labels": [lbl.name for lbl in t.labels],
                "images": t.images,
            }
            for t in tasks
        ]
        return [TextContent(type="text", text=json.dumps(summary, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _cvat_create_task(
    task_name: str,
    image_dir: str,
    labels: list[dict],
    project_id: int | None = None,
    allow_reuse: bool = True,
) -> list[TextContent]:
    try:
        client = CvatClient()
        existing = client.find_task_by_name(task_name, project_id=project_id)
        if existing is not None and allow_reuse:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"status": "reused", "task_id": existing.id, "name": existing.name},
                        indent=2,
                    ),
                )
            ]

        task_id = client.create_task(
            name=task_name,
            image_dir=image_dir,
            labels=labels,
            project_id=project_id,
        )
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"status": "created", "task_id": task_id, "name": task_name}, indent=2
                ),
            )
        ]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _cvat_export_task(
    task_id: int, export_format: str = "CVAT for images 1.1"
) -> list[TextContent]:
    try:
        client = CvatClient()
        path = client.export_task(task_id, export_format=export_format)
        return [
            TextContent(
                type="text", text=json.dumps({"status": "exported", "path": path}, indent=2)
            )
        ]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _sam2_generate(
    task_id: int,
    image_dir: str | None = None,
    protocol_path: str | None = None,
    max_masks_per_image: int = 200,
) -> list[TextContent]:
    try:
        sam2 = Sam2Client()
        if image_dir is None:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": "image_dir is required for SAM2 generation"}, indent=2
                    ),
                )
            ]

        settings = get_settings()
        artifact_dir = str(settings.artifact_root / f"task_{task_id}_sam2")
        result = sam2.generate_candidates(
            image_dir=image_dir,
            artifact_dir=artifact_dir,
            max_masks_per_image=max_masks_per_image,
            protocol_path=protocol_path,
        )
        return [TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _cvat_import(
    task_id: int,
    annotations_path: str,
    format_name: str = "CVAT for images 1.1",
) -> list[TextContent]:
    try:
        client = CvatClient()
        result = client.import_annotations(task_id, annotations_path, format_name=format_name)
        return [
            TextContent(
                type="text", text=json.dumps({"status": "imported", "result": result}, indent=2)
            )
        ]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _annotation_start_run(
    protocol_path: str,
    input_dir: str,
    project_id: int | None = None,
    task_name: str | None = None,
    dry_run: bool = False,
) -> list[TextContent]:
    try:
        result = start_run(
            protocol_path=protocol_path,
            input_dir=input_dir,
            project_id=project_id,
            task_name=task_name,
            dry_run=dry_run,
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


def _annotation_get_report(run_id: str) -> list[TextContent]:
    try:
        report = get_report(run_id)
        return [TextContent(type="text", text=json.dumps(report, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}, indent=2))]


# ── MCP server setup ──────────────────────────────────────────────


def create_server() -> Server:
    """Create and configure the MCP server with all CVAT/SAM2 tools."""
    server = Server("cvat-sam2-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(
        name: str, arguments: list[Any] | dict[str, Any] | None
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        return await _handle_tools(name, arguments)

    return server


def main() -> None:
    """Entry point for the MCP server."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.mcp_log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting CVAT SAM2 MCP server")
    server = create_server()

    async def run():
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
