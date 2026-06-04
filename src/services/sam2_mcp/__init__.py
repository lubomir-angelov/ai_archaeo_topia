"""SAM2 MCP service - map-only segmentation via Model Context Protocol.

SAM 2 MCP is a **map-only segmentation service**.  It operates only on
image inputs that represent maps, map tiles, or cropped map regions.
It does **not** handle PDFs, documents, OCR, legend interpretation,
or generic image analysis.  Those responsibilities belong to the
DeepSeek OCR MCP and the Qwen geospatial agent.

Provides sam2_health, sam2_segment_box, sam2_segment_points, and
sam2_generate_proposals tools.  Operates independently of CVAT; the
downstream SAM2 backend is reached through a configured HTTP endpoint.
"""

from __future__ import annotations

from .schemas import (
    CoordinateSpace,
    InputKind,
    PromptType,
    TileMetadata,
    validate_map_image_path,
)
from .server import create_server, main
from .settings import Settings, get_settings

__all__ = [
    "CoordinateSpace",
    "InputKind",
    "PromptType",
    "Settings",
    "TileMetadata",
    "create_server",
    "get_settings",
    "main",
    "validate_map_image_path",
]
