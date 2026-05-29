"""SAM2 MCP service - standalone segmentation via Model Context Protocol.

Provides sam2_health, sam2_segment_box, sam2_segment_points, and
sam2_generate_proposals tools.  Operates independently of CVAT; the
downstream SAM2 backend is reached through a configured HTTP endpoint.
"""

from __future__ import annotations

from .server import create_server, main
from .settings import Settings, get_settings

__all__ = [
    "Settings",
    "create_server",
    "get_settings",
    "main",
]
