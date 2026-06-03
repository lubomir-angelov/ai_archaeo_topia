"""SAM2 Backend -- self-contained HTTP inference service.

Stateless FastAPI server that runs SAM2 segmentation inference.
Called by the SAM2 MCP adapter (src/services/sam2_mcp/) over HTTP.

Operates independently of CVAT, Nuclio, OCR, and LLM services.
"""

from __future__ import annotations

from .settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
]
