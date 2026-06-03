"""Custom exceptions for the SAM2 MCP service."""

from __future__ import annotations


class Sam2McpError(Exception):
    """Base exception for all SAM2 MCP errors."""


class ImageError(Sam2McpError):
    """Raised when an image cannot be loaded or validated."""


class ValidationError(Sam2McpError):
    """Raised when input parameters fail validation."""


class BackendError(Sam2McpError):
    """Raised when the SAM2 backend is unreachable or returns an error."""


class OutputError(Sam2McpError):
    """Raised when writing output artifacts fails."""


class MapOnlyError(Sam2McpError):
    """Raised when a non-map input is provided to the map-only service.

    SAM 2 MCP accepts only map images, map tiles, and map crops.
    PDFs, documents, and non-map images must be rejected at the
    boundary and routed to the appropriate service (e.g. OCR MCP).
    """
