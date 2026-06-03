"""Custom exceptions for the annotation pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all annotation pipeline errors."""


class PdfError(PipelineError):
    """Raised when PDF extraction fails."""


class Sam2McpError(PipelineError):
    """Raised when SAM2 MCP calls fail."""


class CsvError(PipelineError):
    """Raised when CSV writing fails."""


class ValidationError(PipelineError):
    """Raised when run validation fails."""


class ArtifactError(PipelineError):
    """Raised when artifact paths or writes fail."""
