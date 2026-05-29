"""Custom exceptions for the SAM2 backend service."""

from __future__ import annotations


class Sam2BackendError(Exception):
    """Base exception for all SAM2 backend errors."""


class ImageError(Sam2BackendError):
    """Raised when an image cannot be loaded or validated."""


class ValidationError(Sam2BackendError):
    """Raised when input parameters fail validation."""


class ModelError(Sam2BackendError):
    """Raised when the SAM2 model is unavailable or inference fails."""


class OutputError(Sam2BackendError):
    """Raised when writing output artifacts fails."""
