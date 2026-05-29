"""Environment-based configuration for the SAM2 backend service."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables with SAM2_BACKEND_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="SAM2_BACKEND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8080, ge=1, le=65535, description="Bind port")

    # Model
    model_cfg: str = Field(
        default="",
        description="Path to SAM2 model config file (e.g. sam2_hiera_l.yaml)",
    )
    checkpoint: str = Field(
        default="",
        description="Path to SAM2 model checkpoint file",
    )
    device: str = Field(
        default="auto",
        description="Inference device: auto, cuda, or cpu",
    )

    # Mode
    mode: str = Field(
        default="mock",
        description="Backend mode: mock or sam2",
    )

    # Output
    output_dir: str = Field(
        default="./artifacts/sam2_backend",
        description="Base directory for generated mask files",
    )

    # Constraints
    max_image_pixels: int = Field(
        default=10_000,
        ge=1,
        description="Maximum image dimension (width or height) in pixels",
    )
    max_proposals_hard_cap: int = Field(
        default=500,
        ge=1,
        description="Absolute upper bound on proposals per image",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Python logging level")

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v.lower() not in ("auto", "cuda", "cpu"):
            raise ValueError(f"Invalid device: {v!r}. Must be auto, cuda, or cpu.")
        return v.lower()

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v.lower() not in ("mock", "sam2"):
            raise ValueError(f"Invalid mode: {v!r}. Must be mock or sam2.")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        if v.upper() not in logging._nameToLevel:
            raise ValueError(f"Invalid log level: {v!r}")
        return v.upper()

    @property
    def output_path(self) -> Path:
        """Resolved output directory path."""
        p = Path(self.output_dir)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    @property
    def resolved_device(self) -> str:
        """Return the actual device string after auto-resolution."""
        if self.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return self.device


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear cached settings (useful for tests)."""
    global _settings
    _settings = None
