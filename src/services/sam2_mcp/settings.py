"""Environment-based configuration for the SAM2 MCP service."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables with SAM2_MCP_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="SAM2_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SAM2 backend
    backend_url: str = Field(
        default="http://127.0.0.1:8080",
        description="HTTP endpoint of the SAM2 inference backend",
    )
    backend_timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds for SAM2 backend calls",
    )

    # Image constraints
    max_image_size: int = Field(
        default=10_000,
        description="Maximum allowed image dimension (width or height) in pixels",
    )

    # Output
    output_dir: str = Field(
        default="./artifacts/sam2_mcp",
        description="Directory for generated masks and proposal artifacts",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Python logging level")

    # Proposal limits
    max_proposals_per_image: int = Field(
        default=50,
        description="Maximum proposals generated per image",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        import logging

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
