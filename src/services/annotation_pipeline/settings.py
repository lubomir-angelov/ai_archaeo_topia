"""Configuration for the annotation pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ANNOTATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sam2_mcp_url: str = Field(
        default="mock",
        description="SAM2 MCP backend URL or 'mock'",
    )
    sam2_mode: str = Field(
        default="mock",
        description="SAM2 backend mode: mock or sam2",
    )
    output_root: str = Field(
        default="./data/annotations/runs",
        description="Root directory for annotation run outputs",
    )
    protocol_dir: str = Field(
        default="docs/annotation",
        description="Path to annotation protocol directory",
    )
    review_status: str = Field(
        default="pending_review",
        description="Review status for machine-generated annotations",
    )
    max_proposals: int = Field(
        default=50,
        description="Max proposals per image",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, process only first few clips",
    )
    dry_run_clips: int = Field(
        default=5,
        description="Max clips to process in dry-run mode",
    )
    log_level: str = Field(default="INFO", description="Python logging level")

    @property
    def output_path(self) -> Path:
        """Resolved output directory path."""
        p = Path(self.output_root)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    @property
    def protocol_path(self) -> Path:
        """Resolved protocol directory path."""
        p = Path(self.protocol_dir)
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
