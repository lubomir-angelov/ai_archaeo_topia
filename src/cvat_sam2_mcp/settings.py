"""Environment-based configuration for the CVAT SAM2 MCP server."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CVAT_SAM2_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # CVAT connection
    cvat_base_url: str = Field(default="http://127.0.0.1:8080")
    cvat_username: str = Field(default="")
    cvat_password: str = Field(default="")
    cvat_org: str | None = Field(default=None)

    # Nuclio / SAM2
    nuclio_dashboard_url: str = Field(default="http://127.0.0.1:8070")
    sam2_function_name: str = Field(
        default="pth-facebookresearch-sam-vit-h",
    )

    # Artifacts
    annotation_artifact_root: str = Field(
        default="./artifacts/annotation_runs",
    )

    # Logging
    mcp_log_level: str = Field(default="INFO")

    @property
    def artifact_root(self) -> Path:
        """Resolved artifact root path."""
        root = Path(self.annotation_artifact_root)
        if not root.is_absolute():
            root = Path.cwd() / root
        return root


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
