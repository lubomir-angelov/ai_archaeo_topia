"""Smoke and unit tests for the sam2_mcp service.

Tests run against the mock backend so no GPU or live SAM2 is required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from services.sam2_mcp.client import Sam2BackendClient
from services.sam2_mcp.errors import ImageError, ValidationError
from services.sam2_mcp.schemas import (
    GenerateProposalsInput,
    SegmentBoxInput,
    SegmentPointsInput,
)
from services.sam2_mcp.server import TOOLS, create_server, handle_tools
from services.sam2_mcp.service import Sam2Service
from services.sam2_mcp.settings import Settings, reset_settings

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Reset settings cache after each test."""
    yield
    reset_settings()


@pytest.fixture
def mock_settings() -> Settings:
    """Return settings with mock backend."""
    return Settings(
        backend_url="mock",
        output_dir="/tmp/sam2_mcp_test_output",
        log_level="WARNING",
    )


@pytest.fixture
def service(mock_settings: Settings) -> Sam2Service:
    """Return a service instance configured with mock backend."""
    reset_settings()
    with patch(
        "services.sam2_mcp.settings._settings",
        mock_settings,
    ):
        return Sam2Service()


@pytest.fixture
def small_image(tmp_path: Path) -> Path:
    """Create a tiny 100x100 PNG for testing."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    path = tmp_path / "test_image.png"
    img.save(str(path))
    return path


@pytest.fixture
def image_dir_with_files(tmp_path: Path) -> Path:
    """Create a directory with a few test images."""
    from PIL import Image

    d = tmp_path / "images"
    d.mkdir()
    for i in range(3):
        img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
        img.save(str(d / f"img_{i}.png"))
    return d


# ── Settings tests ────────────────────────────────────────────────


class TestSettings:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.backend_url == "http://127.0.0.1:8080"
        assert s.backend_timeout == 120.0
        assert s.max_image_size == 10_000
        assert s.log_level == "INFO"

    def test_env_vars(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SAM2_MCP_BACKEND_URL": "http://sam2.test:9000",
                "SAM2_MCP_BACKEND_TIMEOUT": "60",
                "SAM2_MCP_MAX_IMAGE_SIZE": "5000",
                "SAM2_MCP_OUTPUT_DIR": "/custom/output",
                "SAM2_MCP_LOG_LEVEL": "DEBUG",
            },
        ):
            s = Settings()
            assert s.backend_url == "http://sam2.test:9000"
            assert s.backend_timeout == 60.0
            assert s.max_image_size == 5000
            assert s.output_dir == "/custom/output"
            assert s.log_level == "DEBUG"

    def test_output_path_resolves(self) -> None:
        s = Settings(output_dir="./relative/path")
        assert s.output_path.is_absolute()

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValueError):
            Settings(log_level="NOTALEVEL")


# ── Client tests ──────────────────────────────────────────────────


class TestClient:
    def test_mock_health(self) -> None:
        c = Sam2BackendClient(backend_url="mock")
        h = c.check_health()
        assert h["reachable"] is True
        assert h["mode"] == "mock"

    def test_mock_segment_box(self) -> None:
        c = Sam2BackendClient(backend_url="mock")
        item = c.segment_box("/tmp/img.png", [10, 20, 50, 60], "test")
        assert item.source == "sam2_mcp"
        assert item.status == "pending_review"
        assert item.bbox == [10, 20, 50, 60]
        assert len(item.polygon) == 4

    def test_mock_segment_points(self) -> None:
        c = Sam2BackendClient(backend_url="mock")
        item = c.segment_points("/tmp/img.png", [[30, 40], [50, 60]], [1, 0], "test")
        assert item.source == "sam2_mcp"
        assert item.status == "pending_review"
        assert len(item.polygon) == 4

    def test_mock_segment_points_no_fg(self) -> None:
        c = Sam2BackendClient(backend_url="mock")
        item = c.segment_points("/tmp/img.png", [[30, 40]], [0], "test")
        assert item.confidence == 0.0
        assert item.polygon == []

    def test_mock_proposals(self) -> None:
        c = Sam2BackendClient(backend_url="mock")
        items = c.generate_proposals("/tmp/img.png", max_proposals=5)
        assert len(items) == 3
        for it in items:
            assert it.source == "sam2_mcp"
            assert it.status == "pending_review"

    def test_discover_images(self, image_dir_with_files: Path) -> None:
        found = Sam2BackendClient.discover_images(str(image_dir_with_files))
        assert len(found) == 3
        assert "img_0.png" in found[0]

    def test_discover_images_empty(self, tmp_path: Path) -> None:
        found = Sam2BackendClient.discover_images(str(tmp_path))
        assert found == []

    def test_discover_images_no_dir(self) -> None:
        found = Sam2BackendClient.discover_images("/nonexistent")
        assert found == []


# ── Service tests ─────────────────────────────────────────────────


class TestService:
    def test_health_check_mock(self, service: Sam2Service) -> None:
        h = service.health_check()
        assert h.ok is True
        assert h.service == "sam2_mcp"

    def test_segment_box_success(self, service: Sam2Service, small_image: Path) -> None:
        inp = SegmentBoxInput(
            image_path=str(small_image),
            bbox=[10, 10, 50, 50],
            label="mound",
        )
        out = service.segment_box(inp)
        assert out.image_path == str(small_image)
        assert out.bbox == [10, 10, 50, 50]
        assert out.source == "sam2_mcp"
        assert out.status == "pending_review"
        assert out.label == "mound"
        assert len(out.polygon) >= 0

    def test_segment_box_invalid_bbox(self, service: Sam2Service, small_image: Path) -> None:
        inp = SegmentBoxInput(
            image_path=str(small_image),
            bbox=[60, 10, 50, 50],
        )
        with pytest.raises(ValidationError, match="x_min.*must be < x_max"):
            service.segment_box(inp)

    def test_segment_box_outside_image(self, service: Sam2Service, small_image: Path) -> None:
        inp = SegmentBoxInput(
            image_path=str(small_image),
            bbox=[0, 0, 200, 200],
        )
        with pytest.raises(ValidationError, match="outside image dimensions"):
            service.segment_box(inp)

    def test_segment_box_missing_image(self, service: Sam2Service) -> None:
        inp = SegmentBoxInput(
            image_path="/nonexistent/image.png",
            bbox=[0, 0, 50, 50],
        )
        with pytest.raises(ImageError, match="not found"):
            service.segment_box(inp)

    def test_segment_points_success(self, service: Sam2Service, small_image: Path) -> None:
        inp = SegmentPointsInput(
            image_path=str(small_image),
            points=[[30, 30], [60, 60]],
            point_labels=[1, 0],
            label="symbol",
        )
        out = service.segment_points(inp)
        assert out.source == "sam2_mcp"
        assert out.status == "pending_review"
        assert out.label == "symbol"
        assert len(out.polygon) >= 0

    def test_segment_points_empty_points(self, service: Sam2Service, small_image: Path) -> None:
        inp = SegmentPointsInput(
            image_path=str(small_image),
            points=[],
            point_labels=[],
        )
        with pytest.raises(ValidationError, match="must not be empty"):
            service.segment_points(inp)

    def test_segment_points_length_mismatch(self, service: Sam2Service, small_image: Path) -> None:
        inp = SegmentPointsInput(
            image_path=str(small_image),
            points=[[30, 30]],
            point_labels=[1, 0],
        )
        with pytest.raises(ValidationError, match="length mismatch"):
            service.segment_points(inp)

    def test_proposals_single_image(self, service: Sam2Service, small_image: Path) -> None:
        inp = GenerateProposalsInput(
            image_path=str(small_image),
            max_proposals=5,
            label_hint="mound",
        )
        out = service.generate_proposals(inp)
        assert len(out.items) >= 1
        for item in out.items:
            assert item.source == "sam2_mcp"
            assert item.status == "pending_review"

    def test_proposals_directory(self, service: Sam2Service, image_dir_with_files: Path) -> None:
        inp = GenerateProposalsInput(
            image_dir=str(image_dir_with_files),
            max_proposals=2,
        )
        out = service.generate_proposals(inp)
        assert len(out.items) >= 1

    def test_proposals_neither_path(self, service: Sam2Service) -> None:
        inp = GenerateProposalsInput()
        with pytest.raises(ValidationError, match="Either image_path or image_dir"):
            service.generate_proposals(inp)

    def test_proposals_both_paths(self, service: Sam2Service, small_image: Path) -> None:
        inp = GenerateProposalsInput(
            image_path=str(small_image),
            image_dir="/tmp",
        )
        with pytest.raises(ValidationError, match="only one of"):
            service.generate_proposals(inp)


# ── MCP server tests ──────────────────────────────────────────────


class TestMCPServer:
    def test_tools_defined(self) -> None:
        names = [t.name for t in TOOLS]
        assert "sam2_health" in names
        assert "sam2_segment_box" in names
        assert "sam2_segment_points" in names
        assert "sam2_generate_proposals" in names

    def test_create_server(self) -> None:
        srv = create_server()
        assert srv is not None

    def test_tool_schemas(self) -> None:
        tool_map = {t.name: t for t in TOOLS}
        assert tool_map["sam2_health"].inputSchema.get("required", []) == []
        assert "image_path" in tool_map["sam2_segment_box"].inputSchema["required"]
        assert "bbox" in tool_map["sam2_segment_box"].inputSchema["required"]
        assert "image_path" in tool_map["sam2_segment_points"].inputSchema["required"]
        assert "points" in tool_map["sam2_segment_points"].inputSchema["required"]
        assert "point_labels" in tool_map["sam2_segment_points"].inputSchema["required"]

    def test_health_tool_call(self) -> None:
        import asyncio

        reset_settings()
        with patch(
            "services.sam2_mcp.settings._settings",
            Settings(backend_url="mock", log_level="WARNING"),
        ):
            service = Sam2Service()
            result = asyncio.run(handle_tools(service, "sam2_health", {}))
            assert len(result) == 1
            payload = json.loads(result[0].text)
            assert payload["ok"] is True
            assert payload["service"] == "sam2_mcp"

    def test_segment_box_tool_call(self, tmp_path: Path) -> None:
        import asyncio

        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img_path = tmp_path / "tool_test.png"
        img.save(str(img_path))

        reset_settings()
        with patch(
            "services.sam2_mcp.settings._settings",
            Settings(backend_url="mock", log_level="WARNING"),
        ):
            service = Sam2Service()
            result = asyncio.run(
                handle_tools(
                    service,
                    "sam2_segment_box",
                    {
                        "image_path": str(img_path),
                        "bbox": [10, 10, 50, 50],
                        "label": "test_mound",
                    },
                )
            )
            assert len(result) == 1
            payload = json.loads(result[0].text)
            assert payload["source"] == "sam2_mcp"
            assert payload["status"] == "pending_review"
            assert payload["bbox"] == [10, 10, 50, 50]
            assert "image_path" in payload
            assert "polygon" in payload

    def test_unknown_tool(self) -> None:
        import asyncio

        reset_settings()
        with patch(
            "services.sam2_mcp.settings._settings",
            Settings(backend_url="mock", log_level="WARNING"),
        ):
            service = Sam2Service()
            result = asyncio.run(handle_tools(service, "nonexistent_tool", {}))
            assert len(result) == 1
            payload = json.loads(result[0].text)
            assert "error" in payload


# ── Package tests ─────────────────────────────────────────────────


class TestPackage:
    def test_imports(self) -> None:
        import services.sam2_mcp  # noqa: F401
        import services.sam2_mcp.client  # noqa: F401
        import services.sam2_mcp.errors  # noqa: F401
        import services.sam2_mcp.schemas  # noqa: F401
        import services.sam2_mcp.server  # noqa: F401
        import services.sam2_mcp.service  # noqa: F401
        import services.sam2_mcp.settings  # noqa: F401
