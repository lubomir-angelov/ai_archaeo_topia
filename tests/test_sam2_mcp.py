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
from services.sam2_mcp.errors import BackendError, ImageError, ValidationError
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
        assert s.backend_url == "http://127.0.0.1:8181"
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
        with pytest.raises(ValidationError, match="not found"):
            SegmentBoxInput(
                image_path="/nonexistent/image.png",
                bbox=[0, 0, 50, 50],
            )

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


# ── Live backend error handling tests ─────────────────────────────


class TestLiveBackendErrors:
    """Test error handling when the live backend misbehaves."""

    def test_health_unreachable(self) -> None:
        c = Sam2BackendClient(
            backend_url="http://127.0.0.1:59999",
            timeout=0.1,
        )
        h = c.check_health()
        assert h["reachable"] is False
        assert h["mode"] == "live"

    def test_health_timeout(self) -> None:
        c = Sam2BackendClient(
            backend_url="http://127.0.0.1:59999",
            timeout=0.01,
        )
        h = c.check_health()
        assert h["reachable"] is False

    def test_segment_box_backend_error_status(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        c = Sam2BackendClient(backend_url="http://fake.test", timeout=1.0)
        c._session = mock_session
        c._is_mock = False

        with pytest.raises(BackendError, match="HTTP 500"):
            c.segment_box(str(tmp_path / "x.png"), [0, 0, 10, 10])

    def test_segment_box_invalid_json(self) -> None:
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Expecting value")
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        c = Sam2BackendClient(backend_url="http://fake.test", timeout=1.0)
        c._session = mock_session
        c._is_mock = False

        with pytest.raises(BackendError, match="invalid JSON"):
            c.segment_box("/tmp/x.png", [0, 0, 10, 10])

    def test_segment_box_invalid_bbox_in_response(self) -> None:
        from unittest.mock import MagicMock

        bad_data = {
            "image_path": "/tmp/x.png",
            "bbox": [1, 2, 3],
            "polygon": [],
            "mask_path": "",
            "confidence": 0.5,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_data
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        c = Sam2BackendClient(backend_url="http://fake.test", timeout=1.0)
        c._session = mock_session
        c._is_mock = False

        with pytest.raises(BackendError, match="invalid segment data"):
            c.segment_box("/tmp/x.png", [0, 0, 10, 10])

    def test_segment_box_mask_path_escape(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        output_dir = str(tmp_path / "output")
        bad_mask = "/etc/passwd"
        good_data = {
            "image_path": "/tmp/x.png",
            "bbox": [0.0, 0.0, 10.0, 10.0],
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "mask_path": bad_mask,
            "confidence": 0.5,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = good_data
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        c = Sam2BackendClient(
            backend_url="http://fake.test",
            timeout=1.0,
            output_dir=output_dir,
        )
        c._session = mock_session
        c._is_mock = False

        with pytest.raises(BackendError, match="outside output_dir"):
            c.segment_box("/tmp/x.png", [0, 0, 10, 10])

    def test_proposal_invalid_response(self) -> None:
        from unittest.mock import MagicMock

        bad_data = {"items": "not_a_list"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_data
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        c = Sam2BackendClient(backend_url="http://fake.test", timeout=1.0)
        c._session = mock_session
        c._is_mock = False

        with pytest.raises(BackendError, match="invalid proposal data"):
            c.generate_proposals("/tmp/x.png")

    def test_proposal_mask_path_escape(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        output_dir = str(tmp_path / "output")
        bad_mask = "/tmp/escaped.png"
        good_data = {
            "items": [
                {
                    "image_path": "/tmp/x.png",
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "mask_path": bad_mask,
                    "confidence": 0.5,
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = good_data
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        c = Sam2BackendClient(
            backend_url="http://fake.test",
            timeout=1.0,
            output_dir=output_dir,
        )
        c._session = mock_session
        c._is_mock = False

        with pytest.raises(BackendError, match="outside output_dir"):
            c.generate_proposals("/tmp/x.png")

    def test_proposal_confidence_out_of_range(self) -> None:
        from unittest.mock import MagicMock

        bad_data = {
            "items": [
                {
                    "image_path": "/tmp/x.png",
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "polygon": [],
                    "mask_path": "",
                    "confidence": 1.5,
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_data
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        c = Sam2BackendClient(backend_url="http://fake.test", timeout=1.0)
        c._session = mock_session
        c._is_mock = False

        with pytest.raises(BackendError, match="invalid proposal data"):
            c.generate_proposals("/tmp/x.png")


# ── Backend contract schema tests ─────────────────────────────────


class TestBackendSchemas:
    """Test that backend contract schemas validate correctly."""

    def test_health_response_valid(self) -> None:
        from services.sam2_mcp.schemas import HealthResponse

        h = HealthResponse(ok=True, service="sam2_backend", model="sam2", device="cuda")
        assert h.ok is True
        assert h.device == "cuda"

    def test_segment_response_valid(self) -> None:
        from services.sam2_mcp.schemas import SegmentResponse

        s = SegmentResponse(
            image_path="/tmp/x.png",
            bbox=[0.0, 0.0, 10.0, 10.0],
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            mask_path="/tmp/out/mask.png",
            confidence=0.92,
        )
        assert s.confidence == 0.92
        assert len(s.bbox) == 4

    def test_segment_response_bad_confidence(self) -> None:
        from services.sam2_mcp.schemas import SegmentResponse

        with pytest.raises(ValueError):
            SegmentResponse(
                image_path="/tmp/x.png",
                bbox=[0.0, 0.0, 10.0, 10.0],
                confidence=-0.1,
            )

    def test_segment_response_bad_bbox_length(self) -> None:
        from services.sam2_mcp.schemas import SegmentResponse

        with pytest.raises(ValueError):
            SegmentResponse(
                image_path="/tmp/x.png",
                bbox=[0.0, 0.0],
            )

    def test_proposal_response_valid(self) -> None:
        from services.sam2_mcp.schemas import ProposalResponse

        p = ProposalResponse(
            items=[
                {
                    "image_path": "/tmp/x.png",
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "polygon": [],
                    "mask_path": "",
                    "confidence": 0.5,
                }
            ]
        )
        assert len(p.items) == 1

    def test_proposal_request_valid(self) -> None:
        from services.sam2_mcp.schemas import ProposalRequest

        r = ProposalRequest(image_path="/tmp/x.png", max_proposals=25)
        assert r.max_proposals == 25
        assert r.label_hint == ""

    def test_segment_bbox_request_valid(self) -> None:
        from services.sam2_mcp.schemas import SegmentBboxRequest

        r = SegmentBboxRequest(
            image_path="/tmp/x.png",
            bbox=[10.0, 20.0, 50.0, 60.0],
            label="mound",
        )
        assert r.prompt_type == "bbox"
        assert r.label == "mound"

    def test_segment_points_request_valid(self) -> None:
        from services.sam2_mcp.schemas import SegmentPointsRequest

        r = SegmentPointsRequest(
            image_path="/tmp/x.png",
            points=[[30.0, 40.0]],
            point_labels=[1],
        )
        assert r.prompt_type == "points"


# ── Map-only boundary tests ──────────────────────────────────────


class TestMapOnlyBoundary:
    """Tests that SAM 2 MCP enforces map-only input boundary."""

    def test_rejects_pdf_input(self, tmp_path: Path) -> None:
        """PDF input must be rejected by map-only validation."""
        from services.sam2_mcp.schemas import SegmentBoxInput

        pdf_path = tmp_path / "map.pdf"
        pdf_path.write_text("%PDF-1.4 fake")

        with pytest.raises(Exception, match="not a map image"):
            SegmentBoxInput(
                image_path=str(pdf_path),
                bbox=[10, 10, 50, 50],
            )

    def test_rejects_docx_input(self, tmp_path: Path) -> None:
        """DOCX input must be rejected by map-only validation."""
        from services.sam2_mcp.schemas import SegmentBoxInput

        docx_path = tmp_path / "document.docx"
        docx_path.write_text("fake docx")

        with pytest.raises(Exception, match="not a map image"):
            SegmentBoxInput(
                image_path=str(docx_path),
                bbox=[10, 10, 50, 50],
            )

    def test_rejects_txt_input(self, tmp_path: Path) -> None:
        """Text file input must be rejected."""
        from services.sam2_mcp.schemas import SegmentBoxInput

        txt_path = tmp_path / "readme.txt"
        txt_path.write_text("not an image")

        with pytest.raises(Exception, match="not a map image"):
            SegmentBoxInput(
                image_path=str(txt_path),
                bbox=[10, 10, 50, 50],
            )

    def test_accepts_png_input(self, tmp_path: Path) -> None:
        """PNG image should be accepted as valid map image."""
        from PIL import Image

        from services.sam2_mcp.schemas import SegmentBoxInput

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img_path = tmp_path / "map_tile.png"
        img.save(str(img_path))

        inp = SegmentBoxInput(
            image_path=str(img_path),
            bbox=[10, 10, 50, 50],
        )
        assert inp.image_path == str(img_path)

    def test_accepts_jpeg_input(self, tmp_path: Path) -> None:
        """JPEG image should be accepted as valid map image."""
        from PIL import Image

        from services.sam2_mcp.schemas import SegmentBoxInput

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img_path = tmp_path / "map_tile.jpg"
        img.save(str(img_path))

        inp = SegmentBoxInput(
            image_path=str(img_path),
            bbox=[10, 10, 50, 50],
        )
        assert inp.image_path == str(img_path)

    def test_accepts_tiff_input(self, tmp_path: Path) -> None:
        """TIFF image should be accepted as valid map image."""
        from PIL import Image

        from services.sam2_mcp.schemas import SegmentBoxInput

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img_path = tmp_path / "map_tile.tiff"
        img.save(str(img_path))

        inp = SegmentBoxInput(
            image_path=str(img_path),
            bbox=[10, 10, 50, 50],
        )
        assert inp.image_path == str(img_path)

    def test_points_input_rejects_pdf(self, tmp_path: Path) -> None:
        """Point prompts must also reject PDF input."""
        from services.sam2_mcp.schemas import SegmentPointsInput

        pdf_path = tmp_path / "map.pdf"
        pdf_path.write_text("%PDF-1.4 fake")

        with pytest.raises(Exception, match="not a map image"):
            SegmentPointsInput(
                image_path=str(pdf_path),
                points=[[30, 30]],
                point_labels=[1],
            )

    def test_proposals_input_rejects_pdf(self, tmp_path: Path) -> None:
        """Proposals input must also reject PDF input."""
        from services.sam2_mcp.schemas import GenerateProposalsInput

        pdf_path = tmp_path / "map.pdf"
        pdf_path.write_text("%PDF-1.4 fake")

        with pytest.raises(Exception, match="not a map image"):
            GenerateProposalsInput(
                image_path=str(pdf_path),
            )


# ── Tile metadata tests ──────────────────────────────────────────


class TestTileMetadata:
    """Tests for tile metadata and coordinate space handling."""

    def test_tile_metadata_creation(self) -> None:
        from services.sam2_mcp.schemas import TileMetadata

        tm = TileMetadata(
            tile_id="tile_001",
            tile_offset_x=1024,
            tile_offset_y=2048,
            tile_width=1024,
            tile_height=1024,
            parent_image_path="/maps/full_map.tif",
            source_map_sheet="K-35-50",
        )
        assert tm.tile_id == "tile_001"
        assert tm.tile_offset_x == 1024
        assert tm.parent_image_path == "/maps/full_map.tif"

    def test_box_input_with_tile_metadata(self, tmp_path: Path) -> None:
        """Box input should accept tile metadata."""
        from PIL import Image

        from services.sam2_mcp.schemas import InputKind, SegmentBoxInput, TileMetadata

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img_path = tmp_path / "tile.png"
        img.save(str(img_path))

        tm = TileMetadata(
            tile_id="t1",
            tile_offset_x=0,
            tile_offset_y=0,
            tile_width=100,
            tile_height=100,
        )
        inp = SegmentBoxInput(
            image_path=str(img_path),
            bbox=[10, 10, 50, 50],
            input_kind=InputKind.MAP_TILE,
            tile_metadata=tm,
            artifact_id="artifact_001",
            run_id="run_20240101",
            class_hint="mound",
        )
        assert inp.input_kind == InputKind.MAP_TILE
        assert inp.tile_metadata.tile_id == "t1"
        assert inp.artifact_id == "artifact_001"
        assert inp.run_id == "run_20240101"
        assert inp.class_hint == "mound"

    def test_coordinate_space_resolution_tile(self) -> None:
        """Tile input with metadata should resolve to tile_pixel."""
        from services.sam2_mcp.schemas import InputKind, TileMetadata
        from services.sam2_mcp.service import Sam2Service

        tm = TileMetadata(tile_id="t1")
        result = Sam2Service._resolve_coordinate_space(InputKind.MAP_TILE, tm)
        assert result.value == "tile_pixel"

    def test_coordinate_space_resolution_image(self) -> None:
        """Map image without tile metadata should resolve to image_pixel."""
        from services.sam2_mcp.schemas import InputKind
        from services.sam2_mcp.service import Sam2Service

        result = Sam2Service._resolve_coordinate_space(InputKind.MAP_IMAGE, None)
        assert result.value == "image_pixel"

    def test_coordinate_space_resolution_crop(self) -> None:
        """Map crop without tile metadata should resolve to image_pixel."""
        from services.sam2_mcp.schemas import InputKind
        from services.sam2_mcp.service import Sam2Service

        result = Sam2Service._resolve_coordinate_space(InputKind.MAP_CROP, None)
        assert result.value == "image_pixel"


# ── Provenance tests ─────────────────────────────────────────────


class TestProvenance:
    """Tests for provenance tracking in segmentation output."""

    def test_box_output_has_provenance_fields(self, tmp_path: Path) -> None:
        """Box output should include artifact_id, run_id, coordinate_space."""
        from services.sam2_mcp.schemas import (
            CoordinateSpace,
            SegmentBoxOutput,
        )

        out = SegmentBoxOutput(
            image_path="/tmp/img.png",
            bbox=[10, 10, 50, 50],
            polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
            artifact_id="artifact_001",
            run_id="run_20240101",
            coordinate_space=CoordinateSpace.TILE_PIXEL,
            class_hint="mound",
        )
        assert out.artifact_id == "artifact_001"
        assert out.run_id == "run_20240101"
        assert out.coordinate_space == CoordinateSpace.TILE_PIXEL
        assert out.class_hint == "mound"

    def test_output_derives_wkt(self) -> None:
        """Output should derive WKT from polygon vertices."""
        from services.sam2_mcp.schemas import SegmentBoxOutput

        out = SegmentBoxOutput(
            image_path="/tmp/img.png",
            bbox=[10, 10, 50, 50],
            polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
        )
        assert "POLYGON" in out.polygon_wkt
        assert "10.0" in out.polygon_wkt

    def test_output_derives_geojson(self) -> None:
        """Output should derive GeoJSON from polygon vertices."""
        from services.sam2_mcp.schemas import SegmentBoxOutput

        out = SegmentBoxOutput(
            image_path="/tmp/img.png",
            bbox=[10, 10, 50, 50],
            polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
        )
        assert "Feature" in out.polygon_geojson
        assert "Polygon" in out.polygon_geojson

    def test_proposals_output_has_provenance(self) -> None:
        """Proposals output should include artifact_id and run_id."""
        from services.sam2_mcp.schemas import ProposalsOutput

        out = ProposalsOutput(
            items=[],
            artifact_id="artifact_001",
            run_id="run_20240101",
        )
        assert out.artifact_id == "artifact_001"
        assert out.run_id == "run_20240101"


# ── WKT and GeoJSON helper tests ─────────────────────────────────


class TestWktGeojson:
    """Tests for WKT and GeoJSON conversion helpers."""

    def test_polygon_to_wkt(self) -> None:
        from services.sam2_mcp.schemas import polygon_to_wkt

        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        wkt = polygon_to_wkt(poly)
        assert wkt == "POLYGON((0 0 10 0 10 10 0 10))"

    def test_polygon_to_wkt_empty(self) -> None:
        from services.sam2_mcp.schemas import polygon_to_wkt

        assert polygon_to_wkt([]) == ""

    def test_polygon_to_geojson(self) -> None:
        from services.sam2_mcp.schemas import polygon_to_geojson

        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        gj = polygon_to_geojson(poly)
        assert gj["type"] == "Feature"
        assert gj["geometry"]["type"] == "Polygon"

    def test_polygon_to_geojson_empty(self) -> None:
        from services.sam2_mcp.schemas import polygon_to_geojson

        assert polygon_to_geojson([]) == {}


# ── Enum tests ────────────────────────────────────────────────────


class TestEnums:
    """Tests for InputKind, PromptType, and CoordinateSpace enums."""

    def test_input_kind_values(self) -> None:
        from services.sam2_mcp.schemas import InputKind

        assert InputKind.MAP_IMAGE.value == "map_image"
        assert InputKind.MAP_TILE.value == "map_tile"
        assert InputKind.MAP_CROP.value == "map_crop"

    def test_prompt_type_values(self) -> None:
        from services.sam2_mcp.schemas import PromptType

        assert PromptType.POINT.value == "point"
        assert PromptType.BOX.value == "box"
        assert PromptType.AUTO.value == "auto"

    def test_coordinate_space_values(self) -> None:
        from services.sam2_mcp.schemas import CoordinateSpace

        assert CoordinateSpace.TILE_PIXEL.value == "tile_pixel"
        assert CoordinateSpace.IMAGE_PIXEL.value == "image_pixel"
        assert CoordinateSpace.MAP_PIXEL.value == "map_pixel"


# ── Tool definition tests ────────────────────────────────────────


class TestToolDefinitions:
    """Tests that MCP tool definitions match map-only architecture."""

    def test_tools_have_map_only_descriptions(self) -> None:
        """Tool descriptions should mention map-only constraint."""
        from services.sam2_mcp.server import TOOLS

        tool_map = {t.name: t for t in TOOLS}
        box_desc = tool_map["sam2_segment_box"].description
        assert "map" in box_desc.lower()

        points_desc = tool_map["sam2_segment_points"].description
        assert "map" in points_desc.lower()

        proposals_desc = tool_map["sam2_generate_proposals"].description
        assert "map" in proposals_desc.lower()

    def test_tools_have_input_kind_in_schema(self) -> None:
        """Tool schemas should include input_kind property."""
        from services.sam2_mcp.server import TOOLS

        tool_map = {t.name: t for t in TOOLS}
        for tool_name in ["sam2_segment_box", "sam2_segment_points", "sam2_generate_proposals"]:
            props = tool_map[tool_name].inputSchema.get("properties", {})
            assert "input_kind" in props, f"{tool_name} missing input_kind"

    def test_tools_have_provenance_fields(self) -> None:
        """Tool schemas should include artifact_id and run_id."""
        from services.sam2_mcp.server import TOOLS

        tool_map = {t.name: t for t in TOOLS}
        for tool_name in ["sam2_segment_box", "sam2_segment_points", "sam2_generate_proposals"]:
            props = tool_map[tool_name].inputSchema.get("properties", {})
            assert "artifact_id" in props, f"{tool_name} missing artifact_id"
            assert "run_id" in props, f"{tool_name} missing run_id"
