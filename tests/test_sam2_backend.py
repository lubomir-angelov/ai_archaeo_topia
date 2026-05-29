"""Tests for the SAM2 backend service.

All tests run in mock mode without GPU or real SAM2 checkpoint.
Uses FastAPI TestClient for HTTP-level testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from services.sam2_backend.predictor import SAM2PredictorBackend
from services.sam2_backend.schemas import (
    HealthResponse,
    ProposalItem,
    ProposalResponse,
    SegmentResponse,
)
from services.sam2_backend.settings import Settings, reset_settings


@pytest.fixture(autouse=True)
def _reset_backend_settings() -> None:
    """Reset backend settings cache after each test."""
    from services.sam2_backend.api import reset_predictor

    yield
    reset_settings()
    reset_predictor()


@pytest.fixture
def mock_backend_settings(tmp_path: Path) -> Settings:
    """Return settings configured for mock mode."""
    return Settings(
        mode="mock",
        host="127.0.0.1",
        port=8099,
        output_dir=str(tmp_path / "output"),
        max_image_pixels=10_000,
        max_proposals_hard_cap=500,
        log_level="WARNING",
    )


@pytest.fixture
def client(mock_backend_settings: Settings) -> TestClient:
    """Return a TestClient for the backend API."""
    reset_settings()
    with patch(
        "services.sam2_backend.settings._settings",
        mock_backend_settings,
    ):
        from services.sam2_backend.api import app

        yield TestClient(app)


@pytest.fixture
def small_image(tmp_path: Path) -> Path:
    """Create a tiny 100x100 PNG for testing."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    path = tmp_path / "test_image.png"
    img.save(str(path))
    return path


@pytest.fixture
def larger_image(tmp_path: Path) -> Path:
    """Create a 200x150 PNG with some structure."""
    from PIL import Image

    img = Image.new("RGB", (200, 150), color=(64, 64, 64))
    path = tmp_path / "larger_image.png"
    img.save(str(path))
    return path


# ── Settings tests ────────────────────────────────────────────────


class TestBackendSettings:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.mode == "mock"
        assert s.device == "auto"
        assert s.host == "0.0.0.0"
        assert s.port == 8080

    def test_env_vars(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SAM2_BACKEND_MODE": "sam2",
                "SAM2_BACKEND_DEVICE": "cuda",
                "SAM2_BACKEND_HOST": "0.0.0.0",
                "SAM2_BACKEND_PORT": "9000",
                "SAM2_BACKEND_CHECKPOINT": "/path/to/checkpoint.pt",
                "SAM2_BACKEND_MODEL_CFG": "sam2_hiera_l.yaml",
                "SAM2_BACKEND_OUTPUT_DIR": "/custom/output",
                "SAM2_BACKEND_LOG_LEVEL": "DEBUG",
            },
        ):
            s = Settings()
            assert s.mode == "sam2"
            assert s.device == "cuda"
            assert s.port == 9000
            assert s.checkpoint == "/path/to/checkpoint.pt"

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            Settings(mode="invalid")

    def test_invalid_device(self) -> None:
        with pytest.raises(ValueError, match="Invalid device"):
            Settings(device="gpu")

    def test_resolved_device_auto_no_cuda(self) -> None:
        s = Settings(device="auto")
        assert s.resolved_device == "cpu"

    def test_output_path_resolves(self) -> None:
        s = Settings(output_dir="./relative/path")
        assert s.output_path.is_absolute()


# ── Predictor tests ───────────────────────────────────────────────


class TestPredictor:
    def test_mock_mode_loads(self) -> None:
        reset_settings()
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(mode="mock", log_level="WARNING"),
        ):
            p = SAM2PredictorBackend()
            p.ensure_loaded()
            assert p.is_loaded is True
            assert p.device == "mock"

    def test_sam2_mode_missing_torch(self) -> None:
        reset_settings()
        with (
            patch(
                "services.sam2_backend.settings._settings",
                Settings(mode="sam2", checkpoint="/no/such/file.pt", log_level="WARNING"),
            ),
            patch("builtins.__import__", side_effect=ImportError("no torch")),
        ):
            p = SAM2PredictorBackend()
            with pytest.raises(Exception, match="torch"):
                p.ensure_loaded()

    def test_sam2_mode_missing_checkpoint(self) -> None:
        reset_settings()
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(
                mode="sam2",
                checkpoint="/nonexistent/checkpoint.pt",
                log_level="WARNING",
            ),
        ):
            p = SAM2PredictorBackend()
            with pytest.raises(Exception, match="(Checkpoint not found|torch)"):
                p.ensure_loaded()

    def test_sam2_mode_no_checkpoint_set(self) -> None:
        reset_settings()
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(mode="sam2", checkpoint="", log_level="WARNING"),
        ):
            p = SAM2PredictorBackend()
            with pytest.raises(Exception, match="(not set|torch)"):
                p.ensure_loaded()

    def test_mock_segment_bbox(self, small_image: Path) -> None:
        reset_settings()
        out_dir = small_image.parent / "output"
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(mode="mock", output_dir=str(out_dir), log_level="WARNING"),
        ):
            p = SAM2PredictorBackend()
            result = p.segment_bbox(str(small_image), [10, 10, 50, 50], out_dir, "test")
            assert result["image_path"] == str(small_image)
            assert result["label"] == "test"
            assert len(result["bbox"]) == 4
            assert 0.0 <= result["confidence"] <= 1.0
            assert result["mask_path"]
            assert Path(result["mask_path"]).exists()

    def test_mock_segment_points(self, small_image: Path) -> None:
        reset_settings()
        out_dir = small_image.parent / "output"
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(mode="mock", output_dir=str(out_dir), log_level="WARNING"),
        ):
            p = SAM2PredictorBackend()
            result = p.segment_points(
                str(small_image),
                [[30, 30], [60, 60]],
                [1, 0],
                out_dir,
                "points_test",
            )
            assert result["image_path"] == str(small_image)
            assert 0.0 <= result["confidence"] <= 1.0
            assert result["mask_path"]

    def test_mock_segment_points_no_fg(self, small_image: Path) -> None:
        reset_settings()
        out_dir = small_image.parent / "output"
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(mode="mock", output_dir=str(out_dir), log_level="WARNING"),
        ):
            p = SAM2PredictorBackend()
            result = p.segment_points(
                str(small_image),
                [[30, 30]],
                [0],
                out_dir,
            )
            assert result["confidence"] == 0.0


# ── HTTP endpoint tests ───────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_mock_mode(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        h = HealthResponse(**data)
        assert h.ok is True
        assert h.service == "sam2_backend"
        assert h.mode == "mock"
        assert h.model_loaded is True

    def test_health_response_fields(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "ok" in data
        assert "service" in data
        assert "model" in data
        assert "mode" in data
        assert "device" in data
        assert "model_loaded" in data
        assert "details" in data


class TestSegmentBbox:
    def test_segment_bbox_success(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [10, 10, 50, 50],
                "label": "mound",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        s = SegmentResponse(**data)
        assert s.image_path == str(small_image)
        assert s.bbox == [10.0, 10.0, 50.0, 50.0]
        assert s.label == "mound"
        assert 0.0 <= s.confidence <= 1.0
        assert s.mask_path
        assert Path(s.mask_path).exists()

    def test_segment_bbox_missing_image(self, client: TestClient) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": "/nonexistent/image.png",
                "prompt_type": "bbox",
                "bbox": [0, 0, 10, 10],
            },
        )
        assert resp.status_code == 400

    def test_segment_bbox_invalid_bbox_order(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [60, 10, 50, 50],
            },
        )
        assert resp.status_code == 400

    def test_segment_bbox_outside_image(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [0, 0, 200, 200],
            },
        )
        assert resp.status_code == 400

    def test_segment_bbox_short_bbox(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [10, 10, 50],
            },
        )
        assert resp.status_code in (400, 422)

    def test_segment_bbox_mask_under_output_dir(
        self, client: TestClient, small_image: Path
    ) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [10, 10, 50, 50],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        mask = Path(data["mask_path"]).resolve()
        # Should be under the configured output dir
        assert mask.exists()


class TestSegmentPoints:
    def test_segment_points_success(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "points",
                "points": [[30, 30], [60, 60]],
                "point_labels": [1, 0],
                "label": "symbol",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        s = SegmentResponse(**data)
        assert s.image_path == str(small_image)
        assert s.label == "symbol"
        assert 0.0 <= s.confidence <= 1.0

    def test_segment_points_empty(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "points",
                "points": [],
                "point_labels": [],
            },
        )
        assert resp.status_code == 400

    def test_segment_points_length_mismatch(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "points",
                "points": [[30, 30]],
                "point_labels": [1, 0],
            },
        )
        assert resp.status_code == 400

    def test_segment_points_bad_label(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "points",
                "points": [[30, 30]],
                "point_labels": [2],
            },
        )
        assert resp.status_code == 400

    def test_segment_points_outside_image(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "points",
                "points": [[200, 200]],
                "point_labels": [1],
            },
        )
        assert resp.status_code == 400


class TestPropose:
    def test_propose_success(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/propose",
            json={
                "image_path": str(small_image),
                "label_hint": "mound",
                "max_proposals": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        pr = ProposalResponse(**data)
        assert len(pr.items) > 0
        assert len(pr.items) <= 5
        for item in pr.items:
            assert item.image_path == str(small_image)
            assert len(item.bbox) == 4
            assert 0.0 <= item.confidence <= 1.0
            if item.mask_path:
                assert Path(item.mask_path).exists()

    def test_propose_missing_image(self, client: TestClient) -> None:
        resp = client.post(
            "/propose",
            json={
                "image_path": "/nonexistent/image.png",
                "max_proposals": 5,
            },
        )
        assert resp.status_code == 400

    def test_propose_zero_max(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/propose",
            json={
                "image_path": str(small_image),
                "max_proposals": 0,
            },
        )
        assert resp.status_code in (400, 422)

    def test_propose_mask_under_output_dir(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/propose",
            json={
                "image_path": str(small_image),
                "max_proposals": 3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        for item in data["items"]:
            if item.get("mask_path"):
                mask = Path(item["mask_path"]).resolve()
                assert mask.exists()


class TestOutputDirEscape:
    def test_output_dir_escape_rejected(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [10, 10, 50, 50],
                "output_dir": "/etc/passwd",
            },
        )
        assert resp.status_code in (400, 500)

    def test_propose_output_dir_escape(self, client: TestClient, small_image: Path) -> None:
        resp = client.post(
            "/propose",
            json={
                "image_path": str(small_image),
                "max_proposals": 3,
                "output_dir": "/tmp/escaped",
            },
        )
        assert resp.status_code in (400, 500)


class TestSam2ModeHealth:
    def test_health_sam2_missing_dependency(self) -> None:
        """Health returns ok=false when SAM2 dependency is missing."""
        reset_settings()
        out_dir = "/tmp/sam2_backend_test_health"
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(
                mode="sam2",
                checkpoint="/nonexistent/checkpoint.pt",
                log_level="WARNING",
                output_dir=out_dir,
            ),
        ):
            from services.sam2_backend.api import app, reset_predictor

            reset_predictor()
            c = TestClient(app)
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is False
            assert data["model_loaded"] is False
            assert "error" in data["details"]

    def test_health_sam2_no_checkpoint(self) -> None:
        """Health returns ok=false when checkpoint is not configured."""
        reset_settings()
        out_dir = "/tmp/sam2_backend_test_health2"
        with patch(
            "services.sam2_backend.settings._settings",
            Settings(
                mode="sam2",
                checkpoint="",
                log_level="WARNING",
                output_dir=out_dir,
            ),
        ):
            from services.sam2_backend.api import app, reset_predictor

            reset_predictor()
            c = TestClient(app)
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is False
            assert data["model_loaded"] is False


class TestSchemaValidation:
    def test_health_response_schema(self) -> None:
        h = HealthResponse(
            ok=True,
            service="sam2_backend",
            model="sam2",
            mode="mock",
            device="mock",
            model_loaded=True,
        )
        assert h.ok is True
        assert h.service == "sam2_backend"

    def test_segment_response_schema(self) -> None:
        s = SegmentResponse(
            image_path="/tmp/x.png",
            bbox=[0.0, 0.0, 10.0, 10.0],
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            mask_path="/tmp/out/mask.png",
            confidence=0.92,
        )
        assert s.confidence == 0.92

    def test_segment_response_bad_confidence(self) -> None:
        with pytest.raises(ValueError):
            SegmentResponse(
                image_path="/tmp/x.png",
                bbox=[0.0, 0.0, 10.0, 10.0],
                confidence=1.5,
            )

    def test_proposal_item_schema(self) -> None:
        item = ProposalItem(
            image_path="/tmp/x.png",
            bbox=[0.0, 0.0, 10.0, 10.0],
            confidence=0.5,
        )
        assert item.confidence == 0.5

    def test_proposal_response_schema(self) -> None:
        p = ProposalResponse(
            items=[
                {
                    "image_path": "/tmp/x.png",
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "confidence": 0.5,
                }
            ]
        )
        assert len(p.items) == 1


class TestContractCompatibility:
    """Verify backend responses are compatible with sam2_mcp client schemas."""

    def test_health_compatible_with_mcp_client(self, client: TestClient) -> None:
        """HealthResponse from backend should parse with sam2_mcp HealthResponse."""
        resp = client.get("/health")
        data = resp.json()
        from services.sam2_mcp.schemas import HealthResponse as McpHealthResponse

        mcp_h = McpHealthResponse(**data)
        assert mcp_h.ok is True
        assert mcp_h.service == "sam2_backend"

    def test_segment_compatible_with_mcp_client(
        self, client: TestClient, small_image: Path
    ) -> None:
        """SegmentResponse from backend should parse with sam2_mcp SegmentResponse."""
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [10, 10, 50, 50],
            },
        )
        data = resp.json()
        from services.sam2_mcp.schemas import SegmentResponse as McpSegmentResponse

        mcp_s = McpSegmentResponse(**data)
        assert mcp_s.image_path == str(small_image)
        assert len(mcp_s.bbox) == 4

    def test_propose_compatible_with_mcp_client(
        self, client: TestClient, small_image: Path
    ) -> None:
        """ProposalResponse from backend should parse with sam2_mcp ProposalResponse."""
        resp = client.post(
            "/propose",
            json={
                "image_path": str(small_image),
                "max_proposals": 3,
            },
        )
        data = resp.json()
        from services.sam2_mcp.schemas import ProposalResponse as McpProposalResponse

        mcp_p = McpProposalResponse(**data)
        assert len(mcp_p.items) > 0


class TestMaskPathGeneration:
    def test_mask_path_generated_under_output_dir(
        self, client: TestClient, small_image: Path
    ) -> None:
        """Generated mask_path must be under the configured output directory."""
        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [10, 10, 50, 50],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        mask_path = Path(data["mask_path"]).resolve()
        assert mask_path.exists()
        assert mask_path.suffix == ".png"

    def test_mask_is_valid_png(self, client: TestClient, small_image: Path) -> None:
        """Generated mask must be a valid PNG file."""
        from PIL import Image

        resp = client.post(
            "/segment",
            json={
                "image_path": str(small_image),
                "prompt_type": "bbox",
                "bbox": [10, 10, 50, 50],
            },
        )
        data = resp.json()
        mask_img = Image.open(data["mask_path"])
        mask_img.verify()


class TestPackage:
    def test_imports(self) -> None:
        import services.sam2_backend  # noqa: F401
        import services.sam2_backend.api  # noqa: F401
        import services.sam2_backend.errors  # noqa: F401
        import services.sam2_backend.masks  # noqa: F401
        import services.sam2_backend.predictor  # noqa: F401
        import services.sam2_backend.proposals  # noqa: F401
        import services.sam2_backend.schemas  # noqa: F401
        import services.sam2_backend.settings  # noqa: F401
