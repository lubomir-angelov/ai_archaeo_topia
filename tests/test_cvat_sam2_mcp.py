"""Pre-flight smoke tests for cvat_sam2_mcp.

These tests check that the package loads, settings parse correctly,
and models serialize. They do NOT require CVAT or SAM2 to be running.
"""

import json
import os
from unittest.mock import patch

import pytest

from cvat_sam2_mcp.annotation_runner import parse_protocol, protocol_to_cvat_labels
from cvat_sam2_mcp.models import (
    AnnotationProtocol,
    AnnotationRunReport,
    CvatLabel,
    CvatProjectInfo,
    CvatTaskInfo,
    LabelType,
    Sam2Candidate,
    Sam2GenerationResult,
)
from cvat_sam2_mcp.sam2_client import Sam2Client
from cvat_sam2_mcp.server import TOOLS, create_server

# ── Settings tests ────────────────────────────────────────────────


class TestSettings:
    def test_default_settings(self):
        from cvat_sam2_mcp.settings import get_settings

        settings = get_settings()
        assert settings.cvat_base_url == "http://127.0.0.1:8080"
        assert settings.nuclio_dashboard_url == "http://127.0.0.1:8070"
        assert settings.sam2_function_name == "pth-facebookresearch-sam-vit-h"
        assert settings.mcp_log_level == "INFO"

    def test_custom_env_vars(self):
        from cvat_sam2_mcp.settings import Settings

        with patch.dict(
            os.environ,
            {
                "CVAT_SAM2_CVAT_BASE_URL": "http://cvat.example.com",
                "CVAT_SAM2_CVAT_USERNAME": "testuser",
                "CVAT_SAM2_NUCLIO_DASHBOARD_URL": "http://nuclio.test:8080",
                "CVAT_SAM2_SAM2_FUNCTION_NAME": "my-sam2-func",
                "CVAT_SAM2_MCP_LOG_LEVEL": "DEBUG",
            },
        ):
            settings = Settings()
            assert settings.cvat_base_url == "http://cvat.example.com"
            assert settings.cvat_username == "testuser"
            assert settings.nuclio_dashboard_url == "http://nuclio.test:8080"
            assert settings.sam2_function_name == "my-sam2-func"
            assert settings.mcp_log_level == "DEBUG"


# ── Model tests ───────────────────────────────────────────────────


class TestModels:
    def test_annotation_protocol(self):
        protocol = AnnotationProtocol(
            protocol_version="test_v1",
            labels=[
                {"name": "symbol", "type": "mask", "color": "#ff0000"},
            ],
            hard_negatives=["text", "legend"],
        )
        assert protocol.protocol_version == "test_v1"
        assert len(protocol.labels) == 1
        assert protocol.labels[0].type == LabelType.mask

    def test_run_report(self):
        report = AnnotationRunReport(
            run_id="test_run", protocol_path="/tmp/prot.yaml", input_dir="/tmp/imgs"
        )
        report.add_step("step1", "success", {"detail": "ok"})
        assert len(report.steps) == 1
        assert report.steps[0].step == "step1"
        assert report.steps[0].status == "success"

    def test_run_report_json(self):
        report = AnnotationRunReport(run_id="r1", protocol_path="p.yaml", input_dir="d")
        json_str = report.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["run_id"] == "r1"

    def test_cvat_task_info(self):
        task = CvatTaskInfo(id=1, name="test", status="ready", images=10)
        assert task.id == 1
        assert task.images == 10

    def test_cvat_project_info(self):
        proj = CvatProjectInfo(id=1, name="arch", labels=[CvatLabel(name="sym", color="#00ff00")])
        assert proj.name == "arch"
        assert proj.labels[0].name == "sym"

    def test_sam2_result(self):
        result = Sam2GenerationResult(
            status="success",
            total_images=5,
            candidates=[Sam2Candidate(image_name="img.png", mask_count=3, artifact_path="/tmp/x")],
        )
        assert result.total_images == 5
        assert result.status == "success"


# ── Protocol parser tests ─────────────────────────────────────────


class TestProtocolParser:
    def test_parse_json_protocol(self, tmp_path):
        proto_file = tmp_path / "protocol.json"
        proto_file.write_text(
            json.dumps(
                {
                    "protocol_version": "test_v1",
                    "labels": [{"name": "arch_symbol", "type": "mask", "color": "#ff0000"}],
                    "hard_negatives": ["text"],
                }
            )
        )
        protocol = parse_protocol(str(proto_file))
        assert protocol.protocol_version == "test_v1"
        assert len(protocol.labels) == 1

    def test_parse_yaml_protocol(self, tmp_path):
        proto_file = tmp_path / "protocol.yaml"
        proto_file.write_text("""
protocol_version: archaeology_symbols_v1
labels:
  - name: archaeological_symbol_candidate
    type: mask
    color: "#ff0000"
hard_negatives:
  - text
  - legend
  - grid
sam2:
  mode: candidate_masks
  max_masks_per_image: 200
  min_area_px: 20
cvat:
  import_mode: candidate_only
  overwrite_verified: false
""")
        protocol = parse_protocol(str(proto_file))
        assert protocol.protocol_version == "archaeology_symbols_v1"
        assert len(protocol.labels) == 1
        assert protocol.labels[0].name == "archaeological_symbol_candidate"
        assert len(protocol.hard_negatives) == 3
        assert protocol.sam2.max_masks_per_image == 200

    def test_parse_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_protocol("/nonexistent/protocol.yaml")

    def test_protocol_to_cvat_labels(self):
        protocol = AnnotationProtocol(
            labels=[
                {"name": "sym1", "type": "mask", "color": "#ff0000"},
                {"name": "sym2", "type": "polygon", "color": "#00ff00"},
            ]
        )
        labels = protocol_to_cvat_labels(protocol)
        assert len(labels) == 2
        assert labels[0]["title"] == "sym1"
        assert labels[0]["type"] == "mask"
        assert labels[1]["type"] == "polygon"


# ── MCP server tests ──────────────────────────────────────────────


class TestMCPServer:
    def test_tools_defined(self):
        tool_names = [t.name for t in TOOLS]
        expected = [
            "cvat_health",
            "cvat_list_projects",
            "cvat_list_tasks",
            "cvat_create_task_from_directory",
            "cvat_export_task",
            "sam2_generate_candidates",
            "cvat_import_candidate_annotations",
            "annotation_start_run",
            "annotation_get_report",
        ]
        for name in expected:
            assert name in tool_names, f"Missing tool: {name}"

    def test_create_server(self):
        server = create_server()
        assert server is not None

    def test_tool_schemas(self):
        tool_map = {t.name: t for t in TOOLS}
        # cvat_health has no required args
        assert tool_map["cvat_health"].inputSchema.get("required", []) == []
        # annotation_start_run requires protocol_path and input_dir
        assert "protocol_path" in tool_map["annotation_start_run"].inputSchema.get("required", [])
        assert "input_dir" in tool_map["annotation_start_run"].inputSchema.get("required", [])


# ── SAM2 client tests ─────────────────────────────────────────────


class TestSam2Client:
    def test_sam2_client_init(self):
        client = Sam2Client()
        assert client.function_name == "pth-facebookresearch-sam-vit-h"

    def test_generate_candidates_no_dir(self):
        client = Sam2Client()
        result = client.generate_candidates(
            image_dir="/nonexistent/dir",
            artifact_dir="/tmp/test_artifacts",
        )
        assert result.status == "error"
        assert "not found" in result.error

    def test_generate_candidates_empty_dir(self, tmp_path):
        client = Sam2Client()
        empty_dir = tmp_path / "empty_imgs"
        empty_dir.mkdir()
        result = client.generate_candidates(
            image_dir=str(empty_dir),
            artifact_dir=str(tmp_path / "artifacts"),
        )
        assert result.status == "error"
        assert "No images" in result.error


# ── Import/compile tests ──────────────────────────────────────────


class TestPackage:
    def test_package_imports(self):
        import cvat_sam2_mcp  # noqa: F401
        import cvat_sam2_mcp.annotation_runner  # noqa: F401
        import cvat_sam2_mcp.cvat_client  # noqa: F401
        import cvat_sam2_mcp.models  # noqa: F401
        import cvat_sam2_mcp.sam2_client  # noqa: F401
        import cvat_sam2_mcp.server  # noqa: F401
        import cvat_sam2_mcp.settings  # noqa: F401

    def test_settings_artifact_root_resolves(self):
        from cvat_sam2_mcp.settings import get_settings

        settings = get_settings()
        root = settings.artifact_root
        assert root.is_absolute()
