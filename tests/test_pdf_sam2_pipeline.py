"""Tests for the PDF → SAM2 annotation pipeline.

All tests use mock backend, no GPU or real SAM2 required.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from PIL import Image

from services.annotation_pipeline.artifacts import (
    create_run_directory,
    render_review_image,
    save_clip_image,
)
from services.annotation_pipeline.csv_writer import (
    append_annotation_csv,
    polygon_to_json,
    write_annotation_csv,
)
from services.annotation_pipeline.errors import (
    ArtifactError,
    CsvError,
    PdfError,
    PipelineError,
    ValidationError,
)
from services.annotation_pipeline.metadata import (
    ANNOTATION_COLUMNS,
    load_sidecar_metadata,
    merge_sidecar,
    report_missing_metadata,
)
from services.annotation_pipeline.pdf_sam2_runner import (
    build_annotation_rows,
    extract_clips_from_pdf,
    run_pipeline,
)
from services.annotation_pipeline.schemas import (
    AnnotationRow,
    ClipMetadata,
    Sam2Proposal,
)
from services.annotation_pipeline.settings import Settings, get_settings, reset_settings
from services.annotation_pipeline.validation import (
    validate_bbox_in_image,
    validate_run,
)


@pytest.fixture(autouse=True)
def _reset_pipeline_settings() -> None:
    """Reset settings cache after each test."""
    yield
    reset_settings()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Create a run directory structure."""
    return create_run_directory(tmp_path, "test_run")


@pytest.fixture
def sample_clip(run_dir: Path) -> ClipMetadata:
    """Create a sample clip metadata."""
    return ClipMetadata(
        source_pdf="/tmp/test.pdf",
        pdf_page=1,
        clip_index=0,
        clip_image_path="",
        annotator="AG",
        sample_id="AG_001_K-35-50_UNKNOWN.png",
        sheet_25k="K-35-50",
        sheet_5k="UNKNOWN",
        province="",
        width=200,
        height=150,
    )


@pytest.fixture
def sample_clip_with_image(run_dir: Path, tmp_path: Path) -> tuple[ClipMetadata, Path]:
    """Create a clip with an actual image file."""
    img_path = tmp_path / "test_clip.png"
    img = Image.new("RGB", (200, 150), color=(128, 128, 128))
    img.save(str(img_path))

    clip = ClipMetadata(
        source_pdf=str(tmp_path / "test.pdf"),
        pdf_page=1,
        clip_index=0,
        clip_image_path=str(img_path),
        annotator="AG",
        sample_id="AG_001_K-35-50_UNKNOWN.png",
        sheet_25k="K-35-50",
        sheet_5k="UNKNOWN",
        province="Test Province",
        width=200,
        height=150,
    )
    return clip, img_path


@pytest.fixture
def sample_proposal() -> Sam2Proposal:
    """Create a sample proposal."""
    return Sam2Proposal(
        bbox=[10.0, 20.0, 50.0, 60.0],
        polygon=[[10, 20], [50, 20], [50, 60], [10, 60]],
        confidence=0.85,
        label="mound",
    )


# ── Settings tests ────────────────────────────────────────────────


class TestPipelineSettings:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.sam2_mcp_url == "mock"
        assert s.sam2_mode == "mock"
        assert s.review_status == "pending_review"
        assert s.dry_run is False

    def test_output_path_resolves(self) -> None:
        s = Settings(output_root="./relative/path")
        assert s.output_path.is_absolute()

    def test_protocol_path_resolves(self) -> None:
        s = Settings(protocol_dir="./docs/annotation")
        assert s.protocol_path.is_absolute()


# ── Artifact tests ────────────────────────────────────────────────


class TestArtifacts:
    def test_create_run_directory(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        assert (run_dir / "raw" / "images").exists()
        assert (run_dir / "annotations").exists()
        assert (run_dir / "review" / "samples_for_review").exists()
        assert (run_dir / "masks").exists()

    def test_create_run_directory_dry_run(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run", dry_run=True)
        assert run_dir.name == "test_run_dry_run"
        assert (run_dir / "raw" / "images").exists()

    def test_save_clip_image(self, tmp_path: Path, sample_clip: ClipMetadata) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        img = Image.new("RGB", (200, 150), color=(100, 100, 100))
        saved = save_clip_image(sample_clip, img, run_dir)
        assert saved.exists()
        assert saved.suffix == ".png"
        assert sample_clip.clip_image_path

    def test_render_review_image(
        self, tmp_path: Path, sample_clip: ClipMetadata, sample_proposal: Sam2Proposal
    ) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        img = Image.new("RGB", (200, 150), color=(100, 100, 100))
        saved = render_review_image(sample_clip, img, [sample_proposal], run_dir)
        assert saved.exists()
        assert saved.suffix == ".png"

    def test_save_clip_image_path_under_run_dir(
        self, tmp_path: Path, sample_clip: ClipMetadata
    ) -> None:
        """Saved clip image path must stay under run directory."""
        run_dir = create_run_directory(tmp_path, "test_run")
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        saved = save_clip_image(sample_clip, img, run_dir)
        saved.resolve().relative_to(run_dir.resolve())


# ── CSV writer tests ──────────────────────────────────────────────


class TestCsvWriter:
    def test_write_annotation_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test.csv"
        rows = [
            AnnotationRow(
                sample_id="AG_001_obj0",
                annotator="AG",
                image_path="/tmp/img.png",
                label="mound",
                bbox_x_min=10.0,
                bbox_y_min=20.0,
                bbox_x_max=50.0,
                bbox_y_max=60.0,
                confidence=0.85,
                review_status="pending_review",
            ),
        ]
        written = write_annotation_csv(rows, csv_path)
        assert written == 1
        assert csv_path.exists()

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert csv_rows[0]["sample_id"] == "AG_001_obj0"
        assert csv_rows[0]["label"] == "mound"

    def test_write_csv_has_required_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "test_cols.csv"
        rows = [AnnotationRow(sample_id="test")]
        write_annotation_csv(rows, csv_path)

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "sample_id" in fieldnames
        assert "bbox_x_min" in fieldnames
        assert "review_status" in fieldnames
        assert "annotation_source" in fieldnames

    def test_append_csv_skips_existing(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "append_test.csv"
        row = AnnotationRow(
            sample_id="AG_001_obj0",
            label="mound",
            bbox_x_min=10.0,
            bbox_y_min=20.0,
            bbox_x_max=50.0,
            bbox_y_max=60.0,
        )
        write_annotation_csv([row], csv_path)
        appended = append_annotation_csv([row], csv_path)
        assert appended == 0

    def test_append_csv_adds_new(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "append_test2.csv"
        row1 = AnnotationRow(
            sample_id="AG_001_obj0",
            label="mound",
            bbox_x_min=10.0,
            bbox_y_min=20.0,
            bbox_x_max=50.0,
            bbox_y_max=60.0,
        )
        row2 = AnnotationRow(
            sample_id="AG_002_obj0",
            label="mound",
            bbox_x_min=15.0,
            bbox_y_min=25.0,
            bbox_x_max=55.0,
            bbox_y_max=65.0,
        )
        write_annotation_csv([row1], csv_path)
        appended = append_annotation_csv([row2], csv_path)
        assert appended == 1

    def test_polygon_to_json(self) -> None:
        poly = [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]]
        result = polygon_to_json(poly)
        parsed = json.loads(result)
        assert parsed == poly

    def test_polygon_to_json_empty(self) -> None:
        assert polygon_to_json([]) == ""


# ── Metadata tests ────────────────────────────────────────────────


class TestMetadata:
    def test_load_csv_sidecar(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "sidecar.csv"
        with open(sidecar, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "province"])
            writer.writeheader()
            writer.writerow({"sample_id": "AG_001", "province": "Dobrich"})
        result = load_sidecar_metadata(str(sidecar))
        assert "AG_001" in result
        assert result["AG_001"]["province"] == "Dobrich"

    def test_load_json_sidecar(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "sidecar.json"
        data = {"AG_001": {"province": "Sofia"}}
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(data, f)
        result = load_sidecar_metadata(str(sidecar))
        assert "AG_001" in result

    def test_load_sidecar_missing_file(self) -> None:
        result = load_sidecar_metadata("/nonexistent/file.csv")
        assert result == {}

    def test_merge_sidecar(self, sample_clip: ClipMetadata) -> None:
        sidecar = {
            "AG_001_K-35-50_UNKNOWN.png": {
                "province": "Test Province",
                "sheet_25k": "K-35-50-B-b",
            }
        }
        merge_sidecar(sample_clip, sidecar)
        assert sample_clip.province == "Test Province"

    def test_report_missing_metadata(self) -> None:
        clips = [
            ClipMetadata(
                source_pdf="/tmp/test.pdf",
                pdf_page=1,
                clip_index=0,
                sample_id="AG_001.png",
                province="",
                sheet_25k="",
            ),
        ]
        missing = report_missing_metadata(clips)
        assert any("province" in m for m in missing)
        assert any("sheet_25k" in m for m in missing)


# ── Validation tests ──────────────────────────────────────────────


class TestValidation:
    def test_validate_run_valid_csv(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        img_path = run_dir / "raw" / "images" / "AG" / "test.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img.save(str(img_path))

        csv_path = run_dir / "annotations" / "test.csv"
        rows = [
            AnnotationRow(
                sample_id="AG_001_obj0",
                image_path=str(img_path),
                label="mound",
                bbox_x_min=10.0,
                bbox_y_min=10.0,
                bbox_x_max=50.0,
                bbox_y_max=50.0,
                review_status="pending_review",
                annotation_status="annotated",
            ),
        ]
        write_annotation_csv(rows, csv_path)

        result = validate_run(csv_path, run_dir)
        assert result["errors"] == 0
        assert result["total_rows"] == 1

    def test_validate_run_duplicate_ids(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        csv_path = run_dir / "annotations" / "test.csv"
        rows = [
            AnnotationRow(
                sample_id="dup_id",
                label="mound",
                bbox_x_min=10.0,
                bbox_y_min=10.0,
                bbox_x_max=50.0,
                bbox_y_max=50.0,
                review_status="pending_review",
            ),
            AnnotationRow(
                sample_id="dup_id",
                label="mound",
                bbox_x_min=10.0,
                bbox_y_min=10.0,
                bbox_x_max=50.0,
                bbox_y_max=50.0,
                review_status="pending_review",
            ),
        ]
        write_annotation_csv(rows, csv_path)

        result = validate_run(csv_path, run_dir)
        assert result["errors"] > 0
        assert any("Duplicate" in e for e in result["error_messages"])

    def test_validate_run_invalid_bbox(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        csv_path = run_dir / "annotations" / "test.csv"
        rows = [
            AnnotationRow(
                sample_id="bad_bbox",
                label="mound",
                bbox_x_min=50.0,
                bbox_y_min=10.0,
                bbox_x_max=10.0,
                bbox_y_max=50.0,
                review_status="pending_review",
            ),
        ]
        write_annotation_csv(rows, csv_path)

        result = validate_run(csv_path, run_dir)
        assert result["errors"] > 0

    def test_validate_run_invalid_label(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        csv_path = run_dir / "annotations" / "test.csv"
        rows = [
            AnnotationRow(
                sample_id="bad_label",
                label="unknown_label",
                bbox_x_min=10.0,
                bbox_y_min=10.0,
                bbox_x_max=50.0,
                bbox_y_max=50.0,
                review_status="pending_review",
            ),
        ]
        write_annotation_csv(rows, csv_path)

        result = validate_run(csv_path, run_dir)
        assert result["warnings"] > 0

    def test_validate_bbox_in_image(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        img.save(str(img_path))

        assert validate_bbox_in_image([10, 10, 50, 50], str(img_path)) is True
        assert validate_bbox_in_image([0, 0, 200, 200], str(img_path)) is False
        assert validate_bbox_in_image([50, 10, 10, 50], str(img_path)) is False

    def test_validate_run_missing_csv(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        with pytest.raises(ValidationError, match="CSV not found"):
            validate_run(tmp_path / "nonexistent.csv", run_dir)

    def test_validate_run_path_escape(self, tmp_path: Path) -> None:
        run_dir = create_run_directory(tmp_path, "test_run")
        csv_path = run_dir / "annotations" / "test.csv"
        rows = [
            AnnotationRow(
                sample_id="escape_test",
                image_path="/etc/passwd",
                label="mound",
                bbox_x_min=10.0,
                bbox_y_min=10.0,
                bbox_x_max=50.0,
                bbox_y_max=50.0,
                review_status="pending_review",
            ),
        ]
        write_annotation_csv(rows, csv_path)

        result = validate_run(csv_path, run_dir)
        assert result["errors"] > 0
        assert any("escapes" in e for e in result["error_messages"])


# ── Annotation row builder tests ──────────────────────────────────


class TestAnnotationRowBuilder:
    def test_build_rows_with_proposals(self) -> None:
        clips = [
            ClipMetadata(
                source_pdf="/tmp/test.pdf",
                pdf_page=1,
                clip_index=0,
                clip_image_path="/tmp/clip.png",
                annotator="AG",
                sample_id="AG_001.png",
                sheet_25k="K-35-50",
            )
        ]
        proposals = {
            "AG_001.png": [
                Sam2Proposal(
                    bbox=[10, 20, 50, 60],
                    confidence=0.85,
                    label="mound",
                ),
            ]
        }
        rows = build_annotation_rows(clips, proposals, "mock", "pending_review")
        assert len(rows) == 1
        assert rows[0].sample_id == "AG_001.png_obj0"
        assert rows[0].label == "mound"
        assert rows[0].review_status == "pending_review"
        assert rows[0].model_backend == "mock"

    def test_build_rows_no_proposals(self) -> None:
        clips = [
            ClipMetadata(
                source_pdf="/tmp/test.pdf",
                pdf_page=1,
                clip_index=0,
                clip_image_path="/tmp/clip.png",
                annotator="AG",
                sample_id="AG_001.png",
            )
        ]
        rows = build_annotation_rows(clips, {}, "mock", "pending_review")
        assert len(rows) == 1
        assert rows[0].sample_id == "AG_001.png"
        assert rows[0].label == ""

    def test_build_rows_multiple_proposals(self) -> None:
        clips = [
            ClipMetadata(
                source_pdf="/tmp/test.pdf",
                pdf_page=1,
                clip_index=0,
                clip_image_path="/tmp/clip.png",
                annotator="AG",
                sample_id="AG_001.png",
            )
        ]
        proposals = {
            "AG_001.png": [
                Sam2Proposal(bbox=[10, 20, 50, 60], confidence=0.85, label="mound"),
                Sam2Proposal(bbox=[60, 70, 100, 110], confidence=0.75, label="mound"),
            ]
        }
        rows = build_annotation_rows(clips, proposals, "mock", "pending_review")
        assert len(rows) == 2
        assert rows[0].sample_id == "AG_001.png_obj0"
        assert rows[1].sample_id == "AG_001.png_obj1"


# ── Pipeline integration tests ────────────────────────────────────


class TestPipelineRunner:
    def test_rejects_missing_pdf(self, tmp_path: Path) -> None:
        with pytest.raises(PdfError, match="PDF not found"):
            run_pipeline(
                pdf_path="/nonexistent/file.pdf",
                output_root=str(tmp_path),
                run_id="test",
            )

    def test_rejects_missing_protocol_dir(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n")
        with pytest.raises(ValidationError, match="Protocol directory not found"):
            run_pipeline(
                pdf_path=str(pdf_path),
                output_root=str(tmp_path),
                run_id="test",
                protocol_dir="/nonexistent/protocol",
            )

    def test_dry_run_creates_isolated_folder(self, tmp_path: Path) -> None:
        """Dry run should create a _dry_run suffixed directory."""
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF")
        doc.save(str(pdf_path))
        doc.close()

        report = run_pipeline(
            pdf_path=str(pdf_path),
            output_root=str(tmp_path),
            run_id="test_run",
            dry_run=True,
            protocol_dir="docs/annotation",
        )
        assert "test_run_dry_run" in report.output_root

    def test_pipeline_report_has_required_fields(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test PDF")
        doc.save(str(pdf_path))
        doc.close()

        report = run_pipeline(
            pdf_path=str(pdf_path),
            output_root=str(tmp_path),
            run_id="test_run",
            protocol_dir="docs/annotation",
        )
        assert report.run_id == "test_run"
        assert report.pdf_path == str(pdf_path)
        assert report.sam2_mode == "mock"

    @pytest.mark.skip(reason="fitz/PyMuPDF is lenient with invalid PDF content")
    def test_rejects_invalid_pdf(self, tmp_path: Path) -> None:
        """Pipeline should reject a file that is not a valid PDF."""
        pdf_path = tmp_path / "fake.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b"not a pdf at all\x00\x00\x00\x00")
        with pytest.raises(PdfError, match="Cannot open PDF"):
            run_pipeline(
                pdf_path=str(pdf_path),
                output_root=str(tmp_path),
                run_id="test",
                protocol_dir="docs/annotation",
            )


# ── Package import tests ──────────────────────────────────────────


class TestPackage:
    def test_imports(self) -> None:
        import services.annotation_pipeline  # noqa: F401
        import services.annotation_pipeline.artifacts  # noqa: F401
        import services.annotation_pipeline.csv_writer  # noqa: F401
        import services.annotation_pipeline.errors  # noqa: F401
        import services.annotation_pipeline.metadata  # noqa: F401
        import services.annotation_pipeline.pdf_sam2_runner  # noqa: F401
        import services.annotation_pipeline.schemas  # noqa: F401
        import services.annotation_pipeline.settings  # noqa: F401
        import services.annotation_pipeline.validation  # noqa: F401
