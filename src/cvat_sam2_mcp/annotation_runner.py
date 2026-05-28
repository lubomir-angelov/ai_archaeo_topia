"""Annotation protocol parser and pipeline orchestration.

Supports YAML protocol files (PyYAML is already a dependency).
Falls back to simple JSON if YAML parsing fails.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import time

from .cvat_client import CvatClient
from .models import (
    AnnotationProtocol,
    AnnotationRunReport,
)
from .sam2_client import Sam2Client
from .settings import get_settings

logger = logging.getLogger(__name__)


# ── Protocol parsing ──────────────────────────────────────────────


def parse_protocol(path: str) -> AnnotationProtocol:
    """Parse an annotation protocol from a YAML or JSON file.

    Tries YAML first (PyYAML is a project dependency), falls back to JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    content = p.read_text(encoding="utf-8")

    # Try YAML first
    try:
        import yaml

        data = yaml.safe_load(content)
        if isinstance(data, dict):
            return AnnotationProtocol(**data)
    except ImportError:
        logger.warning("PyYAML not available, trying JSON")
    except Exception:
        logger.debug("YAML parse failed, trying JSON")

    # Try JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return AnnotationProtocol(**data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse protocol from {path}: {exc}") from exc

    raise ValueError(f"Protocol file {path} did not contain a valid YAML/JSON mapping")


def protocol_to_cvat_labels(protocol: AnnotationProtocol) -> list[dict]:
    """Convert protocol labels to CVAT label format."""
    type_map = {
        "mask": "mask",
        "rectangle": "rectangle",
        "polygon": "polygon",
        "points": "points",
    }
    labels = []
    for lbl in protocol.labels:
        labels.append(
            {
                "title": lbl.name,
                "color": lbl.color,
                "type": type_map.get(lbl.type.value, "mask"),
                "attributes": [],
            }
        )
    return labels


# ── Pipeline orchestration ────────────────────────────────────────


def _make_run_id() -> str:
    return f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def start_run(
    protocol_path: str,
    input_dir: str,
    project_id: int | None = None,
    task_name: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the full annotation pipeline.

    Steps:
    1. Parse/validate protocol
    2. Create or reuse CVAT task
    3. Export task data
    4. Run SAM2 candidate generation
    5. Import candidates (unless dry_run)
    6. Write report JSON

    Returns a compact dict with run_id, task_id, artifact paths, status.
    """
    settings = get_settings()
    run_id = _make_run_id()
    run_dir = settings.artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = AnnotationRunReport(
        run_id=run_id,
        protocol_path=protocol_path,
        input_dir=input_dir,
        project_id=project_id,
        dry_run=dry_run,
    )

    cvat = CvatClient()
    sam2 = Sam2Client()

    try:
        # Step 1: Parse protocol
        t0 = time()
        protocol = parse_protocol(protocol_path)
        report.add_step(
            "parse_protocol",
            "success",
            {"version": protocol.protocol_version, "labels": len(protocol.labels)},
        )
        logger.info("Step 1 done: protocol parsed (%.1fs)", time() - t0)

        # Save protocol snapshot
        proto_snap = run_dir / f"protocol_snapshot_{protocol.protocol_version}.yaml"
        import yaml

        proto_snap.write_text(
            yaml.dump(protocol.model_dump(mode="json"), default_flow_style=False)
        )

        # Step 2: Create or reuse task
        t0 = time()
        cvat_labels = protocol_to_cvat_labels(protocol)

        if task_name is None:
            task_name = f"archaeo_auto_{run_id}"

        existing = cvat.find_task_by_name(task_name, project_id=project_id)
        if existing is not None:
            report.add_step(
                "create_task", "reused", {"task_id": existing.id, "name": existing.name}
            )
            logger.info("Step 2 done: reused task %d", existing.id)
            task_id = existing.id
        else:
            # Validate input dir
            img_path = Path(input_dir)
            if not img_path.is_dir():
                raise ValueError(f"Input directory does not exist: {input_dir}")

            image_files = list(img_path.iterdir())
            image_count = len(
                [
                    f
                    for f in image_files
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
                ]
            )
            if image_count == 0:
                raise ValueError(f"No images found in {input_dir}")

            if dry_run:
                report.add_step(
                    "create_task",
                    "dry_run",
                    {
                        "task_name": task_name,
                        "image_count": image_count,
                        "labels": cvat_labels,
                    },
                )
                logger.info(
                    "Step 2 done: dry-run, would create task '%s' with %d images",
                    task_name,
                    image_count,
                )
                task_id = None
            else:
                task_id = cvat.create_task(
                    name=task_name,
                    image_dir=input_dir,
                    labels=cvat_labels,
                    project_id=project_id,
                )
                report.add_step("create_task", "created", {"task_id": task_id, "name": task_name})
                logger.info("Step 2 done: created task %d", task_id)

        report.task_id = task_id

        # Step 3: Export task
        t0 = time()
        if dry_run or task_id is None:
            report.add_step("export_task", "skipped_dry_run")
        else:
            export_path = cvat.export_task(task_id, artifact_dir=str(run_dir))
            report.add_step("export_task", "success", {"export_path": export_path})
            logger.info("Step 3 done: exported to %s (%.1fs)", export_path, time() - t0)

        # Step 4: SAM2 candidate generation
        t0 = time()
        if dry_run:
            report.add_step(
                "sam2_generate",
                "dry_run",
                {
                    "image_dir": input_dir,
                    "max_masks": protocol.sam2.max_masks_per_image,
                },
            )
            logger.info("Step 4 done: dry-run, would run SAM2 on %s", input_dir)
        else:
            sam2_dir = run_dir / "sam2_artifacts"
            sam2_result = sam2.generate_candidates(
                image_dir=input_dir,
                artifact_dir=str(sam2_dir),
                max_masks_per_image=protocol.sam2.max_masks_per_image,
                protocol_path=protocol_path,
            )
            # Save SAM2 result
            (run_dir / "sam2_candidates.json").write_text(
                json.dumps(sam2_result.model_dump(), indent=2)
            )
            report.add_step(
                "sam2_generate",
                sam2_result.status,
                {
                    "total_images": sam2_result.total_images,
                    "successful": len([c for c in sam2_result.candidates if c.mask_count > 0]),
                    "failed": len([c for c in sam2_result.candidates if c.mask_count == 0]),
                },
            )
            logger.info("Step 4 done: SAM2 status=%s (%.1fs)", sam2_result.status, time() - t0)

        # Step 5: Import candidates
        t0 = time()
        if dry_run:
            report.add_step("import_annotations", "skipped_dry_run")
            report.status = "dry_run_complete"
        elif task_id is not None:
            # Look for SAM2 artifacts to import
            sam2_artifacts = run_dir / "sam2_artifacts"
            if sam2_artifacts.exists() and (sam2_artifacts / "sam2_candidates.json").exists():
                import_result = cvat.import_annotations(
                    task_id=task_id,
                    annotations_path=str(sam2_artifacts / "sam2_candidates.json"),
                )
                report.add_step("import_annotations", "success", import_result)
                report.status = "complete"
            else:
                report.add_step("import_annotations", "skipped_no_artifacts")
                report.status = "incomplete"
            logger.info("Step 5 done: import (%.1fs)", time() - t0)
        else:
            report.add_step("import_annotations", "skipped_no_task")
            report.status = "incomplete"

        # Step 6: Write report
        report.completed_at = datetime.now(UTC).isoformat()
        if report.status == "pending":
            report.status = "complete"

        # Build next steps
        next_steps = []
        if dry_run:
            next_steps = [
                "Review the dry-run report",
                "Run without --dry-run to execute the pipeline",
            ]
        elif report.status == "complete":
            next_steps = [
                "Review candidate annotations in CVAT",
                f"Open CVAT task {task_id} for human verification",
                "Approve or reject candidate masks",
            ]
        else:
            next_steps = [
                "Check logs for errors",
                "Verify SAM2 function is deployed and reachable",
                "Re-run the pipeline after fixing issues",
            ]
        report.next_steps = next_steps

        report_path = run_dir / "run_report.json"
        report_path.write_text(report.model_dump_json(indent=2))
        logger.info("Report written to %s", report_path)

    except Exception as exc:
        report.status = "error"
        report.completed_at = datetime.now(UTC).isoformat()
        report.add_step("pipeline_error", "error", {"error": str(exc)})
        report.next_steps = ["Check logs for details", "Fix the error and re-run"]
        report_path = run_dir / "run_report.json"
        report_path.write_text(report.model_dump_json(indent=2))
        logger.error("Pipeline failed: %s", exc)
        raise

    return report.model_dump()


def get_report(run_id: str) -> dict:
    """Read and return a saved run report."""
    settings = get_settings()
    report_path = settings.artifact_root / run_id / "run_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    return json.loads(report_path.read_text())


def list_runs() -> list[str]:
    """List all run IDs found under the artifact root."""
    settings = get_settings()
    root = settings.artifact_root
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])
