"""Core segmentation service logic.

Validates inputs, calls the backend client, converts masks to polygons,
and writes output artifacts under the configured output directory.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .client import Sam2BackendClient
from .errors import BackendError, ImageError, OutputError, ValidationError
from .schemas import (
    GenerateProposalsInput,
    HealthOutput,
    ProposalsOutput,
    SegmentationItem,
    SegmentBoxInput,
    SegmentBoxOutput,
    SegmentPointsInput,
    SegmentPointsOutput,
)
from .settings import get_settings

logger = logging.getLogger(__name__)


class Sam2Service:
    """Business-logic layer for SAM2 segmentation operations."""

    def __init__(self, client: Sam2BackendClient | None = None) -> None:
        self.client = client or Sam2BackendClient()
        self.settings = get_settings()
        logger.info(
            "Sam2Service initialised (backend=%s, output=%s)",
            self.client.backend_url,
            self.settings.output_dir,
        )

    # ── health ────────────────────────────────────────────────────

    def health_check(self) -> HealthOutput:
        """Check service and backend health."""
        details = self.client.check_health()
        ok = details.get("reachable", False)
        result = HealthOutput(
            ok=ok,
            service="sam2_mcp",
            backend=self.client.backend_url,
            details=details,
        )
        logger.info("Health check result: ok=%s backend=%s", ok, self.client.backend_url)
        return result

    # ── box segmentation ─────────────────────────────────────────

    def segment_box(self, inp: SegmentBoxInput) -> SegmentBoxOutput:
        """Segment an image using a bounding-box prompt."""
        self._validate_image_path(inp.image_path)
        self._validate_bbox(inp.bbox, inp.image_path)
        item = self.client.segment_box(inp.image_path, inp.bbox, inp.label)
        self._validate_segmentation_item(item, "segment_box")
        return self._save_and_return_box(item, inp.image_path)

    # ── point segmentation ───────────────────────────────────────

    def segment_points(self, inp: SegmentPointsInput) -> SegmentPointsOutput:
        """Segment an image using point prompts."""
        self._validate_image_path(inp.image_path)
        self._validate_points(inp.points, inp.point_labels, inp.image_path)
        item = self.client.segment_points(inp.image_path, inp.points, inp.point_labels, inp.label)
        self._validate_segmentation_item(item, "segment_points")
        return self._save_and_return_points(item, inp.image_path, inp.points, inp.point_labels)

    # ── proposals ────────────────────────────────────────────────

    def generate_proposals(self, inp: GenerateProposalsInput) -> ProposalsOutput:
        """Generate candidate segmentation proposals."""
        if inp.image_path is None and inp.image_dir is None:
            raise ValidationError("Either image_path or image_dir must be provided")
        if inp.image_path is not None and inp.image_dir is not None:
            raise ValidationError("Provide only one of image_path or image_dir, not both")

        image_paths: list[str] = []
        if inp.image_path is not None:
            self._validate_image_path(inp.image_path)
            image_paths = [inp.image_path]
        else:
            image_paths = self.client.discover_images(inp.image_dir)
            if not image_paths:
                raise ValidationError(f"No images found in directory: {inp.image_dir}")

        max_prop = min(inp.max_proposals, self.settings.max_proposals_per_image)
        all_items: list[SegmentationItem] = []
        for img_path in image_paths:
            try:
                items = self.client.generate_proposals(
                    img_path, max_proposals=max_prop, label_hint=inp.label_hint
                )
                for item in items:
                    self._validate_segmentation_item(item, "generate_proposals")
                all_items.extend(items)
            except BackendError as exc:
                logger.warning("Backend error for %s: %s", img_path, exc)

        logger.info("Generated %d proposals for %d image(s)", len(all_items), len(image_paths))
        return ProposalsOutput(items=all_items)

    # ── output validation ────────────────────────────────────────

    def _validate_segmentation_item(self, item: SegmentationItem, context: str) -> None:
        """Validate a segmentation item returned by the backend."""
        if item.source != "sam2_mcp":
            item.source = "sam2_mcp"
        if item.status != "pending_review":
            item.status = "pending_review"

        if len(item.bbox) != 4:
            raise BackendError(
                f"Invalid bbox in {context} response: expected 4 values, got {len(item.bbox)}"
            )
        for v in item.bbox:
            if not isinstance(v, (int, float)):
                raise BackendError(f"Non-numeric bbox value in {context} response: {v!r}")

        for _i, pt in enumerate(item.polygon):
            if len(pt) != 2:
                raise BackendError(
                    f"Invalid polygon vertex in {context} response: expected [x, y], got {pt!r}"
                )
            for v in pt:
                if not isinstance(v, (int, float)):
                    raise BackendError(f"Non-numeric polygon value in {context} response: {v!r}")

        if item.mask_path:
            self._validate_mask_path(item.mask_path, context)

    def _validate_mask_path(self, mask_path: str, context: str) -> None:
        """Ensure mask_path is under the configured output directory."""
        out = self.settings.output_path.resolve()
        mask = Path(mask_path).resolve()
        try:
            mask.relative_to(out)
        except ValueError:
            raise BackendError(
                f"Backend returned mask_path outside output_dir in {context}: "
                f"{mask_path!r} (output_dir={self.settings.output_dir})"
            ) from None

    # ── validation helpers ────────────────────────────────────────

    def _validate_image_path(self, image_path: str) -> None:
        """Validate that an image file exists and is readable."""
        p = Path(image_path)
        if not p.exists():
            raise ImageError(f"Image not found: {image_path}")
        if not p.is_file():
            raise ImageError(f"Not a file: {image_path}")
        try:
            img = Image.open(p)
            img.verify()
        except Exception as exc:
            raise ImageError(f"Cannot read image {image_path}: {exc}") from exc

        img = Image.open(p)
        w, h = img.size
        if w > self.settings.max_image_size or h > self.settings.max_image_size:
            raise ImageError(
                f"Image {w}x{h} exceeds max dimension {self.settings.max_image_size}px"
            )

    def _validate_bbox(self, bbox: list[float], image_path: str) -> None:
        """Validate bounding-box coordinates."""
        if len(bbox) != 4:
            raise ValidationError("bbox must have exactly 4 values")
        for v in bbox:
            if not isinstance(v, (int, float)):
                raise ValidationError(f"bbox values must be numeric, got {type(v).__name__}")
        xmin, ymin, xmax, ymax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if xmin >= xmax:
            raise ValidationError(f"bbox x_min ({xmin}) must be < x_max ({xmax})")
        if ymin >= ymax:
            raise ValidationError(f"bbox y_min ({ymin}) must be < y_max ({ymax})")

        img = Image.open(image_path)
        w, h = img.size
        if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
            raise ValidationError(
                f"bbox [{xmin},{ymin},{xmax},{ymax}] outside image dimensions {w}x{h}"
            )

    def _validate_points(
        self,
        points: list[list[float]],
        point_labels: list[int],
        image_path: str,
    ) -> None:
        """Validate point prompts."""
        if len(points) == 0:
            raise ValidationError("points must not be empty")
        if len(points) != len(point_labels):
            raise ValidationError(
                f"points ({len(points)}) and point_labels ({len(point_labels)}) length mismatch"
            )
        for i, pt in enumerate(points):
            if len(pt) != 2:
                raise ValidationError(f"points[{i}] must have exactly 2 values")
            for v in pt:
                if not isinstance(v, (int, float)):
                    raise ValidationError(f"point values must be numeric, got {type(v).__name__}")
        for lbl in point_labels:
            if lbl not in (0, 1):
                raise ValidationError(f"point_labels must be 0 or 1, got {lbl}")

        img = Image.open(image_path)
        w, h = img.size
        for i, pt in enumerate(points):
            x, y = float(pt[0]), float(pt[1])
            if x < 0 or y < 0 or x > w or y > h:
                raise ValidationError(f"points[{i}] ({x},{y}) outside image dimensions {w}x{h}")

    # ── artifact persistence ─────────────────────────────────────

    def _make_mask_filename(self, image_path: str, prompt_hash: str) -> str:
        """Build a deterministic mask filename."""
        stem = Path(image_path).stem
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        short_hash = prompt_hash[:12]
        return f"mask_{stem}_{ts}_{short_hash}.png"

    def _prompt_hash(self, data: Any) -> str:
        """Create a short hash from prompt data for deterministic naming."""
        raw = str(data).encode()
        return hashlib.sha256(raw).hexdigest()

    def _write_mask(self, mask_array: Any, filename: str) -> str:
        """Write a binary mask to the output directory. Returns path."""
        out = self.settings.output_path
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        try:
            if hasattr(mask_array, "save"):
                mask_array.save(str(path), "PNG")
            else:
                img = Image.fromarray(mask_array)
                img.save(str(path), "PNG")
        except Exception as exc:
            raise OutputError(f"Failed to write mask {path}: {exc}") from exc
        logger.info("Wrote mask to %s", path)
        return str(path)

    def _save_and_return_box(self, item: SegmentationItem, image_path: str) -> SegmentBoxOutput:
        """Persist mask (if available) and return structured output."""
        mask_path = item.mask_path
        if not mask_path:
            prompt_h = self._prompt_hash(item.bbox)
            fname = self._make_mask_filename(image_path, prompt_h)
            try:
                img = Image.open(image_path)
                w, h = img.size
                xmin, ymin, xmax, ymax = item.bbox
                mask = Image.new("L", (w, h), 0)
                mask.putpixel((int(xmin), int(ymin)), 255)
                mask.putpixel((int(xmax), int(ymax)), 255)
                mask_path = self._write_mask(mask, fname)
            except Exception as exc:
                logger.warning("Failed to write mask: %s", exc)

        result = SegmentBoxOutput(
            image_path=item.image_path,
            label=item.label,
            bbox=item.bbox,
            mask_path=mask_path,
            polygon=item.polygon,
            confidence=item.confidence,
            source="sam2_mcp",
            status="pending_review",
        )
        logger.info(
            "segment_box result: image=%s mask=%s confidence=%.4f",
            result.image_path,
            mask_path,
            result.confidence,
        )
        return result

    def _save_and_return_points(
        self,
        item: SegmentationItem,
        image_path: str,
        points: list[list[float]],
        point_labels: list[int],
    ) -> SegmentPointsOutput:
        """Persist mask (if available) and return structured output."""
        mask_path = item.mask_path
        if not mask_path:
            prompt_h = self._prompt_hash(str(points) + str(point_labels))
            fname = self._make_mask_filename(image_path, prompt_h)
            try:
                img = Image.open(image_path)
                w, h = img.size
                mask = Image.new("L", (w, h), 0)
                for pt in points:
                    px, py = int(pt[0]), int(pt[1])
                    if 0 <= px < w and 0 <= py < h:
                        mask.putpixel((px, py), 255)
                mask_path = self._write_mask(mask, fname)
            except Exception as exc:
                logger.warning("Failed to write mask: %s", exc)

        result = SegmentPointsOutput(
            image_path=item.image_path,
            label=item.label,
            points=points,
            point_labels=point_labels,
            mask_path=mask_path,
            polygon=item.polygon,
            confidence=item.confidence,
            source="sam2_mcp",
            status="pending_review",
        )
        logger.info(
            "segment_points result: image=%s mask=%s confidence=%.4f",
            result.image_path,
            mask_path,
            result.confidence,
        )
        return result
