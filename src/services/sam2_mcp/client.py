"""SAM2 backend HTTP client with mock fallback.

Talks to a configured SAM2 inference HTTP endpoint.  When the backend
is unreachable or when SAM2_MCP_BACKEND_URL is set to the sentinel
value ``mock``, the client falls back to a deterministic mock that
returns plausible masks for testing without a GPU backend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

from .errors import BackendError
from .schemas import (
    HealthResponse,
    ProposalResponse,
    SegmentationItem,
    SegmentResponse,
)
from .settings import get_settings

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class Sam2BackendClient:
    """HTTP client for the SAM2 inference backend."""

    def __init__(
        self,
        backend_url: str | None = None,
        timeout: float | None = None,
        output_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self.backend_url = (backend_url or settings.backend_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.backend_timeout
        self.output_dir = output_dir or str(settings.output_path)
        self._session = requests.Session()
        self._is_mock = self.backend_url.lower() == "mock"
        logger.info(
            "Sam2BackendClient initialised (mode=%s, url=%s, timeout=%.1f, output=%s)",
            "mock" if self._is_mock else "live",
            self.backend_url,
            self.timeout,
            self.output_dir,
        )

    # ── health ────────────────────────────────────────────────────

    def check_health(self) -> dict[str, Any]:
        """Probe the backend and return a health dict."""
        if self._is_mock:
            logger.debug("Mock health check (always healthy)")
            return {
                "mode": "mock",
                "backend_url": self.backend_url,
                "reachable": True,
            }

        logger.info("Live health check for %s/health", self.backend_url)
        try:
            resp = self._session.get(
                f"{self.backend_url}/health",
                timeout=min(self.timeout, 5.0),
            )
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    HealthResponse(**data)
                    logger.info("Health check OK (status=%d)", resp.status_code)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Health returned invalid JSON: %s", exc)
                    data = resp.text[:200]
                return {
                    "mode": "live",
                    "backend_url": self.backend_url,
                    "reachable": True,
                    "status_code": resp.status_code,
                    "response": data,
                }
            logger.warning("Health check failed (status=%d)", resp.status_code)
            return {
                "mode": "live",
                "backend_url": self.backend_url,
                "reachable": False,
                "status_code": resp.status_code,
                "response": resp.text[:200],
            }
        except requests.ConnectionError as exc:
            logger.error("Health check connection error: %s", exc)
            return {
                "mode": "live",
                "backend_url": self.backend_url,
                "reachable": False,
                "error": "connection_refused",
            }
        except requests.Timeout as exc:
            logger.error("Health check timeout: %s", exc)
            return {
                "mode": "live",
                "backend_url": self.backend_url,
                "reachable": False,
                "error": "timeout",
            }

    # ── segmentation ──────────────────────────────────────────────

    def segment_box(
        self,
        image_path: str,
        bbox: list[float],
        label: str = "",
    ) -> SegmentationItem:
        """Segment using a bounding-box prompt."""
        if self._is_mock:
            logger.debug("Mock segment_box for %s", image_path)
            return self._mock_segment(image_path, bbox, label=label)
        logger.info("Live segment_box for %s bbox=%s", image_path, bbox)
        return self._live_segment_box(image_path, bbox, label)

    def segment_points(
        self,
        image_path: str,
        points: list[list[float]],
        point_labels: list[int],
        label: str = "",
    ) -> SegmentationItem:
        """Segment using point prompts."""
        if self._is_mock:
            logger.debug("Mock segment_points for %s (%d points)", image_path, len(points))
            return self._mock_segment_points(image_path, points, point_labels, label)
        logger.info("Live segment_points for %s (%d points)", image_path, len(points))
        return self._live_segment_points(image_path, points, point_labels, label)

    def generate_proposals(
        self,
        image_path: str,
        max_proposals: int = 50,
        label_hint: str = "",
    ) -> list[SegmentationItem]:
        """Generate candidate proposals for an image."""
        if self._is_mock:
            logger.debug("Mock proposals for %s (max=%d)", image_path, max_proposals)
            return self._mock_proposals(image_path, max_proposals, label_hint)
        logger.info(
            "Live proposals for %s (max=%d, hint=%s)",
            image_path,
            max_proposals,
            label_hint,
        )
        return self._live_proposals(image_path, max_proposals, label_hint)

    # ── live backend calls ────────────────────────────────────────

    def _live_segment_box(
        self,
        image_path: str,
        bbox: list[float],
        label: str,
    ) -> SegmentationItem:
        payload = {
            "image_path": image_path,
            "prompt_type": "bbox",
            "bbox": bbox,
            "label": label,
            "output_dir": self.output_dir,
        }
        resp = self._post_with_validation(
            f"{self.backend_url}/segment",
            payload,
            "segment_box",
        )
        data = self._parse_json_response(resp, "segment_box")
        seg = self._validate_segment_response(data, image_path)
        self._validate_mask_path(seg.mask_path, "segment_box")
        return self._to_segmentation_item(seg)

    def _live_segment_points(
        self,
        image_path: str,
        points: list[list[float]],
        point_labels: list[int],
        label: str,
    ) -> SegmentationItem:
        payload = {
            "image_path": image_path,
            "prompt_type": "points",
            "points": points,
            "point_labels": point_labels,
            "label": label,
            "output_dir": self.output_dir,
        }
        resp = self._post_with_validation(
            f"{self.backend_url}/segment",
            payload,
            "segment_points",
        )
        data = self._parse_json_response(resp, "segment_points")
        seg = self._validate_segment_response(data, image_path)
        self._validate_mask_path(seg.mask_path, "segment_points")
        return self._to_segmentation_item(seg)

    def _live_proposals(
        self,
        image_path: str,
        max_proposals: int,
        label_hint: str,
    ) -> list[SegmentationItem]:
        payload = {
            "image_path": image_path,
            "max_proposals": max_proposals,
            "label_hint": label_hint,
            "output_dir": self.output_dir,
        }
        resp = self._post_with_validation(
            f"{self.backend_url}/propose",
            payload,
            "propose",
        )
        data = self._parse_json_response(resp, "propose")
        return self._validate_proposal_response(data, image_path)

    # ── HTTP helpers ──────────────────────────────────────────────

    def _post_with_validation(
        self, url: str, payload: dict[str, Any], op_name: str
    ) -> requests.Response:
        """POST with status-code validation."""
        resp = self._session.post(
            url,
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            body = resp.text[:300]
            logger.error("Backend %s returned status %d: %s", op_name, resp.status_code, body)
            raise BackendError(f"SAM2 backend {op_name} returned HTTP {resp.status_code}: {body}")
        return resp

    @staticmethod
    def _parse_json_response(resp: requests.Response, op_name: str) -> dict[str, Any]:
        """Parse JSON from response with clear error."""
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Backend %s returned invalid JSON: %s", op_name, exc)
            raise BackendError(f"SAM2 backend {op_name} returned invalid JSON: {exc}") from exc

    # ── response validators ───────────────────────────────────────

    @staticmethod
    def _validate_segment_response(data: dict[str, Any], expected_image: str) -> SegmentResponse:
        """Validate and parse a segment response."""
        try:
            return SegmentResponse(**data)
        except ValueError as exc:
            logger.error("Invalid segment response: %s", exc)
            raise BackendError(f"SAM2 backend returned invalid segment data: {exc}") from exc

    def _validate_proposal_response(
        self, data: dict[str, Any], expected_image: str
    ) -> list[SegmentationItem]:
        """Validate and parse a proposal response."""
        try:
            resp = ProposalResponse(**data)
        except ValueError as exc:
            logger.error("Invalid proposal response: %s", exc)
            raise BackendError(f"SAM2 backend returned invalid proposal data: {exc}") from exc

        items: list[SegmentationItem] = []
        for i, item in enumerate(resp.items):
            self._validate_mask_path(item.mask_path, f"proposal[{i}]")
            items.append(
                SegmentationItem(
                    image_path=item.image_path,
                    bbox=item.bbox,
                    polygon=item.polygon,
                    mask_path=item.mask_path,
                    confidence=item.confidence,
                    label=item.label,
                    source="sam2_mcp",
                    status="pending_review",
                )
            )
        logger.info("Proposal response validated (%d items)", len(items))
        return items

    def _validate_mask_path(self, mask_path: str, context: str) -> None:
        """Ensure mask_path is under the configured output directory."""
        if not mask_path:
            return
        out = Path(self.output_dir).resolve()
        mask = Path(mask_path).resolve()
        try:
            mask.relative_to(out)
        except ValueError:
            logger.error(
                "mask_path %s is outside output_dir %s (%s)", mask_path, self.output_dir, context
            )
            raise BackendError(
                f"Backend returned mask_path outside output_dir: {mask_path!r}"
            ) from None

    @staticmethod
    def _to_segmentation_item(seg: SegmentResponse) -> SegmentationItem:
        """Convert backend SegmentResponse to internal SegmentationItem."""
        return SegmentationItem(
            image_path=seg.image_path,
            bbox=seg.bbox,
            polygon=seg.polygon,
            mask_path=seg.mask_path,
            confidence=seg.confidence,
            label=seg.label,
            source="sam2_mcp",
            status="pending_review",
        )

    # ── mock implementations ──────────────────────────────────────

    @staticmethod
    def _mock_segment(
        image_path: str,
        bbox: list[float],
        label: str = "",
    ) -> SegmentationItem:
        """Return a mock segmentation tightly fitting the input bbox."""
        xmin, ymin, xmax, ymax = bbox
        margin = 0.02
        polygon = [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ]
        return SegmentationItem(
            image_path=image_path,
            bbox=bbox,
            polygon=polygon,
            confidence=round(0.85 - margin, 4),
            label=label,
            source="sam2_mcp",
            status="pending_review",
        )

    @staticmethod
    def _mock_segment_points(
        image_path: str,
        points: list[list[float]],
        point_labels: list[int],
        label: str = "",
    ) -> SegmentationItem:
        """Return a mock segmentation around foreground points."""
        fg_points = [p for p, lbl in zip(points, point_labels, strict=True) if lbl == 1]
        if not fg_points:
            return SegmentationItem(
                image_path=image_path,
                bbox=[0.0, 0.0, 0.0, 0.0],
                polygon=[],
                confidence=0.0,
                label=label,
                source="sam2_mcp",
                status="pending_review",
            )
        xs = [p[0] for p in fg_points]
        ys = [p[1] for p in fg_points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        spread = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            0.1,
        )
        half = spread * 0.6
        bbox = [round(cx - half, 4), round(cy - half, 4), round(cx + half, 4), round(cy + half, 4)]
        polygon = [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ]
        return SegmentationItem(
            image_path=image_path,
            bbox=bbox,
            polygon=polygon,
            confidence=0.75,
            label=label,
            source="sam2_mcp",
            status="pending_review",
        )

    @staticmethod
    def _mock_proposals(
        image_path: str,
        max_proposals: int,
        label_hint: str = "",
    ) -> list[SegmentationItem]:
        """Return 1-3 mock proposals for the image."""
        n = min(max_proposals, 3)
        items: list[SegmentationItem] = []
        for i in range(n):
            offset = i * 0.25
            bbox = [
                round(0.1 + offset, 4),
                round(0.1 + offset * 0.5, 4),
                round(0.25 + offset, 4),
                round(0.25 + offset * 0.5, 4),
            ]
            polygon = [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ]
            items.append(
                SegmentationItem(
                    image_path=image_path,
                    bbox=bbox,
                    polygon=polygon,
                    confidence=round(0.7 - i * 0.1, 4),
                    label=label_hint or f"proposal_{i}",
                    source="sam2_mcp",
                    status="pending_review",
                )
            )
        return items

    @staticmethod
    def discover_images(directory: str) -> list[str]:
        """Return sorted list of image file paths in a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []
        return sorted(
            str(f)
            for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
