"""SAM2 backend HTTP client with mock fallback.

Talks to a configured SAM2 inference HTTP endpoint.  When the backend
is unreachable or when SAM2_MCP_BACKEND_URL is set to the sentinel
value ``mock``, the client falls back to a deterministic mock that
returns plausible masks for testing without a GPU backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from .errors import BackendError
from .schemas import SegmentationItem
from .settings import get_settings

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class Sam2BackendClient:
    """HTTP client for the SAM2 inference backend."""

    def __init__(
        self,
        backend_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.backend_url = (backend_url or settings.backend_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.backend_timeout
        self._session = requests.Session()
        self._is_mock = self.backend_url.lower() == "mock"

    # ── health ────────────────────────────────────────────────────

    def check_health(self) -> dict[str, Any]:
        """Probe the backend and return a health dict."""
        if self._is_mock:
            return {
                "mode": "mock",
                "backend_url": self.backend_url,
                "reachable": True,
            }

        try:
            resp = self._session.get(
                f"{self.backend_url}/health",
                timeout=min(self.timeout, 5.0),
            )
            return {
                "mode": "live",
                "backend_url": self.backend_url,
                "reachable": resp.status_code == 200,
                "status_code": resp.status_code,
                "response": resp.json() if resp.status_code == 200 else resp.text[:200],
            }
        except requests.ConnectionError:
            return {
                "mode": "live",
                "backend_url": self.backend_url,
                "reachable": False,
                "error": "connection_refused",
            }
        except requests.Timeout:
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
            return self._mock_segment(image_path, bbox, label=label)
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
            return self._mock_segment_points(image_path, points, point_labels, label)
        return self._live_segment_points(image_path, points, point_labels, label)

    def generate_proposals(
        self,
        image_path: str,
        max_proposals: int = 50,
        label_hint: str = "",
    ) -> list[SegmentationItem]:
        """Generate candidate proposals for an image."""
        if self._is_mock:
            return self._mock_proposals(image_path, max_proposals, label_hint)
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
        }
        resp = self._session.post(
            f"{self.backend_url}/segment",
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise BackendError(f"SAM2 backend returned {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return SegmentationItem(**data)

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
        }
        resp = self._session.post(
            f"{self.backend_url}/segment",
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise BackendError(f"SAM2 backend returned {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return SegmentationItem(**data)

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
        }
        resp = self._session.post(
            f"{self.backend_url}/propose",
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise BackendError(f"SAM2 backend returned {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        items = data.get("items", data if isinstance(data, list) else [])
        return [SegmentationItem(**item) for item in items]

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
