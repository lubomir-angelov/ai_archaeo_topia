"""SAM2 / Nuclio function client.

Invokes the deployed SAM2 GPU function via the Nuclio dashboard API.
The SAM2 function returns embedding features (blob), not masks directly.
Mask generation from features is handled by CVAT's serverless interactor.

TODO: If a dedicated SAM2 mask-generation endpoint becomes available,
replace the feature-extraction approach with direct mask generation.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import requests

from .models import Sam2Candidate, Sam2GenerationResult
from .settings import get_settings

logger = logging.getLogger(__name__)


class Sam2Client:
    """Client for invoking the SAM2 Nuclio function."""

    def __init__(self, nuclio_url: str | None = None, function_name: str | None = None):
        settings = get_settings()
        self.nuclio_url = (nuclio_url or settings.nuclio_dashboard_url).rstrip("/")
        self.function_name = function_name or settings.sam2_function_name
        self._session = requests.Session()

    def discover_function(self) -> dict[str, Any] | None:
        """Discover the SAM2 function via Nuclio dashboard API.

        Returns the function metadata dict or None if not found.
        """
        try:
            resp = self._session.get(f"{self.nuclio_url}/api/functions", timeout=10)
            resp.raise_for_status()
            functions = resp.json()
            for name, fn in functions.items():
                if self.function_name in name or name == self.function_name:
                    logger.info("Found SAM2 function: %s", name)
                    return fn
            logger.warning(
                "SAM2 function '%s' not found in Nuclio. Available: %s",
                self.function_name,
                list(functions.keys()),
            )
        except requests.ConnectionError:
            logger.error("Cannot reach Nuclio dashboard at %s", self.nuclio_url)
        except requests.HTTPError as exc:
            logger.error("Nuclio API error: %s", exc)
        return None

    def check_reachable(self) -> bool:
        """Check if the SAM2 function is reachable."""
        return self.discover_function() is not None

    def generate_candidates(
        self,
        image_dir: str,
        artifact_dir: str,
        max_masks_per_image: int = 200,
        protocol_path: str | None = None,
    ) -> Sam2GenerationResult:
        """Run SAM2 candidate mask generation for images in a directory.

        For each image:
        1. Encode as base64
        2. Send to SAM2 Nuclio function
        3. Save embedding blob and metadata

        Returns a Sam2GenerationResult with counts and artifact paths.
        """
        img_path = Path(image_dir)
        if not img_path.is_dir():
            return Sam2GenerationResult(
                status="error",
                total_images=0,
                error=f"Image directory not found: {image_dir}",
            )

        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [
                f
                for f in img_path.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
            ]
        )

        if not image_files:
            return Sam2GenerationResult(
                status="error",
                total_images=0,
                error=f"No images found in {image_dir}",
            )

        # Discover function to get port
        fn_meta = self.discover_function()
        if fn_meta is None:
            return Sam2GenerationResult(
                status="error",
                total_images=len(image_files),
                error=(
                    f"SAM2 function '{self.function_name}' not found in Nuclio at {self.nuclio_url}. "
                    "Deploy it first with 'make cvat-deploy-sam2-gpu' or set SAM2_FUNCTION_NAME."
                ),
            )

        # Try to get the function's HTTP port from Nuclio spec
        func_port = None
        spec = fn_meta.get("spec", {})
        triggers = spec.get("triggers", {})
        for trigger_name, trigger in triggers.items():
            if "myHttpTrigger" in trigger_name or trigger.get("kind") == "http":
                func_port = trigger.get("hostPort") or trigger.get("attributes", {}).get(
                    "hostPort"
                )
                break

        # Also check status for port
        status = fn_meta.get("status", {})
        if func_port is None:
            func_port = status.get("port", None)

        if func_port is None:
            return Sam2GenerationResult(
                status="error",
                total_images=len(image_files),
                error=(
                    "Could not determine HTTP port for SAM2 function. "
                    "Check Nuclio dashboard or deploy with nuctl."
                ),
            )

        # Invoke SAM2 function for each image
        candidates: list[Sam2Candidate] = []
        invoke_url = f"http://localhost:{func_port}"

        for img_file in image_files:
            try:
                raw = img_file.read_bytes()
                encoded = base64.b64encode(raw).decode("ascii")

                payload = {"image": encoded}
                resp = self._session.post(
                    invoke_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()

                blob = result.get("blob", "")
                blob_path = artifact_path / f"{img_file.stem}_sam2_blob.b64"
                blob_path.write_text(blob)

                candidates.append(
                    Sam2Candidate(
                        image_name=img_file.name,
                        mask_count=min(
                            max_masks_per_image, 1
                        ),  # SAM2 returns features, masks derived later
                        artifact_path=str(blob_path),
                    )
                )

            except Exception as exc:
                logger.error("SAM2 failed for %s: %s", img_file.name, exc)
                candidates.append(
                    Sam2Candidate(
                        image_name=img_file.name,
                        mask_count=0,
                        artifact_path="",
                    )
                )

        # Save candidates manifest
        manifest_path = artifact_path / "sam2_candidates.json"
        manifest = {
            "function_name": self.function_name,
            "total_images": len(image_files),
            "successful": len([c for c in candidates if c.mask_count > 0]),
            "failed": len([c for c in candidates if c.mask_count == 0]),
            "candidates": [c.model_dump() for c in candidates],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        status = "success" if manifest["failed"] == 0 else "partial"
        if manifest["successful"] == 0:
            status = "error"

        return Sam2GenerationResult(
            status=status,
            total_images=len(image_files),
            candidates=candidates,
        )
