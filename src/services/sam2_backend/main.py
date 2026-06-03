"""SAM2 backend service entry point.

Usage:
    python -m services.sam2_backend.main

Or via Makefile:
    make sam2-backend-run
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from .settings import get_settings

logger = logging.getLogger("sam2_backend")


def main() -> None:
    """Start the SAM2 backend FastAPI server."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    logger.info(
        "Starting SAM2 backend (mode=%s, host=%s, port=%d, device=%s)",
        settings.mode,
        settings.host,
        settings.port,
        settings.resolved_device,
    )

    uvicorn.run(
        "services.sam2_backend.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
