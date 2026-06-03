"""FastAPI routes for the SAM2 backend service.

Exposes GET /health, POST /segment, and POST /propose endpoints
matching the contract in docs/automation/SAM2_BACKEND_CONTRACT.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .errors import ImageError, ModelError, OutputError, Sam2BackendError, ValidationError
from .predictor import SAM2PredictorBackend
from .proposals import generate_proposals
from .schemas import (
    ErrorResponse,
    HealthResponse,
    ProposalItem,
    ProposalRequest,
    ProposalResponse,
    SegmentBboxRequest,
    SegmentPointsRequest,
    SegmentResponse,
)
from .settings import get_settings

logger = logging.getLogger(__name__)

app = FastAPI(title="SAM2 Backend", version="0.1.0")

_predictor: SAM2PredictorBackend | None = None


def get_predictor() -> SAM2PredictorBackend:
    """Return the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = SAM2PredictorBackend()
    return _predictor


def reset_predictor() -> None:
    """Reset the predictor singleton (for tests)."""
    global _predictor
    _predictor = None


# ── Health ────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> HealthResponse:
    """Check backend availability and model status."""
    settings = get_settings()
    predictor = get_predictor()

    details: dict[str, Any] = {}

    if settings.mode == "mock":
        return HealthResponse(
            ok=True,
            service="sam2_backend",
            model="sam2",
            mode="mock",
            device="mock",
            model_loaded=True,
            checkpoint="",
            details=details,
        )

    # sam2 mode
    try:
        predictor.ensure_loaded()
        return HealthResponse(
            ok=True,
            service="sam2_backend",
            model="sam2",
            mode="sam2",
            device=predictor.device,
            model_loaded=predictor.is_loaded,
            checkpoint=settings.checkpoint,
            details=details,
        )
    except ModelError as exc:
        details["error"] = str(exc)
        return HealthResponse(
            ok=False,
            service="sam2_backend",
            model="sam2",
            mode="sam2",
            device=predictor.device,
            model_loaded=False,
            checkpoint=settings.checkpoint,
            details=details,
        )


# ── Segment ───────────────────────────────────────────────────────


@app.post("/segment")
async def segment(request: Request) -> SegmentResponse:
    """Run segmentation with a bounding-box or point prompt.

    Accepts either SegmentBboxRequest or SegmentPointsRequest payload.
    """
    settings = get_settings()
    predictor = get_predictor()

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(400, detail=f"Invalid JSON: {exc}") from exc

    prompt_type = body.get("prompt_type", "bbox")

    if prompt_type == "bbox":
        try:
            req = SegmentBboxRequest(**body)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        # Validate image
        try:
            from .masks import validate_image_path

            validate_image_path(req.image_path, settings.max_image_pixels)
        except ImageError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        # Validate bbox
        _validate_bbox(req.bbox, req.image_path)

        # Resolve output dir
        out_dir = _resolve_output_dir(req.output_dir, settings)

        # Run inference
        try:
            result = predictor.segment_bbox(req.image_path, req.bbox, out_dir, req.label)
        except ModelError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        except OutputError as exc:
            raise HTTPException(500, detail=str(exc)) from exc

    elif prompt_type == "points":
        try:
            req = SegmentPointsRequest(**body)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        # Validate image
        try:
            from .masks import validate_image_path

            validate_image_path(req.image_path, settings.max_image_pixels)
        except ImageError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        # Validate points
        _validate_points(req.points, req.point_labels, req.image_path)

        # Resolve output dir
        out_dir = _resolve_output_dir(req.output_dir, settings)

        # Run inference
        try:
            result = predictor.segment_points(
                req.image_path,
                req.points,
                req.point_labels,
                out_dir,
                req.label,
            )
        except ModelError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        except OutputError as exc:
            raise HTTPException(500, detail=str(exc)) from exc

    else:
        raise HTTPException(
            400,
            detail=f"Invalid prompt_type: {prompt_type!r}. Must be 'bbox' or 'points'.",
        )

    # Validate mask_path stays under output dir
    if result.get("mask_path"):
        _validate_mask_path(result["mask_path"], out_dir)

    return SegmentResponse(**result)


# ── Propose ───────────────────────────────────────────────────────


@app.post("/propose")
async def propose(req: ProposalRequest) -> ProposalResponse:
    """Generate candidate segmentation proposals for an image."""
    settings = get_settings()

    # Validate image
    try:
        from .masks import validate_image_path

        validate_image_path(req.image_path, settings.max_image_pixels)
    except ImageError as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    # Validate max_proposals
    if req.max_proposals <= 0:
        raise HTTPException(400, detail="max_proposals must be positive")

    # Resolve output dir
    out_dir = _resolve_output_dir(req.output_dir, settings)

    # Generate proposals
    try:
        items = generate_proposals(
            req.image_path,
            out_dir,
            req.label_hint,
            req.max_proposals,
            settings.mode,
        )
    except ValidationError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except OutputError as exc:
        raise HTTPException(500, detail=str(exc)) from exc

    # Validate each item
    validated: list[ProposalItem] = []
    for item in items:
        if item.get("mask_path"):
            _validate_mask_path(item["mask_path"], out_dir)
        validated.append(ProposalItem(**item))

    return ProposalResponse(items=validated)


# ── Validation helpers ────────────────────────────────────────────


def _validate_bbox(bbox: list[float], image_path: str) -> None:
    """Validate bounding-box coordinates."""
    if len(bbox) != 4:
        raise HTTPException(400, detail="bbox must have exactly 4 values")
    for v in bbox:
        if not isinstance(v, (int, float)):
            raise HTTPException(400, detail=f"bbox values must be numeric, got {type(v).__name__}")

    xmin, ymin, xmax, ymax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if xmin >= xmax:
        raise HTTPException(400, detail=f"bbox x_min ({xmin}) must be < x_max ({xmax})")
    if ymin >= ymax:
        raise HTTPException(400, detail=f"bbox y_min ({ymin}) must be < y_max ({ymax})")

    from .masks import validate_image_path

    w, h = validate_image_path(image_path)
    if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
        raise HTTPException(
            400,
            detail=f"bbox [{xmin},{ymin},{xmax},{ymax}] outside image dimensions {w}x{h}",
        )


def _validate_points(
    points: list[list[float]],
    point_labels: list[int],
    image_path: str,
) -> None:
    """Validate point prompts."""
    if len(points) == 0:
        raise HTTPException(400, detail="points must not be empty")
    if len(points) != len(point_labels):
        raise HTTPException(
            400,
            detail=f"points ({len(points)}) and point_labels ({len(point_labels)}) length mismatch",
        )
    for i, pt in enumerate(points):
        if len(pt) != 2:
            raise HTTPException(400, detail=f"points[{i}] must have exactly 2 values")
        for v in pt:
            if not isinstance(v, (int, float)):
                raise HTTPException(
                    400, detail=f"point values must be numeric, got {type(v).__name__}"
                )
    for lbl in point_labels:
        if lbl not in (0, 1):
            raise HTTPException(400, detail=f"point_labels must be 0 or 1, got {lbl}")

    from .masks import validate_image_path

    w, h = validate_image_path(image_path)
    for i, pt in enumerate(points):
        x, y = float(pt[0]), float(pt[1])
        if x < 0 or y < 0 or x > w or y > h:
            raise HTTPException(
                400,
                detail=f"points[{i}] ({x},{y}) outside image dimensions {w}x{h}",
            )


def _resolve_output_dir(
    requested_dir: str,
    settings: Any,
) -> Path:
    """Resolve and validate the output directory."""
    from .masks import validate_output_dir

    return validate_output_dir(requested_dir, settings.output_path)


def _validate_mask_path(mask_path: str, output_dir: Path) -> None:
    """Ensure mask_path is under the output directory."""
    if not mask_path:
        return
    mask = Path(mask_path).resolve()
    base = output_dir.resolve()
    try:
        mask.relative_to(base)
    except ValueError:
        raise HTTPException(
            500,
            detail=f"mask_path {mask_path!r} escapes output_dir {base}",
        ) from None


# ── Custom error handler ──────────────────────────────────────────


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.__class__.__name__, detail=exc.detail).model_dump(),
    )


@app.exception_handler(Sam2BackendError)
async def _backend_error_handler(request: Request, exc: Sam2BackendError) -> JSONResponse:
    """Return structured error responses for custom backend errors."""
    status = 400
    if isinstance(exc, (ModelError,)):
        status = 503
    elif isinstance(exc, OutputError):
        status = 500
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )
