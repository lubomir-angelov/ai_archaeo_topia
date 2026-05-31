"""Artifact management for the annotation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .errors import ArtifactError
from .schemas import ClipMetadata, Sam2Proposal

logger = logging.getLogger(__name__)

# Label-to-color mapping for review images
LABEL_COLORS = {
    "mound": "lime",
    "hard_negative_symbol": "orange",
    "uncertain_ignore": "red",
}
DEFAULT_COLOR = "cyan"


def create_run_directory(
    output_root: Path,
    run_id: str,
    dry_run: bool = False,
) -> Path:
    """Create the run-specific output directory structure.

    Follows protocol folder layout:
        run_<RUN_ID>/
          raw/images/
          annotations/
          review/samples_for_review/
          masks/

    Args:
        output_root: Base output directory.
        run_id: Unique run identifier.
        dry_run: If True, append _dry_run suffix.

    Returns:
        Path to the run directory.

    Raises:
        ArtifactError: If directory creation fails.
    """
    suffix = "_dry_run" if dry_run else ""
    run_dir = output_root / f"{run_id}{suffix}"

    try:
        (run_dir / "raw" / "images").mkdir(parents=True, exist_ok=True)
        (run_dir / "annotations").mkdir(parents=True, exist_ok=True)
        (run_dir / "review" / "samples_for_review").mkdir(parents=True, exist_ok=True)
        (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ArtifactError(f"Failed to create run directory {run_dir}: {e}") from e

    logger.info("Created run directory: %s", run_dir)
    return run_dir


def save_clip_image(
    clip: ClipMetadata,
    pil_img: Image.Image,
    run_dir: Path,
) -> Path:
    """Save extracted clip image to run directory.

    Args:
        clip: Clip metadata.
        pil_img: PIL image to save.
        run_dir: Run output directory.

    Returns:
        Path to saved image.

    Raises:
        ArtifactError: If save fails.
    """
    annotator = clip.annotator or "UNK"
    img_dir = run_dir / "raw" / "images" / annotator
    img_dir.mkdir(parents=True, exist_ok=True)

    filename = clip.sample_id or f"clip_p{clip.pdf_page}_i{clip.clip_index}.png"
    if not filename.lower().endswith(".png"):
        filename += ".png"

    img_path = img_dir / filename
    try:
        pil_img.save(str(img_path), "PNG")
    except Exception as e:
        raise ArtifactError(f"Failed to save clip image {img_path}: {e}") from e

    clip.clip_image_path = str(img_path)
    logger.debug("Saved clip image: %s", img_path)
    return img_path


def render_review_image(
    clip: ClipMetadata,
    pil_img: Image.Image,
    proposals: list[Sam2Proposal],
    run_dir: Path,
) -> Path:
    """Render review image with bounding boxes drawn.

    Args:
        clip: Clip metadata.
        pil_img: Source PIL image.
        proposals: SAM2 proposals to draw.
        run_dir: Run output directory.

    Returns:
        Path to saved review image.

    Raises:
        ArtifactError: If rendering fails.
    """
    review = pil_img.copy()
    draw = ImageDraw.Draw(review)

    # Red border
    draw.rectangle(
        [0, 0, review.width - 1, review.height - 1],
        outline="red",
        width=3,
    )

    # Draw proposal boxes
    for prop in proposals:
        if len(prop.bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = prop.bbox
        color = LABEL_COLORS.get(prop.label, DEFAULT_COLOR)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

        # Label text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        label_text = f"{prop.label} {prop.confidence:.2f}"
        draw.text((xmin + 2, max(0, ymin - 16)), label_text, fill=color, font=font)

    # Info bar
    info_text = f"P{clip.pdf_page} | {clip.sample_id}"
    draw.rectangle(
        [0, review.height - 22, min(400, review.width), review.height],
        fill="black",
    )
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, review.height - 20), info_text, fill="white", font=font)

    # Save
    review_dir = run_dir / "review" / "samples_for_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_filename = f"review_{clip.sample_id}"
    if not review_filename.lower().endswith(".png"):
        review_filename += ".png"
    review_path = review_dir / review_filename

    try:
        review.save(str(review_path), "PNG")
    except Exception as e:
        raise ArtifactError(f"Failed to save review image {review_path}: {e}") from e

    clip.clip_image_path = str(review_path) if not clip.clip_image_path else clip.clip_image_path
    logger.debug("Saved review image: %s", review_path)
    return review_path


def save_proposal_mask(
    proposal: Sam2Proposal,
    sample_id: str,
    run_dir: Path,
    index: int,
) -> str:
    """Create a simple binary mask from proposal bbox.

    Args:
        proposal: SAM2 proposal with bbox.
        sample_id: Sample identifier.
        run_dir: Run output directory.
        index: Proposal index for naming.

    Returns:
        Path to saved mask.
    """
    mask_dir = run_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if len(proposal.bbox) != 4:
        return ""

    # Create a minimal mask (the real mask comes from backend in live mode)
    xmin, ymin, xmax, ymax = proposal.bbox
    w = max(1, int(xmax) - int(xmin))
    h = max(1, int(ymax) - int(ymin))

    mask = Image.new("L", (w, h), 0)
    mask.save(str(mask_dir / f"mask_{sample_id}_{index}.png"), "PNG")
    mask_path = str(mask_dir / f"mask_{sample_id}_{index}.png")
    return mask_path
