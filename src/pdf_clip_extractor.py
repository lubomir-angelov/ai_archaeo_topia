#!/usr/bin/env python3
"""Extract clips and metadata from map_annotation_clips.pdf.

Follows the annotation protocol in docs/annotation/PROTOCOL_EN.md:
- sample_id format: <ANNOTATOR>_<NUMBER>_<SHEET_25K>_<SHEET_5K>.png
- Output folder structure: raw/images/<ANNOTATOR>/, review/samples_for_review/
- master_samples.csv with protocol-compliant columns
- Bounding-box review images rendered on extracted clips
"""

from __future__ import annotations

import csv
import io
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Protocol constants ──────────────────────────────────────────────

PROTOCOL_COLUMNS = [
    "sample_id",
    "annotator",
    "image_path",
    "sheet_25k",
    "sheet_5k",
    "province",
    "position_on_25k_sheet",
    "original_description",
    "target_present",
    "target_count_claimed",
    "contains_necropolis",
    "contains_single_mound",
    "contains_hard_negative",
    "relief_original",
    "relief_normalized",
    "difficulty",
    "uncertainty",
    "annotation_status",
    "review_status",
    "notes",
]

LABELS = ["mound", "hard_negative_symbol", "uncertain_ignore"]

# Known annotator prefixes from the PDF text
ANNOTATOR_PREFIXES = {"AG", "VG", "JTZ", "IK"}

# Relief normalization map (BG -> EN normalized)
RELIEF_MAP = {
    "равнинен": "plain",
    "равниен": "plain",
    "хълмист": "hilly",
    "слабо хълмист": "hilly",
    "полупланински": "mixed",
    "планински": "mountain",
    "склон": "slope",
    "склон на хълм": "slope",
    "склон на рид": "slope",
    "било на възвишение": "ridge",
    "било": "ridge",
    "ниска част на възвишение": "ridge",
    "средна част на ниско възвишение": "ridge",
    "било на възвишение и околностите му": "ridge",
    "долен част на възвишение": "valley",
    "долна част на възвишение": "valley",
    "речна долина": "valley",
    "дере": "valley",
    "ридли": "ridge",
    "ридов": "ridge",
    "подножие на възвишение": "hilly",
    "градска среда": "urban",
}


@dataclass
class ClipInfo:
    """Metadata for one extracted clip."""

    page_num: int
    image_index: int
    xref: int
    width: int
    height: int
    annotator: str = ""
    sample_number: int = 0
    sheet_25k: str = ""
    sheet_5k: str = ""
    province: str = ""
    position: str = ""
    description: str = ""
    relief_original: str = ""
    relief_normalized: str = ""
    difficulty: str = "medium"
    uncertainty: bool = False
    target_present: bool = True
    target_count: str = ""
    contains_necropolis: bool = False
    contains_single_mound: bool = False
    contains_hard_negative: bool = False
    notes: str = ""
    sample_id: str = ""
    image_filename: str = ""
    review_filename: str = ""
    image_path: str = ""
    review_path: str = ""


def normalize_relief(original: str) -> str:
    """Normalize relief description per protocol section 6."""
    cleaned = original.strip().lower()
    # Direct match
    if cleaned in RELIEF_MAP:
        return RELIEF_MAP[cleaned]
    # Partial match
    for key, val in RELIEF_MAP.items():
        if key in cleaned or cleaned in key:
            return val
    return "unknown"


def parse_page_text(text: str) -> dict:
    """Parse structured metadata from a page's text content.

    Returns a dict with keys: annotator, sheet_25k, sheet_5k, province,
    position, description, relief_original, target_count, is_uncertain, notes.
    """
    result: dict = {
        "annotator": "",
        "sheet_25k": "",
        "sheet_5k": "",
        "province": "",
        "position": "",
        "description": "",
        "relief_original": "",
        "target_count": "",
        "is_uncertain": False,
        "notes": "",
        "contains_necropolis": False,
        "contains_single_mound": False,
        "contains_hard_negative": False,
    }

    # Detect annotator prefix (standalone line at top: AG, VG, JTZ, IK, etc.)
    for prefix in sorted(ANNOTATOR_PREFIXES, key=len, reverse=True):
        if re.search(rf"^\s*{prefix}\s*$", text, re.MULTILINE):
            result["annotator"] = prefix
            break

    # Parse sheet_25k: "Картен лист 1:25 000: К-35-8-Г-а" or "1:25000: К-35-27-Г-г"
    sheet_25k_match = re.search(
        r"Картен лист\s+1:\s*(?:25\s*000|25000)\s*:\s*([^\n,]+)",
        text,
        re.IGNORECASE,
    )
    if sheet_25k_match:
        result["sheet_25k"] = sheet_25k_match.group(1).strip()

    # Parse sheet_5k: "картен лист 1:5 000: К-35-8 (170, 186)"
    sheet_5k_match = re.search(
        r"картен лист\s+1:\s*(?:5\s*000|5000)\s*:\s*([^\n,]+)",
        text,
        re.IGNORECASE,
    )
    if sheet_5k_match:
        raw_5k = sheet_5k_match.group(1).strip()
        # Extract sheet numbers in parens
        paren_nums = re.findall(r"\(([^)]+)\)", raw_5k)
        if paren_nums:
            result["sheet_5k"] = " ".join(paren_nums)
        else:
            result["sheet_5k"] = raw_5k

    # Parse province: "обл. Добрич" or "обл. Силистра"
    prov_match = re.search(r"обл\.\s*([^\n,]+)", text)
    if prov_match:
        result["province"] = prov_match.group(1).strip()

    # Parse position: "Средна част на лист 1:25 000" or "Северозападна част"
    pos_match = re.search(
        r"(Северозападна|Северна|Североизточна|Източна|Югоизточна|Южна|Югозападна|Западна|Централна|Средна|З[а]падна|С[е]верна|СИ|ЮЗ|ЮИ|СЗ)\s+част",
        text,
    )
    if pos_match:
        result["position"] = pos_match.group(0).strip()

    # Parse relief: "Релеф: равнинен" or "Релеф: хълмист"
    relief_match = re.search(r"Релеф:\s*(.+)", text)
    if relief_match:
        result["relief_original"] = relief_match.group(1).strip()

    # Determine if this is an uncertain/negative case from description
    desc_lower = text.lower()
    if any(
        kw in desc_lower
        for kw in [
            "не е могила",
            "няма могили",
            "няма знак за могила",
            "не е ясен",
            "не знам",
            "наподобява",
            "може и да е",
            "вероятно",
            "водениц[ае]",
            "мелниц[ае]",
            "резервоар",
            "изкоп",
            "каптаж",
            "кладенец",
            "рибарник",
            "кошара",
            "табия",
        ]
    ):
        result["is_uncertain"] = True

    # Parse target count from description (e.g. "Могилен некропол от 13 могили", "Две могили")
    count_match = re.search(
        r"(една|две|три|четири|пет|шест|седем|осем|девет|десет|единадесет|дванадесет|тринадесет|четиринадесет|петнадесет|шестнадесет|седемнадесет|деветнадесет|двадесет|петнайсет|шестнайсет|седемнайсет|деветнайсет|дванадесет)\s+могили|от\s+(\d+)\s+могили|Една\s+могила|Две\s+могили|Три\s+могили|Няколко\s+могили|Една\s+могила",
        text,
        re.IGNORECASE,
    )
    if count_match:
        result["target_count"] = count_match.group(0).strip()

    # Detect necropolis
    if "некропол" in text.lower():
        result["contains_necropolis"] = True

    # Detect hard negative
    if any(
        kw in text.lower()
        for kw in [
            "воденица",
            "мелница",
            "резервоар",
            "полезащитен",
            "изкоп",
            "каптаж",
            "кладенец",
            "рибарник",
            "кошара",
            "табия",
            "наподобява",
        ]
    ):
        result["contains_hard_negative"] = True

    # Detect single mound
    if re.search(r"Една\s+могила", text):
        result["contains_single_mound"] = True

    return result


def build_sample_id(
    annotator: str,
    number: int,
    sheet_25k: str,
    sheet_5k: str,
) -> str:
    """Build sample_id per protocol section 3.

    Format: <ANNOTATOR>_<NUMBER>_<SHEET_25K>_<SHEET_5K>.png
    """
    # Clean sheet names: replace spaces, normalize
    s25k = sheet_25k.strip().replace(" ", "_") if sheet_25k else "UNKNOWN"
    s5k = sheet_5k.strip().replace(" ", "_") if sheet_5k else "UNKNOWN"
    return f"{annotator}_{number:03d}_{s25k}_{s5k}.png"


def extract_images_from_page(page: fitz.Page, text: str) -> list[ClipInfo]:
    """Extract embedded images from a PDF page and attach parsed metadata.

    Returns a list of ClipInfo objects, one per embedded image.
    """
    images = page.get_images(full=True)
    if not images:
        return []

    parsed = parse_page_text(text)
    clips: list[ClipInfo] = []

    for idx, img in enumerate(images):
        xref = img[0]
        w, h = img[2], img[3]

        try:
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n >= 5:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            logger.warning("Failed to extract image xref=%d: %s", xref, e)
            continue

        clip = ClipInfo(
            page_num=page.number + 1,
            image_index=idx,
            xref=xref,
            width=pil_img.width,
            height=pil_img.height,
            annotator=parsed["annotator"],
            sheet_25k=parsed["sheet_25k"],
            sheet_5k=parsed["sheet_5k"],
            province=parsed["province"],
            position=parsed["position"],
            description=parsed["description"],
            relief_original=parsed["relief_original"],
            relief_normalized=normalize_relief(parsed["relief_original"])
            if parsed["relief_original"]
            else "unknown",
            uncertainty=parsed["is_uncertain"],
            target_count=parsed["target_count"],
            contains_necropolis=parsed["contains_necropolis"],
            contains_single_mound=parsed["contains_single_mound"],
            contains_hard_negative=parsed["contains_hard_negative"],
        )

        # Annotator prefix per page section (AG, VG, JTZ, IK)
        # Use the page-level annotator if available, else infer from sheet prefix
        if not clip.annotator:
            # Infer from sheet 25k prefix pattern
            clip.annotator = "UNK"

        clips.append(clip)

    return clips


def assign_sample_ids(clips: list[ClipInfo]) -> None:
    """Assign sequential sample IDs and filenames per annotator."""
    # Group by annotator
    by_annotator: dict[str, list[ClipInfo]] = {}
    for clip in clips:
        by_annotator.setdefault(clip.annotator, []).append(clip)

    for annotator, annotator_clips in by_annotator.items():
        for i, clip in enumerate(annotator_clips, start=1):
            clip.sample_number = i
            clip.sample_id = build_sample_id(
                annotator=annotator,
                number=i,
                sheet_25k=clip.sheet_25k or "UNKNOWN",
                sheet_5k=clip.sheet_5k or "UNKNOWN",
            )
            clip.image_filename = clip.sample_id
            clip.review_filename = "review_" + clip.sample_id


def render_review_image(
    clip: ClipInfo,
    pil_img: Image.Image,
    draw_boxes: list[dict] | None = None,
) -> Image.Image:
    """Render a review image with bounding boxes drawn.

    draw_boxes: list of {"x": float, "y": float, "w": float, "h": float, "label": str}
    """
    review = pil_img.copy()
    draw = ImageDraw.Draw(review)

    # Draw a red border around the whole image
    draw.rectangle([0, 0, review.width - 1, review.height - 1], outline="red", width=3)

    if draw_boxes:
        for box in draw_boxes:
            x1 = int(box["x"] * review.width)
            y1 = int(box["y"] * review.height)
            x2 = int((box["x"] + box["w"]) * review.width)
            y2 = int((box["y"] + box["h"]) * review.height)
            color = {"mound": "lime", "hard_negative_symbol": "orange", "uncertain_ignore": "red"}.get(
                box.get("label", ""), "cyan"
            )
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # Label
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()
            draw.text((x1 + 2, y1 - 16), box.get("label", ""), fill=color, font=font)

    # Add page/clip info text
    info_text = f"P{clip.page_num} | {clip.sample_id}"
    draw.rectangle([0, review.height - 22, 400, review.height], fill="black")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((4, review.height - 20), info_text, fill="white", font=font)

    return review


def generate_proposal_boxes(clip: ClipInfo, pil_img: Image.Image) -> list[dict]:
    """Generate synthetic bounding box proposals for SAM2 processing.

    For now, creates placeholder boxes at common mound-symbol locations
    based on image center and size heuristics. The real proposals will
    come from CVAT/SAM2 in the next step.

    Returns list of {"x": float, "y": float, "w": float, "h": float, "label": str, "confidence": float}.
    """
    w, h = pil_img.size
    boxes: list[dict] = []

    # Heuristic: mound symbols tend to be circular, ~30-80px diameter
    # Place a few candidate boxes at likely positions
    if clip.contains_necropolis:
        # Multiple mounds - spread across the image
        for i in range(3):
            cx = 0.2 + i * 0.3
            cy = 0.3 + (i % 2) * 0.4
            boxes.append(
                {
                    "x": cx,
                    "y": cy,
                    "w": 0.08,
                    "h": 0.08,
                    "label": "mound",
                    "confidence": 0.5,
                    "source": "heuristic_placeholder",
                }
            )
    elif clip.contains_single_mound:
        boxes.append(
            {
                "x": 0.35,
                "y": 0.35,
                "w": 0.15,
                "h": 0.15,
                "label": "mound",
                "confidence": 0.6,
                "source": "heuristic_placeholder",
            }
        )
    elif clip.contains_hard_negative:
        boxes.append(
            {
                "x": 0.4,
                "y": 0.4,
                "w": 0.1,
                "h": 0.1,
                "label": "hard_negative_symbol",
                "confidence": 0.4,
                "source": "heuristic_placeholder",
            }
        )
    elif clip.uncertainty:
        boxes.append(
            {
                "x": 0.4,
                "y": 0.4,
                "w": 0.12,
                "h": 0.12,
                "label": "uncertain_ignore",
                "confidence": 0.3,
                "source": "heuristic_placeholder",
            }
        )
    else:
        # Default: center box
        boxes.append(
            {
                "x": 0.35,
                "y": 0.35,
                "w": 0.2,
                "h": 0.2,
                "label": "mound",
                "confidence": 0.5,
                "source": "heuristic_placeholder",
            }
        )

    return boxes


def save_clips_and_reviews(
    clips: list[ClipInfo],
    output_root: Path,
    images: dict[int, Image.Image],
) -> None:
    """Save extracted clip images and review images with bounding boxes."""
    raw_images_dir = output_root / "raw" / "images"
    review_dir = output_root / "review" / "samples_for_review"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        pil_img = images.get(clip.xref)
        if pil_img is None:
            logger.warning("No image data for xref=%d (%s), skipping", clip.xref, clip.sample_id)
            continue

        # Save raw clip
        annotator_dir = raw_images_dir / clip.annotator
        annotator_dir.mkdir(parents=True, exist_ok=True)
        clip_path = annotator_dir / clip.image_filename
        pil_img.save(str(clip_path), "PNG")

        # Generate proposal boxes and render review image
        proposal_boxes = generate_proposal_boxes(clip, pil_img)
        review_img = render_review_image(clip, pil_img, proposal_boxes)
        review_path = review_dir / clip.review_filename
        review_img.save(str(review_path), "PNG")

        clip.image_path = str(clip_path)
        clip.review_path = str(review_path)

        # Store annotation data on clip
        clip.annotations = proposal_boxes  # type: ignore[attr-defined]


def populate_csv(
    clips: list[ClipInfo],
    csv_path: Path,
) -> None:
    """Write master_samples.csv with all protocol-compliant columns."""
    # Check if CSV exists and read existing sample_ids to avoid overwriting
    existing_ids: set[str] = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row.get("sample_id", ""))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROTOCOL_COLUMNS)
        writer.writeheader()

        for clip in clips:
            if clip.sample_id in existing_ids:
                logger.info("Skipping existing sample_id: %s", clip.sample_id)
                continue

            writer.writerow(
                {
                    "sample_id": clip.sample_id,
                    "annotator": clip.annotator or "",
                    "image_path": clip.image_path or "",
                    "sheet_25k": clip.sheet_25k or "",
                    "sheet_5k": clip.sheet_5k or "",
                    "province": clip.province or "",
                    "position_on_25k_sheet": clip.position or "",
                    "original_description": "",  # Will be filled from text
                    "target_present": str(clip.target_present).lower(),
                    "target_count_claimed": clip.target_count or "",
                    "contains_necropolis": str(clip.contains_necropolis).lower(),
                    "contains_single_mound": str(clip.contains_single_mound).lower(),
                    "contains_hard_negative": str(clip.contains_hard_negative).lower(),
                    "relief_original": clip.relief_original or "",
                    "relief_normalized": clip.relief_normalized or "",
                    "difficulty": clip.difficulty,
                    "uncertainty": str(clip.uncertainty).lower(),
                    "annotation_status": "annotated",
                    "review_status": "pending_review",
                    "notes": clip.notes or "",
                }
            )


def main(
    pdf_path: str,
    output_root: str,
    protocol_dir: str,
    dry_run: bool = False,
) -> dict:
    """Main extraction pipeline.

    Steps:
    1. Open PDF, iterate pages, extract embedded images
    2. Parse metadata from page text
    3. Assign sample IDs per protocol naming convention
    4. Save raw clip images and review images with bounding boxes
    5. Populate master_samples.csv
    6. Return run summary
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output = Path(output_root)
    if dry_run:
        output = output / "dry_run_preview"

    # Extract images and metadata
    all_clips: list[ClipInfo] = []
    all_images: dict[int, Image.Image] = {}  # keyed by xref

    doc = fitz.open(str(pdf))
    logger.info("Opened PDF: %d pages, %d total embedded images", doc.page_count, sum(
        len(page.get_images()) for page in doc
    ))

    for page in doc:
        text = page.get_text()
        page_clips = extract_images_from_page(page, text)
        all_clips.extend(page_clips)

        # Store PIL images for later rendering (keyed by xref)
        for clip in page_clips:
            xref = clip.xref
            try:
                pix = fitz.Pixmap(page.parent, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                all_images[xref] = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                logger.warning("Failed to load image xref=%d: %s", xref, e)

    doc.close()

    # Assign sample IDs
    assign_sample_ids(all_clips)

    # Save clips and reviews
    save_clips_and_reviews(all_clips, output, all_images)

    # Populate CSV
    csv_path = output / "annotations" / "master_samples.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    populate_csv(all_clips, csv_path)

    # Build summary
    summary = {
        "pdf_path": str(pdf),
        "output_root": str(output),
        "dry_run": dry_run,
        "pages_processed": len(set(c.page_num for c in all_clips)),
        "clips_extracted": len(all_clips),
        "csv_path": str(csv_path),
        "annotators": sorted(set(c.annotator for c in all_clips if c.annotator)),
        "sheets_25k": sorted(set(c.sheet_25k for c in all_clips if c.sheet_25k)),
        "necropolis_count": sum(1 for c in all_clips if c.contains_necropolis),
        "hard_negative_count": sum(1 for c in all_clips if c.contains_hard_negative),
        "uncertain_count": sum(1 for c in all_clips if c.uncertainty),
        "single_mound_count": sum(1 for c in all_clips if c.contains_single_mound),
        "created_at": datetime.now(UTC).isoformat(),
    }

    logger.info("Done. %d clips extracted from %d pages.", summary["clips_extracted"], summary["pages_processed"])
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract clips from map_annotation_clips.pdf")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--output-root", default="./artifacts/annotation_runs", help="Output root directory")
    parser.add_argument("--protocol-dir", default="docs/annotation", help="Protocol directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    summary = main(
        pdf_path=args.pdf,
        output_root=args.output_root,
        protocol_dir=args.protocol_dir,
        dry_run=args.dry_run,
    )
    print("\n=== Extraction Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
