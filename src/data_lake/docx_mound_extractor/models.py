"""Data models for the docx mound extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AnnotationStatus = Literal[
    "not_started",
    "in_progress",
    "annotated",
    "needs_review",
    "reviewed",
    "rejected",
]

ReviewStatus = Literal[
    "pending_review",
    "accepted",
    "corrected",
    "rejected",
]

Difficulty = Literal["easy", "medium", "hard", "very_hard"]


@dataclass
class RawTextBlock:
    """One numbered sample block extracted from docx text stream."""

    block_number: int
    raw_text: str
    text_before_image: str = ""
    text_after_image: str = ""


@dataclass
class DocxImageRef:
    """Reference to an embedded image inside a docx."""

    rid: str
    media_path: str
    sample_number: int
    image_index_in_sample: int = 0


@dataclass
class DocxSample:
    """One extracted sample with full metadata."""

    annotator: str
    sample_number: int
    sample_id: str
    image_path: str
    image_width: int = 0
    image_height: int = 0
    image_index_in_sample: int = 0

    sheet_25k: str = ""
    sheet_5k: str = ""
    province: str = ""
    position_on_25k_sheet: str = ""
    original_description: str = ""
    relief_original: str = ""
    relief_normalized: str = "unknown"

    target_present: bool = True
    target_count_claimed: int = 0
    contains_necropolis: bool = False
    contains_single_mound: bool = False
    contains_hard_negative: bool = False

    difficulty: Difficulty = "medium"
    uncertainty: bool = False
    annotation_status: AnnotationStatus = "not_started"
    review_status: ReviewStatus = "pending_review"
    notes: str = ""


@dataclass
class FileStats:
    """Extraction stats for a single docx file."""

    filename: str
    annotator: str
    total_images: int
    total_samples: int
    samples_with_multiple_images: int
    provinces: list[str] = field(default_factory=list)


@dataclass
class ExtractionReport:
    """Summary report for the full extraction run."""

    source_dir: str
    output_dir: str
    files_processed: int
    total_samples: int
    total_images: int
    file_stats: list[FileStats] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
