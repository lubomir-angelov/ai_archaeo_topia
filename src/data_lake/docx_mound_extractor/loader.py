"""Load, parse, and combine docx samples into DocxSample objects.

Orchestration stage: coordinates extraction, image-to-block assignment,
and metadata parsing for a single docx file.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

from .extractor import DocxExtractor
from .models import DocxImageRef, DocxSample, FileStats, RawTextBlock
from .parser import assign_difficulty, parse_text_block

logger = logging.getLogger(__name__)


class DocxLoader:
    """Loads one docx, produces a list of DocxSample objects."""

    def __init__(self, filepath: Path, output_images_dir: Path) -> None:
        self.filepath = filepath
        self.annotator = filepath.stem
        self.output_images_dir = output_images_dir

    def load(self) -> tuple[list[DocxSample], FileStats]:
        """Full load pipeline for one docx.

        Returns (samples, stats).
        """
        blocks: list[RawTextBlock] = []
        image_refs: list[DocxImageRef] = []
        image_cache: dict[str, bytes] = {}

        with DocxExtractor(self.filepath) as ext:
            rid_map = ext.get_rid_to_media_map()
            blocks = ext.extract_text_blocks()
            image_refs = ext.assign_images_to_blocks(blocks, rid_map)

            # Cache all image bytes in single archive open
            for ref in image_refs:
                if ref.media_path:
                    image_cache[ref.media_path] = ext.extract_image_bytes(ref.media_path)

        if not blocks:
            logger.warning("No text blocks found in %s", self.filepath.name)
            return [], FileStats(
                filename=self.filepath.name,
                annotator=self.annotator,
                total_images=0,
                total_samples=0,
                samples_with_multiple_images=0,
            )

        refs_by_sample: dict[int, list[DocxImageRef]] = defaultdict(list)
        for ref in image_refs:
            refs_by_sample[ref.sample_number].append(ref)

        samples: list[DocxSample] = []
        multi_img_count = 0
        provinces: set[str] = set()

        for block in blocks:
            parsed = parse_text_block(block)
            refs = refs_by_sample.get(block.block_number, [])

            if not refs:
                logger.warning(
                    "%s: sample #%d has no image, skipping",
                    self.annotator,
                    block.block_number,
                )
                continue

            if len(refs) > 1:
                multi_img_count += 1
                logger.info(
                    "%s: sample #%d has %d images",
                    self.annotator,
                    block.block_number,
                    len(refs),
                )

            for img_idx, ref in enumerate(refs):
                sample = self._build_sample(block, parsed, ref, img_idx, image_cache)
                assign_difficulty(sample)
                samples.append(sample)
                if parsed["province"]:
                    provinces.add(parsed["province"])

        stats = FileStats(
            filename=self.filepath.name,
            annotator=self.annotator,
            total_images=len(image_refs),
            total_samples=len(samples),
            samples_with_multiple_images=multi_img_count,
            provinces=sorted(provinces),
        )
        return samples, stats

    def _build_sample(
        self,
        block: RawTextBlock,
        parsed: dict,
        ref: DocxImageRef,
        img_idx: int,
        image_cache: dict[str, bytes],
    ) -> DocxSample:
        """Construct a DocxSample and save the image to disk."""
        img_dir = self.output_images_dir / self.annotator
        img_dir.mkdir(parents=True, exist_ok=True)

        s25k = self._sanitize(parsed["sheet_25k"]) or "UNKNOWN"
        s5k = self._sanitize(parsed["sheet_5k"]) or "UNKNOWN"

        if img_idx == 0:
            fname = f"{self.annotator}_{ref.sample_number:03d}_{s25k}_{s5k}.png"
        else:
            fname = f"{self.annotator}_{ref.sample_number:03d}_img{img_idx + 1}_{s25k}_{s5k}.png"

        img_path = img_dir / fname

        img_bytes = image_cache.get(ref.media_path, b"")
        pil_img = Image.open(BytesIO(img_bytes))
        pil_img.save(str(img_path), "PNG")

        sample_id = fname.replace(".png", "")

        return DocxSample(
            annotator=self.annotator,
            sample_number=ref.sample_number,
            sample_id=sample_id,
            image_path=f"raw/images/{self.annotator}/{fname}",
            image_width=pil_img.width,
            image_height=pil_img.height,
            image_index_in_sample=img_idx,
            sheet_25k=parsed["sheet_25k"],
            sheet_5k=parsed["sheet_5k"],
            province=parsed["province"],
            position_on_25k_sheet=parsed["position_on_25k_sheet"],
            original_description=parsed["original_description"],
            relief_original=parsed["relief_original"],
            relief_normalized=parsed["relief_normalized"],
            target_present=parsed["target_present"],
            target_count_claimed=parsed["target_count_claimed"],
            contains_necropolis=parsed["contains_necropolis"],
            contains_single_mound=parsed["contains_single_mound"],
            contains_hard_negative=parsed["contains_hard_negative"],
            uncertainty=parsed["uncertainty"],
            notes=parsed["notes"],
        )

    @staticmethod
    def _sanitize(val: str) -> str:
        """Remove path-breaking characters from a string."""
        out = val.replace(" ", "_")
        out = re.sub(r"[/\\:*?\"<>|,;()']", "", out)
        out = re.sub(r"_+", "_", out)
        return out.strip("_")
