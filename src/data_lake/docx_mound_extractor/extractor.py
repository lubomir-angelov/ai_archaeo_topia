"""Extract raw images and text blocks from docx files.

Read-only stage: opens each .docx as a zip archive, parses document.xml for
text runs and image references, and returns structured raw data.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .models import DocxImageRef, RawTextBlock

logger = logging.getLogger(__name__)


class DocxExtractor:
    """Extracts images and text from a single .docx file."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.annotator = filepath.stem
        self._zip: zipfile.ZipFile | None = None

    def open(self) -> None:
        """Open the docx zip archive."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"docx not found: {self.filepath}")
        self._zip = zipfile.ZipFile(self.filepath)

    def close(self) -> None:
        """Close the zip archive."""
        if self._zip:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> DocxExtractor:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_rid_to_media_map(self) -> dict[str, str]:
        """Parse _rels/document.xml.rels and return rId -> zip-path mapping.

        Targets in the rels file are relative to word/, so we prefix with
        'word/' to get the actual archive member path.
        """
        if not self._zip:
            raise RuntimeError("Call open() first")
        tree = ET.parse(self._zip.open("word/_rels/document.xml.rels"))
        root = tree.getroot()
        rmap: dict[str, str] = {}
        for rel in root.iter(
            "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
        ):
            target = rel.get("Target", "")
            if target and not target.startswith("word/"):
                target = f"word/{target}"
            rmap[rel.get("Id")] = target
        return rmap

    def get_all_text(self) -> str:
        """Concatenate all <w:t> text runs from document.xml."""
        if not self._zip:
            raise RuntimeError("Call open() first")
        tree = ET.parse(self._zip.open("word/document.xml"))
        root = tree.getroot()
        parts: list[str] = []
        for t in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
            if t.text:
                parts.append(t.text)
        return "".join(parts)

    def get_embed_positions(self) -> list[tuple[int, str]]:
        """Return list of (xml_byte_position, rId) for all r:embed references."""
        if not self._zip:
            raise RuntimeError("Call open() first")
        raw = self._zip.read("word/document.xml").decode("utf-8")
        return [(m.start(), m.group(1)) for m in re.finditer(r'r:embed="([^"]+)"', raw)]

    def get_text_run_positions(self) -> list[tuple[int, str]]:
        """Return list of (text_stream_position, content) for all <w:t> runs."""
        if not self._zip:
            raise RuntimeError("Call open() first")
        raw = self._zip.read("word/document.xml").decode("utf-8")
        return [(m.start(), m.group(1)) for m in re.finditer(r"<w:t[^>]*>([^<]*)</w:t>", raw)]

    def extract_text_blocks(self) -> list[RawTextBlock]:
        """Split concatenated text into numbered sample blocks.

        Each block starts at a numbered entry like "1.", "2.", etc.
        """
        text = self.get_all_text()
        matches = list(re.finditer(r"(\d+)\.", text))
        if not matches:
            return []

        blocks: list[RawTextBlock] = []
        for i, m in enumerate(matches):
            num = int(m.group(1))
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            blocks.append(
                RawTextBlock(
                    block_number=num,
                    raw_text=text[start:end],
                )
            )
        return blocks

    def assign_images_to_blocks(
        self,
        blocks: list[RawTextBlock],
        rid_map: dict[str, str],
    ) -> list[DocxImageRef]:
        """Map each r:embed in document.xml to its preceding numbered block.

        Uses XML-level positions to determine which image belongs to which block.
        """
        embeds = self.get_embed_positions()
        text_runs = self.get_text_run_positions()

        # Build text_stream positions for each numbered entry
        # Anchor on "N. Картен лист" to avoid matching numbers inside sheet IDs
        num_text_positions: list[tuple[int, int]] = []
        for block in blocks:
            for tr_pos, tr_text in text_runs:
                idx = tr_text.find(f"{block.block_number}.")
                if idx >= 0:
                    text_stream_pos = sum(len(t) for p, t in text_runs if p < tr_pos) + idx
                    num_text_positions.append((block.block_number, text_stream_pos))
                    break

        # Build text_stream positions for each embed
        embed_text_positions: list[tuple[str, int]] = []
        for em_pos, rid in embeds:
            ts_pos = sum(len(t) for p, t in text_runs if p < em_pos)
            embed_text_positions.append((rid, ts_pos))

        # Assign each embed to the latest numbered block before it
        refs: list[DocxImageRef] = []
        block_idx_counter: dict[int, int] = {}

        for rid, ts_pos in sorted(embed_text_positions, key=lambda x: x[1]):
            best_num = 0
            best_ts = -1
            for num, nts in num_text_positions:
                if nts <= ts_pos and (nts > best_ts or num > best_num):
                    best_ts = nts
                    best_num = num
            if best_num == 0:
                best_num = blocks[0].block_number if blocks else 1

            idx = block_idx_counter.get(best_num, 0)
            block_idx_counter[best_num] = idx + 1
            refs.append(
                DocxImageRef(
                    rid=rid,
                    media_path=rid_map.get(rid, ""),
                    sample_number=best_num,
                    image_index_in_sample=idx,
                )
            )
        return refs

    def extract_image_bytes(self, media_path: str) -> bytes:
        """Read image bytes from the docx archive."""
        if not self._zip:
            raise RuntimeError("Call open() first")
        return self._zip.read(media_path)
