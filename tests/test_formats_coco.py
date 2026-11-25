"""Tests for CocoDocument, especially the decoder that replaces two others."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from archeo_topia.formats import CocoDocument
from archeo_topia.formats.coco import (
    CATEGORIES,
    HARD_NEGATIVE,
    MOUND,
    decode_mask,
    sheet_of,
)

V003 = Path("annotation/cvat/v0.0.3/instances_default.json")


class TestSheetOf:
    """One rule, previously written three times."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("K-35-51-B-a_3.png", "K-35-51-B-a"),
            ("K-35-51-B-a_3", "K-35-51-B-a"),
            ("L-35-139-V-v_1.png", "L-35-139-V-v"),
            ("K-34-8-G-a_12.png", "K-34-8-G-a"),
        ],
    )
    def test_splits_on_the_last_underscore(self, name: str, expected: str) -> None:
        """Sheet ids contain hyphens but not underscores, so the last one wins."""
        assert sheet_of(name) == expected


class TestDecodeMask:
    """The union of the two decoders this replaces."""

    def test_single_polygon(self) -> None:
        annotation = {"segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]], "bbox": [2, 2, 6, 6]}
        mask = decode_mask(annotation, 10, 10)
        assert mask.shape == (10, 10)
        assert mask.any()
        assert mask[5, 5]

    def test_flat_polygon_without_a_wrapping_list(self) -> None:
        annotation = {"segmentation": [2, 2, 8, 2, 8, 8, 2, 8], "bbox": [2, 2, 6, 6]}
        assert decode_mask(annotation, 10, 10)[5, 5]

    def test_multi_polygon_unions_every_ring(self) -> None:
        """build_detection_windows.decode_mask read only segmentation[0].

        A two-part symbol therefore lost its second component silently.
        """
        annotation = {
            "segmentation": [[0, 0, 3, 0, 3, 3, 0, 3], [6, 6, 9, 6, 9, 9, 6, 9]],
            "bbox": [0, 0, 9, 9],
        }
        mask = decode_mask(annotation, 10, 10)
        assert mask[1, 1] and mask[7, 7]

    def test_bbox_fallback_for_a_box_only_annotation(self) -> None:
        """Hard negatives carry a box and no segmentation; there are 542 of them."""
        mask = decode_mask({"segmentation": [], "bbox": [2, 2, 4, 4]}, 10, 10)
        assert mask.any()
        assert mask[4, 4]

    def test_no_geometry_at_all_gives_an_empty_mask(self) -> None:
        assert not decode_mask({"segmentation": [], "bbox": None}, 10, 10).any()

    def test_uncompressed_rle_is_column_major(self) -> None:
        """COCO RLE is Fortran order; reading it row-major transposes the mask."""
        # 4x4, first column set: counts start with background 0, then 4 on.
        annotation = {"segmentation": {"size": [4, 4], "counts": [0, 4, 12]}}
        mask = decode_mask(annotation, 4, 4)
        assert mask[:, 0].all()
        assert not mask[:, 1:].any()

    def test_compressed_rle_is_refused_loudly(self) -> None:
        annotation = {"segmentation": {"size": [4, 4], "counts": "abcd"}}
        with pytest.raises(ValueError, match="compressed"):
            decode_mask(annotation, 4, 4)

    def test_rle_counts_must_cover_the_image(self) -> None:
        annotation = {"segmentation": {"size": [4, 4], "counts": [0, 3]}}
        with pytest.raises(ValueError, match="invalid RLE counts"):
            decode_mask(annotation, 4, 4)

    def test_negative_rle_count_is_refused(self) -> None:
        annotation = {"segmentation": {"size": [2, 2], "counts": [-1, 5]}}
        with pytest.raises(ValueError, match="negative"):
            decode_mask(annotation, 2, 2)


@pytest.mark.skipif(not V003.exists(), reason="needs the v0.0.3 export")
class TestAgainstTheRealExport:
    """Agreement with the decoder being replaced, on every real annotation."""

    def test_matches_build_detection_windows_on_every_annotation(self) -> None:
        from archeo_topia.datasets.build_detection_windows import decode_mask as old

        document = CocoDocument.load(V003)
        images = document.images_by_id
        checked = {"polygon": 0, "rle": 0}
        for annotation in document.annotations:
            image = images[annotation["image_id"]]
            segmentation = annotation.get("segmentation")
            if not segmentation:
                # 533 annotations carry a box and no segmentation. The old
                # decoder returned an empty mask for every one of them; the
                # bbox fallback is tested separately above.
                continue
            kind = "rle" if isinstance(segmentation, dict) else "polygon"
            mine = decode_mask(annotation, image["width"], image["height"])
            theirs = old(annotation, image["width"], image["height"])
            assert np.array_equal(mine, theirs), f"annotation {annotation['id']} differs"
            checked[kind] += 1
        assert checked == {"polygon": 20, "rle": 161}

    def test_indexes_and_counts(self) -> None:
        document = CocoDocument.load(V003)
        assert len(document.images) == 12
        assert len(document.annotations) == 714
        assert document.counts_by_category() == {
            MOUND: 169,
            HARD_NEGATIVE: 542,
            "uncertain_ignore": 3,
        }
        assert set(document.images_by_id) == {i["id"] for i in document.images}
        grouped = document.annotations_by_image()
        assert sum(len(v) for v in grouped.values()) == 714


class TestBuilding:
    """Writing a document CVAT will accept."""

    def test_empty_is_shaped_like_a_cvat_export(self) -> None:
        document = CocoDocument.empty("proposals")
        assert document.categories == [dict(c) for c in CATEGORIES]
        assert document.category_ids == {MOUND: 1, HARD_NEGATIVE: 2, "uncertain_ignore": 3}
        assert document.document["info"]["description"] == "proposals"

    def test_add_image_and_annotation(self, tmp_path: Path) -> None:
        document = CocoDocument.empty()
        image = document.add_image("K-35-1-A-a_1.png", 100, 80)
        document.add_annotation(image["id"], MOUND, [10, 10, 20, 18], [[10, 10, 30, 10, 30, 28]])
        annotation = document.annotations[0]
        assert annotation["category_id"] == 1
        assert annotation["area"] == pytest.approx(360.0)
        assert annotation["iscrowd"] == 0

        out = document.save(tmp_path / "c.json")
        assert CocoDocument.load(out).counts_by_category()[MOUND] == 1

    def test_image_records_carry_cvats_optional_fields(self) -> None:
        document = CocoDocument.empty()
        record = document.add_image("a.png", 10, 10)
        assert set(record) >= {"license", "flickr_url", "coco_url", "date_captured"}
