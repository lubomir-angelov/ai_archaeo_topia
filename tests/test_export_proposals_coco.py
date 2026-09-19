"""Tests for the detector-proposal to CVAT COCO exporter (MapSAM v0.6 step 3)."""

from __future__ import annotations

import numpy as np

from archeo_topia.datasets.export_proposals_coco import (
    annotation_record,
    assign_to_clips,
    mask_to_polygon,
)
from archeo_topia.formats import LabelSchema

MANIFEST = {
    "sheet_id": "K-35-1-A-a",
    "clips": [
        {"file": "K-35-1-A-a_1.png", "offset_xy": [0, 0], "size": [100, 80]},
        {"file": "K-35-1-A-a_2.png", "offset_xy": [100, 0], "size": [100, 80]},
        {"file": "K-35-1-A-a_3.png", "offset_xy": [0, 80], "size": [100, 80]},
        {"file": "K-35-1-A-a_4.png", "offset_xy": [100, 80], "size": [100, 80]},
    ],
}


class TestAssignToClips:
    """The sweep runs over the parent sheet; CVAT annotates clips."""

    def test_each_candidate_lands_in_exactly_one_clip(self) -> None:
        candidates = [
            {"x": 10.0, "y": 10.0, "score": 0.9},
            {"x": 150.0, "y": 10.0, "score": 0.8},
            {"x": 10.0, "y": 120.0, "score": 0.7},
            {"x": 150.0, "y": 120.0, "score": 0.6},
        ]
        assigned = assign_to_clips(candidates, MANIFEST)
        assert [len(v) for v in assigned.values()] == [1, 1, 1, 1]

    def test_coordinates_are_translated_into_clip_space(self) -> None:
        assigned = assign_to_clips([{"x": 150.0, "y": 120.0, "score": 0.5}], MANIFEST)
        local = assigned["K-35-1-A-a_4.png"][0]
        assert (local["x"], local["y"]) == (50.0, 40.0)

    def test_the_box_is_translated_with_the_centre(self) -> None:
        candidate = {"x": 150.0, "y": 10.0, "score": 0.5, "box": [140.0, 0.0, 160.0, 20.0]}
        local = assign_to_clips([candidate], MANIFEST)["K-35-1-A-a_2.png"][0]
        assert local["box"] == [40.0, 0.0, 60.0, 20.0]
        assert local["x"] == 50.0

    def test_a_candidate_outside_every_clip_is_dropped(self) -> None:
        assigned = assign_to_clips([{"x": 999.0, "y": 999.0, "score": 0.5}], MANIFEST)
        assert sum(len(v) for v in assigned.values()) == 0

    def test_clip_boundaries_do_not_double_count(self) -> None:
        """A candidate exactly on a seam belongs to one clip, not two."""
        assigned = assign_to_clips([{"x": 100.0, "y": 80.0, "score": 0.5}], MANIFEST)
        assert sum(len(v) for v in assigned.values()) == 1


class TestMaskToPolygon:
    """Masks become polygons because that is what the existing annotations are."""

    def test_a_square_mask_becomes_a_polygon(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        polygon = mask_to_polygon(mask, (0, 0))
        assert polygon is not None
        assert len(polygon) % 2 == 0
        assert min(polygon[0::2]) == 8 and max(polygon[0::2]) == 23

    def test_the_window_offset_is_applied(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        polygon = mask_to_polygon(mask, (100, 200))
        assert min(polygon[0::2]) == 108
        assert min(polygon[1::2]) == 208

    def test_an_empty_mask_returns_none(self) -> None:
        assert mask_to_polygon(np.zeros((32, 32), dtype=bool), (0, 0)) is None

    def test_a_degenerate_mask_returns_none(self) -> None:
        """CVAT will not accept a shape with fewer than three points."""
        mask = np.zeros((32, 32), dtype=bool)
        mask[5, 5] = True
        assert mask_to_polygon(mask, (0, 0)) is None

    def test_the_largest_component_wins(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[4:8, 4:8] = True
        mask[30:60, 30:60] = True
        polygon = mask_to_polygon(mask, (0, 0))
        assert min(polygon[0::2]) >= 30


class TestAnnotationRecord:
    """Every proposal must arrive labelled as a proposal."""

    def test_provenance_marks_it_as_a_model_proposal(self) -> None:
        record = annotation_record(1, 1, 1, [0.0, 0.0, 25.0, 23.0], None, 0.7)
        assert record["attributes"]["annotation_provenance"] == "model_proposal_accepted"

    def test_water_line_is_unreviewed_not_guessed(self) -> None:
        record = annotation_record(1, 1, 1, [0.0, 0.0, 25.0, 23.0], None, 0.7)
        assert record["attributes"]["water_line_crossing"] == "unreviewed"

    def test_boolean_attributes_default_false(self) -> None:
        record = annotation_record(1, 1, 1, [0.0, 0.0, 25.0, 23.0], None, 0.7)
        schema = LabelSchema.load("annotation/cvat/labels.json")
        booleans = [name for name, kind in schema.gis_field_types("mound").items() if kind is bool]
        assert booleans, "the schema declares no boolean attributes"
        assert all(record["attributes"][name] is False for name in booleans)

    def test_confidence_is_carried_so_a_reviewer_can_sort(self) -> None:
        record = annotation_record(1, 1, 1, [0.0, 0.0, 25.0, 23.0], None, 0.58612)
        assert record["attributes"]["detector_confidence"] == 0.5861

    def test_negatives_carry_a_negative_type(self) -> None:
        record = annotation_record(1, 1, 2, [0.0, 0.0, 25.0, 23.0], None, 0.1)
        assert record["attributes"]["negative_type"] == "other"

    def test_mounds_do_not_carry_a_negative_type(self) -> None:
        record = annotation_record(1, 1, 1, [0.0, 0.0, 25.0, 23.0], None, 0.9)
        assert "negative_type" not in record["attributes"]

    def test_a_polygon_is_written_as_a_single_ring(self) -> None:
        record = annotation_record(
            1, 1, 1, [0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 0.0, 4.0, 4.0], 0.9
        )
        assert record["segmentation"] == [[0.0, 0.0, 4.0, 0.0, 4.0, 4.0]]

    def test_a_boxless_proposal_has_empty_segmentation(self) -> None:
        record = annotation_record(1, 1, 2, [0.0, 0.0, 25.0, 23.0], None, 0.1)
        assert record["segmentation"] == []


class TestSchemaAgreement:
    """The exporter's attributes now come from the schema, not a parallel list.

    This used to be a drift test between a literal block in this module and
    ``annotation/cvat/labels.json``. It is now a statement of the property that
    made the drift impossible: the exporter reads the schema, so every declared
    attribute is emitted and no undeclared one can be.
    """

    def test_emitted_attributes_are_exactly_what_the_schema_declares(self) -> None:
        schema = LabelSchema.load("annotation/cvat/labels.json")
        mound = annotation_record(1, 1, 1, [0.0, 0.0, 25.0, 23.0], None, 0.9)
        negative = annotation_record(2, 1, 2, [0.0, 0.0, 25.0, 23.0], None, 0.1)
        assert set(mound["attributes"]) == set(schema.defaults("mound"))
        assert set(negative["attributes"]) == set(schema.defaults("hard_negative_symbol"))

    def test_a_new_schema_attribute_reaches_the_export_without_an_edit(self) -> None:
        """The property the old drift test could only check after the fact."""
        schema = LabelSchema.load("annotation/cvat/labels.json")
        schema.label("mound")["attributes"].append(
            {
                "name": "invented_for_this_test",
                "input_type": "checkbox",
                "mutable": True,
                "values": ["false"],
                "default_value": "false",
            }
        )
        record = annotation_record(1, 1, 1, [0.0, 0.0, 25.0, 23.0], None, 0.9, schema=schema)
        assert record["attributes"]["invented_for_this_test"] is False
