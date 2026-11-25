"""Tests for LabelSchema: the schema of record and its three projections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archeo_topia.formats import LabelSchema, LabelValidationError
from archeo_topia.formats.labels import (
    REVIEW_STATUS,
    REVIEW_STATUS_VALUES,
    validate,
)

SCHEMA = Path("annotation/cvat/labels.json")
RAW = Path("annotation/cvat/labels_cvat_raw.json")
needs_schema = pytest.mark.skipif(not SCHEMA.exists(), reason="needs the label schema")


class TestRoundTrip:
    """JSON and YAML, because this is the one document a person edits."""

    def test_json_round_trip(self, tmp_path: Path) -> None:
        schema = LabelSchema(
            {"labels": [{"name": "a", "color": "", "type": "tag", "attributes": []}]}
        )
        out = tmp_path / "s.json"
        schema.save(out)
        assert LabelSchema.load(out).label_names == ["a"]

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        schema = LabelSchema(
            {"labels": [{"name": "a", "color": "", "type": "tag", "attributes": []}]}
        )
        out = tmp_path / "s.yaml"
        schema.save(out)
        assert "labels:" in out.read_text()
        assert LabelSchema.load(out).label_names == ["a"]


class TestValidation:
    """Ported from CVAT's browser-side checks, which are stricter than its API."""

    def test_a_wrapper_object_is_not_a_label_array(self) -> None:
        """The mistake that produces 'Label name must be a string' in the UI."""
        with pytest.raises(LabelValidationError, match="non-empty string"):
            validate([{"schema_version": "v0.0.3", "labels": []}])

    def test_empty_values_are_rejected_for_every_input_type(self) -> None:
        """CVAT's REST serializer accepts this; its label editor does not."""
        labels = [
            {
                "name": "mound",
                "color": "#33ddff",
                "type": "polygon",
                "attributes": [
                    {
                        "name": "detector_confidence",
                        "input_type": "text",
                        "mutable": True,
                        "values": [],
                        "default_value": "",
                    }
                ],
            }
        ]
        with pytest.raises(LabelValidationError, match="non-empty array"):
            validate(labels)

    def test_unknown_label_type_is_rejected(self) -> None:
        with pytest.raises(LabelValidationError, match="unknown label type"):
            validate([{"name": "m", "color": "", "type": "polygone", "attributes": []}])

    def test_duplicate_attribute_names_are_rejected(self) -> None:
        attribute = {
            "name": "dup",
            "input_type": "checkbox",
            "mutable": True,
            "values": ["false"],
            "default_value": "false",
        }
        labels = [
            {"name": "m", "color": "", "type": "tag", "attributes": [attribute, dict(attribute)]}
        ]
        with pytest.raises(LabelValidationError, match="attribute names must be unique"):
            validate(labels)

    def test_bad_colour_is_rejected(self) -> None:
        with pytest.raises(LabelValidationError, match="color value is invalid"):
            validate([{"name": "m", "color": "blue", "type": "tag", "attributes": []}])


class TestCoercion:
    """One attribute, three tools, three representations."""

    @pytest.fixture
    def schema(self) -> LabelSchema:
        return LabelSchema(
            {
                "labels": [
                    {
                        "name": "mound",
                        "color": "#33ddff",
                        "type": "polygon",
                        "attributes": [
                            {
                                "name": "flag",
                                "input_type": "checkbox",
                                "mutable": True,
                                "values": ["false"],
                                "default_value": "false",
                            },
                            {
                                "name": "score",
                                "input_type": "number",
                                "mutable": True,
                                "values": ["0"],
                                "default_value": "0",
                            },
                            {
                                "name": "note",
                                "input_type": "text",
                                "mutable": True,
                                "values": [""],
                                "default_value": "",
                            },
                        ],
                    }
                ]
            }
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("true", True), ("false", False), (1, True), (0, False), (True, True), ("yes", True)],
    )
    def test_checkbox_from_every_tool(
        self, schema: LabelSchema, value: object, expected: bool
    ) -> None:
        """CVAT gives 'true', a GeoPackage gives 1, our writer gives True."""
        assert schema.coerce("mound", {"flag": value})["flag"] is expected

    def test_number_becomes_float(self, schema: LabelSchema) -> None:
        assert schema.coerce("mound", {"score": "0.87"})["score"] == pytest.approx(0.87)

    def test_unparseable_number_falls_back_rather_than_raising(self, schema: LabelSchema) -> None:
        assert schema.coerce("mound", {"score": ""})["score"] == 0.0

    def test_unknown_attributes_are_dropped(self, schema: LabelSchema) -> None:
        assert "bogus" not in schema.coerce("mound", {"bogus": 1})

    def test_missing_attributes_take_their_default(self, schema: LabelSchema) -> None:
        assert set(schema.coerce("mound", {})) == {"flag", "score", "note"}

    def test_gis_field_types(self, schema: LabelSchema) -> None:
        assert schema.gis_field_types("mound") == {"flag": bool, "score": float, "note": str}


@needs_schema
class TestTheRealSchema:
    """The committed schema of record."""

    @pytest.fixture(scope="class")
    def schema(self) -> LabelSchema:
        return LabelSchema.load(SCHEMA)

    def test_it_validates(self, schema: LabelSchema) -> None:
        schema.validate()

    def test_three_labels(self, schema: LabelSchema) -> None:
        assert schema.label_names == ["mound", "hard_negative_symbol", "uncertain_ignore"]

    def test_review_status_is_declared_on_every_label(self, schema: LabelSchema) -> None:
        """A reviewer's verdict is a separate axis from who drew the shape.

        If a rejection were recorded by deleting the feature, a rejected
        detection would be indistinguishable from one never sent, and the
        recall denominator could not be reconstructed.
        """
        for label in schema.label_names:
            attribute = schema.attribute(label, REVIEW_STATUS)
            assert attribute["values"] == list(REVIEW_STATUS_VALUES)
            assert attribute["default_value"] == "unreviewed"
            assert schema.defaults(label)[REVIEW_STATUS] == "unreviewed"

    def test_provenance_is_kept_separate_from_review_status(self, schema: LabelSchema) -> None:
        provenance = schema.attribute("mound", "annotation_provenance")
        assert "human_added" in provenance["values"]
        assert REVIEW_STATUS not in provenance["values"]

    def test_defaults_are_typed_for_gis(self, schema: LabelSchema) -> None:
        defaults = schema.defaults("mound")
        assert defaults["blurred_or_bad_print"] is False
        assert defaults["water_line_crossing"] == "none"

    @pytest.mark.skipif(not RAW.exists(), reason="needs the generated CVAT array")
    def test_the_generated_array_matches_the_schema(self, schema: LabelSchema) -> None:
        assert json.loads(RAW.read_text()) == schema.to_cvat()
