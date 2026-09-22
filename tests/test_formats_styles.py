"""Tests for the QGIS styles generated from the schema of record.

The point of generating rather than hand-writing the style is that a reviewer's
dropdown cannot offer a value CVAT would reject. These tests hold that
property, and check that embedding leaves a GeoPackage readable.
"""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from archeo_topia.formats.labels import LabelSchema
from archeo_topia.formats.styles import (
    LOCKED_FIELDS,
    STATUS_COLOURS,
    SYMBOL_STYLE,
    apply_to_bundle,
    build_point_style,
    embed,
)

SCHEMA = Path("annotation/cvat/labels.json")
needs_schema = pytest.mark.skipif(not SCHEMA.exists(), reason="needs the label schema")


def body(document: str) -> ET.Element:
    """Parse a .qml past its DOCTYPE.

    QGIS writes ``<!DOCTYPE qgis PUBLIC 'http://mapserver.org/~lars/qgis.dtd'``,
    and ElementTree refuses the ``~`` in a public id. The declaration is what
    QGIS itself emits, so the test drops it rather than the module changing it.
    """
    return ET.fromstring(document[document.index("<qgis") :])


def schema_of(attributes: list[dict]) -> LabelSchema:
    """Build a one-label schema carrying the given attributes."""
    return LabelSchema(
        {
            "schema_version": "test",
            "labels": [
                {"name": "mound", "color": "", "type": "any", "attributes": attributes}
            ],
        }
    )


class TestPointStyle:
    """The form a reviewer meets."""

    def test_checkbox_attribute_becomes_a_checkbox(self) -> None:
        schema = schema_of(
            [
                {
                    "name": "has_trig_point",
                    "mutable": True,
                    "input_type": "checkbox",
                    "values": ["false"],
                    "default_value": "false",
                }
            ]
        )
        field = body(build_point_style(schema)).find(
            './/fieldConfiguration/field[@name="has_trig_point"]'
        )
        assert field is not None
        assert field.find("editWidgetSetup").get("type") == "CheckBox"

    def test_select_offers_exactly_the_schema_values(self) -> None:
        schema = schema_of(
            [
                {
                    "name": "water_line_crossing",
                    "mutable": True,
                    "input_type": "select",
                    "values": ["none", "surface", "underground"],
                    "default_value": "none",
                }
            ]
        )
        field = body(build_point_style(schema)).find(
            './/fieldConfiguration/field[@name="water_line_crossing"]'
        )
        assert field.find("editWidgetSetup").get("type") == "ValueMap"
        offered = {
            option.get("name")
            for option in field.findall(".//Option/Option")
            if option.get("name") not in (None, "map")
        }
        assert offered == {"none", "surface", "underground"}

    def test_a_new_schema_value_reaches_the_form(self) -> None:
        """The regression the generation exists to prevent."""
        attribute = {
            "name": "review_status",
            "mutable": True,
            "input_type": "select",
            "values": ["unreviewed", "confirmed"],
            "default_value": "unreviewed",
        }
        before = build_point_style(schema_of([attribute]))
        attribute["values"] = [*attribute["values"], "needs_fieldwork"]
        after = build_point_style(schema_of([attribute]))
        assert "needs_fieldwork" not in before
        assert "needs_fieldwork" in after

    def test_text_attributes_get_no_widget(self) -> None:
        """detector_confidence is model output; it is locked, not offered."""
        schema = schema_of(
            [
                {
                    "name": "detector_confidence",
                    "mutable": True,
                    "input_type": "text",
                    "values": [""],
                    "default_value": "",
                }
            ]
        )
        root = body(build_point_style(schema))
        assert root.findall(".//fieldConfiguration/field") == []
        locked = {field.get("name") for field in root.findall(".//editable/field")}
        assert "detector_confidence" in locked

    def test_bookkeeping_fields_are_locked(self) -> None:
        root = body(build_point_style(schema_of([])))
        locked = {field.get("name") for field in root.findall(".//editable/field")}
        assert locked == set(LOCKED_FIELDS)

    def test_every_review_status_has_a_category(self) -> None:
        root = body(build_point_style(schema_of([])))
        values = [category.get("value") for category in root.findall(".//category")]
        assert values == list(STATUS_COLOURS)
        assert root.find(".//renderer-v2").get("attr") == "review_status"

    def test_symbol_style_draws_no_fill(self) -> None:
        """A filled polygon reads as ground extent, which it is not."""
        root = body(SYMBOL_STYLE)
        options = {
            option.get("name"): option.get("value")
            for option in root.findall(".//Option/Option")
        }
        assert options["style"] == "no"
        assert options["color"] == "0,0,0,0"


class TestEmbed:
    """Styles travel inside the container, and leave it readable."""

    @staticmethod
    def make_gpkg(path: Path) -> None:
        """A minimal stand-in carrying the one table embedding writes to."""
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, "
            "data_type TEXT, identifier TEXT, description TEXT)"
        )
        connection.execute("CREATE TABLE mound_points (mound_id TEXT)")
        connection.execute("INSERT INTO mound_points VALUES ('a-00001')")
        connection.commit()
        connection.close()

    def test_embeds_both_layers_as_default(self, tmp_path: Path) -> None:
        gpkg = tmp_path / "sheet.gpkg"
        self.make_gpkg(gpkg)
        embed(gpkg, {"mound_points": "<qml/>", "mound_symbols": "<qml/>"}, "v0.0.3")
        connection = sqlite3.connect(gpkg)
        rows = connection.execute(
            "SELECT f_table_name, useAsDefault FROM layer_styles ORDER BY f_table_name"
        ).fetchall()
        connection.close()
        assert rows == [("mound_points", 1), ("mound_symbols", 1)]

    def test_is_idempotent(self, tmp_path: Path) -> None:
        """Re-running must not leave a stale vocabulary behind."""
        gpkg = tmp_path / "sheet.gpkg"
        self.make_gpkg(gpkg)
        embed(gpkg, {"mound_points": "<old/>"}, "v1")
        embed(gpkg, {"mound_points": "<new/>"}, "v2")
        connection = sqlite3.connect(gpkg)
        rows = connection.execute("SELECT styleQML FROM layer_styles").fetchall()
        connection.close()
        assert rows == [("<new/>",)]

    def test_registers_the_table_in_gpkg_contents(self, tmp_path: Path) -> None:
        gpkg = tmp_path / "sheet.gpkg"
        self.make_gpkg(gpkg)
        embed(gpkg, {"mound_points": "<qml/>"}, "v0.0.3")
        connection = sqlite3.connect(gpkg)
        row = connection.execute(
            "SELECT data_type FROM gpkg_contents WHERE table_name = 'layer_styles'"
        ).fetchone()
        assert row == ("attributes",)
        # The data is still there, which is the point of checking at all.
        assert connection.execute("SELECT count(*) FROM mound_points").fetchone() == (1,)
        connection.close()

    @needs_schema
    def test_apply_to_bundle_walks_the_delivery(self, tmp_path: Path) -> None:
        for name in ("sheets/a.gpkg", "set_aside/b.gpkg"):
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            self.make_gpkg(path)
        assert apply_to_bundle(tmp_path, LabelSchema.load(SCHEMA)) == 2


@needs_schema
class TestAgainstTheRealSchema:
    """The delivered form matches what an annotator sees in CVAT."""

    def test_every_select_and_checkbox_is_represented(self) -> None:
        schema = LabelSchema.load(SCHEMA)
        expected = {
            attribute["name"]
            for attribute in schema.attributes("mound")
            if attribute["input_type"] in ("checkbox", "select")
        }
        root = body(build_point_style(schema))
        present = {field.get("name") for field in root.findall(".//fieldConfiguration/field")}
        assert present == expected

    def test_water_line_crossing_keeps_its_four_states(self) -> None:
        """Two booleans cannot express these; the select is the whole point."""
        schema = LabelSchema.load(SCHEMA)
        field = body(build_point_style(schema)).find(
            './/fieldConfiguration/field[@name="water_line_crossing"]'
        )
        offered = {
            option.get("name")
            for option in field.findall(".//Option/Option")
            if option.get("name") not in (None, "map")
        }
        assert offered == {"none", "surface", "underground", "unreviewed"}
