#!/usr/bin/env python3
"""The annotation schema of record, and the projections each tool needs.

``annotation/cvat/labels.json`` describes the three classes and their
attributes. Three consumers want three different shapes of the same thing:

* **CVAT** wants a bare JSON array carrying only the four keys its label editor
  knows, and rejects anything else.
* **GIS** wants flat feature properties whose JSON types let ``ogr2ogr`` infer
  a sensible column type.
* **The exporters** want the attribute bag at its defaults, and a way to
  normalise a bag coming back from either tool.

Before this class those three lived in three places, and the third was a
literal Python block in ``export_proposals_coco`` duplicating the schema, with
a test whose only job was to notice when the two drifted apart.

**Validation mirrors CVAT's browser-side check, not its REST serializer.** The
two disagree and the stricter one is the one a person meets first. CVAT's
``LabelSerializer`` accepts ``values: []`` on a text attribute; the label
editor in the browser refuses it with *"attribute values must be a non-empty
array"* before the request is ever sent. The rules below are ported from
``cvat-ui/src/components/labels-editor/common.ts`` so a schema that would be
refused in the UI fails here instead.

Usage:
    python -m archeo_topia.formats.labels \\
        --schema annotation/cvat/labels.json \\
        --output annotation/cvat/labels_cvat_raw.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Keys CVAT's label editor accepts on an attribute. Anything else is dropped;
#: the schema of record carries prose that CVAT has no field for.
ATTRIBUTE_KEYS = ("name", "mutable", "input_type", "values", "default_value")

#: Input types CVAT recognises.
INPUT_TYPES = ("checkbox", "number", "text", "radio", "select")

#: Shape types CVAT recognises.
LABEL_TYPES = (
    "any",
    "rectangle",
    "polygon",
    "polyline",
    "points",
    "ellipse",
    "cuboid",
    "cuboid_3d",
    "mask",
    "tag",
    "skeleton",
)

#: Attributes CVAT supplies itself; they are not part of the schema of record.
BUILTIN_ATTRIBUTES = ("occluded", "rotation")

#: What a reviewer concluded, as opposed to who drew the shape. Kept separate
#: from ``annotation_provenance`` on purpose: if a rejection were expressed by
#: deleting the feature, a rejected detection would be indistinguishable from
#: one that was never sent, and the recall denominator could not be
#: reconstructed afterwards.
REVIEW_STATUS = "review_status"
REVIEW_STATUS_VALUES = ("unreviewed", "confirmed", "rejected", "corrected", "added")


class LabelValidationError(ValueError):
    """A label set CVAT's editor would refuse."""


def validate_attribute(attribute: dict[str, Any], label: str) -> None:
    """Check one attribute against CVAT's browser-side rules.

    Args:
        attribute: The attribute spec.
        label: Owning label name, for the error message.

    Raises:
        LabelValidationError: If CVAT's label editor would reject it.
    """
    name = attribute.get("name")
    if not isinstance(name, str) or not name.strip():
        raise LabelValidationError(f"{label}: attribute name must be a non-empty string")

    if str(attribute.get("input_type", "")).lower() not in INPUT_TYPES:
        raise LabelValidationError(
            f"{label}.{name}: unknown input type {attribute.get('input_type')!r}"
        )

    if not isinstance(attribute.get("mutable"), bool):
        raise LabelValidationError(f"{label}.{name}: mutable must be a boolean")

    values = attribute.get("values")
    if not isinstance(values, list) or not values:
        # The rule a text attribute trips: CVAT requires a non-empty array for
        # every input type, and a text attribute's single entry is its empty
        # default.
        raise LabelValidationError(f"{label}.{name}: attribute values must be a non-empty array")

    if any(not isinstance(value, str) for value in values):
        raise LabelValidationError(f"{label}.{name}: each attribute value must be a string")

    trimmed = [value.strip() for value in values]
    if len(set(trimmed)) != len(trimmed):
        raise LabelValidationError(f"{label}.{name}: attribute values must be unique")

    default = attribute.get("default_value")
    if default and default not in values and str(attribute["input_type"]).lower() != "text":
        raise LabelValidationError(f"{label}.{name}: invalid default value {default!r}")


def validate_label(label: dict[str, Any]) -> None:
    """Check one label against CVAT's browser-side rules.

    Args:
        label: The label spec.

    Raises:
        LabelValidationError: If CVAT's label editor would reject it.
    """
    name = label.get("name")
    if not isinstance(name, str) or not name.strip():
        raise LabelValidationError("label name must be a non-empty string")

    color = label.get("color")
    if color and (not isinstance(color, str) or not _is_hex(color)):
        raise LabelValidationError(f"{name}: color value is invalid")

    if label.get("type") not in LABEL_TYPES:
        raise LabelValidationError(f"{name}: unknown label type {label.get('type')!r}")

    attributes = label.get("attributes")
    if not isinstance(attributes, list):
        raise LabelValidationError(f"{name}: attributes must be an array")

    for attribute in attributes:
        validate_attribute(attribute, name)

    names = [a["name"].strip() for a in attributes]
    if len(set(names)) != len(names):
        raise LabelValidationError(f"{name}: attribute names must be unique")


def _is_hex(color: str) -> bool:
    """Return whether a string is a ``#rrggbb`` colour or empty."""
    if color == "":
        return True
    return (
        len(color) == 7
        and color[0] == "#"
        and all(c in "0123456789abcdefABCDEF" for c in color[1:])
    )


def validate(labels: list[dict[str, Any]]) -> None:
    """Check a whole label set as CVAT's editor would.

    Args:
        labels: The array CVAT's label editor would receive.

    Raises:
        LabelValidationError: On the first problem found.
    """
    if not isinstance(labels, list):
        raise LabelValidationError("the label set must be a JSON array, not an object")
    for label in labels:
        validate_label(label)


@dataclass
class LabelSchema:
    """The annotation schema of record.

    Attributes:
        document: The parsed schema, with ``schema_version``, ``note``,
            ``builtin_attributes_excluded`` and ``labels``.
    """

    document: dict[str, Any]

    # ---- loading and saving -------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> LabelSchema:
        """Read a schema from JSON or YAML, chosen by file extension.

        YAML is offered here and nowhere else in this package. This is the one
        document a person edits by hand, so comments and readability are worth
        a second parser; COCO and GeoJSON are machine-generated and have
        readers that expect JSON.

        Args:
            path: ``.json``, ``.yaml`` or ``.yml``.

        Returns:
            The schema.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            return cls(yaml.safe_load(text))
        return cls(json.loads(text))

    def save(self, path: str | Path) -> None:
        """Write the schema as JSON or YAML, chosen by file extension.

        Args:
            path: Destination.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            path.write_text(
                yaml.safe_dump(self.document, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(self.document, indent=2) + "\n", encoding="utf-8")

    # ---- lookups ------------------------------------------------------------

    @property
    def labels(self) -> list[dict[str, Any]]:
        """The label specs, in declaration order."""
        return list(self.document.get("labels", []))

    @property
    def label_names(self) -> list[str]:
        """The label names, in declaration order."""
        return [label["name"] for label in self.labels]

    def label(self, name: str) -> dict[str, Any]:
        """Return one label spec.

        Args:
            name: Label name.

        Returns:
            The spec.

        Raises:
            KeyError: If the label is not declared.
        """
        for label in self.labels:
            if label["name"] == name:
                return label
        raise KeyError(f"no label {name!r}; have {self.label_names}")

    def attributes(self, label: str) -> list[dict[str, Any]]:
        """Return one label's attribute specs.

        Args:
            label: Label name.

        Returns:
            The attribute specs.
        """
        return list(self.label(label).get("attributes", []))

    def attribute(self, label: str, name: str) -> dict[str, Any]:
        """Return one attribute spec.

        Args:
            label: Label name.
            name: Attribute name.

        Returns:
            The spec.

        Raises:
            KeyError: If the attribute is not declared on that label.
        """
        for attribute in self.attributes(label):
            if attribute["name"] == name:
                return attribute
        raise KeyError(f"no attribute {name!r} on {label!r}")

    # ---- projections --------------------------------------------------------

    def to_cvat(self) -> list[dict[str, Any]]:
        """Derive CVAT's paste-ready label array.

        Returns:
            The array to paste into CVAT's raw label editor.

        Raises:
            LabelValidationError: If the result would be refused.
        """
        labels = [
            {
                "name": label["name"],
                "color": label["color"],
                "type": label["type"],
                "attributes": [
                    {key: attribute[key] for key in ATTRIBUTE_KEYS}
                    for attribute in label["attributes"]
                ],
            }
            for label in self.labels
        ]
        validate(labels)
        return labels

    def validate(self) -> None:
        """Check the schema as CVAT's editor would.

        Raises:
            LabelValidationError: On the first problem found.
        """
        self.to_cvat()

    def defaults(self, label: str) -> dict[str, Any]:
        """Return the attribute bag at its declared defaults.

        Values are returned as the Python types that serialise into JSON types
        ``ogr2ogr`` infers usefully: a checkbox becomes ``bool`` so GeoPackage
        gets ``Integer(Boolean)``, a number becomes ``float``, everything else
        stays a string.

        Args:
            label: Label name.

        Returns:
            Attribute name to default value.
        """
        return {
            attribute["name"]: _typed(attribute, attribute.get("default_value", ""))
            for attribute in self.attributes(label)
        }

    def gis_field_types(self, label: str) -> dict[str, type]:
        """Return the Python type each attribute takes in a GIS layer.

        Args:
            label: Label name.

        Returns:
            Attribute name to ``bool``, ``float`` or ``str``.
        """
        return {attribute["name"]: _python_type(attribute) for attribute in self.attributes(label)}

    def coerce(self, label: str, attributes: dict[str, Any]) -> dict[str, Any]:
        """Normalise an attribute bag arriving from CVAT or from a GIS.

        The same checkbox comes back as ``"true"`` from a CVAT export, as ``1``
        from a GeoPackage and as ``True`` from our own writer. Unless they are
        brought to one type, a reviewer's answer means different things
        depending on which tool they used.

        Unknown attributes are dropped, and declared attributes that are
        missing take their default, so the result always matches the schema.

        Args:
            label: Label name.
            attributes: The incoming bag.

        Returns:
            A bag matching the schema exactly.
        """
        out = self.defaults(label)
        for attribute in self.attributes(label):
            name = attribute["name"]
            if name in attributes:
                out[name] = _typed(attribute, attributes[name])
        return out


def _python_type(attribute: dict[str, Any]) -> type:
    """Return the Python type an attribute's values take."""
    input_type = str(attribute.get("input_type", "")).lower()
    if input_type == "checkbox":
        return bool
    if input_type == "number":
        return float
    return str


def _typed(attribute: dict[str, Any], value: Any) -> Any:
    """Coerce one value to its attribute's Python type.

    Args:
        attribute: The attribute spec.
        value: The incoming value, from any of the three tools.

    Returns:
        The value as ``bool``, ``float`` or ``str``.
    """
    target = _python_type(attribute)
    if target is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    if target is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return "" if value is None else str(value)


def main() -> None:
    """Write the paste-ready label array beside the schema of record."""
    parser = argparse.ArgumentParser(description="Derive CVAT's label array from the schema")
    parser.add_argument("--schema", default="annotation/cvat/labels.json")
    parser.add_argument("--output", default="annotation/cvat/labels_cvat_raw.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    schema = LabelSchema.load(args.schema)
    labels = schema.to_cvat()
    Path(args.output).write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "wrote %s: %s",
        args.output,
        ", ".join(f"{label['name']} ({len(label['attributes'])} attributes)" for label in labels),
    )


if __name__ == "__main__":
    main()
