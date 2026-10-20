#!/usr/bin/env python3
"""Migrate the CVAT annotation schema to v0.0.3 (MapSAM v0.6).

Two changes, both cheap and both aimed at the same thing: making a domain rule
that has already caught twelve mislabels into a check a machine can run.

**``crossed_by_water_line`` becomes ``water_line_crossing``.** A mound cannot be
crossed by a *surface* watercourse -- water runs along terrain lows and a mound
is a raised feature -- so the attribute being true on a ``mound`` was a strong
data-error signal. It found all twelve watermills labelled as mounds on
``K-34-35-B-g``. But an *underground* line is a pipe and can run beneath
anything, and the boolean conflates the two, so a hit meant one of three things
and only a person looking at the map could say which.

The replacement is a single select rather than the two booleans the v0.6 plan
proposed, because two booleans cannot express the state most instances are
actually in. 105 hard negatives and 2 uncertain regions carry the old attribute
and **nobody has reviewed whether their crossing is surface or underground**.
Splitting into two booleans would force a value onto every one of them and
manufacture 105 assertions no annotator ever made. A fourth state,
``unreviewed``, records the truth instead, and it doubles as a work queue.

Values: ``none``, ``surface``, ``underground``, ``unreviewed``.

The mechanical check the split restores is then one line: a ``mound`` with
``surface`` is a data error, and a ``mound`` with ``unreviewed`` is a mound
nobody has checked.

**One corrected attribute.** ``K-35-51-B-a_3`` at (314, 15.5), COCO id 528, is a
real mound beside a lake with no water line crossing it; the annotator confirmed
the attribute was set in error, and v0.5 reported the number without it because
it was known wrong. It becomes ``none``. The other flagged mound,
``K-35-8-G-a_1`` at (955.5, 627.5), COCO id 22, was reviewed and is a real mound
with a genuine underground line, so it becomes ``underground``.

Nothing else changes: no annotation is added, removed or relabelled, and
v0.0.1 and v0.0.2 are left exactly as they are so the pre- and post-correction
experiments stay side by side.

Usage:
    python -m archeo_topia.datasets.migrate_annotation_schema \\
        --source annotation/cvat/v0.0.2/instances_default.json \\
        --output annotation/cvat/v0.0.3/instances_default.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OLD_ATTRIBUTE = "crossed_by_water_line"
NEW_ATTRIBUTE = "water_line_crossing"

NONE = "none"
SURFACE = "surface"
UNDERGROUND = "underground"
UNREVIEWED = "unreviewed"

VALUES = (NONE, SURFACE, UNDERGROUND, UNREVIEWED)

#: Reviewed mounds, keyed by COCO annotation id. These are the only two mounds
#: in v0.0.2 carrying the old attribute, and both were resolved in v0.5 --
#: see ``docs/mapsam/v005/RESULTS.md``, "Known data notes".
REVIEWED: dict[int, tuple[str, str]] = {
    22: (UNDERGROUND, "K-35-8-G-a_1 (955.5, 627.5): real mound, genuine underground line"),
    528: (NONE, "K-35-51-B-a_3 (314, 15.5): real mound, attribute set in error"),
}


def migrate_attributes(
    annotation: dict[str, Any], category: str
) -> tuple[dict[str, Any], str]:
    """Rewrite one annotation's water-line attribute.

    Args:
        annotation: A COCO annotation.
        category: Its category name.

    Returns:
        The new attribute dict, and the value assigned.
    """
    attributes = dict(annotation.get("attributes", {}))
    crossed = bool(attributes.pop(OLD_ATTRIBUTE, False))

    if not crossed:
        value = NONE
    elif annotation["id"] in REVIEWED:
        value = REVIEWED[annotation["id"]][0]
    elif category == "mound":
        # A mound carrying the old attribute that is not in REVIEWED would be a
        # case nobody has looked at. There are none in v0.0.2, and if a later
        # export adds one it must surface rather than be quietly resolved.
        value = UNREVIEWED
    else:
        # Hard negatives and uncertain regions: the attribute was recorded, the
        # surface/underground distinction was never asked for.
        value = UNREVIEWED

    attributes[NEW_ATTRIBUTE] = value
    return attributes, value


def migrate(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the schema migration to a whole COCO document.

    Args:
        document: The parsed v0.0.2 export.

    Returns:
        The migrated document and a report of what changed.
    """
    categories = {c["id"]: c["name"] for c in document["categories"]}
    counts: dict[str, dict[str, int]] = {}
    corrections: list[str] = []

    for annotation in document["annotations"]:
        category = categories[annotation["category_id"]]
        attributes, value = migrate_attributes(annotation, category)
        annotation["attributes"] = attributes
        counts.setdefault(category, dict.fromkeys(VALUES, 0))[value] += 1
        if annotation["id"] in REVIEWED:
            corrections.append(f"id {annotation['id']} -> {value}: {REVIEWED[annotation['id']][1]}")

    report = {
        "source_schema": OLD_ATTRIBUTE,
        "target_schema": NEW_ATTRIBUTE,
        "annotations": len(document["annotations"]),
        "by_category": counts,
        "reviewed_corrections": corrections,
        "mounds_flagged_for_review": counts.get("mound", {}).get(SURFACE, 0)
        + counts.get("mound", {}).get(UNREVIEWED, 0),
    }
    return document, report


def water_line_violations(document: dict[str, Any]) -> list[int]:
    """Return mounds whose water-line value needs a human.

    This is the check the migration exists to restore. It is one-way: it says
    nothing about symbols *not* crossed by a water line.

    Args:
        document: A migrated COCO document.

    Returns:
        COCO annotation ids of mounds marked ``surface`` or ``unreviewed``.
    """
    categories = {c["id"]: c["name"] for c in document["categories"]}
    return [
        a["id"]
        for a in document["annotations"]
        if categories[a["category_id"]] == "mound"
        and a.get("attributes", {}).get(NEW_ATTRIBUTE) in {SURFACE, UNREVIEWED}
    ]


def main() -> None:
    """Migrate an export and write the result alongside a report."""
    parser = argparse.ArgumentParser(description="Migrate the CVAT annotation schema")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    source = Path(args.source)
    output = Path(args.output)
    document = json.loads(source.read_text(encoding="utf-8"))

    before = len(document["annotations"])
    document, report = migrate(document)
    assert len(document["annotations"]) == before, "migration must not change the annotation count"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document), encoding="utf-8")
    (output.parent / "MIGRATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger.info("wrote %s (%d annotations)", output, before)
    for line in report["reviewed_corrections"]:
        logger.info("  %s", line)
    logger.info("mounds still needing review: %d", report["mounds_flagged_for_review"])


if __name__ == "__main__":
    main()
