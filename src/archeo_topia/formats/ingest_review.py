#!/usr/bin/env python3
"""Read the team's reviewed GeoPackage back, and turn it into evidence.

The delivery goes out as a GeoPackage, the team confirms, rejects, corrects and
adds features in QGIS or ArcGIS, and it comes back. This module reads it and
produces two things: an evaluation of the model against their verdicts, and a
COCO file so those verdicts land in CVAT beside the existing annotations.

**Integrity is checked before anything is counted.** A hand-edited file will
have surprises, and a silently miscounted one produces a number that looks
authoritative and is not. Checked: unknown or duplicated ``mound_id``, features
that moved off the sheet, a CRS the GIS changed on save, and features still
marked ``unreviewed``.

**What the resulting numbers mean, and do not.** Reviewers see model proposals,
so the result measures **precision** well: every proposal has a verdict.
Recall is bounded only over what a reviewer independently noticed and added,
which is not the same as an exhaustive search, so it is reported as a floor and
labelled as one. This is field verification, not the frozen blind test, and it
cannot substitute for it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from archeo_topia.formats.coco import CocoDocument
from archeo_topia.formats.features import FeatureCollection
from archeo_topia.formats.georeference import SheetReference
from archeo_topia.formats.labels import REVIEW_STATUS, LabelSchema

logger = logging.getLogger(__name__)

#: Verdicts a reviewer can record.
CONFIRMED = "confirmed"
REJECTED = "rejected"
CORRECTED = "corrected"
ADDED = "added"
UNREVIEWED = "unreviewed"

#: A feature that moved further than this from its original position is treated
#: as relocated rather than nudged. Generous next to a ~53 m mound symbol.
MOVED_THRESHOLD_M = 25.0


@dataclass
class ReviewReport:
    """What came back, and whether it can be trusted.

    Attributes:
        sheet_id: The sheet reviewed.
        counts: Verdict counts.
        problems: Integrity failures, empty when the file is sound.
        moved: Mound ids whose geometry the reviewer relocated, with distance.
        precision: Confirmed over all reviewed proposals, or None if none were.
        recall_floor: Confirmed over confirmed plus added, or None.
    """

    sheet_id: str
    counts: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    moved: dict[str, float] = field(default_factory=dict)
    precision: float | None = None
    recall_floor: float | None = None

    @property
    def sound(self) -> bool:
        """Whether the file passed every integrity check."""
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dict."""
        return {
            "sheet_id": self.sheet_id,
            "counts": self.counts,
            "problems": self.problems,
            "moved": self.moved,
            "precision": self.precision,
            "recall_floor": self.recall_floor,
            "scope": (
                "Field verification by domain experts on model-proposed candidates. Precision "
                "is measured over every proposal. Recall is a floor: it counts only the mounds "
                "a reviewer independently added, which is not an exhaustive search. Not the "
                "frozen blind test set and not a substitute for it."
            ),
        }


def check(
    returned: FeatureCollection,
    original: FeatureCollection,
    reference: SheetReference | None = None,
) -> ReviewReport:
    """Compare a returned collection against what was sent.

    Args:
        returned: The reviewed features.
        original: The features as delivered.
        reference: The sheet, used to check features are still on it.

    Returns:
        The report.
    """
    sheet_id = next(
        (f.properties.get("sheet_id") for f in returned.features if f.properties.get("sheet_id")),
        reference.sheet_id if reference else "?",
    )
    report = ReviewReport(sheet_id=sheet_id)

    sent = {f.properties["mound_id"]: f for f in original.features if f.properties.get("mound_id")}

    if returned.crs and original.crs and returned.crs != original.crs:
        report.problems.append(
            f"CRS changed on save: sent {original.crs}, received {returned.crs}. "
            "Reproject back before ingesting, or the coordinates will not line up."
        )

    seen: Counter[str] = Counter()
    for feature in returned.features:
        mound_id = feature.properties.get("mound_id")
        if not mound_id:
            continue
        seen[mound_id] += 1

    duplicated = sorted(m for m, n in seen.items() if n > 1)
    if duplicated:
        report.problems.append(
            f"{len(duplicated)} duplicated mound_id: {duplicated[:5]}"
            f"{'...' if len(duplicated) > 5 else ''}. mound_id is the join key and the "
            "identity this path matches on; it must not be edited or copied."
        )

    counts: Counter[str] = Counter()
    for feature in returned.features:
        mound_id = feature.properties.get("mound_id")
        status = str(feature.properties.get(REVIEW_STATUS, UNREVIEWED) or UNREVIEWED)

        if not mound_id or mound_id not in sent:
            # No id, or one we never sent: a feature the reviewer drew.
            counts[ADDED if status in (ADDED, UNREVIEWED) else status] += 1
            continue

        counts[status] += 1
        origin = sent[mound_id].geometry.get("coordinates")
        current = feature.geometry.get("coordinates")
        if origin and current:
            distance = math.dist(origin[:2], current[:2])
            if distance > MOVED_THRESHOLD_M:
                report.moved[mound_id] = round(distance, 2)

        if reference and reference.georeferenced and current:
            pixel = reference.from_ground(current[0], current[1])
            if pixel and not (
                0 <= pixel[0] < reference.size[0] and 0 <= pixel[1] < reference.size[1]
            ):
                report.problems.append(f"{mound_id} now falls outside the sheet")

    missing = sorted(set(sent) - set(seen))
    if missing:
        # Deleting rather than marking rejected is the failure mode review_status
        # exists to prevent: a deleted feature is indistinguishable from one that
        # was never sent, so the denominator cannot be reconstructed.
        report.problems.append(
            f"{len(missing)} feature(s) delivered but absent from the return: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}. Mark a wrong proposal "
            f"{REVIEW_STATUS}=rejected rather than deleting it, so it still counts in the "
            "denominator."
        )

    if counts.get(UNREVIEWED):
        report.problems.append(
            f"{counts[UNREVIEWED]} feature(s) still marked unreviewed; they are excluded "
            "from the figures below."
        )

    report.counts = dict(sorted(counts.items()))
    judged = counts[CONFIRMED] + counts[CORRECTED] + counts[REJECTED]
    if judged:
        report.precision = round((counts[CONFIRMED] + counts[CORRECTED]) / judged, 4)
    found = counts[CONFIRMED] + counts[CORRECTED]
    if found + counts[ADDED]:
        report.recall_floor = round(found / (found + counts[ADDED]), 4)
    return report


def to_coco(
    returned: FeatureCollection,
    reference: SheetReference,
    schema: LabelSchema,
    keep: tuple[str, ...] = (CONFIRMED, CORRECTED, ADDED),
) -> CocoDocument:
    """Turn accepted verdicts into a COCO document for CVAT.

    Args:
        returned: The reviewed features.
        reference: The sheet's clip geometry.
        schema: Label schema.
        keep: Which verdicts become mound annotations.

    Returns:
        The document.
    """
    accepted = FeatureCollection(
        features=[
            f for f in returned.features if str(f.properties.get(REVIEW_STATUS, "")) in keep
        ],
        crs=returned.crs,
        name=returned.name,
    )
    return CocoDocument.from_features(
        accepted,
        reference,
        schema=schema,
        description=f"Field-reviewed mounds for {reference.sheet_id}",
    )


def main() -> None:
    """Ingest a reviewed GeoPackage and write a report plus a COCO file."""
    parser = argparse.ArgumentParser(description="Ingest a reviewed GeoPackage")
    parser.add_argument("--returned", required=True, help="The reviewed .gpkg")
    parser.add_argument("--original", required=True, help="The .gpkg as delivered")
    parser.add_argument("--clips-dir", required=True, help="The sheet's clip directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", default="annotation/cvat/labels.json")
    parser.add_argument("--layer", default="mound_points")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    returned = FeatureCollection.from_geopackage(args.returned, args.layer)
    original = FeatureCollection.from_geopackage(args.original, args.layer)
    reference = SheetReference.from_clips_json(args.clips_dir)
    schema = LabelSchema.load(args.schema)

    report = check(returned, original, reference)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "review_report.json").write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )

    for problem in report.problems:
        logger.warning("%s", problem)
    logger.info("verdicts: %s", report.counts)
    if report.precision is not None:
        logger.info("precision %.3f, recall floor %s", report.precision, report.recall_floor)

    to_coco(returned, reference, schema).save(output / "instances_default.json")
    logger.info("wrote %s", output)


if __name__ == "__main__":
    main()
