"""Tests for the v0.0.3 schema migration and the label-schema artifact (v0.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archeo_topia.datasets.migrate_annotation_schema import (
    NEW_ATTRIBUTE,
    OLD_ATTRIBUTE,
    REVIEWED,
    UNDERGROUND,
    UNREVIEWED,
    VALUES,
    migrate,
    water_line_violations,
)

V002 = Path("annotation/cvat/v0.0.2/instances_default.json")
V003 = Path("annotation/cvat/v0.0.3/instances_default.json")
LABELS = Path("annotation/cvat/labels.json")
SPLITS = Path("configs/splits/v0_6_splits.json")


@pytest.mark.skipif(not V002.exists(), reason="needs the v0.0.2 export")
class TestMigration:
    """v0.0.3 must differ from v0.0.2 only where it is supposed to."""

    @pytest.fixture(scope="class")
    def migrated(self) -> dict:
        return migrate(json.loads(V002.read_text(encoding="utf-8")))[0]

    @pytest.fixture(scope="class")
    def source(self) -> dict:
        return json.loads(V002.read_text(encoding="utf-8"))

    def test_annotation_count_is_unchanged(self, migrated: dict, source: dict) -> None:
        assert len(migrated["annotations"]) == len(source["annotations"]) == 714

    def test_category_counts_are_unchanged(self, migrated: dict) -> None:
        names = {c["id"]: c["name"] for c in migrated["categories"]}
        counts: dict[str, int] = {}
        for annotation in migrated["annotations"]:
            counts[names[annotation["category_id"]]] = (
                counts.get(names[annotation["category_id"]], 0) + 1
            )
        assert counts == {"mound": 169, "hard_negative_symbol": 542, "uncertain_ignore": 3}

    def test_geometry_is_untouched(self, migrated: dict, source: dict) -> None:
        for new, old in zip(migrated["annotations"], source["annotations"], strict=True):
            assert new["id"] == old["id"]
            assert new["bbox"] == old["bbox"]
            assert new["segmentation"] == old["segmentation"]

    def test_old_attribute_is_gone_and_new_one_is_everywhere(self, migrated: dict) -> None:
        for annotation in migrated["annotations"]:
            assert OLD_ATTRIBUTE not in annotation["attributes"]
            assert annotation["attributes"][NEW_ATTRIBUTE] in VALUES

    def test_no_other_attribute_changes(self, migrated: dict, source: dict) -> None:
        for new, old in zip(migrated["annotations"], source["annotations"], strict=True):
            a, b = dict(old["attributes"]), dict(new["attributes"])
            a.pop(OLD_ATTRIBUTE, None)
            b.pop(NEW_ATTRIBUTE, None)
            assert a == b, f"attribute drift on annotation {new['id']}"

    def test_the_two_reviewed_mounds_resolve_as_recorded(self, migrated: dict) -> None:
        """v0.5 resolved both flagged mounds; the migration must carry those verdicts.

        ``K-35-8-G-a_1`` is a real mound with a genuine underground line, and
        ``K-35-51-B-a_3`` is a real mound whose attribute was set in error --
        v0.5 reported its figures knowing that and left the fix to this export.
        """
        by_id = {a["id"]: a for a in migrated["annotations"]}
        for annotation_id, (expected, _) in REVIEWED.items():
            assert by_id[annotation_id]["attributes"][NEW_ATTRIBUTE] == expected
        assert by_id[22]["attributes"][NEW_ATTRIBUTE] == UNDERGROUND

    def test_no_mound_needs_review(self, migrated: dict) -> None:
        """The check the split exists to restore, run on the corrected export.

        A mound marked ``surface`` is a data error and a mound marked
        ``unreviewed`` is one nobody has checked. Both must be empty here,
        because v0.5 resolved the only two candidates.
        """
        assert water_line_violations(migrated) == []

    def test_unreviewed_is_used_rather_than_guessed(self, migrated: dict) -> None:
        """105 hard negatives carried the old flag and were never classified.

        Two booleans would have forced a value onto every one of them. The
        fourth state records that nobody asked the question.
        """
        names = {c["id"]: c["name"] for c in migrated["categories"]}
        unreviewed = [
            a
            for a in migrated["annotations"]
            if a["attributes"][NEW_ATTRIBUTE] == UNREVIEWED
            and names[a["category_id"]] == "hard_negative_symbol"
        ]
        assert len(unreviewed) == 105

    @pytest.mark.skipif(not V003.exists(), reason="needs the committed v0.0.3 export")
    def test_the_committed_export_matches_a_fresh_migration(self, migrated: dict) -> None:
        assert json.loads(V003.read_text(encoding="utf-8")) == migrated


@pytest.mark.skipif(not LABELS.exists(), reason="needs the label schema")
class TestLabelSchema:
    """The schema artifact must describe the export it was generated from."""

    @pytest.fixture(scope="class")
    def labels(self) -> dict:
        return json.loads(LABELS.read_text(encoding="utf-8"))

    def test_three_classes(self, labels: dict) -> None:
        assert [label["name"] for label in labels["labels"]] == [
            "mound",
            "hard_negative_symbol",
            "uncertain_ignore",
        ]

    def test_every_class_carries_provenance(self, labels: dict) -> None:
        for label in labels["labels"]:
            names = {a["name"] for a in label["attributes"]}
            assert "annotation_provenance" in names
            assert NEW_ATTRIBUTE in names

    def test_water_line_values_match_the_migration(self, labels: dict) -> None:
        for label in labels["labels"]:
            attribute = next(a for a in label["attributes"] if a["name"] == NEW_ATTRIBUTE)
            assert attribute["values"] == list(VALUES)

    @pytest.mark.skipif(not V003.exists(), reason="needs the committed v0.0.3 export")
    def test_schema_covers_every_attribute_in_the_export(self, labels: dict) -> None:
        """The drift this file exists to prevent: an attribute in the data and not the schema."""
        document = json.loads(V003.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in document["categories"]}
        declared = {
            label["name"]: {a["name"] for a in label["attributes"]} for label in labels["labels"]
        }
        builtin = set(labels["builtin_attributes_excluded"])
        for annotation in document["annotations"]:
            category = names[annotation["category_id"]]
            used = set(annotation["attributes"]) - builtin
            assert used <= declared[category], (
                f"{category} annotation {annotation['id']} uses "
                f"{sorted(used - declared[category])}, absent from labels.json"
            )


@pytest.mark.skipif(not SPLITS.exists(), reason="needs the splits file")
class TestSplits:
    """Grouping by 1:100k parent, and what that forces."""

    @pytest.fixture(scope="class")
    def splits(self) -> dict:
        return json.loads(SPLITS.read_text(encoding="utf-8"))

    def test_every_sheet_has_exactly_one_split(self, splits: dict) -> None:
        assigned = [s for names in splits["splits"].values() for s in names]
        assert len(assigned) == len(set(assigned)) == len(splits["sheets"]) == 63

    def test_no_parent_spans_two_splits(self, splits: dict) -> None:
        """The whole point of the grouping.

        Adjacent 1:25k quadrants of one 1:50k sheet share terrain, survey
        campaign, print run and scan batch, so a parent appearing in both train
        and test would be a leak the sheet ids alone would not reveal.
        """
        seen: dict[str, str] = {}
        for sheet, meta in splits["sheets"].items():
            parent, split = meta["parent"], meta["split"]
            assert seen.setdefault(parent, split) == split, (
                f"parent {parent} spans {seen[parent]} and {split} (at {sheet})"
            )

    def test_the_adjacent_quadrant_pair_is_grouped(self, splits: dict) -> None:
        sheets = splits["sheets"]
        assert sheets["K-35-51-B-g"]["split"] == sheets["K-35-51-B-a"]["split"]

    def test_the_four_frozen_sheets_are_in_test(self, splits: dict) -> None:
        frozen = [s for s, m in splits["sheets"].items() if m["frozen_blind_test"]]
        assert sorted(frozen) == [
            "K-35-22-A-v",
            "K-35-39-G-v",
            "K-35-39-V-g",
            "L-35-139-V-v",
        ]
        assert all(splits["sheets"][s]["split"] == "test" for s in frozen)

    def test_the_unannotated_sibling_of_two_frozen_sheets_is_held_out(self, splits: dict) -> None:
        """K-35-39-A-g shares parent K-35-39 with two blind test sheets.

        It is not annotated and contributes no evaluation instances, but
        training on it would be exactly the leak the parent grouping prevents.
        """
        assert splits["sheets"]["K-35-39-A-g"]["split"] == "test"

    def test_the_annotated_sheets_are_trainable(self, splits: dict) -> None:
        for sheet in ("K-35-51-B-a", "K-35-8-G-a", "K-34-35-B-g"):
            assert splits["sheets"][sheet]["annotated"] is True
            assert splits["sheets"][sheet]["split"] == "train"

    def test_scale_and_series_are_recorded(self, splits: dict) -> None:
        """A nomenclature-read-as-scale error propagated through nine documents."""
        for meta in splits["sheets"].values():
            assert meta["scale"] == "1:25000"
            assert meta["series"] == "Bulgarian archival topographic"

    def test_the_loso_preset_still_matches_the_code(self, splits: dict) -> None:
        from archeo_topia.datasets.build_detection_windows import FOLDS

        assert splits["presets"]["loso_v0_5"]["folds"] == FOLDS
