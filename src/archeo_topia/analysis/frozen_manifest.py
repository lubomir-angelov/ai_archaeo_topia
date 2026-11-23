#!/usr/bin/env python3
"""Pin every input of the frozen blind-test openings by sha256 (protocol C6).

``frozen_manifest.json`` is the only place the full hash table exists. Every
opening verifies it first and stops on any mismatch, so a changed ground truth,
clip, checkpoint or windows dataset can never reach a published number.

Entries carry a ``root`` (``repo`` or ``lake``) and a path relative to it, so
the committed file holds no host path. Hashing is byte-level and touches no
model output: writing or verifying a manifest is not an opening.

**A manifest is written once and extended once.** ``extend`` appends the refit
detector and its ``selection.json`` for opening 2 and nothing else. It first
re-verifies every existing entry, then records the sha256 of the pre-extension
file and the number of its entries under ``base``. ``verify`` re-derives that
hash from ``entries[:N]``, so the opening-1 entries are provably the bytes
opening 1 verified, and an extended manifest still pins them.

Usage:
    python -m archeo_topia.analysis.frozen_manifest write \\
        --lake-root "$LAKE" --clips-dir cleaned/map_clips/_frozen \\
        --ground-truth annotation/cvat/frozen_test/v0.0.1/instances_Test.json \\
        --source-md annotation/cvat/frozen_test/v0.0.1/SOURCE.md \\
        --detector-run artifacts/detection/v0_5_yolo26s_gtfix \\
        --mapsam-root artifacts/models/mapsam --mapsam-prefix v0_4r_loso512 \\
        --sam-checkpoint models/checkpoints/sam/sam_vit_b_01ec64.pth \\
        --dataset data/curated/datasets/mapsam_det_frozen_v001 \\
        --output docs/mapsam/v006/analysis/frozen_manifest.json

    python -m archeo_topia.analysis.frozen_manifest extend \\
        --manifest docs/mapsam/v006/analysis/frozen_manifest.json --lake-root "$LAKE" \\
        --selection artifacts/detection/v0_6_refit/selection/selection.json

    python -m archeo_topia.analysis.frozen_manifest verify \\
        --manifest docs/mapsam/v006/analysis/frozen_manifest.json --lake-root "$LAKE"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, get_args

from archeo_topia.analysis.mapsam_end_to_end import decoder_checkpoint_path
from archeo_topia.analysis.runtime import write_record
from archeo_topia.datasets.build_detection_windows import (
    FOLDS,
    REPO_ROOT,
    SPLITS_FILE,
    build_output_problems,
    load_split_document,
    sha256_file,
)
from archeo_topia.formats.georeference import SheetReference
from archeo_topia.training.train_mound_detector import (
    fold_weights_path,
    require_selection,
)

logger = logging.getLogger(__name__)

#: Where an entry's path is relative to.
Root = Literal["repo", "lake"]

#: What an entry pins.
Role = Literal[
    "ground_truth",
    "clips_json",
    "clip",
    "detector",
    "decoder",
    "sam_base",
    "windows_build_report",
    "refit_detector",
    "refit_selection",
]

#: Roles that appear exactly once in a manifest.
SINGLE_ROLES: tuple[str, ...] = ("ground_truth", "sam_base", "windows_build_report")

#: Roles ``extend`` appends after the base entries, in this order.
REFIT_ROLES: tuple[str, ...] = ("refit_detector", "refit_selection")

#: First cell of the ``SOURCE.md`` table row that carries the ground-truth hash.
SOURCE_HASH_ROW = "`instances_Test.json` sha256"

_HEX_DIGITS = frozenset("0123456789abcdef")
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class ManifestSources:
    """Where the inputs to pin live.

    Attributes:
        repo_root: Root that ``repo`` entries are relative to.
        lake_root: Root that ``lake`` entries are relative to.
        clips_dir: Lake-relative directory holding ``<sheet>/clips.json`` and
            the clips.
        ground_truth: Repo-relative ground-truth COCO JSON.
        source_md: Repo-relative ``SOURCE.md`` recording the ground-truth hash.
        detector_run: Repo-relative detector run directory (``<fold>/weights``).
        mapsam_root: Repo-relative directory of MapSAM decoder runs.
        mapsam_prefix: Decoder run-name prefix.
        sam_checkpoint: Repo-relative base SAM checkpoint.
        dataset: Repo-relative frozen windows dataset.
        splits_file: Splits file declaring which sheets are frozen.
    """

    repo_root: Path
    lake_root: Path
    clips_dir: Path
    ground_truth: Path
    source_md: Path
    detector_run: Path
    mapsam_root: Path
    mapsam_prefix: str
    sam_checkpoint: Path
    dataset: Path
    splits_file: Path


def parse_source_hash(source_md: Path) -> str:
    """Read the ground-truth sha256 out of ``SOURCE.md``.

    The hash sits in one Markdown table row whose first cell is exactly
    :data:`SOURCE_HASH_ROW` and whose second cell is the digest in backticks.

    Args:
        source_md: The provenance file.

    Returns:
        The lowercase 64-character digest.

    Raises:
        ValueError: If zero or several rows match, or the cell is not a digest.
    """
    cells_by_row = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in source_md.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("|")
    ]
    rows = [cells for cells in cells_by_row if cells and cells[0] == SOURCE_HASH_ROW]
    if len(rows) != 1:
        raise ValueError(f"{source_md} has {len(rows)} rows for {SOURCE_HASH_ROW}; expected 1")
    if len(rows[0]) != 2:
        raise ValueError(f"{source_md}: the {SOURCE_HASH_ROW} row has {len(rows[0])} cells")
    value = rows[0][1].strip("`")
    if len(value) != _SHA256_LENGTH or not set(value) <= _HEX_DIGITS:
        raise ValueError(f"{source_md}: {value!r} is not a sha256 digest")
    return value


def _entry(role: str, root: Root, base: Path, relative: Path, **keys: str) -> dict[str, str]:
    """Hash one file and describe it.

    Args:
        role: What the file is.
        root: Which root ``relative`` is relative to.
        base: Absolute directory of that root.
        relative: Path below the root.
        **keys: Extra identifying keys such as ``sheet``, ``fold`` or ``file``.

    Returns:
        The manifest entry.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return {
        "role": role,
        **keys,
        "root": root,
        "path": relative.as_posix(),
        "sha256": sha256_file(base / relative),
    }


def manifest_text(manifest: dict[str, Any]) -> str:
    """Return the canonical serialization of a manifest.

    It is the one definition of the bytes a manifest file holds, so the hash
    recorded under ``base`` can be re-derived from the parsed entries.
    """
    return json.dumps(manifest, indent=2) + "\n"


def _require_relative(path: Path, name: str) -> None:
    """Require a path argument that is relative to its root."""
    if path.is_absolute():
        raise ValueError(f"{name} must be relative to its root, got {path}")


def _frozen_sheets(splits_file: Path) -> list[str]:
    """Return the sheets flagged ``frozen_blind_test`` in the splits file."""
    document = load_split_document(splits_file)
    sheets = sorted(s for s, meta in document["sheets"].items() if meta["frozen_blind_test"])
    if not sheets:
        raise ValueError(f"no sheet is frozen_blind_test in {splits_file}")
    return sheets


def _recorded_path_is(recorded: str, expected: Path, repo_root: Path) -> bool:
    """Whether a path recorded in a build report names ``expected``."""
    return (repo_root / recorded).resolve() == (repo_root / expected).resolve()


def _check_build_report(
    sources: ManifestSources,
    sheets: list[str],
    clip_entries: list[dict[str, str]],
    clips_json_entries: list[dict[str, str]],
) -> None:
    """Stop unless the windows dataset is the frozen test build of these sheets.

    The build report records the hash of every source image it cropped and of
    every ``clips.json`` it validated against; both must equal the pinned lake
    files, so the crops are tied to the pixels the manifest pins.
    """
    report_path = sources.repo_root / sources.dataset / "metadata" / "build_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    label_sources = report["inputs"]["label_sources"]
    if len(label_sources) != 1 or not _recorded_path_is(
        label_sources[0]["coco_json"], sources.ground_truth, sources.repo_root
    ):
        problems.append(
            f"built from {[s['coco_json'] for s in label_sources]!r}, not the ground truth "
            f"{sources.ground_truth} alone"
        )
    if report["listing"]["plan"] != "test":
        problems.append(f"listing plan is {report['listing']['plan']!r}, not 'test'")
    elif sorted(report["listing"]["listings"]["test"]["sheets"]) != sheets:
        problems.append(
            f"test listing covers {report['listing']['listings']['test']['sheets']}, "
            f"the frozen sheets are {sheets}"
        )
    pinned_images = {e["file"]: e["sha256"] for e in clip_entries}
    built_images = report["inputs"]["source_images_sha256"]
    differing = sorted(
        name
        for name in pinned_images.keys() | built_images.keys()
        if pinned_images.get(name) != built_images.get(name)
    )
    if differing:
        problems.append(f"source image hashes differ from the pinned clips: {differing}")
    pinned_jsons = {e["sheet"]: e["sha256"] for e in clips_json_entries}
    built_jsons = report["inputs"]["clips_json_sha256"]
    differing = sorted(
        name
        for name in pinned_jsons.keys() | built_jsons.keys()
        if pinned_jsons.get(name) != built_jsons.get(name)
    )
    if differing:
        problems.append(f"clips.json hashes differ from the pinned files: {differing}")
    if problems:
        raise ValueError(
            f"{report_path} is not the frozen windows build:\n  " + "\n  ".join(problems)
        )


def write_manifest(sources: ManifestSources, output: Path) -> dict[str, Any]:
    """Hash every frozen input and write the manifest.

    Nothing is written unless the ground truth matches the hash ``SOURCE.md``
    records, the windows dataset is the frozen test build and still matches its
    own report, and ``output`` does not exist.

    Args:
        sources: Where the inputs live.
        output: Manifest to write; must not exist.

    Returns:
        The manifest.

    Raises:
        FileExistsError: If ``output`` exists.
        ValueError: On any mismatch, naming it.
    """
    if output.exists():
        raise FileExistsError(f"{output} already exists; a manifest is written once")
    for name in (
        "clips_dir",
        "ground_truth",
        "source_md",
        "detector_run",
        "mapsam_root",
        "sam_checkpoint",
        "dataset",
    ):
        _require_relative(getattr(sources, name), name)
    repo, lake = sources.repo_root, sources.lake_root

    recorded = parse_source_hash(repo / sources.source_md)
    ground_truth = _entry("ground_truth", "repo", repo, sources.ground_truth)
    if ground_truth["sha256"] != recorded:
        raise ValueError(
            f"{sources.ground_truth} hashes to {ground_truth['sha256']}, "
            f"{sources.source_md} records {recorded}"
        )

    sheets = _frozen_sheets(sources.splits_file)
    clips_json_entries = []
    clip_entries = []
    for sheet in sheets:
        sheet_dir = sources.clips_dir / sheet
        clips_json = sheet_dir / "clips.json"
        clips_json_entries.append(_entry("clips_json", "lake", lake, clips_json, sheet=sheet))
        reference = SheetReference.from_clips_json(lake / clips_json)
        if reference.sheet_id != sheet or not reference.clips:
            raise ValueError(
                f"{lake / clips_json} names sheet {reference.sheet_id!r} with "
                f"{len(reference.clips)} clips, expected sheet {sheet!r} with clips"
            )
        clip_entries += [
            _entry("clip", "lake", lake, sheet_dir / clip.file, sheet=sheet, file=clip.file)
            for clip in reference.clips
        ]
    _check_build_report(sources, sheets, clip_entries, clips_json_entries)
    problems = build_output_problems(repo / sources.dataset)
    if problems:
        raise ValueError(
            "the windows dataset differs from its build report:\n  " + "\n  ".join(problems)
        )

    entries = [ground_truth, *clips_json_entries]
    entries += clip_entries
    entries += [
        _entry(
            "detector",
            "repo",
            repo,
            fold_weights_path(sources.detector_run, fold),
            fold=fold,
        )
        for fold in FOLDS
    ]
    entries += [
        _entry(
            "decoder",
            "repo",
            repo,
            decoder_checkpoint_path(sources.mapsam_root, sources.mapsam_prefix, fold),
            fold=fold,
        )
        for fold in FOLDS
    ]
    entries.append(_entry("sam_base", "repo", repo, sources.sam_checkpoint))
    entries.append(
        _entry(
            "windows_build_report",
            "repo",
            repo,
            sources.dataset / "metadata" / "build_report.json",
        )
    )

    manifest = {"entries": entries}
    write_record(output, manifest_text(manifest))
    logger.info("wrote %d entries to %s", len(entries), output)
    return manifest


def _base_problems(manifest: dict[str, Any]) -> list[str]:
    """Check the refit extension of a manifest, or that there is none."""
    entries = manifest["entries"]
    roles = [e.get("role") for e in entries]
    if "base" not in manifest:
        return [
            f"entry {i}: role {r!r} needs a base" for i, r in enumerate(roles) if r in REFIT_ROLES
        ]
    base = manifest["base"]
    count = base.get("entries") if isinstance(base, dict) else None
    if (
        not isinstance(base, dict)
        or set(base) != {"entries", "sha256"}
        or type(count) is not int
        or not 0 < count < len(entries)
    ):
        return [f"base {base!r} is not {{entries: N with 0 < N < {len(entries)}, sha256}}"]
    problems: list[str] = []
    digest = base["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != _SHA256_LENGTH
        or not set(digest) <= _HEX_DIGITS
    ):
        problems.append(f"base sha256 {digest!r} is not a digest")
    if any(role in REFIT_ROLES for role in roles[:count]):
        problems.append("the base entries hold a refit role")
    if tuple(roles[count:]) != REFIT_ROLES:
        problems.append(
            f"entries after the base are {roles[count:]}, expected {list(REFIT_ROLES)}"
        )
    if not problems and _base_sha256(entries[:count]) != digest:
        problems.append("the entries before the extension do not hash to the recorded base sha256")
    return problems


def _base_sha256(entries: list[dict[str, Any]]) -> str:
    """Return the hash of the canonical text of a manifest holding only these entries."""
    return hashlib.sha256(manifest_text({"entries": entries}).encode("utf-8")).hexdigest()


def _shape_problems(manifest: dict[str, Any]) -> list[str]:
    """Check that a manifest is structurally what ``write_manifest`` and ``extend`` produce."""
    if "entries" not in manifest or set(manifest) not in ({"entries"}, {"entries", "base"}):
        return [f"top-level keys {sorted(manifest)} are not 'entries' with an optional 'base'"]
    problems: list[str] = []
    entries = manifest["entries"]
    required = {"role", "root", "path", "sha256"}
    for index, entry in enumerate(entries):
        where = f"entry {index}"
        missing = sorted(required - entry.keys())
        if missing:
            problems.append(f"{where}: missing keys {missing}")
            continue
        if entry["role"] not in get_args(Role):
            problems.append(f"{where}: unknown role {entry['role']!r}")
        if entry["root"] not in get_args(Root):
            problems.append(f"{where}: unknown root {entry['root']!r}")
        path = PurePosixPath(entry["path"])
        if not entry["path"] or path.is_absolute() or ".." in path.parts:
            problems.append(f"{where}: path {entry['path']!r} is not a relative path")
        digest = entry["sha256"]
        if len(digest) != _SHA256_LENGTH or not set(digest) <= _HEX_DIGITS:
            problems.append(f"{where}: sha256 {digest!r} is not a digest")
    counts = {
        role: sum(1 for e in entries if required <= e.keys() and e["role"] == role)
        for role in SINGLE_ROLES
    }
    problems += [
        f"role {role!r} appears {n} times, expected 1" for role, n in counts.items() if n != 1
    ]
    return problems + _base_problems(manifest)


def verify_manifest(manifest_path: Path, lake_root: Path, repo_root: Path) -> dict[str, Any]:
    """Recompute every pinned hash and the windows dataset's own output hashes.

    Every mismatch is collected and raised together, so one run shows the whole
    list.

    Args:
        manifest_path: The manifest.
        lake_root: Root that ``lake`` entries are relative to.
        repo_root: Root that ``repo`` entries are relative to.

    Returns:
        The parsed manifest.

    Raises:
        ValueError: If the manifest is malformed or anything differs or is
            missing.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = _shape_problems(manifest)
    if shape:
        raise ValueError(f"{manifest_path} is malformed:\n  " + "\n  ".join(shape))

    roots = {"repo": repo_root, "lake": lake_root}
    problems: list[str] = []
    for entry in manifest["entries"]:
        file = roots[entry["root"]] / entry["path"]
        try:
            actual = sha256_file(file)
        except FileNotFoundError:
            problems.append(f"{entry['role']}: {file} is missing")
            continue
        if actual != entry["sha256"]:
            problems.append(
                f"{entry['role']}: {file} hashes to {actual}, manifest has {entry['sha256']}"
            )
    report_entry = next(e for e in manifest["entries"] if e["role"] == "windows_build_report")
    report_path = roots[report_entry["root"]] / report_entry["path"]
    if report_path.is_file():
        problems += build_output_problems(report_path.parent.parent)
    if problems:
        raise ValueError(f"{manifest_path} does not match its inputs:\n  " + "\n  ".join(problems))
    return manifest


def extend_manifest(
    manifest_path: Path, lake_root: Path, repo_root: Path, selection: Path
) -> dict[str, Any]:
    """Append the refit detector and its selection record to a verified manifest.

    The extension is staged in ``<manifest>.extending``, verified in full, and
    moved over the manifest with one ``os.replace``, which is the only place a
    manifest file is ever replaced. Nothing is deleted on failure: a leftover
    ``.extending`` file stops the next attempt, so the operator looks at it.

    Args:
        manifest_path: A verified, canonical, not yet extended manifest.
        lake_root: Root that ``lake`` entries are relative to.
        repo_root: Root that ``repo`` entries are relative to.
        selection: Repo-relative ``selection.json`` of the refit.

    Returns:
        The extended manifest.

    Raises:
        ValueError: If the manifest does not verify, is already extended, is
            not in canonical form, or the selection is not valid (see
            ``require_selection``).
        FileExistsError: If a staging file is left over.
    """
    manifest = verify_manifest(manifest_path, lake_root, repo_root)
    if "base" in manifest:
        raise ValueError(f"{manifest_path} is already extended; a manifest is extended once")
    text = manifest_path.read_text(encoding="utf-8")
    if text != manifest_text(manifest):
        raise ValueError(
            f"{manifest_path} is not in canonical form, so its base hash is unprovable"
        )
    base_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    _require_relative(selection, "selection")
    chosen = json.loads((repo_root / selection).read_text(encoding="utf-8"))
    require_selection(chosen, repo_root)
    chosen_path = Path(chosen["chosen"]["path"])
    _require_relative(chosen_path, "chosen path")
    refit_detector = _entry(
        "refit_detector", "repo", repo_root, chosen_path, checkpoint=chosen["chosen"]["file"]
    )
    refit_selection = _entry("refit_selection", "repo", repo_root, selection)
    extended = {
        "entries": [*manifest["entries"], refit_detector, refit_selection],
        "base": {"entries": len(manifest["entries"]), "sha256": base_sha256},
    }
    staged = manifest_path.with_name(manifest_path.name + ".extending")
    write_record(staged, manifest_text(extended))
    try:
        verify_manifest(staged, lake_root, repo_root)
        if sha256_file(manifest_path) != base_sha256:
            raise ValueError(f"{manifest_path} changed while it was being extended")
        os.replace(staged, manifest_path)
    except BaseException:
        logger.error("extension failed; the staged file is kept at %s", staged)
        raise
    logger.info("extended %s to %d entries", manifest_path, len(extended["entries"]))
    return extended


def main() -> None:
    """Parse arguments and write or verify a manifest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    write = commands.add_parser("write", help="hash the inputs and write a new manifest")
    write.add_argument("--lake-root", type=Path, required=True)
    write.add_argument("--clips-dir", type=Path, required=True)
    write.add_argument("--ground-truth", type=Path, required=True)
    write.add_argument("--source-md", type=Path, required=True)
    write.add_argument("--detector-run", type=Path, required=True)
    write.add_argument("--mapsam-root", type=Path, required=True)
    write.add_argument("--mapsam-prefix", required=True)
    write.add_argument("--sam-checkpoint", type=Path, required=True)
    write.add_argument("--dataset", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)

    extend = commands.add_parser("extend", help="append the refit detector and its selection")
    extend.add_argument("--manifest", type=Path, required=True)
    extend.add_argument("--lake-root", type=Path, required=True)
    extend.add_argument("--selection", type=Path, required=True)

    verify = commands.add_parser("verify", help="recompute every hash in a manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--lake-root", type=Path, required=True)

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    if args.command == "write":
        write_manifest(
            ManifestSources(
                repo_root=REPO_ROOT,
                lake_root=args.lake_root,
                clips_dir=args.clips_dir,
                ground_truth=args.ground_truth,
                source_md=args.source_md,
                detector_run=args.detector_run,
                mapsam_root=args.mapsam_root,
                mapsam_prefix=args.mapsam_prefix,
                sam_checkpoint=args.sam_checkpoint,
                dataset=args.dataset,
                splits_file=SPLITS_FILE,
            ),
            args.output,
        )
    elif args.command == "extend":
        extended = extend_manifest(args.manifest, args.lake_root, REPO_ROOT, args.selection)
        print(f"base sha256 {extended['base']['sha256']}")
        print(f"new sha256  {sha256_file(args.manifest)}")
        print(f"{len(extended['entries'])} entries")
    else:
        manifest = verify_manifest(args.manifest, args.lake_root, REPO_ROOT)
        print(f"verified {len(manifest['entries'])} entries")


if __name__ == "__main__":
    main()
