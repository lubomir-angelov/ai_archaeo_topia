"""Fitness test: protocol writers create records once, through ``analysis.runtime``.

``exist_ok=True`` is how a protocol record gets silently overwritten or merged
into a stale directory. In the protocol modules it is allowed only where the
directory is not a record, listed below by exact ``(path, function)``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "archeo_topia"

PROTOCOL_MODULES = (
    "datasets/build_detection_windows.py",
    "analysis/mapsam_end_to_end.py",
    "training/train_mound_detector.py",
    "analysis/reproduce_v0_5_scoring.py",
    "analysis/frozen_test_opening.py",
    "analysis/frozen_manifest.py",
    "analysis/paired_comparison.py",
    "formats/cut_review_export.py",
)

#: Each entry is a reviewed ``exist_ok=True`` that does not touch a record.
_ALLOWLIST = {
    # The Ultralytics project directory, which the library manages itself.
    ("training/train_mound_detector.py", "train_fold"),
    # The label directory is the training run's directory; its records are
    # protected by write_record.
    ("training/train_mound_detector.py", "evaluate_checkpoint"),
    # The parent of the staging and output directories, not a record.
    ("analysis/frozen_test_opening.py", "run_opening"),
}


def _exist_ok_sites(rel_path: str) -> set[tuple[str, str]]:
    tree = ast.parse((SRC / rel_path).read_text(encoding="utf-8"))
    sites: set[tuple[str, str]] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if (
                        keyword.arg == "exist_ok"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        sites.add((rel_path, function.name))
    return sites


def test_exist_ok_appears_only_at_reviewed_sites() -> None:
    found = {site for rel in PROTOCOL_MODULES for site in _exist_ok_sites(rel)}

    assert found <= _ALLOWLIST, f"unreviewed exist_ok=True: {sorted(found - _ALLOWLIST)}"


def test_every_allowlisted_site_still_exists() -> None:
    found = {site for rel in PROTOCOL_MODULES for site in _exist_ok_sites(rel)}

    assert found >= _ALLOWLIST, f"stale allowlist entries: {sorted(_ALLOWLIST - found)}"
