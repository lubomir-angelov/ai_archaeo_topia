#!/usr/bin/env python3
"""Rewrite legacy full-model checkpoints as trainable-only checkpoints.

Checkpoints written before the partial-checkpoint change stored the whole SAM
state dict.  With the image encoder frozen that is ~358 MB per file, of which
~342 MB is a byte-identical copy of the stock SAM weights the model was built
from.  This tool rewrites them to hold only the parameters that were trained.

Every file is verified before the original is replaced.  Verification
reconstructs the full state dict from the stock weights plus the rewritten
file, re-read from disk, and requires every tensor to match the original
bit for bit.  A file that fails verification is left untouched.

By default one ``last.pt`` per run keeps its optimizer state, so a migrated
run can still be resumed.  Nothing is deleted until its replacement has been
verified, and the swap itself is atomic.

Usage::

    # report what would change, touching nothing
    python -m archeo_topia.training.migrate_checkpoints \\
        --root artifacts/models/mapsam

    # perform the migration
    python -m archeo_topia.training.migrate_checkpoints \\
        --root artifacts/models/mapsam --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

def trained_module_prefixes(config: dict[str, Any]) -> set[str]:
    """Work out which top-level modules were trained, from a stored config.

    Mirrors the freezing policy in :func:`build_sam_model` so a migrated file
    holds exactly what ``save_checkpoint`` would write today.

    Args:
        config: The ``config`` dict stored inside the checkpoint.

    Returns:
        Names of the top-level modules whose parameters were trainable.
    """
    model_cfg = (config or {}).get("model", {})
    trained: set[str] = set()

    if not model_cfg.get("freeze_image_encoder", True):
        trained.add("image_encoder")
    if model_cfg.get("train_prompt_encoder", False):
        trained.add("prompt_encoder")
    if model_cfg.get("train_mask_decoder", True):
        trained.add("mask_decoder")
    return trained


def split_state_dict(
    state: dict[str, torch.Tensor],
    trained: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a full state dict into trained and recoverable halves.

    Args:
        state: The full model state dict from a legacy checkpoint.
        trained: Top-level module names that were trained.

    Returns:
        Tuple of ``(trained_tensors, frozen_tensors)``.
    """
    keep = {k: v for k, v in state.items() if k.split(".")[0] in trained}
    drop = {k: v for k, v in state.items() if k.split(".")[0] not in trained}
    return keep, drop


def verify_reconstruction(
    original: dict[str, torch.Tensor],
    rewritten: dict[str, torch.Tensor],
    stock: dict[str, torch.Tensor],
) -> list[str]:
    """Check that stock weights plus the rewritten file rebuild the original.

    Args:
        original: Full state dict from the legacy checkpoint.
        rewritten: Trained-only state dict read back from the new file.
        stock: State dict of the base SAM checkpoint.

    Returns:
        Names of tensors that do not match.  Empty means the rewrite is safe.
    """
    reconstructed = dict(stock)
    reconstructed.update(rewritten)

    mismatched = [k for k in original if k not in reconstructed]
    mismatched += [
        k
        for k in original
        if k in reconstructed and not torch.equal(original[k], reconstructed[k])
    ]
    return mismatched


def migrate_file(
    path: Path,
    stock: dict[str, torch.Tensor],
    keep_optimizer: bool,
    apply: bool,
) -> dict[str, Any]:
    """Rewrite one checkpoint, verifying before the original is replaced.

    Args:
        path: Checkpoint to migrate.
        stock: State dict of the base SAM checkpoint.
        keep_optimizer: Retain optimizer state in the rewritten file.
        apply: Perform the rewrite. When ``False`` nothing is written.

    Returns:
        Report dict with ``path``, ``status``, ``before``, ``after``.
    """
    before = path.stat().st_size
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    if ckpt.get("state_dict_scope") == "trainable":
        return {
            "path": path,
            "status": "skipped-already-migrated",
            "before": before,
            "after": before,
        }

    original = ckpt["model_state_dict"]
    trained = trained_module_prefixes(ckpt.get("config", {}))
    if not trained:
        return {
            "path": path,
            "status": "skipped-no-trained-modules",
            "before": before,
            "after": before,
        }

    keep, dropped = split_state_dict(original, trained)
    missing_from_stock = [k for k in dropped if k not in stock]
    if missing_from_stock:
        return {
            "path": path,
            "status": f"refused-{len(missing_from_stock)}-frozen-tensors-absent-from-stock",
            "before": before,
            "after": before,
        }

    payload = {
        **{k: v for k, v in ckpt.items() if k not in {"model_state_dict", "optimizer_state_dict"}},
        "state_dict_scope": "trainable",
        "model_state_dict": keep,
        "has_optimizer_state": keep_optimizer,
    }
    if keep_optimizer and "optimizer_state_dict" in ckpt:
        payload["optimizer_state_dict"] = ckpt["optimizer_state_dict"]

    if not apply:
        tmp = path.with_suffix(".sizecheck")
        torch.save(payload, str(tmp))
        after = tmp.stat().st_size
        tmp.unlink()
        return {"path": path, "status": "would-migrate", "before": before, "after": after}

    # Write beside the original, verify from disk, then swap atomically.  The
    # original is only released once its replacement is proven equivalent.
    tmp = path.with_suffix(".migrating")
    torch.save(payload, str(tmp))

    reread = torch.load(str(tmp), map_location="cpu", weights_only=False)
    mismatched = verify_reconstruction(original, reread["model_state_dict"], stock)
    if mismatched:
        tmp.unlink()
        return {
            "path": path,
            "status": f"FAILED-verification-{len(mismatched)}-tensors",
            "before": before,
            "after": before,
        }

    after = tmp.stat().st_size
    os.replace(str(tmp), str(path))
    return {"path": path, "status": "migrated", "before": before, "after": after}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read from ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="Directory to scan for *.pt checkpoints")
    p.add_argument(
        "--sam-checkpoint",
        default="models/checkpoints/sam/sam_vit_b_01ec64.pth",
        help="Base SAM weights the frozen tensors are recovered from",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration. Without this the run is a dry report.",
    )
    p.add_argument(
        "--no-keep-last",
        action="store_true",
        help="Do not preserve a last.pt with optimizer state per run directory",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for checkpoint migration."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(args.root)
    if not root.exists():
        logger.error("Root not found: %s", root)
        sys.exit(1)

    base = Path(args.sam_checkpoint)
    if not base.exists():
        logger.error("Base SAM checkpoint not found: %s", base)
        sys.exit(1)

    logger.info("Loading base SAM weights: %s", base)
    stock = torch.load(str(base), map_location="cpu", weights_only=False)

    files = sorted(root.glob("**/*.pt"))
    if not files:
        logger.error("No checkpoints found under %s", root)
        sys.exit(1)

    # One resume point per run keeps its optimizer state.
    keep_optimizer_for: set[Path] = set()
    if not args.no_keep_last:
        by_dir: dict[Path, list[Path]] = {}
        for f in files:
            by_dir.setdefault(f.parent, []).append(f)
        for fs in by_dir.values():
            named_last = [f for f in fs if f.name == "last.pt"]
            final = [f for f in fs if f.name == "final.pt"]
            keep_optimizer_for.add((named_last or final or sorted(fs))[-1])

    logger.info(
        "%s %d checkpoints under %s (%.1f GB)",
        "Migrating" if args.apply else "Inspecting",
        len(files),
        root,
        sum(f.stat().st_size for f in files) / 1073741824,
    )
    if not args.apply:
        logger.info("DRY RUN — nothing will be written. Pass --apply to migrate.\n")

    reports = []
    for f in files:
        r = migrate_file(f, stock, f in keep_optimizer_for, args.apply)
        reports.append(r)
        logger.info(
            "  %-58s %6.1f -> %6.1f MB  %s",
            str(f.relative_to(root))[:58],
            r["before"] / 1048576,
            r["after"] / 1048576,
            r["status"],
        )

    before = sum(r["before"] for r in reports)
    after = sum(r["after"] for r in reports)
    failed = [r for r in reports if r["status"].startswith(("FAILED", "refused"))]

    logger.info("\n%s", "=" * 78)
    logger.info("before : %8.2f GB", before / 1073741824)
    logger.info("after  : %8.2f GB", after / 1073741824)
    logger.info(
        "saved  : %8.2f GB (%.1fx smaller)",
        (before - after) / 1073741824,
        before / after if after else 0,
    )
    if failed:
        logger.error("%d file(s) not migrated and left untouched:", len(failed))
        for r in failed:
            logger.error("  %s — %s", r["path"], r["status"])
        sys.exit(1)


if __name__ == "__main__":
    main()
