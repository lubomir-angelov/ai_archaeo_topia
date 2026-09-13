#!/usr/bin/env python3
"""Tests for legacy checkpoint migration.

The migration drops frozen weights that are recoverable from the stock SAM
checkpoint.  Since it removes data from files that can be expensive to
reproduce, the verification path matters more than the happy path.
"""

from __future__ import annotations

from pathlib import Path

import torch

from archeo_topia.training.migrate_checkpoints import (
    migrate_file,
    split_state_dict,
    trained_module_prefixes,
    verify_reconstruction,
)

CONFIG = {
    "model": {
        "freeze_image_encoder": True,
        "train_prompt_encoder": False,
        "train_mask_decoder": True,
    }
}


def _stock() -> dict[str, torch.Tensor]:
    return {
        "image_encoder.w": torch.ones(4),
        "prompt_encoder.w": torch.ones(2),
        "mask_decoder.w": torch.zeros(3),
    }


def _trained() -> dict[str, torch.Tensor]:
    s = _stock()
    s["mask_decoder.w"] = torch.tensor([1.0, 2.0, 3.0])
    return s


def _write_legacy(path: Path, state: dict[str, torch.Tensor], config: dict = CONFIG) -> None:
    torch.save(
        {
            "epoch": 7,
            "model_type": "vit_b",
            "sam_checkpoint": "base.pth",
            "model_state_dict": state,
            "optimizer_state_dict": {"state": {0: {"step": 5.0}}, "param_groups": []},
            "train_loss": 0.1,
            "val_loss": 0.2,
            "config": config,
        },
        str(path),
    )


class TestTrainedModulePrefixes:
    def test_default_policy_trains_only_the_decoder(self) -> None:
        assert trained_module_prefixes(CONFIG) == {"mask_decoder"}

    def test_unfrozen_encoder_is_retained(self) -> None:
        cfg = {"model": {"freeze_image_encoder": False}}
        assert "image_encoder" in trained_module_prefixes(cfg)

    def test_trained_prompt_encoder_is_retained(self) -> None:
        cfg = {"model": {"train_prompt_encoder": True}}
        assert "prompt_encoder" in trained_module_prefixes(cfg)

    def test_missing_config_falls_back_to_defaults(self) -> None:
        assert trained_module_prefixes({}) == {"mask_decoder"}


class TestSplit:
    def test_split_keeps_only_trained_modules(self) -> None:
        keep, drop = split_state_dict(_trained(), {"mask_decoder"})
        assert set(keep) == {"mask_decoder.w"}
        assert set(drop) == {"image_encoder.w", "prompt_encoder.w"}


class TestVerification:
    def test_exact_reconstruction_reports_no_mismatch(self) -> None:
        original = _trained()
        keep, _ = split_state_dict(original, {"mask_decoder"})
        assert verify_reconstruction(original, keep, _stock()) == []

    def test_corrupted_tensor_is_detected(self) -> None:
        original = _trained()
        keep, _ = split_state_dict(original, {"mask_decoder"})
        keep["mask_decoder.w"] = torch.tensor([9.0, 9.0, 9.0])
        assert verify_reconstruction(original, keep, _stock()) == ["mask_decoder.w"]

    def test_drifted_frozen_weights_are_detected(self) -> None:
        """If stock does not match what was frozen, the rewrite is unsafe."""
        original = _trained()
        keep, _ = split_state_dict(original, {"mask_decoder"})
        drifted = _stock()
        drifted["image_encoder.w"] = torch.zeros(4)
        assert verify_reconstruction(original, keep, drifted) == ["image_encoder.w"]


class TestMigrateFile:
    def test_dry_run_leaves_the_file_untouched(self, tmp_path: Path) -> None:
        p = tmp_path / "epoch_1.pt"
        _write_legacy(p, _trained())
        before = p.read_bytes()

        r = migrate_file(p, _stock(), keep_optimizer=False, apply=False)
        assert r["status"] == "would-migrate"
        assert r["after"] < r["before"]
        assert p.read_bytes() == before

    def test_apply_rewrites_and_preserves_trained_weights(self, tmp_path: Path) -> None:
        p = tmp_path / "epoch_1.pt"
        original = _trained()
        _write_legacy(p, original)

        r = migrate_file(p, _stock(), keep_optimizer=False, apply=True)
        assert r["status"] == "migrated"

        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert ck["state_dict_scope"] == "trainable"
        assert set(ck["model_state_dict"]) == {"mask_decoder.w"}
        assert torch.equal(ck["model_state_dict"]["mask_decoder.w"], original["mask_decoder.w"])
        assert ck["epoch"] == 7

    def test_optimizer_state_dropped_unless_requested(self, tmp_path: Path) -> None:
        p = tmp_path / "epoch_1.pt"
        _write_legacy(p, _trained())
        migrate_file(p, _stock(), keep_optimizer=False, apply=True)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert "optimizer_state_dict" not in ck

    def test_optimizer_state_kept_when_requested(self, tmp_path: Path) -> None:
        p = tmp_path / "last.pt"
        _write_legacy(p, _trained())
        migrate_file(p, _stock(), keep_optimizer=True, apply=True)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert ck["optimizer_state_dict"]["state"][0]["step"] == 5.0

    def test_already_migrated_file_is_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "epoch_1.pt"
        _write_legacy(p, _trained())
        migrate_file(p, _stock(), keep_optimizer=False, apply=True)
        again = migrate_file(p, _stock(), keep_optimizer=False, apply=True)
        assert again["status"] == "skipped-already-migrated"

    def test_refuses_when_frozen_tensors_are_absent_from_stock(self, tmp_path: Path) -> None:
        """Without matching stock weights the frozen half is unrecoverable."""
        p = tmp_path / "epoch_1.pt"
        _write_legacy(p, _trained())
        before = p.read_bytes()

        r = migrate_file(p, {"mask_decoder.w": torch.zeros(3)}, keep_optimizer=False, apply=True)
        assert r["status"].startswith("refused-")
        assert p.read_bytes() == before

    def test_no_temp_files_survive(self, tmp_path: Path) -> None:
        p = tmp_path / "epoch_1.pt"
        _write_legacy(p, _trained())
        migrate_file(p, _stock(), keep_optimizer=False, apply=True)
        assert [f.name for f in tmp_path.iterdir()] == ["epoch_1.pt"]
