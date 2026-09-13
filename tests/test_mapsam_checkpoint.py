#!/usr/bin/env python3
"""Tests for partial (trainable-only) checkpoint save and load.

Checkpoints store only the parameters that were actually trained.  With the
SAM image encoder frozen, a full state dict is ~358 MB of which ~342 MB is a
byte-identical copy of the stock weights the model was built from.  These
tests use a small stand-in module so the behaviour can be checked without
loading SAM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from archeo_topia.training.train_mapsam_v0 import (
    file_digest,
    frozen_weights_digest,
    load_checkpoint,
    save_checkpoint,
)


class TinyModel(nn.Module):
    """Stand-in with a frozen 'encoder' and a trainable 'decoder'.

    Initialised from a fixed seed so every instance carries identical frozen
    weights, standing in for models built from the same stock SAM checkpoint.
    Tests that need a mismatched base overwrite those weights explicitly.
    """

    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.image_encoder = nn.Linear(8, 8)
            self.mask_decoder = nn.Linear(8, 4)

    def freeze_encoder(self) -> None:
        """Mark the encoder as frozen, as the training script does."""
        for p in self.image_encoder.parameters():
            p.requires_grad = False


def _save(path: Path, model: nn.Module) -> None:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    save_checkpoint(path, model, opt, 3, "tiny", "base.pth", 0.5, 0.4, {"k": "v"})


class TestPartialSave:
    def test_only_trainable_tensors_are_saved(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        p = tmp_path / "ck.pt"
        _save(p, m)

        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        keys = set(ck["model_state_dict"])
        assert keys == {"mask_decoder.weight", "mask_decoder.bias"}
        assert not any(k.startswith("image_encoder") for k in keys)

    def test_scope_marker_is_written(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        p = tmp_path / "ck.pt"
        _save(p, m)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert ck["state_dict_scope"] == "trainable"
        assert ck["sam_checkpoint"] == "base.pth"
        assert ck["epoch"] == 3

    def test_nothing_frozen_saves_everything(self, tmp_path: Path) -> None:
        """With no module frozen the checkpoint is complete, as before."""
        m = TinyModel()
        p = tmp_path / "ck.pt"
        _save(p, m)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert set(ck["model_state_dict"]) == set(m.state_dict())

    def test_partial_checkpoint_is_smaller(self, tmp_path: Path) -> None:
        frozen, full = TinyModel(), TinyModel()
        frozen.freeze_encoder()
        pf, pu = tmp_path / "partial.pt", tmp_path / "full.pt"
        _save(pf, frozen)
        _save(pu, full)
        assert pf.stat().st_size < pu.stat().st_size


class TestPartialLoad:
    def test_round_trip_restores_trained_weights(self, tmp_path: Path) -> None:
        src = TinyModel()
        src.freeze_encoder()
        with torch.no_grad():
            src.mask_decoder.weight.fill_(0.1234)
        p = tmp_path / "ck.pt"
        _save(p, src)

        dst = TinyModel()
        dst.freeze_encoder()
        load_checkpoint(p, dst)
        assert torch.equal(dst.mask_decoder.weight, src.mask_decoder.weight)

    def test_frozen_weights_come_from_the_live_model(self, tmp_path: Path) -> None:
        """The partial load must not disturb weights it does not carry."""
        src = TinyModel()
        src.freeze_encoder()
        p = tmp_path / "ck.pt"
        _save(p, src)

        dst = TinyModel()
        dst.freeze_encoder()
        before = dst.image_encoder.weight.clone()
        load_checkpoint(p, dst)
        assert torch.equal(dst.image_encoder.weight, before)

    def test_legacy_full_checkpoint_still_loads(self, tmp_path: Path) -> None:
        """Checkpoints written before the change have no scope marker."""
        src = TinyModel()
        with torch.no_grad():
            src.mask_decoder.bias.fill_(2.5)
            src.image_encoder.bias.fill_(3.5)
        p = tmp_path / "legacy.pt"
        torch.save(
            {
                "epoch": 9,
                "model_type": "tiny",
                "sam_checkpoint": "base.pth",
                "model_state_dict": src.state_dict(),
                "optimizer_state_dict": {},
                "train_loss": 0.1,
                "val_loss": 0.2,
                "config": {},
            },
            str(p),
        )

        dst = TinyModel()
        ck = load_checkpoint(p, dst)
        assert ck["epoch"] == 9
        assert torch.equal(dst.mask_decoder.bias, src.mask_decoder.bias)
        assert torch.equal(dst.image_encoder.bias, src.image_encoder.bias)

    def test_unexpected_tensors_raise(self, tmp_path: Path) -> None:
        """A checkpoint for a different architecture must fail loudly."""
        p = tmp_path / "wrong.pt"
        torch.save(
            {
                "epoch": 1,
                "model_type": "other",
                "sam_checkpoint": "base.pth",
                "state_dict_scope": "trainable",
                "model_state_dict": {"not_a_real_module.weight": torch.zeros(2)},
                "optimizer_state_dict": {},
                "train_loss": 0.0,
                "val_loss": 0.0,
                "config": {},
            },
            str(p),
        )
        with pytest.raises(ValueError, match="absent from this model"):
            load_checkpoint(p, TinyModel())

    def test_optimizer_state_round_trips_when_included(self, tmp_path: Path) -> None:
        """The rolling last.pt carries optimizer state so a run can resume."""
        m = TinyModel()
        m.freeze_encoder()
        trainable = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=1e-3)
        sum(p.sum() for p in trainable).backward()
        opt.step()

        p = tmp_path / "last.pt"
        save_checkpoint(p, m, opt, 1, "tiny", "base.pth", 0.1, 0.1, {}, include_optimizer=True)

        dst = TinyModel()
        dst.freeze_encoder()
        dst_opt = torch.optim.AdamW([q for q in dst.parameters() if q.requires_grad], lr=1e-3)
        load_checkpoint(p, dst, dst_opt)

        restored = dst_opt.state_dict()["state"]
        assert len(restored) == len(opt.state_dict()["state"])
        assert all(v["step"] == 1 for v in restored.values())


class TestOptimizerStateSplit:
    """Archival checkpoints are lean; only the rolling last.pt can resume."""

    def test_archival_checkpoint_omits_optimizer(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        p = tmp_path / "epoch_1.pt"
        _save(p, m)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert "optimizer_state_dict" not in ck
        assert ck["has_optimizer_state"] is False

    def test_last_checkpoint_includes_optimizer(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=1e-3)
        p = tmp_path / "last.pt"
        save_checkpoint(p, m, opt, 1, "tiny", "base.pth", 0.1, 0.1, {}, include_optimizer=True)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert "optimizer_state_dict" in ck
        assert ck["has_optimizer_state"] is True

    def test_archival_is_smaller_than_last(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        trainable = [q for q in m.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=1e-3)
        sum(q.sum() for q in trainable).backward()
        opt.step()

        lean, last = tmp_path / "epoch_1.pt", tmp_path / "last.pt"
        save_checkpoint(lean, m, opt, 1, "tiny", "base.pth", 0.1, 0.1, {})
        save_checkpoint(last, m, opt, 1, "tiny", "base.pth", 0.1, 0.1, {}, include_optimizer=True)
        assert lean.stat().st_size < last.stat().st_size

    def test_resuming_from_archival_fails_with_guidance(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        p = tmp_path / "epoch_1.pt"
        _save(p, m)

        dst = TinyModel()
        dst.freeze_encoder()
        dst_opt = torch.optim.AdamW([q for q in dst.parameters() if q.requires_grad], lr=1e-3)
        with pytest.raises(ValueError, match="last.pt"):
            load_checkpoint(p, dst, dst_opt)

    def test_archival_still_loads_for_inference(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        with torch.no_grad():
            m.mask_decoder.weight.fill_(0.75)
        p = tmp_path / "epoch_1.pt"
        _save(p, m)

        dst = TinyModel()
        dst.freeze_encoder()
        load_checkpoint(p, dst, None)
        assert torch.equal(dst.mask_decoder.weight, m.mask_decoder.weight)


class TestFrozenWeightsPinning:
    """A partial checkpoint is only correct against the base it was trained on.

    A substituted base is silently wrong rather than broken - the shapes still
    fit, so the model loads and predicts nonsense - so the mismatch has to be
    caught at load time.
    """

    def test_digest_is_deterministic(self) -> None:
        m = TinyModel()
        sd = m.state_dict()
        names = {"mask_decoder.weight", "mask_decoder.bias"}
        assert frozen_weights_digest(sd, names) == frozen_weights_digest(sd, names)

    def test_digest_ignores_trained_tensors(self) -> None:
        """Training must not change the digest, or every load would fail."""
        m = TinyModel()
        names = {"mask_decoder.weight", "mask_decoder.bias"}
        before = frozen_weights_digest(m.state_dict(), names)
        with torch.no_grad():
            m.mask_decoder.weight.fill_(3.3)
        assert frozen_weights_digest(m.state_dict(), names) == before

    def test_digest_tracks_frozen_tensors(self) -> None:
        m = TinyModel()
        names = {"mask_decoder.weight", "mask_decoder.bias"}
        before = frozen_weights_digest(m.state_dict(), names)
        with torch.no_grad():
            m.image_encoder.weight.fill_(3.3)
        assert frozen_weights_digest(m.state_dict(), names) != before

    def test_checkpoint_records_the_digest(self, tmp_path: Path) -> None:
        m = TinyModel()
        m.freeze_encoder()
        p = tmp_path / "ck.pt"
        _save(p, m)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        assert len(ck["frozen_weights_sha256"]) == 64

    def test_matching_base_loads(self, tmp_path: Path) -> None:
        src = TinyModel()
        src.freeze_encoder()
        p = tmp_path / "ck.pt"
        _save(p, src)

        dst = TinyModel()
        dst.freeze_encoder()
        load_checkpoint(p, dst)
        assert torch.equal(dst.mask_decoder.weight, src.mask_decoder.weight)

    def test_mismatched_base_is_refused(self, tmp_path: Path) -> None:
        src = TinyModel()
        src.freeze_encoder()
        p = tmp_path / "ck.pt"
        _save(p, src)

        wrong = TinyModel()
        wrong.freeze_encoder()
        with torch.no_grad():
            wrong.image_encoder.weight.fill_(99.0)
        with pytest.raises(ValueError, match="different set of frozen weights"):
            load_checkpoint(p, wrong)

    def test_checkpoint_without_digest_still_loads(self, tmp_path: Path) -> None:
        """Files migrated before pinning existed must keep working."""
        m = TinyModel()
        m.freeze_encoder()
        p = tmp_path / "ck.pt"
        _save(p, m)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        del ck["frozen_weights_sha256"]
        torch.save(ck, str(p))

        dst = TinyModel()
        dst.freeze_encoder()
        with torch.no_grad():
            dst.image_encoder.weight.fill_(42.0)  # would fail a digest check
        load_checkpoint(p, dst)  # unverifiable, but must not raise
        assert torch.equal(dst.mask_decoder.weight, m.mask_decoder.weight)


class TestFileDigest:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        import hashlib

        p = tmp_path / "blob.bin"
        data = b"archaeological mound symbols" * 5000
        p.write_bytes(data)
        assert file_digest(p) == hashlib.sha256(data).hexdigest()
