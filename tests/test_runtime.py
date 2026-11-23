"""Tests for device resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from archeo_topia.analysis.runtime import (
    environment_record,
    new_directory,
    resolve_device,
    write_record,
)


def test_cpu_resolves_to_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_without_cuda_raises_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda:0")


def test_a_cuda_index_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        resolve_device("cuda:3")


def test_bare_cuda_is_returned_with_the_index_it_will_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert resolve_device("cuda") == torch.device("cuda", 0)


def test_an_invalid_device_string_raises() -> None:
    with pytest.raises(RuntimeError):
        resolve_device("not-a-device")


def test_environment_record_names_versions_and_the_device() -> None:
    record = environment_record(torch.device("cpu"))

    assert record.keys() == {
        "python",
        "torch",
        "ultralytics",
        "segment_anything",
        "device",
        "device_name",
    }
    assert record["device"] == "cpu"
    assert record["device_name"] is None
    assert record["torch"].split("+")[0] == torch.__version__.split("+")[0]


def test_environment_record_names_the_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")

    record = environment_record(torch.device("cuda", 0))

    assert record["device"] == "cuda:0"
    assert record["device_name"] == "Test GPU"


def test_new_directory_creates_parents_and_refuses_an_existing_directory(tmp_path: Path) -> None:
    created = new_directory(tmp_path / "a" / "b")
    assert created == tmp_path / "a" / "b"
    assert created.is_dir()

    with pytest.raises(FileExistsError, match="written once"):
        new_directory(created)


def test_write_record_creates_a_file_and_refuses_an_existing_one(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    write_record(path, "first")
    assert path.read_text(encoding="utf-8") == "first"

    with pytest.raises(FileExistsError, match="written once"):
        write_record(path, "second")
    assert path.read_text(encoding="utf-8") == "first"
