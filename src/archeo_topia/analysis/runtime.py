"""Device resolution and write-once protocol records.

A metric produced on CPU because CUDA was unavailable is a different
experiment from the one that was asked for, so a requested CUDA device that
cannot be used is an error, not a quiet downgrade.

Protocol records are created, never replaced: :func:`new_directory` and
:func:`write_record` are the only place the overwrite policy is defined.
"""

from __future__ import annotations

import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve a device request to a concrete device or fail.

    Args:
        requested: A torch device string such as ``cuda``, ``cuda:1`` or ``cpu``.

    Returns:
        The device. A bare ``cuda`` is returned with its current index, so the
        recorded device names the GPU that was actually used.

    Raises:
        RuntimeError: If a CUDA device is requested and CUDA is unavailable, or
            the device index does not exist.
    """
    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"device {requested!r} requested but only {torch.cuda.device_count()} CUDA device(s) exist"
        )
    return device


def environment_record(device: torch.device) -> dict[str, Any]:
    """Describe the software and device a run executed on.

    The record sits beside a published number so a later difference in library
    versions can be told apart from a difference in code.

    Args:
        device: The device the run used, as returned by :func:`resolve_device`.

    Returns:
        Python, torch, ultralytics and segment_anything versions, the device
        string, and the CUDA device name (None on a non-CUDA device).

    Raises:
        importlib.metadata.PackageNotFoundError: If a recorded package is not
            installed, since a run that needs it could not have happened.
    """
    return {
        "python": platform.python_version(),
        "torch": version("torch"),
        "ultralytics": version("ultralytics"),
        "segment_anything": version("segment_anything"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def new_directory(path: Path) -> Path:
    """Create a directory that must not already exist.

    Args:
        path: Directory to create, with its parents.

    Returns:
        ``path``.

    Raises:
        FileExistsError: If ``path`` already exists.
    """
    try:
        path.mkdir(parents=True)
    except FileExistsError as error:
        raise FileExistsError(
            f"{path} already exists; protocol outputs are written once"
        ) from error
    return path


def write_record(path: Path, text: str) -> None:
    """Write a file that must not already exist.

    Exclusive create is the guarantee, not a prior existence check.

    Args:
        path: File to create.
        text: UTF-8 content.

    Raises:
        FileExistsError: If ``path`` already exists.
    """
    try:
        handle = open(path, "x", encoding="utf-8")  # noqa: SIM115
    except FileExistsError as error:
        raise FileExistsError(
            f"{path} already exists; protocol outputs are written once"
        ) from error
    with handle:
        handle.write(text)
