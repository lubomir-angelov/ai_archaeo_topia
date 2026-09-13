#!/usr/bin/env python3
"""Prompt-centred input windows for MapSAM.

The v0.2 pipeline feeds SAM a whole ~2400 px map tile resized to 1024, so a
~21 px mound symbol survives as roughly 5 positive pixels at the 256x256
decoder resolution and covers about 0.56 of one 16 px ViT patch.  Cropping a
window around the prompt before the resize is the only lever that raises that
coverage while the image encoder stays frozen.

This is a different mechanism from ``loss.use_bbox_loss_crop``, the "cropon" /
"cropoff" switch in the v0.2 config names.  That one restricts where the
*loss* is computed, at logit resolution, and leaves the encoder input alone.
This one changes what the encoder sees.  Hence "window" rather than "crop".

The window is centred on the prompt point and clamped inside the tile, never
padded: a synthetic border is input the frozen encoder has never seen, whereas
a clamped window is still real map.  Clamping means the prompt is *not* at the
centre near an edge, so callers must use the returned transform rather than
assuming the centre.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


class Window(NamedTuple):
    """A square crop window in original-image pixel coordinates.

    Attributes:
        x0: Left edge, inclusive.
        y0: Top edge, inclusive.
        size: Side length in original-image pixels.
    """

    x0: int
    y0: int
    size: int

    @property
    def x1(self) -> int:
        """Right edge, exclusive."""
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        """Bottom edge, exclusive."""
        return self.y0 + self.size

    def scale_to(self, image_size: int) -> float:
        """Pixels of *image_size* output per pixel of original image.

        Args:
            image_size: Side length the window is resized to.

        Returns:
            The linear scale factor.
        """
        return image_size / self.size


def compute_window(
    center_xy: tuple[float, float],
    image_hw: tuple[int, int],
    window_px: int,
) -> Window:
    """Place a square window on the prompt, clamped inside the image.

    The window is shrunk to fit when *window_px* exceeds either image
    dimension, so a window larger than the tile degrades to the whole tile
    rather than requiring padding.

    Args:
        center_xy: Prompt point ``(x, y)`` in original-image pixels.
        image_hw: Image ``(height, width)`` in pixels.
        window_px: Requested window side length.

    Returns:
        The placed window, guaranteed to lie inside the image.

    Raises:
        ValueError: If *window_px* is not positive, or the image is empty.
    """
    if window_px < 1:
        raise ValueError(f"window_px must be positive, got {window_px}")

    height, width = image_hw
    if height < 1 or width < 1:
        raise ValueError(f"Image must be non-empty, got {image_hw}")

    size = min(window_px, height, width)
    if size < window_px:
        logger.debug(
            "Window %d px exceeds image %dx%d; shrunk to %d px", window_px, width, height, size
        )

    cx, cy = center_xy
    x0 = int(round(cx - size / 2.0))
    y0 = int(round(cy - size / 2.0))

    # Clamp rather than pad: the prompt moves off-centre near an edge, but
    # every pixel the encoder sees is real map.
    x0 = max(0, min(x0, width - size))
    y0 = max(0, min(y0, height - size))
    return Window(x0, y0, size)


def crop_to_window(tensor_chw: torch.Tensor, window: Window) -> torch.Tensor:
    """Crop a ``(C, H, W)`` tensor to *window*.

    Args:
        tensor_chw: Image or mask tensor.
        window: Window to crop to.

    Returns:
        A ``(C, size, size)`` view of the input.

    Raises:
        ValueError: If *tensor_chw* is not 3-dimensional, or the window does
            not fit inside it.
    """
    if tensor_chw.dim() != 3:
        raise ValueError(f"Expected a (C, H, W) tensor, got shape {tuple(tensor_chw.shape)}")

    height, width = tensor_chw.shape[1], tensor_chw.shape[2]
    if window.x1 > width or window.y1 > height or window.x0 < 0 or window.y0 < 0:
        raise ValueError(f"Window {window} does not fit inside a {height}x{width} tensor")

    return tensor_chw[:, window.y0 : window.y1, window.x0 : window.x1]


def point_into_window(
    point_xy: tuple[float, float] | list[float],
    window: Window,
    image_size: int,
) -> torch.Tensor:
    """Map a point from original-image to resized-window coordinates.

    Args:
        point_xy: Point ``(x, y)`` in original-image pixels.
        window: The window the image was cropped to.
        image_size: Side length the window was resized to.

    Returns:
        Float32 tensor ``(2,)`` holding the mapped ``(x, y)``.
    """
    scale = window.scale_to(image_size)
    return torch.tensor(
        [(float(point_xy[0]) - window.x0) * scale, (float(point_xy[1]) - window.y0) * scale],
        dtype=torch.float32,
    )


def bbox_into_window(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    window: Window,
    image_size: int,
) -> torch.Tensor:
    """Map a bounding box from original-image to resized-window coordinates.

    The box is clamped to the output bounds: a mound near a tile edge can have
    a padded bbox that extends past the window, and SAM's prompt encoder
    expects coordinates inside the input.

    Args:
        bbox_xyxy: Box ``(x1, y1, x2, y2)`` in original-image pixels.
        window: The window the image was cropped to.
        image_size: Side length the window was resized to.

    Returns:
        Float32 tensor ``(4,)`` holding the mapped box.
    """
    scale = window.scale_to(image_size)
    mapped = [
        (float(bbox_xyxy[0]) - window.x0) * scale,
        (float(bbox_xyxy[1]) - window.y0) * scale,
        (float(bbox_xyxy[2]) - window.x0) * scale,
        (float(bbox_xyxy[3]) - window.y0) * scale,
    ]
    clamped = [min(max(v, 0.0), float(image_size)) for v in mapped]
    return torch.tensor(clamped, dtype=torch.float32)


def source_pixels_per_mask_pixel(window: Window, mask_size: int) -> float:
    """Area of one predicted-mask pixel, in original-image pixels.

    This is what makes the resolution arms comparable.  An area or boundary
    error measured on the decoder's 256x256 grid means a different physical
    distance in every arm, so IoU rises under cropping even when the model has
    not improved.  Multiplying by this factor puts every arm on the tile's own
    grid.

    Args:
        window: The window the input was cropped to.
        mask_size: Side length of the predicted mask (256 for SAM's decoder).

    Returns:
        Source-image pixels covered by one mask pixel.

    Raises:
        ValueError: If *mask_size* is not positive.
    """
    if mask_size < 1:
        raise ValueError(f"mask_size must be positive, got {mask_size}")
    return (window.size / mask_size) ** 2


def apply_prompt_jitter(
    center_xy: tuple[float, float],
    bbox_xyxy: tuple[float, float, float, float],
    image_hw: tuple[int, int],
    jitter_xy: tuple[float, float] | None,
    jitter_prompts: bool = True,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float, float, float]]:
    """Displace a ground-truth-derived prompt by a fixed offset.

    Every MapSAM number up to v0.3 was measured with the prompt derived from
    the annotation, so it answers "given the correct mound location, can the
    mound be segmented".  A detector will not localize that exactly, and with
    a prompt-centred window the localization error is not only a worse prompt
    but a *displaced input*: 100 px is 4% of a 2400 px tile and 20% of a
    512 px window.  This offsets the prompt so that gap can be measured.

    Coordinates are clamped to the image, because a detector proposal outside
    the tile is not a case worth modelling and SAM's prompt encoder expects
    in-bounds coordinates.  Clamping keeps the box at least one pixel wide.

    Args:
        center_xy: Ground-truth prompt point ``(x, y)``.
        bbox_xyxy: Ground-truth prompt box ``(x1, y1, x2, y2)``.
        image_hw: Image ``(height, width)`` in pixels.
        jitter_xy: Offset ``(dx, dy)`` in original-image pixels, or ``None``
            for no jitter.
        jitter_prompts: When ``True`` the point and box move with the window.
            When ``False`` only the window centre moves, which isolates the
            cost of a displaced input from the cost of a wrong prompt.

    Returns:
        Tuple of ``(window_center, prompt_center, prompt_bbox)``.
    """
    if jitter_xy is None:
        return center_xy, center_xy, bbox_xyxy

    height, width = image_hw
    dx, dy = jitter_xy

    def _clamp(x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, 0.0), float(width - 1)),
            min(max(y, 0.0), float(height - 1)),
        )

    window_center = _clamp(center_xy[0] + dx, center_xy[1] + dy)
    if not jitter_prompts:
        return window_center, center_xy, bbox_xyxy

    x1, y1 = _clamp(bbox_xyxy[0] + dx, bbox_xyxy[1] + dy)
    x2, y2 = _clamp(bbox_xyxy[2] + dx, bbox_xyxy[3] + dy)
    prompt_bbox = (x1, y1, max(x2, x1 + 1.0), max(y2, y1 + 1.0))
    return window_center, window_center, prompt_bbox
