"""Sanity-check script for SAM v1 installation and checkpoint loading."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

_SUPPORTED_MODEL_TYPES: Final[list[str]] = ["vit_b", "vit_l", "vit_h"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``model_type`` and ``checkpoint``.
    """
    parser = argparse.ArgumentParser(
        description="Verify SAM v1 installation and checkpoint loading.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=_SUPPORTED_MODEL_TYPES,
        help="SAM model variant: vit_b, vit_l, or vit_h.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the SAM checkpoint .pth file.",
    )
    return parser.parse_args(argv)


def check_sam_setup(model_type: str, checkpoint_path: str) -> None:
    """Load a SAM checkpoint and log diagnostic information.

    Args:
        model_type: SAM model variant identifier.
        checkpoint_path: Path to the checkpoint file.

    Raises:
        SystemExit: If the checkpoint cannot be loaded or the model type
            is unsupported.
    """
    checkpoint = Path(checkpoint_path)

    if not checkpoint.is_file():
        logger.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)

    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        logger.error(
            "segment_anything is not installed. Install it with:\n"
            "  python -m pip install "
            '"git+https://github.com/facebookresearch/segment-anything.git"',
        )
        raise SystemExit(1) from exc

    if model_type not in _SUPPORTED_MODEL_TYPES:
        logger.error(
            "Unsupported model type '%s'. Choose from: %s",
            model_type,
            ", ".join(_SUPPORTED_MODEL_TYPES),
        )
        sys.exit(1)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SAM model: %s", model_type)
    logger.info("Checkpoint path: %s", checkpoint)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Selected device: %s", device)

    try:
        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    except Exception as exc:
        logger.error("Failed to load checkpoint: %s", exc)
        sys.exit(1)

    model.to(device=device)
    model.eval()

    logger.info("Model loaded successfully.")
    logger.info("Model: %s", model)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI sanity check."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    args = parse_args(argv)
    check_sam_setup(args.model_type, args.checkpoint)


if __name__ == "__main__":
    main()
