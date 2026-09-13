"""Verify SAM installation and checkpoint loading.

CLI entrypoint for checking that the Segment-Anything model can be imported,
a checkpoint file exists, and the model loads on the available device.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = ("vit_b", "vit_l", "vit_h")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Verify SAM installation and checkpoint loading.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=SUPPORTED_MODEL_TYPES,
        help="SAM model type (default: vit_b).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("models/checkpoints/sam/sam_vit_b_01ec64.pth")),
        help="Path to SAM checkpoint file.",
    )
    return parser.parse_args(argv)


def check_sam_setup(model_type: str, checkpoint_path: str) -> bool:
    """Load SAM model and log diagnostics.

    Args:
        model_type: SAM model variant identifier (vit_b, vit_l, vit_h).
        checkpoint_path: Filesystem path to the checkpoint ``.pth`` file.

    Returns:
        True if all checks pass, False otherwise.
    """

    checkpoint = Path(checkpoint_path)

    # -- Verify checkpoint exists --
    if not checkpoint.is_file():
        log.error("Checkpoint not found: %s", checkpoint)
        return False

    log.info("Checkpoint: %s", checkpoint)

    # -- Import SAM --
    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        log.error("Failed to import segment_anything: %s", exc)
        return False

    log.info("segment_anything imported successfully")

    # -- Detect device --
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError as exc:
        log.error("Failed to import torch: %s", exc)
        return False

    device = "cuda" if cuda_available else "cpu"
    log.info("CUDA available: %s", cuda_available)
    log.info("Device: %s", device)

    # -- Load model --
    try:
        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=device)
        model.eval()
    except Exception as exc:
        log.error("Failed to load SAM model (%s): %s", model_type, exc)
        return False

    log.info("Model type: %s", model_type)
    log.info("SAM model loaded successfully on %s", device)
    return True


def main(argv: list[str] | None = None) -> None:
    """Run SAM setup verification."""

    args = parse_args(argv)

    success = check_sam_setup(args.model_type, args.checkpoint)

    if success:
        log.info("SAM setup check PASSED")
    else:
        log.error("SAM setup check FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
