#!/usr/bin/env bash
set -euxo pipefail

# install_sam.sh
# Installs Meta Segment-Anything (SAM v1) from GitHub via pip.

pip_cmd="${SAM_PIP:-pip}"

printf "Installing segment-anything from GitHub...\n" >&2
python -m pip install "git+https://github.com/facebookresearch/segment-anything.git"

printf "Verifying SAM installation...\n" >&2
python -c "from segment_anything import sam_model_registry; print('  sam_model_registry imported successfully')"

printf "SAM installed successfully.\n" >&2
