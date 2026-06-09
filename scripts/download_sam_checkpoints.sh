#!/usr/bin/env bash
set -euxo pipefail

# download_sam_checkpoints.sh
# Downloads Meta Segment-Anything (SAM v1) checkpoints.
# Default: ViT-B only. Use flags for ViT-L, ViT-H, or --all.

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything"

CHECKPOINTS=(
  "vit_b:sam_vit_b_01ec64.pth"
  "vit_l:sam_vit_l_0b3195.pth"
  "vit_h:sam_vit_h_4b8939.pth"
)

OUTPUT_DIR="models/checkpoints/sam"
FORCE=false
DOWNLOAD_VIT_B=false
DOWNLOAD_VIT_L=false
DOWNLOAD_VIT_H=false

usage() {
  printf "Usage: %s [--vit-b] [--vit-l] [--vit-h] [--all] [--force]\n" "${0}" >&2
  printf "  --vit-b   Download ViT-B checkpoint (default)\n" >&2
  printf "  --vit-l   Download ViT-L checkpoint\n" >&2
  printf "  --vit-h   Download ViT-H checkpoint\n" >&2
  printf "  --all     Download all checkpoints\n" >&2
  printf "  --force   Re-download even if file exists\n" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --vit-b)
      DOWNLOAD_VIT_B=true
      shift
      ;;
    --vit-l)
      DOWNLOAD_VIT_L=true
      shift
      ;;
    --vit-h)
      DOWNLOAD_VIT_H=true
      shift
      ;;
    --all)
      DOWNLOAD_VIT_B=true
      DOWNLOAD_VIT_L=true
      DOWNLOAD_VIT_H=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      printf "Unknown option: %s\n" "${1}" >&2
      usage
      ;;
  esac
done

# Default to ViT-B if no model selected
if [[ "${DOWNLOAD_VIT_B}" == false ]] && \
   [[ "${DOWNLOAD_VIT_L}" == false ]] && \
   [[ "${DOWNLOAD_VIT_H}" == false ]]; then
  DOWNLOAD_VIT_B=true
fi

mkdir -p "${OUTPUT_DIR}"

download_checkpoint() {
  local model_type="${1}"
  local filename="${2}"
  local target_path="${OUTPUT_DIR}/${filename}"
  local url="${BASE_URL}/${filename}"

  if [[ "${FORCE}" == false ]] && [[ -s "${target_path}" ]]; then
    printf "Skipping %s (already exists and is non-empty)\n" "${filename}" >&2
    return 0
  fi

  printf "Downloading %s (%s) ...\n" "${model_type}" "${filename}" >&2
  curl -fSL -o "${target_path}" "${url}"
  printf "Downloaded %s to %s\n" "${filename}" "${target_path}" >&2
}

if [[ "${DOWNLOAD_VIT_B}" == true ]]; then
  download_checkpoint "vit_b" "sam_vit_b_01ec64.pth"
fi

if [[ "${DOWNLOAD_VIT_L}" == true ]]; then
  download_checkpoint "vit_l" "sam_vit_l_0b3195.pth"
fi

if [[ "${DOWNLOAD_VIT_H}" == true ]]; then
  download_checkpoint "vit_h" "sam_vit_h_4b8939.pth"
fi

printf "Done.\n" >&2
