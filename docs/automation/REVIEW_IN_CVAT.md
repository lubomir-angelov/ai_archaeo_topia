# Reviewing SAM2 Annotations in CVAT

## Overview

After running the SAM2 proposal pipeline on extracted PDF clips, you can import the resulting masks into CVAT for manual review and refinement. This guide covers the full workflow from SAM2 output to CVAT-ready annotations.

## Prerequisites

- CVAT running and accessible (see `docs/automation/SAM2_BACKEND_SERVICE.md`)
- SAM2 backend running with torch installed
- Extracted images + metadata from `data/sam2_test/`

## Pipeline Summary

```
PDF → Extract images → Generate proposals → SAM2 segmentation → Export to CVAT → Review
```

## Step 1: Run SAM2 Proposals

```bash
# Live backend (requires torch + SAM2 checkpoint)
make sam2-proposals-run \
    INPUT_DIR=data/sam2_test \
    OUTPUT_DIR=data/sam2_test/automated_mcp \
    BACKEND_URL=http://127.0.0.1:8181 \
    MAX_PROPOSALS=10

# Mock backend (no GPU needed)
make sam2-proposals-mock \
    INPUT_DIR=data/sam2_test \
    OUTPUT_DIR=data/sam2_test/automated_mcp_mock \
    MAX_PROPOSALS=10
```

Output structure:
```
automated_mcp/
├── masks/                  # Binary mask PNGs (one per object)
├── review/                 # Overlay images with bounding boxes
├── sam2_annotations.csv    # Full annotation CSV pinned to metadata
└── manifest.json           # Run provenance and stats
```

## Step 2: Inspect Results

Check the manifest for run summary:
```bash
cat data/sam2_test/automated_mcp/manifest.json
```

Key metrics:
- `clips_with_masks`: samples that got at least one mask
- `total_masks`: total segmentation objects generated
- `backend_health`: confirms live/mock mode and model status

Visual review (first 20 samples):
```bash
ls data/sam2_test/automated_mcp/review/
```

## Step 3: Export to CVAT Format

The export script filters by confidence threshold and generates CVAT-compatible XML:

```bash
python3 src/export_sam2_to_cvat.py \
    --csv data/sam2_test/automated_mcp/sam2_annotations.csv \
    --images data/sam2_test/raw/images \
    --output data/sam2_test/cvat_import \
    --min-confidence 0.5
```

Output:
```
cvat_import/
├── annotations.xml   # CVAT for images 1.1 format
└── images/           # Flat image directory (only annotated samples)
```

### Confidence Thresholds

| Threshold | Annotations | Samples | Use case |
|-----------|-------------|---------|----------|
| 0.3 | ~290 | ~150 | Include everything for review |
| 0.5 | ~206 | ~121 | Default — reasonable recall |
| 0.7 | ~116 | ~80 | Higher precision, fewer false positives |
| 0.8 | ~65 | ~50 | Only high-confidence masks |

### Manual Selection

Edit `sam2_annotations.csv` directly to keep/remove specific masks, then re-run the export. The export script reads whatever rows remain in the CSV.

## Step 4: Import into CVAT

### Option A: Upload Annotations (Recommended)

1. In CVAT UI, create a new task
2. Upload all images from `cvat_import/images/`
3. After task creation: **Task → Upload annotations**
4. Select `annotations.xml`
5. Format: **CVAT for images 1.1**
6. Click Upload — polygons will appear as pre-annotations

### Option B: Create Task with Annotations

1. In CVAT UI, create a new task
2. Upload images from `cvat_import/images/`
3. Set labels: `mound`, `hard_negative_symbol`, `uncertain_ignore`
4. After task creation: **Task → Upload annotations**
5. Select `annotations.xml`, format: **CVAT for images 1.1**

## Step 5: Review and Refine

In CVAT:
- **Verify**: Click through each polygon, confirm it matches the archaeological feature
- **Edit**: Adjust polygon vertices if the mask is imprecise
- **Delete**: Remove false positives (roads, legends, decorative marks)
- **Add**: Draw new polygons for missed features
- **Relabel**: Change label if SAM2 misclassified the object

### Common False Positives

- Road segments and grid lines
- Map legends and scale bars
- Decorative cartographic symbols
- Scan noise and artifacts

### Common False Negatives

- Small mounds below proposal box size
- Overlapping features in dense necropolises
- Low-contrast symbols on busy backgrounds

## Metadata Preservation

The `sam2_annotations.csv` preserves all PDF-derived metadata:

| Field | Description |
|-------|-------------|
| `sample_id` | Protocol-compliant ID: `ANNOTATOR_NNN_SHEET25K_SHEET5K.png` |
| `annotator` | Original annotator prefix (AG, VG, IK, UNK) |
| `sheet_25k` | 1:25,000 map sheet identifier |
| `sheet_5k` | 1:5,000 map sheet identifier |
| `province` | Bulgarian province |
| `contains_necropolis` | Multiple mounds expected |
| `contains_single_mound` | Single mound expected |
| `contains_hard_negative` | False positive expected |
| `relief_normalized` | Terrain classification |

This metadata travels with each annotation row, allowing you to trace back from any CVAT annotation to the original PDF page and annotator notes.

## Next Steps

After CVAT review:
1. Export verified annotations from CVAT
2. Use as training data for fine-tuned detection models
3. Compare SAM2 results against manual annotations for evaluation
4. Iterate proposal parameters based on review findings

## Troubleshooting

**CVAT rejects XML import**: Verify format matches "CVAT for images 1.1". Check that all image filenames in the XML exist in the task.

**Missing polygons**: Check confidence threshold — lower `--min-confidence` to include more masks.

**Label mismatch**: Ensure the three labels (`mound`, `hard_negative_symbol`, `uncertain_ignore`) exist in your CVAT task before importing.

**Images not loading**: CVAT expects images in the task directory. The export script copies them flat into `cvat_import/images/` — upload this directory when creating the task.
