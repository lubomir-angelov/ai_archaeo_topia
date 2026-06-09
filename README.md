# ai_archaeo_topia
A repository to track the work on the AI ArchaeoTopia project.  https://archaeotopia.naim.bg/

# GDAL
If you see errors such as:
no module named _gdal_array

You need to re-install GDAL in your python virtual env:
(From: https://gis.stackexchange.com/questions/153199/import-error-no-module-named-gdal-array)

```bash
pip uninstall gdal

# ensure numpy is installed prior to installing gdal
pip install numpy

# ensure setuptools and wheel are installed to do the build in your current environment
pip install setuptools wheel

# install gdal (note the version might be different on your machine!)
pip install --no-build-isolation --no-cache-dir --force-reinstall gdal==3.4.1
```

# SAM2 modes
```bash
Mode A: local/internal mock
- fastest tests
- no HTTP services
- useful for unit tests only

Mode B: architecture mock
- real sam2_mcp process
- real sam2_backend process
- backend in SAM2_BACKEND_MODE=mock
- required for milestone acceptance
```

## Architecture acceptance path
```bash
# Terminal 1
SAM2_BACKEND_MODE=mock make sam2-backend-run

# Terminal 2
export SAM2_MCP_BACKEND_URL="http://127.0.0.1:8181"
make sam2-mcp-run

# Terminal 3
export PDF_PATH="path/to/input.pdf"
export OUTPUT_ROOT="data/annotations/runs"
export RUN_ID="manual_pdf_seed_001"
export SAM2_MCP_URL="http://127.0.0.1:8181"
make annotation-pdf-sam2-mock-dry-run
```

## MapSAM dataset preparation (Step 1)

Convert a CVAT COCO export into a MapSAM-ready binary segmentation dataset with
sheet-level splits (no cross-sheet leakage).

```bash
python -m src.archeo_topia.datasets.prepare_mapsam_coco \
  --coco-json data/curated/datasets/cvat/v0.0.1/annotations/instances_default.json \
  --images-dir data/curated/datasets/cvat/v0.0.1/images/default \
  --output-dir data/curated/datasets/mapsam_v0 \
  --positive-label mound \
  --ignore-label uncertain_ignore \
  --hard-negative-label hard_negative_symbol \
  --split-by sheet
```

Produces:

```
data/curated/datasets/mapsam_v0/
  images/       {train, val, test}/
  masks/        {train, val, test}/       # binary positive (mound = 255)
  ignore_masks/ {train, val, test}/       # binary ignore (uncertain = 255)
  metadata/
    splits.json
    dataset_summary.json
```

Example output:

```
INFO Loading COCO JSON from data/curated/datasets/cvat/v0.0.1/annotations/instances_default.json
INFO Category IDs — positive=1, ignore=3, hard_negative=2
INFO Found 12 images in COCO index
INFO Split assignment: {'train': 2, 'val': 0, 'test': 1}
INFO Dataset written to data/curated/datasets/mapsam_v0
INFO Summary: {
  "total_images": 12,
  "total_sheets": 3,
  "images_per_split": {"train": 8, "val": 0, "test": 4},
  "sheets_per_split": {"train": 2, "val": 0, "test": 1},
  "annotation_counts_by_category": {
    "mound": 180,
    "uncertain_ignore": 3,
    "hard_negative_symbol": 530
  },
  "mound_count_per_split": {"train": 146, "val": 0, "test": 34},
  "hard_negative_count_per_split": {"train": 434, "val": 0, "test": 96},
  "uncertain_ignore_count_per_split": {"train": 3, "val": 0, "test": 0},
  "images_with_no_mound": 1
}
```

Defaults: `--positive-label mound`, `--ignore-label uncertain_ignore`,
`--hard-negative-label hard_negative_symbol`, `--split-by sheet`, 70/15/15
train/val/test split.