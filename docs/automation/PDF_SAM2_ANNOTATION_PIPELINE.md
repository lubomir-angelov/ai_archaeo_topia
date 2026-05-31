# PDF → SAM2 Seed Annotation Pipeline

Seed annotation ingestion pipeline for manually prepared PDFs containing curated archaeological image clips.

**Not the production georeferencing pipeline.** This pass proves data orchestration, metadata preservation, protocol compliance, candidate annotation generation, CSV writing, artifact layout, and validation independently of real SAM2 model quality.

## Purpose

Turn a manually prepared PDF (containing archaeological map image clips + metadata text) into a protocol-compliant annotation run:

```
PDF
→ pdf_clip_extractor logic
→ clip images + metadata
→ sam2_mcp (generate_proposals)
→ sam2_backend (mock mode)
→ annotation CSV
→ validate_annotation_run
```

## Relationship to existing components

| Component | Role |
|-----------|------|
| `src/pdf_clip_extractor.py` | Original PDF clip extraction logic (reused patterns) |
| `src/services/sam2_mcp/client.py` | SAM2 backend HTTP client with mock fallback |
| `src/services/sam2_backend/` | SAM2 inference backend (mock mode for this pass) |
| `src/validate_annotation_run.py` | CSV validation against protocol |
| `docs/annotation/PROTOCOL_EN.md` | Annotation protocol (folder layout, CSV schema, labels) |

## Package layout

```
src/services/annotation_pipeline/
├── __init__.py
├── artifacts.py        # Run directory creation, clip/review image saving
├── csv_writer.py       # CSV writing with protocol columns
├── errors.py           # Pipeline exceptions
├── metadata.py         # Sidecar metadata loading, missing metadata reporting
├── pdf_sam2_runner.py  # Main pipeline runner + CLI
├── schemas.py          # Pydantic models (ClipMetadata, AnnotationRow, RunReport)
├── settings.py         # Pipeline configuration
└── validation.py       # Run validation (bboxes, labels, paths, duplicates)
```

## Required environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_PATH` | *(required)* | Path to input PDF |
| `OUTPUT_ROOT` | `./data/annotations/runs` | Base output directory |
| `RUN_ID` | `run_<timestamp>` | Unique run identifier |
| `SAM2_MCP_URL` | `mock` | SAM2 MCP backend URL or `mock` |

## Prerequisites

1. SAM2 backend running in mock mode:
   ```bash
   SAM2_BACKEND_MODE=mock make sam2-backend-run
   ```

2. SAM2 MCP service (optional when using `mock`):
   ```bash
   export SAM2_MCP_BACKEND_URL="http://127.0.0.1:8181"
   make sam2-mcp-run
   ```

3. Virtual environment activated:
   ```bash
   source ~/venvs/ai_archaeo_topia/bin/activate
   ```

## Dry-run command

Processes first 5 clips only, writes to isolated `_dry_run` folder:

```bash
export PDF_PATH="path/to/input.pdf"
export OUTPUT_ROOT="data/annotations/runs"
export RUN_ID="manual_pdf_seed_001"
export SAM2_MCP_URL="mock"

make annotation-pdf-sam2-mock-dry-run
```

Or directly:

```bash
python -m services.annotation_pipeline.pdf_sam2_runner \
  --pdf "${PDF_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --sam2-mcp-url "${SAM2_MCP_URL}" \
  --sam2-mode mock \
  --dry-run true
```

## Full mock run command

```bash
make annotation-pdf-sam2-mock-run
```

Or directly:

```bash
python -m services.annotation_pipeline.pdf_sam2_runner \
  --pdf "${PDF_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --sam2-mcp-url "${SAM2_MCP_URL}" \
  --sam2-mode mock \
  --dry-run false
```

## Validation

```bash
make annotation-pdf-sam2-validate CSV_PATH="data/annotations/runs/run_001/annotations/annotation_results.csv"
```

Or directly:

```bash
python -m src.validate_annotation_run --csv "${CSV_PATH}"
```

## Output folder layout

Protocol-compliant structure under `data/annotations/runs/<RUN_ID>/`:

```
<OUTPUT_ROOT>/<RUN_ID>/
├── raw/images/<ANNOTATOR>/
│   └── AG_001_K-35-50_UNKNOWN.png
├── annotations/
│   └── annotation_results.csv
├── review/samples_for_review/
│   └── review_AG_001_K-35-50_UNKNOWN.png
└── masks/
    └── mask_AG_001_K-35-50_UNKNOWN_0.png
```

## CSV output

`annotation_results.csv` with protocol-compliant columns:

- `sample_id` — unique identifier per annotated object
- `annotator` — source annotator prefix
- `image_path` / `clip_image_path` — path to extracted clip
- `source_pdf` / `pdf_page` / `clip_index` — provenance
- `annotation_source` — always `sam2_mcp`
- `model_backend` — `mock` or `sam2`
- `label` — `mound`, `hard_negative_symbol`, `uncertain_ignore`
- `bbox_x_min` / `bbox_y_min` / `bbox_x_max` / `bbox_y_max` — pixel coordinates
- `polygon` — JSON polygon vertices
- `mask_path` — path to generated mask
- `confidence` — model confidence [0.0, 1.0]
- `review_status` — always `pending_review` for machine-generated
- Protocol metadata: `sheet_25k`, `sheet_5k`, `province`, `relief_original`, `relief_normalized`, `difficulty`, etc.

## Validation behavior

The runner validates:

- PDF path exists and is readable
- Output directory is writable
- Protocol directory exists
- SAM2 MCP endpoint is reachable (or mock mode confirmed)
- Extracted clip image files exist
- CSV rows reference existing image files
- Bbox coordinates are numeric and inside clip dimensions
- Labels conform to protocol taxonomy
- `review_status` is `pending_review`
- No duplicate sample IDs
- All generated paths stay under the run output directory
- `validate_annotation_run.py` is invoked post-run

## Dry-run mode

- Processes only first 5 clips
- Writes to `<RUN_ID>_dry_run/` folder
- Does not modify existing annotation data
- Runs full validation
- Prints concise summary

## Limitations

1. **Mock mode only** — SAM2 backend returns deterministic grid-based proposals, not real segmentation
2. **No OCR** — metadata extracted from PDF text only; no OCR enrichment
3. **No georeferencing** — pixel coordinates only, no map CRS
4. **No CVAT integration** — output is CSV, not CVAT-compatible format
5. **No database persistence** — outputs are file-based only
6. **No COCO/YOLO export** — CSV output only
7. **Metadata extraction is heuristic** — relies on PDF text patterns (Bulgarian Cyrillic)

## Next steps

### Optional OCR metadata enrichment
- Integrate `ai_co_scientist` OCR service
- Rebuild OCR locally for improved Bulgarian Cyrillic recognition
- Enrich `original_description`, `sheet_25k`, `sheet_5k` from OCR output

### Real SAM2
- Deploy SAM2 backend with GPU: `SAM2_BACKEND_MODE=sam2 make sam2-backend-run`
- Point pipeline to live backend: `--sam2-mcp-url http://127.0.0.1:8181`
- Evaluate proposal quality against manual annotations

### CVAT review/import/export
- Add CVAT dataset export from CSV
- Import annotations into CVAT for expert review
- Export reviewed annotations back to CSV

### Production GeoTIFF/georeferencing pipeline
- Add CRS metadata and georeferencing transforms
- Tile GeoTIFFs and run pipeline on tiles
- Export to COCO/YOLO/GeoJSON formats
- Database persistence for reviewed annotations

## CLI reference

```
python -m services.annotation_pipeline.pdf_sam2_runner [OPTIONS]

Options:
  --pdf PATH              Path to input PDF file  [required]
  --output-root PATH      Base output directory  [default: ./data/annotations/runs]
  --run-id ID             Unique run identifier  [default: run_<timestamp>]
  --sam2-mcp-url URL      SAM2 MCP backend URL or 'mock'  [default: mock]
  --sam2-mode MODE        SAM2 backend mode: mock|sam2  [default: mock]
  --review-status STATUS  Review status for annotations  [default: pending_review]
  --dry-run BOOL          Dry-run mode (first 5 clips)  [default: false]
  --max-proposals N       Max proposals per image  [default: 50]
  --protocol-dir PATH     Protocol directory  [default: docs/annotation]
  --log-level LEVEL       Logging level  [default: INFO]
```

## Makefile targets

| Target | Description |
|--------|-------------|
| `make annotation-pdf-sam2-mock-dry-run` | Dry-run with mock backend |
| `make annotation-pdf-sam2-mock-run` | Full run with mock backend |
| `make annotation-pdf-sam2-validate` | Validate a CSV file |

Environment variables: `PDF_PATH`, `OUTPUT_ROOT`, `RUN_ID`, `SAM2_MCP_URL`, `CSV_PATH`.
