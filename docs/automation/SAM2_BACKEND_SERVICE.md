# SAM2 Backend Service

Self-contained HTTP inference backend for SAM2 segmentation. Called by the
SAM2 MCP adapter (`src/services/sam2_mcp/`) over HTTP. Independent of CVAT,
Nuclio, OCR, and LLM services.

## Purpose

Provides a stateless HTTP server that runs SAM2 segmentation inference.
The MCP adapter acts as a client; this backend handles model loading,
image processing, mask generation, and proposal generation.

## Relationship to sam2_mcp

```
┌─────────────────────┐     HTTP      ┌──────────────────────┐
│  sam2_mcp (MCP)     │ ──────────►   │  sam2_backend (HTTP) │
│  - Tool definitions  │  /health      │  - Model loading     │
│  - Input validation  │  /segment     │  - Inference         │
│  - Client logic      │  /propose     │  - Mask generation   │
│  - MCP transport     │               │  - Proposal gen      │
└─────────────────────┘               └──────────────────────┘
```

- `sam2_mcp` = MCP tool adapter (stdio transport, tool definitions, client)
- `sam2_backend` = HTTP inference service (model, masks, proposals)

## Quick start

### Mock mode (no GPU required)

```bash
# Start backend
SAM2_BACKEND_MODE=mock make sam2-backend-run

# Health check
curl http://localhost:8080/health

# Segment with bbox
curl -s -X POST http://localhost:8080/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.png",
    "prompt_type": "bbox",
    "bbox": [100, 200, 300, 400],
    "label": "mound"
  }' | python3 -m json.tool

# Proposals
curl -s -X POST http://localhost:8080/propose \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.png",
    "max_proposals": 10
  }' | python3 -m json.tool
```

### Real SAM2 mode

```bash
# Install SAM2 package (if not already installed)
pip install sam2

# Set checkpoint path
export SAM2_BACKEND_CHECKPOINT=/path/to/sam2_hiera_large.pt
export SAM2_BACKEND_MODEL_CFG=sam2_hiera_l.yaml
export SAM2_BACKEND_MODE=sam2
export SAM2_BACKEND_DEVICE=auto

# Start backend
make sam2-backend-run
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM2_BACKEND_HOST` | `0.0.0.0` | Bind host |
| `SAM2_BACKEND_PORT` | `8080` | Bind port |
| `SAM2_BACKEND_MODE` | `mock` | `mock` or `sam2` |
| `SAM2_BACKEND_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `SAM2_BACKEND_MODEL_CFG` | `` | SAM2 model config file path |
| `SAM2_BACKEND_CHECKPOINT` | `` | SAM2 checkpoint file path |
| `SAM2_BACKEND_OUTPUT_DIR` | `./artifacts/sam2_backend` | Base output directory |
| `SAM2_BACKEND_MAX_IMAGE_PIXELS` | `10000` | Max image dimension |
| `SAM2_BACKEND_MAX_PROPOSALS_HARD_CAP` | `500` | Hard cap on proposals |
| `SAM2_BACKEND_LOG_LEVEL` | `INFO` | Python logging level |

## Modes

### Mock mode (`SAM2_BACKEND_MODE=mock`)

- Deterministic, no GPU required
- Returns plausible masks/polygons for testing
- Bbox prompts: fills bbox with binary mask, confidence 0.85
- Point prompts: draws circles around foreground points, confidence 0.75
- Proposals: grid-based deterministic proposals, decreasing confidence
- Suitable for CI, smoke tests, and local development

### SAM2 mode (`SAM2_BACKEND_MODE=sam2`)

- Loads actual SAM2 model/checkpoint
- Uses CUDA when configured and available
- Falls back to CPU if CUDA unavailable (unless `device=cuda`)
- Fails clearly at health check if dependency/checkpoint missing

## Device selection

| `SAM2_BACKEND_DEVICE` | Behavior |
|----------------------|----------|
| `auto` | CUDA if available, else CPU |
| `cuda` | Requires CUDA; fails if unavailable |
| `cpu` | Uses CPU |

## Checkpoint setup

Place your SAM2 checkpoint file and set:

```bash
export SAM2_BACKEND_CHECKPOINT=/path/to/sam2_hiera_large.pt
export SAM2_BACKEND_MODEL_CFG=sam2_hiera_l.yaml
```

Common model configs:
- `sam2_hiera_t.yaml` -- tiny, fastest
- `sam2_hiera_s.yaml` -- small
- `sam2_hiera_l.yaml` -- large (default)
- `sam2_hiera_b+.yaml` -- base-plus

## API endpoints

### GET /health

Returns backend status:

```json
{
  "ok": true,
  "service": "sam2_backend",
  "model": "sam2",
  "mode": "mock",
  "device": "mock",
  "model_loaded": true,
  "checkpoint": "",
  "details": {}
}
```

In sam2 mode with missing dependency:

```json
{
  "ok": false,
  "service": "sam2_backend",
  "model": "sam2",
  "mode": "sam2",
  "device": "cpu",
  "model_loaded": false,
  "checkpoint": "",
  "details": {"error": "SAM2_BACKEND_CHECKPOINT is not set"}
}
```

### POST /segment

Segment with bbox or point prompt. See `SAM2_BACKEND_CONTRACT.md` for full spec.

### POST /propose

Generate candidate proposals. See `SAM2_BACKEND_CONTRACT.md` for full spec.

## Makefile targets

| Target | Description |
|--------|-------------|
| `make sam2-backend-run` | Start the backend HTTP service |
| `make sam2-backend-health` | Check backend health via curl |
| `make sam2-backend-mock-smoke` | Run mock-mode smoke test |
| `make sam2-backend-contract-test` | Run backend contract tests |

## Pointing sam2_mcp to this backend

```bash
# Start backend
SAM2_BACKEND_MODE=mock make sam2-backend-run

# In another terminal, configure MCP client
export SAM2_MCP_BACKEND_URL=http://127.0.0.1:8080

# Start MCP server
make sam2-mcp-run
```

## Package structure

```
src/services/sam2_backend/
├── __init__.py      # Package init
├── api.py           # FastAPI routes
├── main.py          # Entry point (uvicorn)
├── settings.py      # Configuration
├── schemas.py       # Pydantic schemas
├── errors.py        # Custom exceptions
├── predictor.py     # SAM2PredictorBackend
├── masks.py         # Mask/polygon utilities
└── proposals.py     # Proposal generation
```

## Known limitations

1. SAM2 mode requires the SAM2 package to be installed separately
2. Proposal generation is Phase 1 (grid/heuristic) -- not semantic
3. Polygon extraction uses bbox fallback when contour approximation yields < 3 points
4. No automatic model download -- checkpoint must be placed manually
5. No Docker Compose service yet (repo doesn't have service containers)
6. Single-model per process -- no model swapping at runtime

## Next steps for PDF annotation integration

1. Connect PDF clip extraction (`src/pdf_clip_extractor.py`) to backend
2. Add batch processing endpoint for multiple images
3. Implement proposal quality scoring
4. Add CVAT-compatible export format
5. Add Docker Compose service definition
6. Integrate with annotation runner pipeline
