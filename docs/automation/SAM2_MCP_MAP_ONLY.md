# SAM2 MCP Service — Map-Only Segmentation

SAM 2 MCP is a **map-only segmentation service**.  It operates only on
image inputs that represent maps, map tiles, or cropped map regions.
It does **not** handle PDFs, documents, OCR, legend interpretation,
or generic image analysis.

## Responsibility boundary

| Responsibility              | Service                    |
|-----------------------------|----------------------------|
| Map-only segmentation       | **SAM 2 MCP** (this service) |
| OCR / PDF-to-text           | DeepSeek OCR MCP           |
| Geospatial reasoning        | Qwen_3.6_27B (Brain)       |
| Document understanding      | Qwen_3.6_35B_A3B           |
| Final annotation decisions  | Qwen_3.6_27B (Brain)       |
| Legend interpretation       | Qwen_3.6_27B (Brain)       |
| Class reasoning             | Qwen_3.6_27B (Brain)       |

## Architecture

```
[PDF / GeoTIFF / Map Image]
          │
          ▼
┌─────────────────────────┐
│  PDF Preprocessing       │  (annotation_pipeline)
│  Extracts images from    │
│  PDF.  SAM 2 never sees  │
│  the raw PDF.            │
└─────────┬───────────────┘
          │  map images only
          ▼
┌─────────────────────────┐
│  SAM 2 MCP               │  (map-only segmentation)
│  - segment_box           │
│  - segment_points        │
│  - generate_proposals    │
│  Returns: masks, polygons│
│          WKT, GeoJSON    │
└─────────┬───────────────┘
          │  geometry artifacts
          ▼
┌─────────────────────────┐
│  Qwen_3.6_27B            │  (Brain / Geospatial Agent)
│  Merges geometry + OCR   │
│  Validates labels        │
│  Produces annotations    │
└─────────────────────────┘
```

## Valid inputs

Accepted image formats:

* PNG
* JPEG/JPG
* TIFF/TIF
* BMP
* WebP

Rejected inputs (with clear error):

* PDF
* DOCX
* TXT
* HTML
* JSON
* CSV
* Any non-image file

## Input kinds

| `input_kind` | Description                              |
|-------------|------------------------------------------|
| `map_image` | Full map image or page                   |
| `map_tile`  | Tiled region of a larger map             |
| `map_crop`  | Cropped region of interest from a map    |

When `input_kind` is `map_tile`, the `tile_metadata` field must be
provided so the geospatial agent can reconstruct full-map coordinates.

## Tile metadata

```json
{
  "tile_id": "tile_001",
  "tile_offset_x": 1024,
  "tile_offset_y": 2048,
  "tile_width": 1024,
  "tile_height": 1024,
  "parent_image_path": "/maps/full_map.tif",
  "source_map_sheet": "K-35-50"
}
```

## MCP tools

### sam2_health

Check service and backend availability.

**Input:** none

**Output:**
```json
{
  "ok": true,
  "service": "sam2_mcp",
  "backend": "http://127.0.0.1:8181",
  "details": {}
}
```

### sam2_segment_box

Segment a **map image, map tile, or map crop** using a bounding-box prompt.

**Input:**
```json
{
  "image_path": "/maps/tile_001.png",
  "bbox": [100, 200, 300, 400],
  "label": "mound",
  "input_kind": "map_tile",
  "artifact_id": "map_K-35-50",
  "run_id": "run_20240101",
  "class_hint": "mound",
  "tile_metadata": {
    "tile_id": "tile_001",
    "tile_offset_x": 0,
    "tile_offset_y": 1024,
    "tile_width": 1024,
    "tile_height": 1024
  }
}
```

**Output:**
```json
{
  "image_path": "/maps/tile_001.png",
  "label": "mound",
  "bbox": [100.0, 200.0, 300.0, 400.0],
  "polygon": [[100, 200], [300, 200], [300, 400], [100, 400]],
  "polygon_wkt": "POLYGON((100 200 300 200 300 400 100 400))",
  "polygon_geojson": "{\"type\":\"Feature\",...}",
  "mask_path": "/output/mask_tile_001_abc123.png",
  "confidence": 0.92,
  "source": "sam2_mcp",
  "status": "pending_review",
  "artifact_id": "map_K-35-50",
  "run_id": "run_20240101",
  "coordinate_space": "tile_pixel",
  "tile_metadata": {...},
  "class_hint": "mound",
  "warnings": [],
  "errors": []
}
```

### sam2_segment_points

Segment a **map image, map tile, or map crop** using point prompts.

**Input:**
```json
{
  "image_path": "/maps/tile_001.png",
  "points": [[150, 250], [200, 300]],
  "point_labels": [1, 0],
  "label": "symbol",
  "input_kind": "map_tile",
  "artifact_id": "map_K-35-50",
  "run_id": "run_20240101"
}
```

**Output:** Same shape as `sam2_segment_box`, plus `points` and
`point_labels` echoed from input.

### sam2_generate_proposals

Generate candidate segmentation proposals for **map images only**.

**Input:**
```json
{
  "image_path": "/maps/tile_001.png",
  "label_hint": "mound",
  "max_proposals": 20,
  "input_kind": "map_image",
  "artifact_id": "map_K-35-50",
  "run_id": "run_20240101"
}
```

**Output:**
```json
{
  "items": [
    {
      "image_path": "/maps/tile_001.png",
      "bbox": [10.0, 20.0, 150.0, 200.0],
      "polygon": [[10, 20], [150, 20], [150, 200], [10, 200]],
      "polygon_wkt": "POLYGON((10 20 150 20 150 200 10 200))",
      "polygon_geojson": "{\"type\":\"Feature\",...}",
      "mask_path": "/output/mask_abc123.png",
      "confidence": 0.85,
      "label": "mound",
      "source": "sam2_mcp",
      "status": "pending_review",
      "artifact_id": "map_K-35-50",
      "run_id": "run_20240101",
      "coordinate_space": "image_pixel"
    }
  ],
  "artifact_id": "map_K-35-50",
  "run_id": "run_20240101",
  "input_kind": "map_image",
  "warnings": [],
  "errors": []
}
```

## Coordinate spaces

| `coordinate_space` | Description                              |
|-------------------|------------------------------------------|
| `tile_pixel`      | Coordinates relative to tile origin      |
| `image_pixel`     | Coordinates relative to full image origin|
| `map_pixel`       | Coordinates in map/georeferenced space   |

When `input_kind` is `map_tile` and `tile_metadata` is provided,
the output `coordinate_space` is `tile_pixel`.  Otherwise it is
`image_pixel`.  The geospatial agent is responsible for converting
tile-local coordinates to full-map or geo-coordinates using the
tile metadata and source raster transform.

## Provenance

Every output includes:

* `artifact_id` — source artifact identifier
* `run_id` — run identifier for batch tracking
* `tile_metadata` — tile-to-global mapping (when applicable)
* `coordinate_space` — coordinate system of geometry
* `source` — always `"sam2_mcp"`
* `status` — always `"pending_review"`

## Error responses

```json
{
  "error": "ValidationError",
  "detail": "Input '/maps/document.pdf' has extension '.pdf' which is not a map image. SAM 2 MCP only accepts map images (PNG, JPEG, TIFF, BMP, WebP). For PDFs, use the DeepSeek OCR MCP or a document preprocessing step."
}
```

## Configuration

| Variable                     | Default                  | Description                              |
|------------------------------|--------------------------|------------------------------------------|
| `SAM2_MCP_BACKEND_URL`       | `http://127.0.0.1:8181`  | Backend HTTP endpoint or `mock`          |
| `SAM2_MCP_BACKEND_TIMEOUT`   | `120.0`                  | Request timeout in seconds               |
| `SAM2_MCP_OUTPUT_DIR`        | `./artifacts/sam2_mcp`   | Directory for generated artifacts        |
| `SAM2_MCP_MAX_IMAGE_SIZE`    | `10000`                  | Maximum image dimension in pixels        |
| `SAM2_MCP_LOG_LEVEL`         | `INFO`                   | Python logging level                     |

Set `SAM2_MCP_BACKEND_URL=mock` for deterministic mock mode.

## Mock mode

When `SAM2_MCP_BACKEND_URL=mock`:

* **Health:** Always returns `reachable: true`
* **Segment box:** Returns polygon matching input bbox, confidence `0.83`
* **Segment points:** Returns polygon around foreground points, confidence `0.75`
* **Proposals:** Returns up to 3 proposals, decreasing confidence from `0.7`

## PDF handling

SAM 2 MCP **does not** accept PDFs.  The correct flow is:

1. **PDF preprocessing** (`annotation_pipeline`): Extract images from PDF
2. **SAM 2 MCP**: Segment the extracted map images
3. **Geospatial agent**: Merge results with OCR and metadata

If a PDF is passed to SAM 2 MCP, it will be rejected with a `ValidationError`
explaining that PDFs should be handled by the DeepSeek OCR MCP or a document
preprocessing step.
