# SAM2 Backend HTTP Contract

This document defines the HTTP API contract between the `src/services/sam2_mcp`
MCP service and the live SAM2 inference backend.  The MCP service acts as a
client; the backend is a stateless HTTP server that runs SAM2 segmentation
inference.

## Endpoints

### GET /health

Check backend availability and model status.

**Request:**

```
GET /health
```

**Success response (200):**

```json
{
  "ok": true,
  "service": "sam2_backend",
  "model": "sam2",
  "device": "cuda",
  "details": {}
}
```

**Fields:**

| Field     | Type    | Required | Description                              |
|-----------|---------|----------|------------------------------------------|
| `ok`      | boolean | yes      | `true` when the backend is healthy       |
| `service` | string  | yes      | Service identifier, must be `"sam2_backend"` |
| `model`   | string  | yes      | Model variant, e.g. `"sam2"`             |
| `device`  | string  | yes      | Inference device: `"cuda"`, `"cpu"`, or `"unknown"` |
| `details` | object  | no       | Arbitrary additional metadata            |

**Error responses:** Any non-200 status code indicates failure. The MCP client
treats connection errors and timeouts as unreachable.

---

### POST /segment

Run segmentation with a bounding-box or point prompt.

**Content-Type:** `application/json`

#### Bounding-box prompt

**Request:**

```json
{
  "image_path": "/absolute/path/to/image.png",
  "prompt_type": "bbox",
  "bbox": [10.0, 20.0, 150.0, 200.0],
  "label": "mound",
  "output_dir": "/absolute/path/to/output"
}
```

**Fields:**

| Field        | Type   | Required | Description                              |
|-------------|--------|----------|------------------------------------------|
| `image_path`| string | yes      | Absolute path to the image on the backend |
| `prompt_type`| string| yes      | Must be `"bbox"` or `"points"`           |
| `bbox`      | array  | yes (if bbox) | `[x_min, y_min, x_max, y_max]` in pixel coordinates |
| `label`     | string | no       | Optional label for the segmentation      |
| `output_dir`| string | no       | Directory for generated mask files       |

#### Point prompt

**Request:**

```json
{
  "image_path": "/absolute/path/to/image.png",
  "prompt_type": "points",
  "points": [[50.0, 60.0], [80.0, 90.0]],
  "point_labels": [1, 0],
  "label": "symbol",
  "output_dir": "/absolute/path/to/output"
}
```

**Fields:**

| Field        | Type   | Required | Description                              |
|-------------|--------|----------|------------------------------------------|
| `image_path`| string | yes      | Absolute path to the image on the backend |
| `prompt_type`| string| yes      | Must be `"bbox"` or `"points"`           |
| `points`    | array  | yes (if points) | `[[x, y], ...]` in pixel coordinates |
| `point_labels`| array| yes (if points) | `1` = foreground, `0` = background |
| `label`     | string | no       | Optional label for the segmentation      |
| `output_dir`| string | no       | Directory for generated mask files       |

**Success response (200):**

```json
{
  "image_path": "/absolute/path/to/image.png",
  "label": "mound",
  "bbox": [10.0, 20.0, 150.0, 200.0],
  "polygon": [[10.0, 20.0], [150.0, 20.0], [150.0, 200.0], [10.0, 200.0]],
  "mask_path": "/absolute/path/to/output/mask_abc123.png",
  "confidence": 0.92,
  "metadata": {}
}
```

**Fields:**

| Field        | Type   | Required | Description                              |
|-------------|--------|----------|------------------------------------------|
| `image_path`| string | yes      | Echo of input image path                 |
| `label`     | string | no       | Echo of input label                      |
| `bbox`      | array  | yes      | `[x_min, y_min, x_max, y_max]` of result |
| `polygon`   | array  | no       | `[[x, y], ...]` polygon vertices         |
| `mask_path` | string | no       | Absolute path to generated mask PNG      |
| `confidence`| number | no       | Float in `[0.0, 1.0]`                    |
| `metadata`  | object | no       | Arbitrary additional metadata            |

**Validation enforced by MCP client:**

- `bbox` must be exactly 4 numeric values
- `polygon` must be a list of `[x, y]` numeric pairs
- `confidence` must be a number in `[0.0, 1.0]`
- `mask_path`, if present, must be under `output_dir`

---

### POST /propose

Generate candidate segmentation proposals for an image.

**Content-Type:** `application/json`

**Request:**

```json
{
  "image_path": "/absolute/path/to/image.png",
  "label_hint": "mound",
  "max_proposals": 50,
  "output_dir": "/absolute/path/to/output"
}
```

**Fields:**

| Field        | Type   | Required | Description                              |
|-------------|--------|----------|------------------------------------------|
| `image_path`| string | yes      | Absolute path to the image on the backend |
| `label_hint`| string | no       | Suggested label for proposals            |
| `max_proposals`| int  | no       | Maximum number of proposals (default 50) |
| `output_dir`| string | no       | Directory for generated mask files       |

**Success response (200):**

```json
{
  "items": [
    {
      "image_path": "/absolute/path/to/image.png",
      "label": "mound",
      "bbox": [10.0, 20.0, 150.0, 200.0],
      "polygon": [[10.0, 20.0], [150.0, 20.0], [150.0, 200.0], [10.0, 200.0]],
      "mask_path": "/absolute/path/to/output/mask_abc123.png",
      "confidence": 0.85,
      "metadata": {}
    }
  ]
}
```

**Fields:**

| Field   | Type  | Required | Description                              |
|---------|-------|----------|------------------------------------------|
| `items` | array | yes      | List of proposal items                   |

Each item in `items` follows the same schema as the POST /segment success
response.

**Validation enforced by MCP client:**

- Same validation as POST /segment for each item
- `mask_path` must be under `output_dir`

---

## Error handling

The MCP client handles the following error conditions:

| Condition                    | Client behavior                                    |
|------------------------------|---------------------------------------------------|
| Backend unreachable          | `BackendError` with `connection_refused` detail   |
| Backend timeout              | `BackendError` with `timeout` detail              |
| Non-200 HTTP status          | `BackendError` with status code and body snippet  |
| Invalid JSON response        | `BackendError` with parse error                   |
| Invalid bbox in response     | `BackendError` with schema validation detail      |
| Invalid polygon in response  | `BackendError` with schema validation detail      |
| Confidence out of range      | `BackendError` with schema validation detail      |
| mask_path outside output_dir | `BackendError` with path escape detail            |

All errors are logged at ERROR or WARNING level before being raised.

## Configuration

The MCP client is configured via environment variables:

| Variable                     | Default                  | Description                              |
|------------------------------|--------------------------|------------------------------------------|
| `SAM2_MCP_BACKEND_URL`       | `http://127.0.0.1:8080`  | Backend HTTP endpoint or `mock`          |
| `SAM2_MCP_BACKEND_TIMEOUT`   | `120.0`                  | Request timeout in seconds               |
| `SAM2_MCP_OUTPUT_DIR`        | `./artifacts/sam2_mcp`   | Directory for generated artifacts        |
| `SAM2_MCP_MAX_IMAGE_SIZE`    | `10000`                  | Maximum image dimension in pixels        |
| `SAM2_MCP_LOG_LEVEL`         | `INFO`                   | Python logging level                     |

Set `SAM2_MCP_BACKEND_URL=mock` to enable the deterministic mock backend
for local testing without a GPU.

## Example curl commands

```bash
# Health check
curl -s http://localhost:8080/health | python3 -m json.tool

# Bounding-box segmentation
curl -s -X POST http://localhost:8080/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/data/maps/map_001.png",
    "prompt_type": "bbox",
    "bbox": [100, 200, 300, 400],
    "label": "mound",
    "output_dir": "/data/output"
  }' | python3 -m json.tool

# Point segmentation
curl -s -X POST http://localhost:8080/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/data/maps/map_001.png",
    "prompt_type": "points",
    "points": [[150, 250], [200, 300]],
    "point_labels": [1, 0],
    "label": "symbol",
    "output_dir": "/data/output"
  }' | python3 -m json.tool

# Proposal generation
curl -s -X POST http://localhost:8080/propose \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/data/maps/map_001.png",
    "label_hint": "mound",
    "max_proposals": 20,
    "output_dir": "/data/output"
  }' | python3 -m json.tool
```

## Mock mode

When `SAM2_MCP_BACKEND_URL=mock`, the client returns deterministic results
without contacting any backend:

- **Health:** Always returns `reachable: true` with `mode: "mock"`
- **Segment box:** Returns a polygon matching the input bbox corners with
  confidence `0.83`
- **Segment points:** Returns a polygon around foreground point centroid
  with confidence `0.75`; returns empty polygon with `0.0` confidence if
  no foreground points
- **Proposals:** Returns up to 3 proposals with decreasing confidence
  starting at `0.7`

Mock mode is suitable for smoke testing, CI, and local development.
