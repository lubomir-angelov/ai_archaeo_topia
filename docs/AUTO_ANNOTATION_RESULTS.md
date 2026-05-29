# Auto-Annotation Run Results

**Run date:** 2026-05-28  
**PDF:** `/home/naim/repos/ai_archaeo_topia/data/map_annotation_clips.pdf`  
**Protocol:** `docs/annotation/PROTOCOL_EN.md` (label schema v0.1)

---

## Protocol

- **Labels:** `mound`, `hard_negative_symbol`, `uncertain_ignore`
- **Sample ID format:** `<ANNOTATOR>_<NUMBER>_<SHEET_25K>_<SHEET_5K>.png`
- **Output structure:** `raw/images/<ANNOTATOR>/`, `review/samples_for_review/`, `annotations/master_samples.csv`
- **Review status:** `pending_review` (all machine-generated candidates)

---

## Outputs

| Deliverable | Path |
|---|---|
| Extracted clip images | `artifacts/annotation_runs/raw/images/<ANNOTATOR>/` |
| Review images (bbox overlays) | `artifacts/annotation_runs/review/samples_for_review/` |
| Master samples CSV | `artifacts/annotation_runs/annotations/master_samples.csv` |
| SAM2 proposals | `artifacts/annotation_runs/sam2_proposals/sam2_proposals.json` |

---

## Numbers

| Metric | Value |
|---|---|
| PDF pages processed | 111 of 112 (page 112 had no images) |
| Clips extracted | **162** |
| Necropolis clips | 43 |
| Hard negative clips | 43 |
| Uncertain clips | 46 |
| Single mound clips | 17 |
| Validation errors | 0 |
| Validation warnings | 0 |

---

## Annotators detected

`AG`, `IK`, `UNK`, `VG` — 4 annotator sections in the PDF. ~120 of 162 clips have `annotator=UNK` (pages 1-41 where the annotator section header was not reliably detected by the parser).

---

## Relief distribution

| Normalized relief | Count |
|---|---|
| plain | 32 |
| mixed | 25 |
| slope | 23 |
| ridge | 20 |
| mountain | 18 |
| hilly | 18 |
| valley | 8 |
| urban | 5 |
| unknown | 13 |

---

## Bounding box proposals

Heuristic placeholder boxes only. SAM2 mask generation could not complete (see blockers below). All boxes marked `source=heuristic_placeholder`, `review_status=pending_review`.

---

## Scripts created for this run

| Script | Purpose |
|---|---|
| `src/pdf_clip_extractor.py` | Extracts embedded images + parses metadata from PDF, writes clips/review images/CSV |
| `src/sam2_proposal_generator.py` | Calls Nuclio SAM2 function for mask proposals |
| `src/validate_annotation_run.py` | Validates CSV columns, image paths, label taxonomy, duplicates |

---

## Commands used

```bash
pip install PyMuPDF

# Dry-run
python -m src.pdf_clip_extractor \
  --pdf data/map_annotation_clips.pdf \
  --output-root ./artifacts/annotation_runs \
  --protocol-dir docs/annotation \
  --dry-run

# Full run
python -m src.pdf_clip_extractor \
  --pdf data/map_annotation_clips.pdf \
  --output-root ./artifacts/annotation_runs \
  --protocol-dir docs/annotation

# Validate
python -m src.validate_annotation_run

# SAM2 proposals
python -m src.sam2_proposal_generator
```

---

## Blockers

### 1. CVAT image upload fails

CVAT behind the Docker proxy does not accept `multipart/form-data` on `/api/tasks/{id}/data`. All upload attempts returned `400 Unsupported media type` or `404 Not Found`. Task creation via JSON API with session key auth works, but image ingestion fails.

### 2. SAM2 mask generation incomplete

The Nuclio SAM2 function (`pth-facebookresearch-sam-vit-h`) is reachable on port 32768 and returns a `blob` (embedding features, ~4MB per image). However, mask generation requires either:

- CVAT's serverless interactor (images must be in a CVAT task), or
- Local PyTorch + SAM2 library (not installed in this environment).

Without either, masks cannot be generated from the blob. Heuristic bounding boxes serve as placeholder candidates.

---

## Assumptions / Protocol gaps

- **UNK annotator:** Pages 1-41 annotator section headers were not reliably parsed. These clips are labeled `annotator=UNK`.
- **Original description:** The `original_description` CSV column is empty. Page text was parsed for structured fields (sheet, province, relief) but the full description text was not extracted.
- **Sheet 5k parsing:** Some values contain unbalanced parentheses (e.g., `К-35-54_(223`) due to PDF text layout. Cosmetic only.
- **Bounding boxes:** Heuristic placeholders (centered, size-varying by case type) rather than SAM2-generated boxes.

---

## Next steps

1. Fix CVAT image upload (may need direct Docker networking or a different upload endpoint)
2. Install PyTorch + SAM2 library for local mask generation, or use the CVAT interactor
3. Fill `original_description` from parsed page text
4. Correct `UNK` annotator assignments by improving page-level annotator detection
