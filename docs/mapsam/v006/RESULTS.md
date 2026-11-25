# MapSAM v0.6 — ingest, tooling, and a delivery the team can work from

Executes steps 1 and 3's tooling from `PLAN.md`, plus two schema changes and a
set of corrections. Steps 2 and 4 were not attempted; see *What was not done*.

**The headline this version planned is not in it.** `PLAN.md` set the
deliverable as the project's first *test* result — the untouched v0.5 detector
and v0.4r decoder measured against a blind, frozen set. That is gated on a
human annotating four sheets with no model output visible, and it has not
happened. Every number below is still development evidence, as every number in
this project has been since v0.1.

What v0.6 did instead is everything that gate does not block: it measured the
new corpus, built the tooling the annotation passes need, fixed the schema
while the protocol was open, and shipped 56 sheets to the team in a form they
can use in the field and review in their own software.

---

## Summary

1. **The domain transferred.** Candidate density on 56 unseen sheets is
   1.31–2.10 per megapixel at confidence 0.25, against 2.55 mounds per
   megapixel on the annotated clips. Same order of magnitude and lower, which
   is what whole sheets containing uninformative terrain should do to a density
   measured on clips chosen for annotation.
2. **The review burden is 0.17–0.30 candidates per window**, against a
   projection of 0.36–0.61. Over 8,671 windows that is 1,613–2,593 candidates
   at confidence 0.25 rather than the ~3,300 projected. The projection was
   pessimistic in the direction that costs reviewer time, and it has been
   replaced by a measurement.
3. **One sheet is an order of magnitude out, and it is central Sofia.**
   `K-34-47-G-v` returns 957 / 140 / 418 candidates from the three checkpoints
   where no other sheet exceeds 110. Excluding it, the three agree to within
   0.14 candidates per megapixel over 55 sheets. Dense urban fabric is a regime
   the corpus contains none of.
4. **The pipeline is healthy on unseen sheets, measured without ground truth.**
   Detector point to mask centroid is median 2.01 px and p90 4.26 px over 4,100
   instances, with 5.2% above the ~5 px band where v0.4 showed segmentation
   starting to degrade. Symbol outline area has median 438 px² against the
   annotated corpus's 445.
5. **56 sheets are delivered as GeoPackages**, 4,100 proposed locations with
   their symbol outlines, in EPSG:25835, with reviewer instructions in English
   and Bulgarian. Nothing in the repository wrote a vector GIS file before this.
6. **The water-line rule became mechanical.** `crossed_by_water_line` is now
   `water_line_crossing`, and the check it restores — a mound crossed by a
   *surface* watercourse is a data error — returns zero flagged mounds on the
   corrected export.

---

## Step 1 — the blind sweep

Full detail in `SWEEP_FINDINGS.md`; artifacts in `analysis/sweep/`. The v0.5
detector, unchanged, over the 56 sheets left after the four frozen ones were
moved aside. All three fold checkpoints, because every fold is equally
out-of-domain on a sheet none has seen.

| | fold A | fold B | fold C |
|---|---:|---:|---:|
| candidates at 0.25 | 2,593 | 1,613 | 2,062 |
| per window at 0.25 | 0.299 | 0.186 | 0.238 |
| per Mpx at 0.25 | 2.10 | 1.31 | 1.67 |
| excluding `K-34-47-G-v`, per Mpx | 1.35 | 1.22 | 1.36 |

8,671 windows over 1,232.4 Mpx — exactly the 9,271 `INPUT_INVENTORY.md`
projects for all 60 sheets, less the 600 belonging to the frozen four.

**This measures candidate density, not precision.** The new sheets carry no
ground truth, so what fraction of these are false is unknowable until the blind
set exists. Reporting it as a false-positive rate would be exactly the circular
measurement the blind set is there to prevent.

Three independently trained checkpoints landing within 0.14 candidates per
megapixel of each other over 55 unseen sheets is the strongest evidence here
that the transfer is real rather than an artifact of one checkpoint. Per-sheet
agreement at 10 px is median 0.766 / 0.813 / 0.904 across the three pairings.

**The alpha filter turned out to be a no-op.** The plan expected roughly 9% of
windows to fall outside the printed frame. Zero were dropped, and on three
sampled sheets no window falls below 50% valid: the clipped frame is diagonal,
so it intersects about a quarter of the windows and empties none. The concern
was correct in principle and measures to zero here.

---

## Schema changes, made while the protocol was open

### `water_line_crossing` replaces `crossed_by_water_line`

A mound cannot be crossed by a **surface** watercourse — water runs along
terrain lows and a mound is raised — and that rule found all twelve watermills
labelled as mounds on `K-34-35-B-g` in v0.5. But an **underground** line is a
pipe and can run beneath anything, and the boolean conflated the two, so a hit
meant one of three things and only a person could say which.

`PLAN.md` asked for two booleans. The migration uses **one select** instead,
with values `none`, `surface`, `underground`, `unreviewed`, because two
booleans cannot express the state most instances are actually in: 105 hard
negatives and 2 uncertain regions carry the old attribute and nobody has ever
been asked whether their crossing is surface or underground. Two booleans would
force a value onto every one and manufacture 105 assertions no annotator made.
The fourth state records the truth and doubles as a work queue.

`annotation/cvat/v0.0.3` applies it. 714 annotations, 169 / 542 / 3 by
category, geometry byte-identical, and the only semantic change is the one
v0.5 left for the next export: `K-35-51-B-a_3` at (314, 15.5), a real mound
whose attribute was set in error. `K-35-8-G-a_1` becomes `underground`, which
is what v0.5's review concluded.

The check the split restores is one line — a mound marked `surface` is a data
error, a mound marked `unreviewed` is one nobody has checked — and both are
empty on v0.0.3.

### `review_status`, and a machine-readable schema

`review_status` — `unreviewed`, `confirmed`, `rejected`, `corrected`, `added` —
is a **separate axis from `annotation_provenance`**. One records who drew a
shape, the other what a reviewer concluded. Collapsing them would make a
rejected detection indistinguishable from one that was never sent, and the
recall denominator impossible to reconstruct afterwards.

`annotation/cvat/labels.json` is now the schema of record: 15 / 16 / 16
attributes across the three classes. Before v0.6 the schema existed only
implicitly inside the COCO exports, which is how `PROTOCOL_EN.md` section 8
drifted away from it. `annotation/archaeology_symbols_v1.yaml` looks like the
schema and is not — it belongs to the unused SAM2 proposal path.

---

## Permanent splits, grouped by 1:100k parent

`configs/splits/v0_6_splits.json`: **45 train / 13 validation / 5 test** over
63 sheets, with scale and series recorded per sheet.

Grouping is on the 1:100k parent, not the 1:25k sheet id. `K-35-51-B-g` and the
annotated `K-35-51-B-a` are adjacent quadrants of the same 1:50k sheet — same
terrain, survey campaign, print run and scan batch — so splitting on the sheet
id is a weaker separation than the ids suggest.

**The grouping forces one thing worth stating.** `K-35-39-A-g` shares parent
`K-35-39` with two of the four frozen sheets, so it lands in test as well. It is
not annotated and contributes no evaluation instances, but training on it would
be exactly the leak the grouping exists to prevent.

The validation split is what finally makes leak-free checkpoint selection
possible. v0.5 had to report `last.pt` for want of one, at a measured cost of
0.812 instead of the 0.891 `best.pt` would have claimed.

---

## The format classes

`src/archeo_topia/formats/` — `LabelSchema`, `CocoDocument`, `SheetReference`,
`FeatureCollection`. Built because the same work is needed by inference,
training and the GIS delivery, and because the COCO handling had spread across
five modules with two independent RLE decoders that disagreed.

**The central decision: COCO is in image pixels and a feature collection is on
the ground**, so a direct conversion between them is undefined. Every
conversion takes a `SheetReference` as a required argument. `AGENTS.md` puts it
as *"avoid hidden coordinate assumptions"*; making the reference impossible to
omit is how that is enforced rather than merely intended.

`SheetReference` closes a gap open since it was written: the clip-to-ground
path is documented in `sheet_clips.py`, specified in `PLAN.md` and **tested** in
`test_sweep_sheets.py` — but no code walked it.

**One decoder replaces two, as their union.** Measured on v0.0.3, the 714
annotations are 161 uncompressed RLE, 20 polygons and **533 carrying a box and
no segmentation** — and box-only is precisely what the old decoder returned a
silently empty mask for. Where the two originals disagreed on RLE size
validation the strict reading wins: a mask declaring a different size than its
image is a data bug, and pasting it into the top-left corner hid it.

**GDAL appears only as a subprocess.** Its Python bindings must match the system
`libgdal` exactly and be built against an installed numpy, or they import and
then fail with `no module named _gdal_array`; `README.md` documents the
recovery. GeoJSON is written with the standard library and `ogr2ogr` converts
it. A test asserts nothing under `formats/` imports `osgeo`, so the decision
cannot quietly erode, and nothing was added to `pyproject.toml`.

### Migrations, each with a guard that ran

| site | guard | result |
|---|---|---|
| `export_proposals_coco` attribute defaults | re-export, compare | attributes now come from the schema; the drift test is retired by construction |
| `build_detection_windows.decode_mask` | rebuild `mapsam_det_v0` and `v1` | `build_report.json` and `annotations.jsonl` byte-identical |
| `mapsam_end_to_end` | re-run fold B end to end | 1.000 at IoU≥0.5, 0.529 / 0.471 at IoU≥0.75, 0.7482 / 0.7455 mean IoU — identical to v0.5 |
| `prepare_mapsam_coco` RLE decoder | rebuild, compare | unchanged |

`decode_mask` keeps a `bbox_fallback` switch set to `False` at the dataset
builders. That protects a published result: the v0.0.1 export carries one mound
with a box and no mask, and v0.1–v0.5 were all computed counting it as having no
geometry. Filling its box instead would silently change the component grouping
and the `mounds_without_geometry` count those runs reported.

---

## The GIS delivery

`data_lake/cleaned/gis/v0_6_foldC/` — 55 sheets in `sheets/`, `K-34-47-G-v` in
`set_aside/`, 15 MB in total, with `README_EN.md` and `README_BG.md`.

Each sheet is one GeoPackage in EPSG:25835 holding two layers:

| layer | what it is |
|---|---|
| `mound_points` | the archaeological location; what field work navigates to |
| `mound_symbols` | the printed cartographic symbol, joined on `mound_id` |

**The `symbol_` prefix is load-bearing.** `TARGET_PIPELINE.md` states it
plainly: *"The MapSAM polygon is the printed cartographic mound symbol. It is
not the physical footprint of the archaeological mound."* Symbol areas run
116–1037 px on a symbol about 21 px across; that spread is drafting variation
and scan condition, not mound size, so a consumer reading area as ground extent
is wrong by an unbounded factor.

| | value |
|---|---:|
| sheets | 56 |
| proposed locations | 4,100 at confidence 0.05; 2,062 at 0.25 |
| symbol outlines | 4,100 — every candidate decoded |
| per sheet | min 7, median 39, max 1,651 |
| outline area px² | min 76, median 438, p90 541, max 816 |
| detector to mask centroid | median 2.01 px, p90 4.26 px, 5.2% above 5 px |
| held out of training | `K-35-39-A-g`, carried on every feature as `split` |

**One sheet in the delivery may not be trained on.** `K-35-39-A-g` shares a
1:100k parent with two of the frozen sheets, so the splits put it in `test`. It
is a working-pool sheet, so it is swept and delivered like any other and the
team can use it in the field — but if their review of it entered training, that
would be exactly the leak the parent grouping exists to prevent. Rather than
leave that in a note, every exported feature carries a `split` field, the
export report lists `held_out_of_training`, and both READMEs say to leave the
field alone.

Formats were chosen on measurement. GeoPackage holds both layers in one file,
declares its CRS unambiguously in both tools, and keeps full-length field
names. GDAL's GeoJSON writer emits a legacy `crs` member that RFC 7946 removed:
QGIS honours it, ArcGIS generally does not and assumes WGS84, so the portable
copy is written with `RFC7946=YES` and reprojected explicitly. **Shapefile is
excluded on evidence** — all 14 `mound` attributes exceed its ten-character
limit, and `crossed_by_contour`, `crossed_by_forestation_line`,
`crossed_by_grid`, `crossed_by_powerline` and `crossed_by_road` all collapse to
`crossed_by`, `crossed__1` … `crossed__4`.

### How it was verified

Counts reproduce the sweep exactly: 4,100 at 0.05 and 2,062 at 0.25, on all 56
sheets, with both layers and EPSG:25835 everywhere.

Geometry was checked by going the whole way back — GeoPackage, to EPSG:25835,
to sheet pixel, to clip pixel — and rendering the result over the source
imagery. The outlines land on the printed symbols, and the round trip returns
the original pixel with zero error. This was done on `K-35-21-G-a` and on
`K-34-10-A-g`, the latter chosen because at easting 148,562 it sits at the
far-western extreme of the corpus, where UTM 35N distortion is largest.

**One caveat visible in that second render.** `K-34-10-A-g`'s candidates sit on
small circular symbols rather than the familiar starburst, consistent with the
print-style variation `SWEEP_FINDINGS.md` already flags for that sheet. Whether
they are mounds, trig points or elevation dots is not something the geometry can
settle, and is exactly what the review is for.

### The return path

`archeo_topia.formats.ingest_review` reads a reviewed GeoPackage and produces an
evaluation plus a COCO file for CVAT. It checks integrity before it counts
anything: a changed CRS, a duplicated `mound_id`, a feature moved off the sheet,
one left `unreviewed`, and a proposal **deleted rather than marked rejected** —
that last being what `review_status` exists to prevent, since a deleted feature
cannot be told from one never sent.

**It has only been tested against simulated edits.** No file a person actually
touched has been through it.

When a real review arrives, scope the result: it is field verification by domain
experts on model-proposed candidates, so it measures **precision** well and
**recall only as a floor**, over what reviewers independently noticed. It is not
the frozen blind test and cannot substitute for it.

---

## Corrections to earlier documents

Four things a reader would otherwise have taken on trust and been wrong about.

- **`PROTOCOL_EN.md` section 8 did not describe the live schema.** It omitted
  `crossed_by_water_line` — the attribute that caught the twelve mislabels —
  along with `crossed_by_forestation_line` and both elevation-mark variants, and
  listed five attributes CVAT does not emit. Its `negative_type` vocabulary was
  a proposal never used. Reconciled, and mirrored into both Bulgarian protocols.
- **`v005/RESULTS.md` duplicated its last three sections**, and the second copy
  was the pre-correction text: it called `K-35-8-G-a_1` unreviewed where the
  first copy records it as resolved.
- **"Seven pre-existing test failures"** appeared verbatim in four documents,
  carried forward without re-measurement. It is **3**, all environment-coupled
  assertions in `test_sam2_backend.py`. The live plan states the measurement;
  the historical documents carry a dated correction rather than a silent
  rewrite.
- **The `sam2-mcp` connection failure is diagnosed and fixed.** It was not the
  import path: a bare `python3` resolves to an interpreter carrying `mcp` 2.2.0,
  which removed the low-level `@server.list_tools` decorator the server uses, so
  `create_server()` raised before the transport came up. The venv has 1.27.1.
  `pyproject.toml` now pins `mcp<2`, and since `.mcp.json` is gitignored the
  working configuration is recorded in `docs/automation/SAM2_MCP_MAP_ONLY.md`.

One defect found and fixed in v0.6's own code: `sweep_sheets` wrote a rounded
pixel beside a ground coordinate derived from the **unrounded** one, so the two
fields in `candidates.jsonl` disagreed by a few millimetres and anyone
recomputing easting from the recorded `x` got a different answer. The sweep was
regenerated; all counts are unchanged.

---

## What this does and does not establish

It is a **candidate-density measurement over 56 unannotated sheets of one
Bulgarian 1:25,000 archival series**, produced by detectors trained on three
sheets of that same series, plus the tooling around it. There is still no frozen
test split, so all of it remains development evidence.

Specifically not established:

- **Any recall or precision figure on the new sheets.** They have no labels.
- **That the candidates are mounds.** Crops show the expected symbol on most
  sheets and something less familiar on at least one. Only review settles it.
- **That the pipeline works on a genuinely different cartographic source.**
  63 sheets of one series cannot test it. The caveat is unchanged from v0.4.
- **That the urban failure mode is a defect.** `K-34-47-G-v` disagrees across
  checkpoints because none of them has a basis for a decision inside a city
  block, which is a description of missing training data, not of a bug.

---

## What was not done

**Step 2 (annotate the blind set)** — not attempted. It is human work, it is
blocking, and it is the only thing standing between this project and a test
result. All four frozen sheets are untouched: nothing in v0.6 rendered,
exported or summarised a proposal on any of them.

**Step 3 (model-assisted annotation)** — the tooling is built and verified end
to end through CVAT's own importer, and one sheet has been imported by hand.
The volume pass itself has not run.

**Step 4 (the first test result)** — blocked on step 2.

**A QGIS `.qml` style** — the plan proposed shipping one so the team's default
view filters to confidence ≥0.25. Not written; they can filter themselves, and
`detector_confidence` is on every feature.

**Retraining on the expanded corpus** — deferred to v0.7 by design, so the
frozen set is touched once rather than twice.

### Known debt

Three `sam2_backend` test failures, all environment-coupled assertions, and
repository-wide ruff errors outside the modules `make lint` covers. Neither
affects any conclusion above.

`data/curated/datasets/mapsam_det_v1/metadata/windows.jsonl` is stale against
its source: it was built before the `negative_type` patch landed in
`annotation/cvat/v0.0.2`, so annotation 235 reads `__undefined__` there and
`other` in the annotations. This pre-dates v0.6 and reproduces on master. v0.5
recorded the patch and noted no figure depends on it.

`src/georeference/` remains orphaned — excluded from the installed package,
importing repo-root `utils.py`, and requiring a `grid_25k.geojson` that is not
present. Its 30.5% pass rate cannot currently be reproduced, let alone improved.
Nothing in v0.6 depends on it; the sheets arrive already georeferenced.

---

## Reproduce

```bash
source ~/venvs/ai_archaeo_topia/bin/activate
LAKE=/mnt/c/Users/lubom/ai_archaeo_topia/data_lake

# Step 1 — sweep the 56 working sheets with all three checkpoints (~6 min)
python -m archeo_topia.analysis.sweep_sheets \
  --source "$LAKE/raw/mound_test_20260915" \
  --weights artifacts/detection/v0_5_yolo26s_gtfix/fold{A,B,C}/weights/last.pt \
  --output-dir artifacts/detection/v0_6_sweep --render-crops 8

# Schema migration to v0.0.3, and the CVAT label array
python -m archeo_topia.datasets.migrate_annotation_schema \
  --source annotation/cvat/v0.0.2/instances_default.json \
  --output annotation/cvat/v0.0.3/instances_default.json
python -m archeo_topia.formats.labels

# Clip the working sheets (~12 min)
for tif in $(find "$LAKE/raw/mound_test_20260915" -name '*_clipped.tif' -not -path '*_frozen*'); do
  sheet=$(basename "$tif" _clipped.tif)
  python -m archeo_topia.datasets.sheet_clips split \
    --source "$tif" --output-dir "$LAKE/cleaned/map_clips/dataset_03/$sheet" --grid 2x2
done

# Symbol outlines (~5 min on one RTX 5090)
python -m archeo_topia.analysis.decode_symbols \
  --candidates artifacts/detection/v0_6_sweep/foldC_last \
  --clips-root "$LAKE/cleaned/map_clips/dataset_03" \
  --output artifacts/detection/v0_6_symbols/foldC.json

# The GIS delivery
python -m archeo_topia.formats.export_gis \
  --candidates artifacts/detection/v0_6_sweep/foldC_last \
  --clips-root "$LAKE/cleaned/map_clips/dataset_03" \
  --symbols artifacts/detection/v0_6_symbols/foldC.json \
  --output "$LAKE/cleaned/gis/v0_6_foldC" --confidence 0.05

ogrinfo -q "$LAKE/cleaned/gis/v0_6_foldC/sheets/K-35-21-G-a.gpkg"
```

Needs `gdal-bin` on `PATH` (`make system-deps`), and `.[sam,detect]` for the
decoder pass. **Do not `pip install GDAL` to satisfy the GIS steps** — nothing
here imports `osgeo`; see the GDAL section of `README.md`.

Environment: torch 2.14.0+cu130, ultralytics 8.4.150, GDAL 3.4.1, one RTX 5090.

## Artifacts

| path | what |
|---|---|
| `analysis/sweep/` | per-sheet sweep summaries for all three checkpoints, cross-checkpoint agreement, aggregates |
| `analysis/gis/export_report.json` | per-sheet delivery counts |
| `analysis/MIGRATION.json` | the v0.0.3 schema migration report |
| `annotation/cvat/v0.0.3/` | the corrected export; v0.0.1 and v0.0.2 untouched |
| `annotation/cvat/labels.json` | the schema of record |
| `configs/splits/v0_6_splits.json` | the permanent splits |
| `artifacts/detection/v0_6_sweep/` | candidates and crops (gitignored) |
| `data_lake/cleaned/gis/v0_6_foldC/` | the delivery |

## Code added in v0.6

| module | what |
|---|---|
| `analysis/sweep_sheets.py` | detector over unlabelled sheets |
| `analysis/decode_symbols.py` | symbol outlines for sweep candidates |
| `datasets/migrate_annotation_schema.py` | v0.0.2 → v0.0.3 |
| `datasets/export_proposals_coco.py` | proposals → CVAT COCO |
| `datasets/sheet_clips.py` | sheet → clips, with geo provenance |
| `formats/` | `LabelSchema`, `CocoDocument`, `SheetReference`, `FeatureCollection`, `export_gis`, `ingest_review` |

684 tests pass; the 3 failures are the pre-existing `sam2_backend` ones.
