# MapSAM v0.6 — status, 26 September 2026

Where the project actually stands. `RESULTS.md` is the v0.6 experimental record
and does not change; this document tracks the current state of play, including
the delivery work done after `RESULTS.md` was written.

**One-line status.** Every model-side target is met and nothing further can be
learned from three annotated sheets. The corpus is swept, the schema is fixed,
the splits are permanent, and 57 sheets are packaged for the team. The project
is waiting on one thing, and it is human: **nobody has annotated the blind
test set, so there is still no test result.**

---

## What was done

### v0.6 proper — see `RESULTS.md` for the full record

| | |
|---|---|
| **Step 1, blind sweep** | The unchanged v0.5 detector over the 56 working sheets with all three fold checkpoints. 8,671 windows, 1,232.4 Mpx. Candidate density 1.31–2.10 per Mpx against 2.55 mounds/Mpx on the annotated clips — same order and lower, which is what whole sheets containing uninformative terrain should do. Excluding central Sofia, the three checkpoints agree to within **0.14 candidates per Mpx over 55 unseen sheets**. |
| **Step 2, annotate the blind set** | **Not attempted.** Human work, and the gate on everything. |
| **Step 3, model-assisted annotation** | Tooling built and verified through CVAT's own importer; one sheet imported by hand. The volume pass has not run. |
| **Step 4, first test result** | Blocked on step 2. |
| **Schema** | `crossed_by_water_line` → `water_line_crossing`, a four-value select (`none` / `surface` / `underground` / `unreviewed`) rather than the two booleans the plan asked for, because two booleans would manufacture 105 assertions no annotator made. `review_status` added as a separate axis from `annotation_provenance`. `annotation/cvat/labels.json` is now the machine-readable schema of record. Applied as `annotation/cvat/v0.0.3`: 714 annotations, geometry byte-identical. |
| **Splits** | `configs/splits/v0_6_splits.json`, grouped on the 1:100k parent rather than the 1:25k sheet id, because adjacent quadrants of one 1:50k sheet share terrain, survey campaign, print run and scan batch. |
| **Format classes** | `archeo_topia.formats` — `LabelSchema`, `CocoDocument`, `SheetReference`, `FeatureCollection`. One mask decoder replaces two that disagreed. GDAL appears only as a subprocess, enforced by a test. |
| **GIS delivery** | 56 sheets as GeoPackages in EPSG:25835, two layers each, verified by a full round trip from GeoPackage back to clip pixel with zero error. |

### After `RESULTS.md` — delivery work, not v0.6 experimental results

State this separation plainly when citing any of it: none of the below is an
experiment, and none of it changes a v0.6 number.

**A 57th sheet was added.** `K-35-75-B-g` arrived on 22 September and went
through the full chain — clip, sweep, symbol decode, export. It is in-domain
and, by the no-ground-truth health monitor, the cleanest sheet in the corpus:

| | K-35-75-B-g | 56-sheet corpus |
|---|---:|---:|
| candidates, three checkpoints at 0.25 | 59 / 58 / 64 | — |
| symbol area, median | 434 px² | 438 px² (annotated: 445) |
| detector point → mask centroid, median | **1.24 px** | 2.01 px |
| p90 | **3.08 px** | 4.26 px |
| instances above the ~5 px band | **0.0%** | 5.2% |

Its 1:100k parent `K-35-75` was already in the corpus through `K-35-75-B-a`,
its adjacent 1:25k quadrant, so the parent grouping decided its split rather
than a judgement call: **train**. The splits are now **46 / 13 / 5 over 64
sheets**.

**A review bundle was packaged for the team**, at
`data_lake/cleaned/gis/v0_7_review_bundle/` — 16 MB:

| | |
|---|---:|
| sheets in the main batch | 56 |
| proposals in the main batch | 2,513 |
| set aside (`K-34-47-G-v`, central Sofia) | 1,651 |
| **total** | **57 sheets, 4,164 proposals** |

Sofia is 40% of the corpus-wide proposal count on its own, against 118 for the
next-largest sheet, which is why it travels in `set_aside/` rather than in the
batch. Quote the main-batch figure, not the total.

**The reviewer's GIS form is generated from the schema.**
`archeo_topia.formats.styles` builds a QGIS `.qml` from
`annotation/cvat/labels.json` and writes it into each GeoPackage's
`layer_styles` table. Two reasons it works this way: generating from the schema
of record means a reviewer's dropdown cannot offer a value CVAT would reject —
the same drift that put five non-existent attributes into `PROTOCOL_EN.md` — and
a `.qml` beside the file is only picked up for single-layer data, so 57
two-layer GeoPackages would otherwise mean loading 114 styles by hand.

Reviewers get the 11 booleans as checkboxes, `review_status`,
`water_line_crossing` and `annotation_provenance` as dropdowns carrying exactly
the schema's values, the nine bookkeeping fields greyed out, and points coloured
by `review_status` so progress is visible without opening the attribute table.
All 57 GeoPackages were re-verified readable afterwards, and a simulated review
still round-trips through `ingest_review`.

**A master's thesis draft was written** from the repository's experiment record,
at `docs/thesis/THESIS_DRAFT_{EN,BG}.md`. English is the source of truth.

---

## Verified state of the gate

The blind test set is intact, checked rather than assumed:

- No file under `data_lake/cleaned/map_clips/dataset_02/` or
  `data_lake/raw/mound_test_20260915/_frozen/` has been modified since it was
  created on 15 September.
- No artifact anywhere under `artifacts/` names `K-35-22-A-v`, `K-35-39-G-v`,
  `K-35-39-V-g` or `L-35-139-V-v`. No model has seen them.
- `annotation/cvat/` still holds exactly `v0.0.1`, `v0.0.2`, `v0.0.3`. No
  export carries an annotation on a frozen sheet.

The frozen data lives in two places, which is easy to miss: the parent
GeoTIFFs at `data_lake/raw/mound_test_20260915/_frozen/` (165 MB, 4 sheets) and
the clips to annotate at `data_lake/cleaned/map_clips/dataset_02/` (120 MB, four
RGBA PNGs plus `clips.json` per sheet). The rasters were moved rather than
deleted because the clips carry no CRS, so without them a test-set detection
could not be put on a map.

**Nothing has come back from the team yet.** `ASSIGNMENTS.csv` has no claimed
rows and `returned/` is empty.

---

## Next steps

Ranked by what blocks what.

1. **Annotate the four frozen sheets, blind.** Four sheets are set aside, the
   splits exist, the clips carry their own geotransform. What is missing is a
   person annotating them with no model output visible. Until that happens every
   number this project has produced — v0.1 through the sweep above — is
   development evidence, and no amount of further engineering changes that.
   Nothing else on this list is worth as much.
2. **Get the review bundle in front of the team.** Decide first whether they
   already hold the supplier-georeferenced sheets: the GeoPackages carry absolute
   EPSG:25835 coordinates, so if they do, the 16 MB bundle is sufficient on its
   own. If they do not, the rasters are needed — 2.5 GB as delivered, or about
   290 MB as JPEG-compressed COGs with overviews (measured: 27 MB → 5.1 MB on one
   sheet). One file per sheet is what makes concurrent review safe; a GeoPackage
   is a single SQLite file and two people editing one copy on shared storage lose
   work silently.
3. **Decide what a rejection is worth before the reviews arrive** — see *Known
   gaps* below. Re-collecting a `negative_type` after the fact means a second
   pass over the same symbols.
4. **Retrain on the expanded corpus**, after the blind set is annotated, so the
   frozen sheets are touched once rather than twice.
5. **A second cartographic series.** Sixty-odd sheets of one Bulgarian 1:25k
   series cannot retire the cross-cartographic caveat; only a different series
   can. Unchanged since v0.4.
6. **Attribute extraction**, whose labels already exist on all 714 annotations.
   With 180 positives the frequent attributes are learnable and the rare ones —
   `crossed_by_powerline` at 4 instances — are not.
7. **A segmentation-quality gate.** Two candidate signals identified, neither
   validated: SAM's discarded `iou_predictions` head, which must be validated
   before it is trusted because it was never in the loss while the decoder around
   it moved, and mask stability under small prompt perturbations.

Still not worth doing, with reasons unchanged from `RESULTS.md`: prompt-jitter
augmentation (contraindicated by the flat recall curve), a second-stage
classifier (0.00–0.36 false positives per window leaves nothing to filter),
further localisation work, and architecture comparisons until a failure exists
that they could address.

---

## Known gaps

**Rejections are counted and then discarded.** `ingest_review.to_coco` keeps
`confirmed`, `corrected` and `added`. A rejected proposal contributes to the
precision figure in `review_report.json` and becomes no annotation of any class —
verified on a simulated review where 10 rejections produced 54 `mound`
annotations and an empty `hard_negative_symbol` category. Those rejections are
the best negatives available, being symbols this detector actually fired on
rather than the 530 hand-picked before any model existed. `to_coco` already takes
a `keep` argument, so routing them is small.

**The GIS form has no `negative_type`.** It is generated from the `mound` label
only, so even if rejections were kept they would arrive untyped — and reporting
error *by negative type* is the whole point of the typed negatives. Also small,
in `styles.py`, but it has to happen before reviewing starts.

**`configs/splits/v0_6_splits.json` is named v0_6 and its `version` field now
reads `v0.7`.** Renaming means touching `DEFAULT_SPLITS` and several documents.
Recorded rather than fixed, because a filename mismatch that is written down is
better than a path change nothing is ready for.

**The v0.7 review bundle has no `docs/` record of its own.** This document is
currently it.

**Unchanged debt from `RESULTS.md`:** three environment-coupled
`sam2_backend` test failures (698 pass, 3 fail, 1 skip as of today);
`mapsam_det_v1/metadata/windows.jsonl` stale against its source in one
attribute, pre-dating v0.6 and depended on by no figure; `src/georeference/`
orphaned, with its 30.5% pass rate not currently reproducible.

---

## Provenance

| what | where |
|---|---|
| v0.6 experimental record | `RESULTS.md` |
| the plan this executed | `PLAN.md` |
| sweep detail | `SWEEP_FINDINGS.md`, `analysis/sweep/` |
| corpus measurements before any model ran | `INPUT_INVENTORY.md` |
| the synthesis across v0.4–v0.5 | `../REPORT_2026-09.md` |
| target architecture | `../../architecture/TARGET_PIPELINE.md` |
| schema of record | `annotation/cvat/labels.json` (v0.0.3) |
| permanent splits | `configs/splits/v0_6_splits.json` (46/13/5 over 64) |
| review bundle | `data_lake/cleaned/gis/v0_7_review_bundle/` |
| frozen blind test set | `data_lake/raw/mound_test_20260915/_frozen/` and `data_lake/cleaned/map_clips/dataset_02/` |
| thesis draft | `../../thesis/THESIS_DRAFT_EN.md` |

Environment: torch 2.14.0+cu130, ultralytics 8.4.150, segment-anything 1.0,
GDAL 3.4.1, one RTX 5090.
