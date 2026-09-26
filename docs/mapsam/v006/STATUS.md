# MapSAM v0.6 — status, 26 September 2026

Where the project actually stands. `RESULTS.md` is the v0.6 experimental record
and does not change; this document tracks the current state of play, including
the delivery work done after `RESULTS.md` was written.

**One-line status.** Every model-side target is met and nothing further can be
learned from three annotated sheets. The corpus is swept, the schema is fixed,
the splits are permanent, 57 sheets are packaged, and **the team has started
reviewing them**. The return path now keeps rejections as well as confirmations,
so the review will produce training data and not only a precision figure. The
project is still waiting on one thing, and it is human: **nobody has annotated
the blind test set, so there is still no test result.**

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

**Rejections stopped being thrown away.** Until 26 September
`ingest_review.to_coco` kept only `confirmed`, `corrected` and `added`. A
rejected proposal contributed to the precision figure in `review_report.json`
and then became no annotation of any class — confirmed on a simulated review
where 10 rejections produced 54 `mound` annotations and an empty
`hard_negative_symbol` category. That discarded the most useful negatives the
project can obtain: symbols this detector actually fired on, as opposed to the
530 picked out by hand before any model existed.

Rejections now come back as `hard_negative_symbol`. The two classes share all 15
of `mound`'s attributes — `negative_type` is the only difference — so everything
a reviewer filled in transfers and only that one field has to be supplied.

Two schema values were needed, and the schema is now **v0.0.4**. Both additions
are additive: no existing annotation changes, no default changes, and the v0.0.3
export was checked to still validate against it.

| added | why |
|---|---|
| `negative_type = unreviewed` | A rejection says "not a mound"; it does not say which kind of symbol it is, and the reviewer's form never asked. `other` is a **real** category carrying 17 annotations, so defaulting to it would make those indistinguishable from symbols nobody has typed. Same reasoning that made `water_line_crossing` a four-value select rather than two booleans, and the value doubles as the work queue. |
| `annotation_provenance = model_proposal_rejected` | `export_gis` stamps every proposal `model_proposal_accepted` at delivery, before anyone has accepted anything, which is a contradiction on a feature that comes back rejected. Provenance is now derived from the verdict rather than trusted from the returned file. |

**No export migration was made, and none is needed.** The latest annotation
export is still `annotation/cvat/v0.0.3` while the schema of record reads
v0.0.4. That is a legitimate state rather than drift: the new values are only
*allowed* values, nothing existing uses them, and the 542 negatives keep the
real types they already have. The first export that will carry them is the one
cut from the returned reviews.

**The delivered GeoPackages were deliberately not changed.** Reviewers hold
downloaded copies and are editing them; adding a `negative_type` field would
mean re-downloading and losing work. They do not need it — they are judging "is
this a mound", not classifying non-mounds — and the typing is cheaper to do
later in CVAT, where the field already exists on the class and an annotator is
looking at the symbol anyway. This is why the fix was possible *after* reviewing
had started: it lives entirely on the return path and applies retroactively to
every file the team sends back.

Verified on the same simulated review that exposed the loss:

```
verdicts: {'confirmed': 9, 'corrected': 45, 'rejected': 10}
precision 0.844, recall floor 1.0
wrote …: {'mound': 54, 'hard_negative_symbol': 10}
```

with every negative carrying `negative_type=unreviewed`,
`annotation_provenance=model_proposal_rejected` and the reviewer's other 15
attributes intact. `to_coco(..., negatives=())` restores the old behaviour, so
output from before this change stays reproducible.

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

**Reviewing is under way and nothing has come back yet.** The team started on
the Drive copy, so progress is not visible from this repository: the local
`ASSIGNMENTS.csv` and `returned/` are the copies that were uploaded, and they
will stay untouched. The Google Sheet is the live one.

---

## Picking this up when the reviews arrive

Per sheet, once a reviewed `.gpkg` lands in `returned/`:

```bash
source ~/venvs/ai_archaeo_topia/bin/activate
LAKE=/mnt/c/Users/lubom/ai_archaeo_topia/data_lake
B="$LAKE/cleaned/gis/v0_7_review_bundle"
SHEET=K-35-21-G-a

python -m archeo_topia.formats.ingest_review \
  --returned "$B/returned/$SHEET.gpkg" \
  --original "$B/sheets/$SHEET.gpkg" \
  --clips-dir "$LAKE/cleaned/map_clips/dataset_03/$SHEET" \
  --output "artifacts/review/v0_7/$SHEET"
```

It writes `review_report.json` and `instances_default.json`, and logs the
verdict counts, precision, recall floor and the per-category annotation counts.

**Read `problems` in the report before anything else.** It is empty when the
file is sound. A non-empty list means a changed CRS, a duplicated or edited
`mound_id`, a feature moved off the sheet, one left `unreviewed`, or — the one
`review_status` exists to catch — a proposal deleted rather than marked
rejected. None of those should be worked around; they need the reviewer.

Then, in order:

1. **Scope the numbers correctly when writing them up.** This is field
   verification by domain experts on model-proposed candidates. It measures
   **precision** well, because every proposal gets a verdict. It measures
   **recall only as a floor**, over what reviewers independently noticed, which
   is not an exhaustive search. It is **not** the frozen blind test and cannot
   substitute for it. `review_report.json` carries that sentence in its `scope`
   field so it travels with the numbers.
2. **Check `K-35-39-A-g` separately if it comes back.** It is a working-pool
   sheet that the parent grouping puts in `test`, because it shares parent
   `K-35-39` with two frozen sheets. Its review is usable in the field and must
   **not** enter training. Every feature carries `split` for this reason.
3. **Import the COCO files into CVAT** to merge the verdicts beside the existing
   annotations, and cut a new annotation export version from the result.
   `annotation/cvat/labels.json` (v0.0.4) is the label set to import.
4. **Type the negatives, optionally.** Every returned rejection arrives as
   `negative_type=unreviewed`, which is the work queue: filtering on it in CVAT
   gives exactly the symbols nobody has classified. Worth doing only if error
   analysis *by negative type* is wanted — the detector is single-class, so
   training treats every negative as background regardless of type.
5. **Then retrain**, and only after the blind set has been annotated, so the
   frozen sheets are touched once rather than twice.

---

## Next steps

Ranked by what blocks what.

1. **Annotate the four frozen sheets, blind.** Four sheets are set aside, the
   splits exist, the clips carry their own geotransform. What is missing is a
   person annotating them with no model output visible. Until that happens every
   number this project has produced — v0.1 through the sweep above — is
   development evidence, and no amount of further engineering changes that.
   Nothing else on this list is worth as much.
2. **Ingest the reviews as they come back** — the procedure is above. Done:
   getting the bundle to the team, and deciding what a rejection is worth.
   If any reviewer turns out not to hold the supplier-georeferenced sheets, they
   need the rasters too: the GeoPackages carry absolute EPSG:25835 coordinates so
   any georeferenced copy of the same sheet lines up, but a plain scan has nothing
   to align to. That is 2.5 GB as delivered, or about 290 MB as JPEG-compressed
   COGs with overviews (measured: 27 MB → 5.1 MB on one sheet).
3. **Retrain on the expanded corpus**, after the blind set is annotated, so the
   frozen sheets are touched once rather than twice.
4. **A second cartographic series.** Sixty-odd sheets of one Bulgarian 1:25k
   series cannot retire the cross-cartographic caveat; only a different series
   can. Unchanged since v0.4.
5. **Attribute extraction**, whose labels already exist on all 714 annotations.
   With 180 positives the frequent attributes are learnable and the rare ones —
   `crossed_by_powerline` at 4 instances — are not.
6. **A segmentation-quality gate.** Two candidate signals identified, neither
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

**The current batch's rejections will arrive untyped.** This is a consequence of
the decision above, not an oversight: `negative_type` is absent from the
reviewers' form, so every rejection comes back `unreviewed` on that field. It
costs nothing for training, because the detector is single-class and negatives
are taught as background regardless of type. It costs the ability to report error
*by* type — "we reject decorative symbols reliably and confuse trig points 40% of
the time" — until someone types them in CVAT. Add `negative_type` to the form for
the **next** batch if that reporting matters; adding it now would make the team
re-download.

**No real reviewed file has been through the return path.** Everything above was
verified against simulated edits made through GDAL. The first genuine return is
also the first test of `ingest_review` against a file a person actually touched,
so read its `problems` list carefully rather than trusting the counts.

**`configs/splits/v0_6_splits.json` is named v0_6 and its `version` field now
reads `v0.7`.** Renaming means touching `DEFAULT_SPLITS` and several documents.
Recorded rather than fixed, because a filename mismatch that is written down is
better than a path change nothing is ready for.

**The v0.7 review bundle has no `docs/` record of its own.** This document is
currently it.

**Unchanged debt from `RESULTS.md`:** three environment-coupled
`sam2_backend` test failures (704 pass, 3 fail, 1 skip as of today);
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
| schema of record | `annotation/cvat/labels.json` (**v0.0.4**), CVAT array in `labels_cvat_raw.json` |
| permanent splits | `configs/splits/v0_6_splits.json` (46/13/5 over 64) |
| review bundle | `data_lake/cleaned/gis/v0_7_review_bundle/` |
| frozen blind test set | `data_lake/raw/mound_test_20260915/_frozen/` and `data_lake/cleaned/map_clips/dataset_02/` |
| thesis draft | `../../thesis/THESIS_DRAFT_EN.md` |

Environment: torch 2.14.0+cu130, ultralytics 8.4.150, segment-anything 1.0,
GDAL 3.4.1, one RTX 5090.
