# MapSAM v0.6 — test protocol for the frozen blind set, pre-registered

Written 22 November 2026 and committed **before any model output or metric exists on the frozen
sheets**. It fixes what will be measured, with which models, at which settings,
and how often the set may be opened. Anything this document does not fix is not
part of the test result, and anything done differently from it is reported as a
deviation.

**Why this document exists.** Every number in v0.1–v0.6 is development
evidence. The four frozen sheets are the project's first ground truth that no
model has seen and no model output helped to draw. They can be spent once. A
test set that is consulted while choosing a threshold, a checkpoint or a
matching rule turns into a second validation set, and then its numbers say
nothing that the development numbers did not.

---

## The set is opened exactly twice

| opening | what is scored | artifacts | when |
|---|---|---|---|
| **1 — `v0_6_test_baseline`** | the v0.5 detector of record, all three fold checkpoints, plus end to end through the v0.4r decoder | `artifacts/detection/v0_6_test_baseline/` | once the prerequisites below have landed and been reviewed |
| **2 — `v0_6_refit`** | the detector refit on the permanent train split, its configuration fixed by this document | `artifacts/detection/v0_6_refit/` | once the refit has been trained and its checkpoint chosen on validation |

**An opening is any run that puts a model output and the frozen ground truth
into the same computation.** Inference and scoring happen in one run per
opening. Every threshold, every checkpoint and the full set of prompt arms are
scored in that single run, so nothing remains to be chosen afterwards.

- A run that fails **before it writes any metric** may be re-run, and the
  failure is recorded. A run that has produced a number *is* the opening.
- Re-running an opening to **reproduce** it is allowed and must give identical
  numbers. If it does not, that is a finding to report. It is not a second
  draw to choose from.
- The frozen set is **not** a validation set. It is not used for checkpoint
  selection, threshold selection, matching-rule selection, early stopping,
  hyperparameter tuning or error-driven data collection. This repeats
  `PLAN.md` step 2 and is restated here because this is where it binds.

**Between the two openings the scoring code is frozen.** The one exception is a
bug fix, meaning code that demonstrably does not do what this document says. A
bug fix lands as its own reviewed change. **Both openings are then re-scored**
with the fixed code, the change and its effect on both are recorded in
`TEST_RESULTS.md`, and the pre-fix numbers are kept beside the post-fix ones.
They are not overwritten. Changing a metric, a radius, a threshold or a
denominator because a result looked wrong is not a bug fix.

**Results go to `docs/mapsam/v006/TEST_RESULTS.md`.** `RESULTS.md` is the v0.6
experimental record and stays closed. `STATUS.md` gets a pointer.

---

## Prerequisites before opening 1

Each item is a defect or a missing capability found in the code. None is
optional, and none may be worked around inside the opening run. The protocol is
committed first, on purpose: the code lands afterwards as reviewed changes that
implement exactly what this text says, and opening 1 runs only after the
reproduction check in C1 passes.

**C1. A test-set entry point.** Prediction and scoring are hard-wired to the
LOSO folds. `predict_fold` reads `<fold>_val.txt`
(`train_mound_detector.py:159`), `evaluate_fold` scores only `FOLDS[fold]`
(`:209-217`), and `run_fold` pairs `detector_run/<fold>` with decoder `<fold>`
and scores only that fold's sheet (`mapsam_end_to_end.py:277-283`, `:303`).
None of these can score a checkpoint over four test sheets.

Required: one scorer, taking an explicit window listing, an explicit sheet set
and explicit weights, plus explicit (detector, decoder) pairs for end to end.
The LOSO path becomes one caller of it, the test path another, and the refit's
validation selection a third. A parallel copy of `evaluate_fold` would leave
three hand-maintained scorers that drift.

Check before opening 1: the generalized scorer reproduces the v0.5 gtfix fold
metrics and the v0.5 end-to-end table exactly on `mapsam_det_v1`. The one
permitted difference is the C3 fix, and it is enumerated.

**C2. No LOSO listings from a frozen or refit build.** `_write_folds` always
writes `foldA/B/C_{train,val}.txt` and YAMLs from the hard-coded `FOLDS`
(`build_detection_windows.py:548-580`, `:109-113`). On a frozen-only export no
window belongs to a LOSO evaluation sheet, so every frozen window would be
listed under `fold*_train.txt`, with a `fold*.yaml` that trains on it beside
it. `build_report.json` would also claim LOSO folds (`:510-513`).

Required: listings follow from a declared split. The frozen build writes a
`test` listing and no training listing of any kind. The refit build writes
train and val from `v0_6_splits.json`, and stops if any `test`-split sheet
appears in either.

**C3. Ignore and hard-negative regions are rebuilt from window-clipped
boxes.** `load_annotations` restores each region from the boxes clipped into
every window it touches, keyed by `(image, annotation_id)`, with the last
window winning (`detection_metrics.py:479-494`; clipping at
`build_detection_windows.py:433-440`). A region crossing a window edge is
restored as whatever fragment the last window held. On `mapsam_det_v1`, ignore
annotation 273 on `K-34-35-B-g_4.png` reconstructs to two different boxes, and
47 of 542 hard-negative regions reconstruct to more than one box.

Required: the builder records each region's **source-coordinate** box once and
the evaluator reads that. The v0.5 published numbers are not regenerated. The
C1 reproduction check reports whether any v0.5 figure would move, and by how
much. The frozen ground truth contains no `uncertain_ignore` regions, so this
fix cannot move a test number through ignores. It still lands before opening
1, because the hard-negative attribution reads the same reconstruction and
because the refit's validation sheets are scored by the same code.

**C4. End to end applies ignore regions.** `run_fold` discards the ignore list
(`mapsam_end_to_end.py:278`), so a candidate inside an `uncertain_ignore`
region is counted in `false_masks` (`:334-336`). End to end gets the same
neutralization as detection, with the count reported. The frozen ground truth
contains no `uncertain_ignore` regions, so this fix cannot move a test
end-to-end number either. It lands before opening 1 because the refit's
validation sheets are scored end to end by the same code.

**C5. The metrics this protocol pre-registers.**

- Wilson intervals on precision and on the end-to-end IoU ≥ 0.5 and ≥ 0.75
  rates.
- F1 at 25 px.
- Median IoU and found-only mean IoU in the output. Found-only is currently
  derived by hand in `../v005/RESULTS.md:337-341`.
- Pooled instance-weighted, macro and per-sheet blocks.
- Explicit per-threshold blocks. `evaluate_fold` stores the **lowest**
  threshold's metrics as the top-level `metrics.json`
  (`train_mound_detector.py:255-262`). The test output keys every block by its
  threshold, so nobody reads the 0.05 block as the headline.
- The paired discordant counts with an exact binomial p-value.

**C6. Input pinning.** A manifest step writes
`docs/mapsam/v006/analysis/frozen_manifest.json` with the sha256 of the
ground-truth JSON, the 16 PNGs, the four `clips.json`, the three baseline
`last.pt`, the three decoder `final.pt`, the SAM base checkpoint and the frozen
windows dataset's `build_report.json`. Every opening verifies it and stops on
mismatch. Hashing is byte-level and is not an opening. The manifest step reads
the `instances_Test.json` sha256 recorded in `SOURCE.md`, recomputes it from the
file, and stops on mismatch before writing anything. `frozen_manifest.json` is
the only place the full hash table exists. No document copies it.

**C7. Builder input guards for the frozen export.**

- Every category name is one of the three in `annotation/cvat/labels.json`.
  Today an unknown name is bucketed silently and never read
  (`build_detection_windows.py:347-352`, `:379-381`).
- The COCO `width` and `height` equal the PNG's. Today windows are planned from
  COCO dimensions and cropped from the PNG (`:375`, `:392-393`).
- All 16 clips named by the four `clips.json` are present in the export, so a
  clip with no annotation is still tiled and its false positives counted.

**C8. Device and attribution honesty.** `torch.device(args.device if
torch.cuda.is_available() else "cpu")` (`mapsam_end_to_end.py:429`) moves to
CPU without saying so. An opening fails if the requested device is
unavailable, and records the device it used. Separately, false-positive
attribution substitutes the literal `"untyped"` for a missing `negative_type`
(`detection_metrics.py:329`). Absence is reported as its own count, not as an
invented label. This is fixed in the same change as C5.

**Refit-only prerequisites (before opening 2, not opening 1).**

- The builder accepts a single `--images-root`
  (`build_detection_windows.py:301-319`). The training clips span
  `data/curated/datasets/mapsam_v02/images` and
  `data_lake/cleaned/map_clips/dataset_03/<sheet>`, so it needs multi-root
  image resolution.
- A split-aware build with the test-split guard of C2.
- Checkpoint selection through the C1 scorer on the validation split.
- `train_fold` reads `<fold>.yaml` (`train_mound_detector.py:113`) and must
  accept a split-named data YAML.

---

## Frozen inputs

| what | path |
|---|---|
| ground truth | `annotation/cvat/frozen_test/v0.0.1/instances_Test.json`, the CVAT COCO export, consensus ground truth |
| clips | `data_lake/cleaned/map_clips/_frozen/<sheet>/<sheet>_{1..4}.png`, 16 RGBA PNGs, with `clips.json` per sheet |
| parent rasters | `data_lake/raw/mound_test_20260915/_frozen/`, used only to place test detections on a map afterwards, never in scoring |
| schema of record | `annotation/cvat/labels.json` (v0.0.4) |
| splits | `configs/splits/v0_6_splits.json` |

**Ground-truth identity** is recorded in
`annotation/cvat/frozen_test/v0.0.1/SOURCE.md`, the one hand-kept record of it.
This document does not copy the hashes.

**Every input is pinned by hash** in `analysis/frozen_manifest.json`: the
ground-truth JSON, the 16 PNGs, the four `clips.json`, the three baseline
detector weights, the three decoder checkpoints, the SAM base checkpoint and
the frozen windows dataset's `build_report.json`. The manifest does not exist yet (C6). No document copies its hashes. Each opening checks the manifest first and **stops on any
mismatch** rather than scoring against an input that has changed.

**Observation on `review_status`.** In `instances_Test.json`, `review_status`
is `confirmed` on 46 of 249 mounds and `unreviewed` on the remaining mounds and
on all 1,240 hard negatives, although the set was annotated jointly with
cross-checking. No scoring path reads `review_status`, so it affects no number.
The `uncertain_ignore` count on the test set is 0.

### The sheets

| sheet | parent | clips | note |
|---|---|---:|---|
| `K-35-22-A-v` | `K-35-22` | 4 | GIS-cut; shares no parent with any annotated sheet |
| `K-35-39-G-v` | `K-35-39` | 4 | GIS-cut |
| `K-35-39-V-g` | `K-35-39` | 4 | GIS-cut |
| `L-35-139-V-v` | `L-35-139` | 4 | programmatic 2×2; the only `L-35` sheet, and the least like anything the detector has seen |

**`K-35-39-A-g` is in the test split and contributes nothing.** It shares the
parent `K-35-39` with two frozen sheets, so the parent grouping holds it out of
training. It was not blind-annotated, so it is not tiled, not scored and not in
any denominator. Its review return from the v0.7 bundle, made by a reviewer who
did not annotate the frozen set, stays out of every training and validation
set.

**No baseline model has seen a test parent.** The three development sheets are
`K-35-51-B-a`, `K-35-8-G-a` and `K-34-35-B-g`, with parents `K-35-51`, `K-35-8`
and `K-34-35`. The test parents are `K-35-22`, `K-35-39` and `L-35-139`.

### The windows dataset

Built **once**, from the frozen clips and the frozen ground truth, by
`archeo_topia.datasets.build_detection_windows` at **window 512, stride 384**.
Windows are clamped flush at the right and bottom edges. Output goes to
`data/curated/datasets/mapsam_det_frozen_v001/`. The same build, pinned by its
own `build_report.json` hash in the manifest, serves both openings. It writes a
**test listing only** and no training listing of any kind (C2). The build stops
unless all 16 clips are present in the export with dimensions matching their
PNGs, and unless every category is one of the three in the schema of record
(C7).

---

## Models of record

### Opening 1 — the baseline

**Detector: `artifacts/detection/v0_5_yolo26s_gtfix`, `last.pt` of folds A, B
and C.** This is experiment 2 of `../v005/RESULTS.md`, trained on the corrected
`v0.0.2` labels, and it is the detector v0.6 actually used as current: it is
the one `PLAN.md` step 1 names and the one the 56-sheet sweep ran. YOLO26-s,
imgsz 512, 150 epochs, batch 16, seed 42, one class, no checkpoint selection.

**`v0_5_yolo26s`, experiment 1 on the uncorrected labels, is not scored.** It
was trained against twelve mills labelled as mounds. Runs before and after a
correction are separate experiments, and that rule is what keeps experiment 1
on record in `../v005/RESULTS.md`. It is not a reason to spend the test set on
a model the project already knows learned from wrong labels. Scoring it would
add a third checkpoint family to the opening and answer no question anyone is
asking.

**All three fold checkpoints are scored and reported separately.** Every fold
is equally unseen on these sheets, and the three are independently trained
replicates. Their spread is the only available measurement of how much of a
test number belongs to the checkpoint and how much to the method. A mean and
min–max across the three are reported **descriptively**, beside the three rows
and never in place of them. **No ensemble is scored.** Merging three
checkpoints' candidates is a method never developed or validated on
development data, and its first evaluation would be on the test set. If an
ensemble is ever wanted, it is first evaluated on validation sheets.

**End to end: detector fold X into decoder `v0_4r_loso512_foldX_pw20`.**
Matching letters are paired because each pair was trained on the same two
development sheets, so each is one leave-one-sheet-out pipeline. Decoder
checkpoint `final.pt`, SAM ViT-B base `sam_vit_b_01ec64.pth`, frozen encoders.
**Both prompt arms are scored:** `fixed`, a 25 × 23 px box with 4 px padding on
the candidate centre, and `detector`, the candidate's own box with 4 px
padding. Prompt window 512 source px, resized to 1024.

**The template-matching baseline is not scored.** `../v005/RESULTS.md` measured
it at about 1% precision wherever recall is useful. Adding it later would be a
third opening.

### Opening 2 — the refit

**Fixed by this document, before opening 1, and not revisited after it.**

| | value |
|---|---|
| model | `yolo26s.pt`, one class |
| label sources | exactly two, with no sheet in both: `annotation/cvat/v0.0.3` for the three development sheets `K-35-51-B-a`, `K-35-8-G-a`, `K-34-35-B-g` (their corrected ground truth of record), and `annotation/cvat/v0.0.5`, the first export cut from the v0.7 review returns, for every other sheet. The build stops if a sheet appears in both, or if a development sheet is missing from v0.0.3 |
| what counts as a mound | in v0.0.3, every `mound` annotation. In v0.0.5, a `mound` annotation whose `review_status` is `confirmed`, `corrected` or `added`. A v0.0.5 `mound` with `review_status` `rejected` is not a target and enters as background, like a hard negative. `hard_negative_symbol` annotations enter as background whatever their `review_status`. `uncertain_ignore` regions are neutralized exactly as in scoring |
| reviewed sheet | a sheet that is present in its label source and has no v0.0.5 `mound` annotation with `review_status` `unreviewed`. Only reviewed sheets contribute windows. A sheet that falls short is excluded whole, never partly, and is named with its reason in the build report |
| training sheets | the reviewed sheets of the `train` split of `configs/splits/v0_6_splits.json` (46 sheets), the three development sheets included |
| selection sheets | the reviewed sheets of the `val` split (13 sheets) |
| never used | every `test`-split sheet, including `K-35-39-A-g`; the build stops if one appears |
| windows | 512 / 384, the same builder |
| imgsz, epochs, batch, seed | 512, 150, 16, 42; early stopping disabled |
| checkpoints | exactly six candidates: the epoch 25, 50, 75, 100 and 125 saves and `last.pt`, which is the epoch-150 state. An epoch-150 periodic save, if one is written, is the same state and not a seventh candidate. Selection stops if any of the six is missing |
| checkpoint selection | each candidate is scored on the selection sheets by the scorer the test uses, at confidence 0.25. Pick the highest pooled validation recall@5px. If hit counts are equal, pick the lower pooled false positives per window. If those are also equal, pick the later epoch. **Ultralytics' `best.pt` is not used**: its fitness is mAP, and this project does not select on mAP. All six scores are reported |
| thresholds, radii | identical to opening 1, not re-tuned on validation |
| end to end | the refit detector into each of the three frozen `v0_4r_loso512` decoders, reported separately; the decoder is not refit |
| artifacts | `artifacts/detection/v0_6_refit/` |

**Why the decoder is held fixed.** Changing both ends at once would make the
number uninterpretable (`mapsam_end_to_end.py:33-34`). With the decoder
constant, the opening-2 minus opening-1 difference is attributable to the
detector. Refitting the decoder on v0.0.5 masks is a new experiment with its
own validation and its own protocol entry.

**What the validation sheets are, and the bias that remains.** The validation
sheets were reviewed thoroughly, including a deliberate search for mounds the
model had missed, so their labels are intended to be on par with the
frozen-set annotation. The residual bias is this: the review started from v0.5
proposals, so it was model-assisted and not blind. Checkpoint selection on
these sheets may therefore slightly favour checkpoints that behave like v0.5.
`TEST_RESULTS.md` reports the `human_added` share on the validation sheets as
the measure of how much the model-assisted start mattered (`PROTOCOL_EN.md`
§8.5). The frozen test set, annotated blind, is the judge.

Hard negatives, including rejections that come back from review as
`hard_negative_symbol`, enter training as background through tiling, exactly as
in v0.5. They are not a second class.

**A refit whose configuration differs from this table is not the pre-registered
refit.** If v0.0.5 turns out not to support it as written, for example because
a label source is missing a sheet the table needs, the deviation is written into
`TEST_RESULTS.md` **before** opening 2, with the reason. If the deviation is
made after opening 1, it says so. The list of excluded sheets and their reasons
is copied from the build report into `TEST_RESULTS.md` before opening 2.
Exclusion under the *reviewed sheet* rule is part of the pre-registered refit,
not a deviation.

---

## Thresholds and matching — the values of record

| | value | from |
|---|---|---|
| confidence operating points | 0.05, 0.10, 0.25, 0.50, 0.70 | `train_mound_detector.THRESHOLDS` |
| **headline operating point** | **0.25**; 0.05 reported second as the recall-oriented point | `../v005/RESULTS.md` tables; end to end at 0.25 |
| detector inference | imgsz 512, every window, emitted from confidence 0.05 | `train_mound_detector` defaults |
| cross-window merge | 10 px centre distance, highest confidence kept, within one clip | `detection_metrics.merge_detections` |
| ground-truth centre | the annotation's bounding-box centre | `build_detection_windows` |
| association | one to one, detections in descending confidence, each claiming its nearest unclaimed mound within **25 px** | `detection_metrics._match` |
| recall radii | 5, 10, 15, 25 px; **5 px primary** | `detection_metrics.RECALL_RADII` |
| localization acceptance | **p90 centre error ≤ 5 px** over matched detections | the v0.4 handoff target |
| false-positive attribution | an unmatched detection within 15 px of a `hard_negative_symbol` box is attributed to that box's `negative_type` | `detection_metrics.ATTRIBUTION_RADIUS` |
| end-to-end association | every candidate whose centre is within 25 px of a mound on both axes is scored against it; each mound keeps its best IoU over those candidates; a mound is missed iff that best IoU is 0 (amended 2026-11-23, below) | `mapsam_end_to_end.run_fold` |
| end-to-end IoU thresholds | 0.5, 0.75 | `mapsam_end_to_end.IOU_THRESHOLDS` |

**Why 0.25 is the headline.** The v0.5 tables of record and the whole
end-to-end measurement are at 0.25. 0.05 is reported second because it is the
setting used to make the models surface their confusions for the review-bundle
GeoPackages. The two are not co-primary, so that nothing can be chosen after
the fact.

**The operating curve stands in for a threshold-free metric.** All five
confidence points are reported for every checkpoint, each with recall@5px,
precision and false positives per window. That is the precision–recall curve at
the resolution the project has always used. **No AP or mAP is reported**,
because `PLAN.md` and `../v005/PLAN.md` rule it out. mAP rewards box overlap,
and the downstream requirement is a point within 5 px.

---

## How the ground-truth classes are scored

| class | role | rule |
|---|---|---|
| `mound` | target | the recall denominator; every mound annotation counts for detection, and those with a decodable mask also count end to end |
| `uncertain_ignore` | ignore region | a detection whose centre falls inside the region's **source-coordinate bounding box** is removed before matching, and counts as neither true nor false positive, in detection and in end to end alike; the count removed is reported (0 expected on the frozen set) |
| `hard_negative_symbol` | non-target | background; a detection on one is a false positive, attributed by `negative_type` |

**`iscrowd` means nothing here.** CVAT exports masks as `iscrowd=1` RLE, and
nothing in this repository runs `COCOeval`. In COCO's own evaluator a crowd
region is an ignore region. In this export it only says the mask is stored as
RLE. No scoring path reads the field, and none may start to.

**Ignore regions are bounding boxes, not masks.** That is the rule the existing
evaluator implements, and it is kept rather than refined now that the test is
in sight.

**Annotations without geometry.** A mound whose segmentation decodes to an
empty mask still counts for detection and is excluded from the end-to-end
denominator. The count is reported per sheet. A CVAT mask export is not expected
to contain any, so a non-zero count is investigated before anything is
published.

---

## Metrics

### Detection, per checkpoint, at each operating point

| metric | reported as |
|---|---|
| **recall@5px** (primary) | rate, hits / n, 95% Wilson interval |
| recall@10, 15, 25 px | rate, hits / n, Wilson |
| **p90 centre error** (acceptance) | px, with median, p99 and max |
| precision, at the 25 px association | matched / (candidates − ignored), Wilson |
| F1 at the 25 px association | from precision and recall@25px; secondary, for readers who expect it |
| false positives | count, per 512 px window, per megapixel, and by `negative_type` |
| attribute subsets | recall@5px for merged components, `has_trig_point`, `blurred_or_bad_print`, `crossed_by_contour`, with n |

**F1 is computed at 25 px, not 5 px**, so that both its inputs come from the
same association. Precision at 5 px would need a second matching pass with a
different meaning, and it is not computed.

**False positives are measured at sheet scale for the first time.** The frozen
clips are whole sheets cut in four, not regions picked for annotation, so these
are the first false-positive-per-megapixel figures in the project that are not
inflated or deflated by clip selection.

### End to end, per detector and decoder pair, per prompt arm, at 0.25

| metric | reported as |
|---|---|
| **IoU ≥ 0.5 rate** | rate, hits / n, Wilson |
| IoU ≥ 0.75 rate | rate, hits / n, Wilson |
| mean IoU, all scorable mounds | missed mounds count 0 |
| median IoU, all scorable mounds | |
| mean IoU, found mounds only | the segmentation quality conditional on detection, as in `../v005/RESULTS.md` |
| missed rate | mounds no candidate mask overlaps (best IoU 0), as in v0.5 |
| false masks | candidates with no mound within 25 px, ignored regions excluded |

**Amendment 2026-11-23 — end-to-end "missed" definition.** Made after the C1
reproduction check failed and before any test number existed. The pre-registered
rule (missed = no candidate within 25 px) disagreed with v0.5 only on fold A
mounds `K-35-51-B-a_3.png` #543 and #552, where a 25 px neighbour's candidate
segmented the neighbour, not the mound. The v0.5 rule was adopted (missed =
best IoU over candidates is exactly 0, found otherwise; mean IoU over found
mounds uses the same found set) so that C1 reproduces v0.5 exactly, with C3 the
only difference. "False masks" and the detector-level metrics are unchanged.

### Per sheet, pooled, and what four sheets can carry

Every metric is reported **per sheet** and **pooled over the four sheets**.

- **Per sheet:** the value with its Wilson interval and n.
- **Pooled, instance-weighted:** all mounds on the four sheets as one sample,
  with a Wilson interval. The interval treats instances as independent. They
  are not, since mounds on one sheet share a print, a scan and an annotator
  pass, so **it understates the uncertainty, and says so where it is printed**.
- **Macro:** the unweighted mean of the four per-sheet values, with the
  min–max range across sheets beside it.

**No interval is computed across sheets.** A bootstrap over four sheets has 35
distinct resamples and a t-interval on four points has three degrees of
freedom. Either would produce an interval with more apparent precision than the
design supports. Between-sheet variation is shown as the four values
themselves and their range. With four sheets that is the honest form.

The **sizing target** from `PLAN.md` was 75–100 mounds, and below 50 the pooled
recall interval is too wide to tell a working detector from a degrading one.
The pooled n is already known, 249 mounds in `instances_Test.json`, so the
pooled figures are not small-sample. **Per-sheet n is where samples are
small.** Each per-sheet value is printed with its n and Wilson interval. A
sheet below 50 mounds is marked as such beside its value. Its interval carries
the uncertainty, and the value is not described as a sheet-level effect.

### Comparing the refit against the baseline

The comparison is **paired**: the same mounds, scored by two detectors. For
recall@5px at the headline operating point, against each baseline checkpoint,
`TEST_RESULTS.md` reports the discordant counts: mounds found by the refit and
missed by the baseline, and the reverse. It also reports the exact two-sided
binomial (McNemar) p-value on those counts.

**Wording is fixed in advance.** The detector claim is decided at confidence
0.25 on pooled recall@5px. Let *R* be the refit's value. Let *B*max, *B*min and
*S* = *B*max − *B*min be the highest and lowest of the three baseline
checkpoints and the spread between them. *p*A, *p*B, *p*C are the paired
two-sided exact McNemar p-values against each checkpoint. Exactly one of these
applies:

1. *R* > *B*max, *p* < 0.05 against all three, and *R* − *B*max ≥ *S*: write
   "**The refit improves recall.**"
2. *R* < *B*min, *p* < 0.05 against all three, and *B*min − *R* ≥ *S*: write
   "**The refit lowers recall.**"
3. Otherwise: write "**No difference between the refit and the baseline is
   detectable on this set.**"

Every case prints *R*, the three baseline values, *S*, the discordant counts and
the three p-values. Rules 1 and 2 require significance against all three
checkpoints at once (an intersection–union test), so 0.05 needs no multiplicity
adjustment. Requiring the margin to reach *S* keeps a difference within
checkpoint-to-checkpoint variation from being reported as an effect of
refitting. No other comparative wording about the refit's detection recall
appears in `TEST_RESULTS.md` or the thesis.

**Everything else is descriptive.** That covers the 0.05 operating point,
precision, F1, false positives and every end-to-end metric. These are reported
side by side with their paired discordant counts. End to end is paired per
decoder: the refit into decoder X against baseline fold X into decoder X. None
of them is described as an improvement or degradation caused by refitting. An
end-to-end claim would need its own rule written here before opening 1.

---

## Ground-truth corrections

If inspecting test-set errors shows that the ground truth is wrong, as
happened on `K-34-35-B-g` in v0.5, the correction is handled as follows.

- **No correction before opening 2 is scored.** Errors on the test set may be
  looked at after opening 1 for reporting, with qualitative overlays, and
  suspected label errors are logged. Nothing is corrected until both openings
  are scored against `v0.0.1`. A correction made in between would mean the
  refit is scored on labels that the baseline's errors shaped.
- A correction is a new version, `annotation/cvat/frozen_test/v0.0.2/`, with
  `v0.0.1` untouched. Both openings are re-scored against it, into new
  artifact paths, and reported as **a separate, labelled experiment** beside
  the original. The `v0.0.1` numbers are never regenerated or removed.
- **`v0.0.1` stays the headline.** A correction found by examining model output
  is model-assisted by construction: only the places where the model and the
  labels disagree get a second look, and every fix moves the labels towards the
  model. The corrected numbers are reported, and the direction of their bias
  is stated beside them.

**Context for how many errors to expect.** The v0.5 fold C label errors came
from an annotation-convention misunderstanding by the annotator, since
corrected. The same symbol type occurs on one of the frozen sheets and was
annotated there under the corrected convention. The frozen set was annotated
jointly and partly peer-reviewed, so errors are expected to be fewer than in
the original single-annotator labels. That is an expectation, not a measured
claim, and no number in this document depends on it.

---

## Disclosure

To be carried verbatim, or nearly so, wherever a test number is cited,
including the thesis.

> The blind set was annotated jointly by two project members, the project lead
> (who also developed the model) and a team reviewer, cross-checking each other
> during annotation, with no model output displayed on the test sheets. The
> team reviewer had previously reviewed model proposals on 25 other sheets of
> the same series (the v0.7 review bundle); the project lead had seen only
> aggregate development results, and developed the model. The result is a single consensus ground truth, so
> no inter-annotator agreement is reported.

What that does and does not license:

- **Recall on this set is blind exhaustive-annotation recall.** The annotators
  searched every clip for every mound with no candidate list in front of them.
  The review bundle's recall is measured on sheets whose review started from
  model proposals (model-assisted), so it is a different quantity. The two are
  never quoted as the same figure.
- **No frozen-set annotator reviewed a bundle sheet sharing a test parent.**
  The team reviewer reviewed 25 bundle sheets, none sharing a 1:100k parent
  with a test sheet (test parents `K-35-22`, `K-35-39`, `L-35-139`).
  `K-35-39-A-g`, the one bundle sheet sharing a parent with test sheets, was
  reviewed by a different reviewer who did not annotate the frozen set, and
  stays excluded from training and validation. The project lead, the other
  annotator, reviewed no bundle sheet and viewed no detector output on
  `K-35-39-A-g` before or during the frozen annotation.
- **Prior exposure is disclosed, not eliminated.** One annotator had reviewed
  this model's proposals on other sheets, and the other developed the model and
  saw its aggregate results; both therefore know broadly what it tends to find. With no output on the test
  sheets, that knowledge cannot copy a prediction into the labels. It could
  still shape what an annotator looks hard for. The design cannot measure this,
  and the text does not claim to have ruled it out.
- **Scope: one cartographic series, four sheets.** Bulgarian archival
  topographic maps at 1:25,000. This is a within-series test result. The
  cross-cartographic caveat open since v0.4 is untouched by it.

---

## What `TEST_RESULTS.md` will contain

One section per opening, each with: the manifest check, the code commit scored
with, the environment (torch, ultralytics, device), the detection table per
checkpoint and per sheet at 0.25 and 0.05, the five-point operating curve, the
end-to-end table per pair and per arm, and false positives by type. Opening 2
adds the validation trajectory that chose its checkpoint, the `human_added`
share on the validation sheets, and the paired comparison. Any deviation from
this document, any bug fix and its re-scoring, and any ground-truth correction
gets its own heading.

---

## Naming

| what | name |
|---|---|
| opening 1 artifacts | `artifacts/detection/v0_6_test_baseline/` |
| opening 2 artifacts | `artifacts/detection/v0_6_refit/` |
| frozen windows dataset | `data/curated/datasets/mapsam_det_frozen_v001/` |
| frozen ground truth | `annotation/cvat/frozen_test/v0.0.1/` |
| refit training export | `annotation/cvat/v0.0.5/` |
| review bundle | `data_lake/cleaned/gis/v0_7_review_bundle/`, path kept as delivered |

**Two labels drift, and both are recorded rather than renamed.** The review
bundle carries `v0_7` because it was packaged as v0.7 delivery work, while the
experiment that consumes it is named here under v0.6. And `STATUS.md` and
`RESULTS.md` call the retrain v0.7 work, while this protocol names its
artifacts `v0_6_refit` because it closes v0.6's step 4. Renaming paths that
reviewers already hold copies of would cost more than a sentence that explains
them.
