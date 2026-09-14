# MapSAM v0.5 — plan

Written at the end of v0.4, revised after a data audit against
`annotation/cvat/v0.0.1/instances_default.json`. Read `../v004/RESULTS.md`
first; this plan assumes its findings and does not repeat the evidence for
them.

> **Status.** Steps 2, 3 and 4 are executed twice — once against the
> `v0.0.1` annotations and once against `v0.0.2`, which corrects twelve
> mislabelled symbols the first run's failures exposed. Both are reported in
> `RESULTS.md` as separate experiments. Step 5 is implemented and unrun.
> Steps 6 and 7 did not trigger: p90 centre error is 2.0–3.7 px on every fold
> of both experiments and the false-positive rate is 0.00–0.36 per window, so
> neither a second-stage filter nor jitter augmentation has a problem to
> solve. With corrected labels, recall@5px is 1.000 / 0.898 / 1.000. **Step 1
> is now the only binding constraint** — both the amount of annotated data and
> its correctness, since one adversarial review of one sheet found twelve
> errors and one omission.

**v0.5 changes what the project is working on.** v0.1–v0.4 asked *"given the
correct mound location, can MapSAM segment it?"* That question is answered
well enough on the data available. v0.5 asks *"can the system find mounds?"*,
which nothing so far has attempted.

## Where v0.4 left things

- **512 px prompt-centred windows generalize across the three sheets.** All
  three leave-one-sheet-out folds improve 64–69% in source-pixel disagreement
  area; the cross-sheet spread falls 119.9 → 24.7 source px². Macro mean
  103.8 px² / 0.7922 decoder IoU, with 99–100% of instances clearing IoU 0.5
  on every sheet.
- **Most of v0.3's apparent domain variation was representational.** Sheets
  differed most where the mound fell furthest below one useful encoder token.
  A residual sheet effect remains and is no longer dominant.
- **Fold A's zero-IoU cases were quantization, not collapse.** 17 of 19 were
  sub-symbol displacements; at 512 all of them are gone, on every fold.
- **The window has ~250 px of placement slack. The prompt has far less.** A
  displaced window is nearly free; a displaced prompt kills the full-tile and
  512 models alike, at the scale of the symbol. The next section states how
  much less, because the round number v0.4 used has been misread since.

### What that is and is not scoped to

It is **consistent within-series conditional segmentation**. All three sheets
are one Soviet 1:50k series. Cross-cartographic generalization is untested and
the 24.7 px² residual may understate it badly. Nothing in v0.1–v0.4 is a test
result; all of it, LOSO included, is development evidence.

---

## The requirement v0.4 handed to v0.5, stated correctly

The jitter control arm separated two things that had always arrived together.
The localization requirement belongs to the **prompted segmentation
mechanism**, not to the crop: the window may be off by up to ~250 source px at
512 with no measurable cost, while the prompt must land on the symbol.

v0.4 summarized the prompt side as "about 15 px". **That number is the failure
boundary, not the acceptance threshold**, and writing it as a tolerance has
already produced one proposal to adopt `recall@15px` as the primary target.
The arm that matters is the one where the prompt and the window move together,
because that is what a detector-driven crop does:

| prompt offset (source px) | disagreement area (px²) | IoU≥0.5 rate |
|---:|---:|---:|
| 0 | 91.8 | 1.00 |
| 5 | 140.5 | 1.00 |
| 10 | 255.0 | 0.69 |
| 15 | 444.6 | **0.02** |
| 25 | 734.7 | 0.00 |

Read down the right-hand column. At 10 px the segmenter already fails on 31%
of instances; at 15 px it fails on 98%. A detector with p90 centre error of
14 px would satisfy "within 15 px" and put most of its tail in a band where
the downstream stage does not work.

So the specification v0.5 is built against is:

- **≤5 px — the working band.** Every instance clears IoU 0.5. Not free: the
  disagreement area is already +53% over a perfect prompt.
- **5–10 px — marginal.** Survival falls to 0.69.
- **>10 px — outside the useful range.** 15 px is effectively a miss.

**The detector target is therefore p90 centre error ≤ 5 source px, and the
primary metric is recall@5px.** Recall at 10, 15 and 25 px is reported because
the curve is what composes with v0.4's tolerance curve, but no acceptance
decision is taken on the 15 px figure.

`docs/architecture/TARGET_PIPELINE.md` carries the same correction.

### The measurement floor this target sits on

"The centre of a mound" has two defensible definitions and they disagree.
Over the 179 mound annotations that carry geometry, the distance between the
annotation bbox centre and the rasterized mask centroid is:

| median | p90 | p99 | max |
|---:|---:|---:|---:|
| 0.95 px | 1.85 px | 3.13 px | 3.28 px |

A 5 px p90 target therefore sits about 2.7× above the annotation's own
definitional ambiguity — tight, but measurable rather than noise-chasing.
Whichever definition the detector is trained against must be the one MapSAM is
prompted with, or up to 3.3 px of the 5 px budget is spent on a convention
mismatch.

---

## Data audit — the open questions are closed

Three things the earlier draft of this plan listed as unresolved or missing
have been checked against the annotation file and the images.

### The supervision is already in the repository

`annotation/cvat/v0.0.1/instances_default.json` holds all 713 annotations:

| Category | n | Geometry | Role |
|---|---:|---|---|
| `mound` | 180 | 160 RLE, 19 polygon, 1 none | positives |
| `hard_negative_symbol` | 530 | 529 bbox-only, 1 polygon | confusable negatives |
| `uncertain_ignore` | 3 | bbox-only | excluded everywhere |

Bounding boxes are present on all 713. **Detection does not need new data to
start.** Step 1 below is still the binding constraint on any robustness
*claim*, but it no longer blocks steps 2–5, and this plan is ordered
accordingly.

### The 180 vs 171 discrepancy is explained

Previously flagged as nine unaccounted-for training annotations. The
accounting is exact:

```text
180  mound annotations
 −1  empty geometry (K-34-35-B-g_3, a 16×8 px symbol truncated at the
     tile's top edge; it has a bbox but no mask)
=179 with geometry
 −8  merge events: touching mound polygons sharing one connected component
=171 training samples
```

One further annotation splits across two components, and one sub-20 px
fragment is dropped by `min_component_area`; these offset each other in the
rendered masks.

Consequences for v0.5, in order of importance:

1. **Detection recall is measured against 180 annotations, not 171 samples.**
   The eight merged pairs are precisely the cases a point-based system will
   collapse into one candidate, so they are a named evaluation subset, not a
   rounding error. `overlaps_other_mound` is true on 13 mounds and is the
   label-side view of the same phenomenon.
2. **The geometry-less annotation is a detection positive and a segmentation
   non-entity.** It can be detected and can never be scored end-to-end. Count
   it in detector recall, exclude it from the end-to-end denominator, and say
   which denominator any given number uses.

### Mound and negative geometry, measured

| | median | min | max |
|---|---:|---:|---:|
| mound bbox width | 25.0 px | 8.6 | 33.0 |
| mound bbox height | 23.0 px | 8.0 | 32.0 |
| hard-negative bbox width | 18.7 px | 6.8 | 66.2 |

Nearest-neighbour distance between mound centres, within an image:

| median | p10 | min |
|---:|---:|---:|
| 71 px | 25 px | 18 px |

No two mounds are closer than 15 px, so a **10–15 px merge radius for
cross-window deduplication is safe on this data**. But 17 of 180 mounds have a
neighbour within 25 px and 2 within 20 px, so **recall@25px must use strict
one-to-one matching** (Hungarian, or greedy by descending confidence with
matched ground truth removed) or it is inflated by detections matching the
wrong mound.

### Per-sheet counts, and what they do to a recall figure

| Sheet | mounds | hard negatives | v0.4 fold that evaluates it |
|---|---:|---:|---|
| K-35-51-B-a | 128 | 307 | fold A |
| K-35-8-G-a | 34 | 96 | fold B |
| K-34-35-B-g | 18 | 127 | fold C |

At a recall of 0.95 the 95% binomial CI half-width is ±3.8 points on fold A,
±7.3 on fold B and ±10.1 on fold C, where a single miss moves the figure 5.6
points. **Every per-fold recall number carries an interval or it will be
over-read.** The macro mean across folds is the headline; fold C alone cannot
support a claim either way.

### Tiling arithmetic on the current clips

At window 512, stride 384 (128 px overlap), over the 12 clips of roughly
2400×2200 px:

| | windows |
|---|---:|
| contain ≥1 mound | 129 |
| contain hard negatives, no mound | **270** |
| empty of both | 81 |
| total | 480 |

Two things follow. The hard negatives **arrive as background automatically**
— more than half of all windows contain one and no mound — so no special
negative-mining or sampling scheme is needed to realize the "teach it that
these are background" design. And only 5 of 180 mounds have no window placing
them at least 20 px from every edge, so **stride 384 is sufficient**; a
tighter stride buys almost nothing and costs inference time linearly.

---

## Constraints carried forward

- Image and prompt encoders stay **frozen**.
- **`pw20` and the loss crop stay frozen.** No further positive-weight search.
- **512 px is the default input representation**, with the 150 loss-crop
  margin at 256 logits.
- Split by sheet, never by sample, for detection as well as segmentation, and
  use v0.4's fold definitions so the two sides stay commensurable:

  | Fold | train sheets | eval sheet |
  |---|---|---|
  | A | K-34-35-B-g, K-35-8-G-a | K-35-51-B-a |
  | B | K-34-35-B-g, K-35-51-B-a | K-35-8-G-a |
  | C | K-35-51-B-a, K-35-8-G-a | K-34-35-B-g |

- **Report the scale-invariant metric.** Any cross-window comparison leads with
  disagreement area in source px². The IoU≥0.5 rate is a threshold on decoder
  IoU and is only readable within one window size.
- Treat the current 512 decoder as a **fixed downstream component** while the
  proposal stage is built. Changing both ends at once makes an end-to-end
  number uninterpretable.
- **Everything operates in source-tile pixel coordinates.** Georeferencing
  stays off the critical path, per `TARGET_PIPELINE.md` stage 5.
- Do not overwrite v0.1–v0.4 artifacts.

### Recorded decision: AGPL dependencies are acceptable

Ultralytics YOLO26 and YOLO11 are AGPL-3.0-or-commercial; this repository is
MIT. The decision, taken explicitly, is to **accept the AGPL dependency**: the
work is carried out by a closed research team, and any user of the system will
be given access to the modified source on request, which is what AGPL §13
requires of a network-deployed derivative.

Two practical consequences to honour rather than rediscover later:

- Keep detector code behind an optional extra (`[detect]`) and out of the MIT
  library surface, so the licence boundary is visible in the packaging rather
  than only in a document.
- If a public network service is ever stood up on top of it, the source-offer
  obligation attaches to that service's users, not only to the team.

This decision is recorded so it is not relitigated each time a detector
dependency is chosen. It does not bind the eventual production detector: a
centre-heatmap head (see Deferred) is a few hundred lines of licence-clean
PyTorch if the constraint ever becomes inconvenient.

---

## The system this plan is building toward

```text
map sheet
  → overlapping 512 px windows     (stride 384; 27% contain a mound)
  → candidate generator            (proposals; recall-oriented)
  → cross-window point merge       (centre distance + confidence)
  → candidate filter/classifier    (hard negatives; precision-oriented)
  → prompt-centred 512 window      (v0.4's representation, ~250 px of slack)
  → MapSAM decoder                 (frozen; 0.79 macro IoU given a good prompt)
  → mound mask
```

The 180 `mound` annotations provide the positives. The 530
`hard_negative_symbol` annotations — decorative symbols, trig points, roads,
text, grid artifacts, colour-pencil marks — are exactly the confusing
negatives this stage needs, and they have never been used. v0.1–v0.4
deliberately kept them out of the decoder loss because "given a valid prompt,
segment the mound" is not the task they address. This is the task they
address.

---

## Step 1 — Acquire and annotate more sheets

**Unchanged in importance, changed in position.** It no longer blocks the work
below, but it still blocks every domain-robustness *claim* the work below
could make. `data/maps`, `data/georeferenced` and `data/cvat_exports` are
empty; `data/curated/datasets` holds only `mapsam_v0` and `mapsam_v02`.

Prefer several more sheets from this Soviet 1:50k series **and** several
genuinely different cartographic sources. All of v0.4's variation is within
one series.

Two data gaps to close while annotating: all three `uncertain_ignore`
annotations live on one sheet, so two of three folds have no ignore-mask
coverage in evaluation at all; and `K-34-35-B-g` contributes 18 mounds, where
one instance moves a recall rate by 5.6 points.

One representativeness gap that detection makes urgent: the 12 clips were
selected for annotation, so a "false positives per unit map area" figure
measured on them does not extrapolate to a whole sheet, which contains large
uninformative regions the clips may under-represent. Until a full sheet is
available, **report false positives per 512 px window and per megapixel, and
label the figure as measured on annotation-selected clips.**

## Step 2 — Detection dataset from the CVAT export

A builder that turns `instances_default.json` plus the 12 clips into a
windowed detection dataset, sheet-grouped on the fold table above.

- Window 512, stride 384, clamped at the right and bottom edges.
- One class, `mound`. Hard negatives are **not** a class: they appear in
  windows with no target, which is what teaches the detector that they are
  background. `hard_negative_symbol` boxes are still carried in the metadata
  so false positives can be attributed to a `negative_type` at evaluation.
- `uncertain_ignore` regions are excluded from both training targets and
  evaluation, in both directions: a detection landing on one is neither a true
  nor a false positive.
- A mound is a target in a window when its centre falls inside; boxes are
  clipped to the window. Record, per window, which mounds are truncated, so
  edge behaviour can be separated from detection failure.
- Emit the inverse transform per window so detections return to source-tile
  coordinates exactly. This is the same coordinate discipline
  `mapsam_window.py` already implements for the segmenter; reuse its
  conventions rather than inventing a second one.

Deliverable: a dataset builder, a manifest per fold, and a short count report
that reproduces the audit numbers above from scratch.

## Step 3 — Template-matching baseline

**Run this before the learned detector, not alongside it.** These are printed
cartographic symbols from one series; normalized cross-correlation against a
handful of representative mound templates is cheap and answers a question the
rest of the plan depends on: how consistent is the symbol within the series,
and how much of the problem is available without learning anything?

Implement with `cv2.matchTemplate` and `TM_CCOEFF_NORMED`. An earlier draft of
this plan specified `scipy.signal.fftconvolve` to avoid adding an OpenCV
dependency; that was wrong — `opencv-python` is already in `requirements.txt`
and in the `sam`/`sam2` extras, and `matchTemplate` *is* the operation being
described. Templates come from training-fold sheets only; evaluation is on the
held-out sheet, same folds, same metrics as step 4.

The result is informative in both directions. If NCC reaches a high
recall@5px cross-sheet, the learned detector has a real bar to clear and the
project has a fallback with no licence, no training and no GPU. If it
collapses on print-quality variation, contour crossings and overlapping
symbology, that is a measured statement of what the learned detector buys.

## Step 4 — YOLO26-s detector

One model, one class, 512 px windows, the three folds. `yolo26s` rather than
`m`/`l`/`x`: with 180 positives, extra capacity is extra variance, not extra
information. `yolo26n` is a reasonable smoke test first. YOLO26's STAL
small-target-aware label assignment is the specific reason to prefer it over
YOLO11 here, and YOLO11 is the fallback if the newer training recipe proves
unstable on a dataset this small.

Train at `imgsz=512` natively as the default, with `imgsz=640` as a one-line
ablation. The upscale puts roughly four stride-8 cells on a 25 px symbol
instead of three, but it also resamples a scanned halftone symbol — the exact
texture the detector has to read — so this is a question to measure rather
than assume. Run the ablation on one fold before committing.

Inference over overlapping windows, then merge in source coordinates on
**centre distance and confidence**, not box NMS: the output that matters is a
point, and box IoU between two 25 px boxes is a noisy quantity. Merge radius
10 px, which the nearest-neighbour audit shows cannot fuse two distinct
annotated mounds.

Confidence threshold is set low deliberately. This stage is recall-oriented;
precision is step 6's problem and the reviewer's.

### Metrics, in decreasing order of what they settle

1. **Recall@5px — primary.** Strict one-to-one matching between detections and
   annotations, per fold, with a binomial CI.
2. **p90 centre-location error** over matched mounds — the acceptance figure,
   target ≤5 px. Median is reported and is not the decision.
3. **Recall@10px, @15px, @25px.** The curve, for composition with v0.4's
   tolerance curve. At 25 px, one-to-one matching is mandatory.
4. **False positives per 512 px window and per megapixel**, labelled as
   measured on annotation-selected clips.
5. **False positives broken down by `negative_type`.** Aggregate precision
   hides the structure here. The 530 negatives are typed — `decorative_symbol`
   246, `trig_point` 136, `road` 97, `text` 29, `colored_pencil` 7, `grid` 6,
   `other` 5, `contour` 1, `__undefined__` 3 — and "we reject decorative
   symbols reliably and confuse trig points 40% of the time" is actionable
   where "precision 0.86" is not.
6. **The trig-point subset specifically.** `trig_point` is the second-largest
   negative class at 136 instances, while `has_trig_point` is true on 42 of
   the 180 mounds. The same printed element is a negative alone and an
   attribute of a positive when it sits on a mound. No quantity of data fixes
   a label geometry that is genuinely context-dependent, so measure it rather
   than average over it.
7. **The merged-pair subset.** The eight components holding two annotations,
   plus the 13 `overlaps_other_mound` mounds: how often does the merge step
   collapse two mounds into one candidate?
8. **mAP50-95.** Recorded for comparability with the literature. It is not the
   system objective and no decision is taken on it.

## Step 5 — End-to-end evaluation

Connect step 4's merged candidates to the frozen 512 decoder immediately —
before any detector tuning beyond a working baseline. Prompts derive from the
candidate point, the window is centred on it and clamped, per fold.

| Metric | Meaning |
|---|---|
| % annotated mounds yielding mask IoU ≥0.50 | practical recall |
| % annotated mounds yielding mask IoU ≥0.75 | strong result |
| missed-mound rate | system failure |
| false GIS-mound rate | system precision |

State the denominator explicitly: 179 mounds with geometry, not 180, and not
171 samples.

This is the first number in the project that answers the actual question.
Expect it below 0.79, and expect the shortfall to be predictable from step 4's
measured error distribution read through the tolerance table above. **If it is
not — if end-to-end quality is worse than the error distribution predicts —
something outside localization is wrong**, and that is worth knowing before
any detector tuning begins. Record
`detector_to_mask_centroid_distance_px` per instance while doing this; it is
the ground-truth-free production monitor described in `TARGET_PIPELINE.md`
stage 4, and this is the run that calibrates it.

## Step 6 — Second-stage classifier, *conditional on step 4*

Only if step 4's false-positive rate is too high to hand to a reviewer, and
only then. A 64/128 px crop around each candidate, classified mound vs
non-mound, is where the 180-vs-530 labelled comparison does its most direct
work, and it can be multi-class over `negative_type` rather than binary.

Do not build it pre-emptively. If the detector's precision is already usable,
a second model is a second thing to train, tune, version and explain.

## Step 7 — Prompt-jitter augmentation, *conditional on step 4*

Deliberately **not** scheduled before the detector exists.

Widening the segmenter's tolerance and tightening the detector's accuracy are
two routes to the same gap, and only one of them is known to be needed. If
step 4's p90 centre error lands at or under 5 px, there is no reason to train
MapSAM to accept larger offsets — and a reason not to: v0.4 step 2 found the
zero-IoU samples sat closer to their neighbours than average (median
nearest-neighbour distance 70 px against 98), and the audit above confirms
mounds can be 18 px apart. A decoder trained to be indifferent to a 30 px
offset is a decoder trained to be less certain which of two adjacent symbols
it was asked about.

So the trigger is step 4's measured error distribution: run this only if the
p90 lands outside 5 px, and then match the augmentation magnitude to the
measured distribution rather than to a round number.

The mechanism is already in place — `MapSamDataset` takes `prompt_jitter_xy`;
training needs a per-sample random draw rather than a fixed offset. Guard
against jittering the target out of the window: at 512 px clipping begins
around 240 px, and v0.4 measured GT area falling 470 → 291 source px² between
offsets 250 and 300. Log mean GT area per epoch.

## Step 8 — Permanent grouped splits

Once enough sheets exist: train / validation / frozen test, split by sheet.
Validation for checkpoint and hyperparameter selection; the frozen test
touched only at milestones.

v0.4 sharpened why. The late-training decision drift its step 2 traced — fold
A going from 2 zero-IoU at epoch 5 to 19 at epoch 50, with the peak logit
correctly placed but negative — is smaller at 512 but not absent; fold C still
loses 6.5% between its epoch-5 peak and epoch 50. It is not worth solving
directly in v0.5. It is worth recording, because early stopping against a real
validation split is the fix, and there is currently no leak-free way to choose
an epoch. The same hazard now applies to the detector, which has its own
checkpoint-selection problem and the same absence of a clean validation split.

---

## Environment

Verified at the time of writing: torch 2.12.0+cu130, CUDA available, one
RTX 5090 with 32 GB. **`ultralytics` and `opencv-python` are not installed.**
Step 3 needs neither; step 4 needs `ultralytics` added under the `[detect]`
extra described in the licence decision above.

## Deferred

**Faster R-CNN + FPN** is the second detector to benchmark if YOLO26
underperforms despite correct tiling — tiny objects, a small dataset, one
class, and accuracy over real-time speed is the regime a two-stage detector
with a feature pyramid handles well. **RT-DETR** is a legitimate architectural
comparison and a worse starting point: very small data, one simple class, and
an already complicated downstream pipeline all argue for the faster
experimental loop first. Run neither until step 4 has reported.

**A centre-heatmap (CenterNet-style) detector.** Conceptually the right
formulation: the requirement is a point and a probability, not a box, and
Gaussian peaks at annotated mound centres train directly against the metric
this plan cares about. It is deferred rather than rejected, because YOLO
yields the same information immediately from the box centre, and because
building a custom architecture before establishing that a standard one fails
is how a project spends a month on the wrong problem. It becomes a justified
v0.6 experiment if step 4's p90 stalls above 5 px or the box objective
misbehaves on symbols this small. It is also the licence-clean option.

**Window sizes below 512 (256, 384) for segmentation.** Deprioritized on
value, not evidence. Nothing in v0.4 shows a tighter window cannot help — even
512 puts only 2.76 tokens on a mound, and at zero prompt error more coverage
might reduce the ~92–116 px² residual further. But that optimizes something
already clearing IoU 0.5 on 99–100% of instances while the pipeline cannot
produce a prompt at all. Reopen if the detector becomes strong and
segmentation becomes the bottleneck again.

Also deferred: calibration; the segmentation-quality gate
(`TARGET_PIPELINE.md` stage 8); attribute extraction (stage 6); new losses;
encoder variants; encoder unfreezing; any further positive-weight search;
box-scale jitter, as opposed to the translation v0.4 measured.

## Known debt, unrelated to MapSAM

Seven pre-existing `sam2_mcp` / `sam2_backend` test failures, and
repository-wide ruff errors outside `src/archeo_topia` (`make lint` is scoped
to `src/services tests`, so `src/archeo_topia` is not covered by the Make
target). Neither affects the v0.4 conclusions; both are worth a separate pass.
