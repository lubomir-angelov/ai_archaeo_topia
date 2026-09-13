# MapSAM v0.5 — first detector: localization solved, recall not

Executes steps 2, 3 and 4 of `PLAN.md`. Step 5 (end-to-end through the frozen
512 decoder) is written and blocked on an install; steps 1, 6, 7 and 8 were not
attempted. See *What was not done* at the end.

Same data as v0.1–v0.4: `annotation/cvat/v0.0.1/instances_default.json`,
3 sheets / 12 clips / 180 mound annotations, split by sheet on the v0.4 fold
definitions.

---

## Summary

1. **The localization requirement is met on every fold, including the one that
   fails.** p90 centre error is 3.30, 3.73 and 2.00 source px against a target
   of ≤5. The detector puts its point inside the working band of v0.4's
   tolerance curve wherever it fires at all.
2. **Recall is the problem, and it is not a localization problem.** The recall
   curve is *flat* in radius on all three folds — recall@5px equals recall@25px
   to within one instance. There is no population of near misses. Every
   detection that lands, lands within about 5 px; every failure is a total
   failure to see the symbol. These are two different problems and the metric
   hierarchy separated them cleanly.
3. **Recall@5px varies enormously by sheet: 1.000, 0.859, 0.389** at the
   recall-oriented operating point. The macro mean over folds is 0.749 and the
   instance-weighted rate is 151/180 = 0.839, but neither number describes any
   sheet. K-34-35-B-g is an outlier by a wide
   margin, and it is also the sheet with only 18 mounds.
4. **The failures concentrate on mounds with no accompanying relative height
   mark and crossed by a road — 0 of 9 found on fold C.** Those two attributes
   are confounded on this sheet at this sample size and cannot be separated.
   The height-mark association independently matches an unexplained v0.4
   finding, which raises it above a post-hoc story but does not establish it.
5. **The template baseline is far behind.** Its best cross-sheet recall@5px is
   0.88, but only at 73 false positives per window; at a tolerable ~2 FP per
   window it manages 0.07–0.18. YOLO26-s reaches the same recall at 0.03 FP
   per window. Whatever the learned detector has trouble with, it is not the
   part that classical correlation was going to solve.

**So the v0.4 handoff specification is satisfied and the problem has moved.**
v0.4 asked for a detector whose p90 centre error lands inside ~5 px. That
exists. What does not exist is a detector that finds the symbol on a sheet
whose mounds are printed without their usual companion glyph.

---

## Step 2 — the detection dataset

`src/archeo_topia/datasets/build_detection_windows.py`. Window 512, stride 384,
clamped flush at the right and bottom edges; one class; hard negatives carried
in metadata rather than trained as a class; `uncertain_ignore` excluded in both
directions.

The builder reproduces the plan's audit from the annotation file:

| | windows |
|---|---:|
| contain ≥1 mound | 129 |
| contain hard negatives, no mound | 270 |
| empty of both | 81 |
| **total** | **480** |

180 mound annotations, 8 merged components, 1 without geometry, 344 target
boxes written (each mound is seen by 1.9 windows on average through the
overlap). Every annotation appears in at least one window, and every one has at
least one window holding its box entire, so no miss below is attributable to
tiling. All untruncated targets round-trip from window to source coordinates
exactly.

The 270 hard-negative-only windows are the point: the design of teaching the
detector that these symbols are background needs no mining or sampling scheme.
It falls out of tiling.

## Step 3 — template-matching baseline

`src/archeo_topia/analysis/detect_template_baseline.py`. Eight k-means template
centroids over 32 px intensity-normalized crops from the fold's *training*
sheets, matched with `cv2.matchTemplate` / `TM_CCOEFF_NORMED`, local maxima,
merged at 10 px.

Best recall@5px per fold, and recall at a false-positive rate a reviewer could
plausibly absorb:

| fold | eval sheet | best recall@5px | at FP/window | recall@5px at ~2 FP/window |
|---|---|---:|---:|---:|
| A | K-35-51-B-a | 0.781 | 85.0 | 0.070 |
| B | K-35-8-G-a | 0.882 | 72.6 | 0.176 |
| C | K-34-35-B-g | 0.389 | 54.3 | 0.111 |

Normalized cross-correlation generates proposals with essentially no
discrimination: precision is around 1% wherever recall is useful. Note that
fold C is the weak fold for the template baseline too, at 0.389 — the same
figure the learned detector reaches, by a completely different route.

**Scope.** This is a cheap reference, not the best achievable classical method.
It has no rotation handling, no multi-scale search and a single global
threshold. A stronger hand-built matcher is not ruled out by this; what is
ruled out is the hope that the obvious cheap version is competitive.

Per-fold detail: `analysis/template/`.

## Step 4 — YOLO26-s

`src/archeo_topia/training/train_mound_detector.py`. `yolo26s.pt`, imgsz 512
native, 150 epochs, batch 16, seed 42, one class, three leave-one-sheet-out
folds. Candidates merged across overlapping windows at 10 px on centre distance
and confidence.

**No checkpoint selection.** The only validation split available is the
held-out sheet, so Ultralytics' `best.pt` would select on the evaluation.
Everything below is `last.pt`. This costs something: see the trajectory table.

### The headline table, at confidence 0.25

| fold | eval sheet | n | recall@5px | 95% CI | p90 error | FP/window |
|---|---|---:|---:|---|---:|---:|
| A | K-35-51-B-a | 128 | 0.812 | 0.736–0.871 | 3.2 px | 0.21 |
| B | K-35-8-G-a | 34 | 1.000 | 0.898–1.000 | 3.7 px | 0.00 |
| C | K-34-35-B-g | 18 | 0.389 | 0.203–0.614 | 2.0 px | 0.02 |

At the recall-oriented operating point (confidence 0.05) fold A reaches 0.859
at 0.52 FP per window. Full curves in `analysis/detector/`.

### The recall curve is flat in radius

| fold | recall@5px | @10px | @15px | @25px |
|---|---:|---:|---:|---:|
| A | 0.859 | 0.883 | 0.883 | 0.883 |
| B | 1.000 | 1.000 | 1.000 | 1.000 |
| C | 0.389 | 0.389 | 0.389 | 0.389 |

This is the most informative single result here. Widening the acceptance radius
five-fold buys at most three instances across 180. The detector does not
produce imprecise points; it produces correct points or nothing. **Any further
work on localization accuracy would be optimizing something that is not
costing anything.**

It also retires a concern the plan carried. Step 7 — prompt-jitter augmentation
to widen the segmenter's tolerance — was made conditional on p90 landing
outside 5 px. It does not, on any fold. The trigger does not fire.

### Where the misses are

Recall@5px conditioned on annotation attributes, at confidence 0.05:

| attribute | fold A (n=128) | fold C (n=18) |
|---|---|---|
| `has_relative_height_mark` yes | 0.872 (117) | 0.857 (7) |
| `has_relative_height_mark` no | 0.727 (11) | **0.091 (11)** |
| `crossed_by_road` yes | 0.750 (32) | **0.167 (12)** |
| `crossed_by_road` no | 0.896 (96) | 0.833 (6) |
| `crossed_by_contour` yes | 0.780 (50) | 0.333 (9) |
| `crossed_by_contour` no | 0.910 (78) | 0.444 (9) |

Fold B is 1.000 on every subset and carries no information here.

On fold C the joint breakdown is stark:

| | not road-crossed | road-crossed |
|---|---|---|
| **has height mark** | 4/4 | 2/3 |
| **no height mark** | 1/2 | **0/9** |

**These two attributes cannot be separated on this data.** Nine of the eleven
no-height-mark mounds are also road-crossed, so the 0/9 cell is the whole
effect and either attribute explains it equally well. Fold A, where the
confound is much weaker, shows both effects in the same direction but small:
0.727 vs 0.872 for the height mark, 0.750 vs 0.896 for roads.

What raises the height-mark association above a post-hoc story is that it was
predicted. v0.4 measured a directional anisotropy in prompt jitter — northward
offsets cost about 1.4× less than westward — and recorded, without
investigating, that the plausible cause was "an associated element (an
elevation label, a neighbouring symbol) on one side". `has_relative_height_mark`
is true on 88% of all mounds but only 39% on K-34-35-B-g. Two independent
measurements now point at the same glyph. That is a hypothesis worth testing
directly, not a finding.

### Why K-34-35-B-g is hard, measured

It is not a mysterious sheet. It is elevated on four axes at once:

| | K-35-8-G-a | K-35-51-B-a | **K-34-35-B-g** |
|---|---:|---:|---:|
| mounds | 34 | 128 | 18 |
| `has_relative_height_mark` | 1.00 | 0.91 | **0.39** |
| `crossed_by_road` | 0.24 | 0.25 | **0.67** |
| `blurred_or_bad_print` | 0.21 | 0.16 | **0.50** |
| `crossed_by_contour` | 0.26 | 0.39 | **0.50** |
| median symbol box | 26.0×24.0 | 24.5×23.0 | 22.5×22.0 |

Half its mounds are badly printed, two thirds are crossed by a road, and most
lack the companion glyph the other two sheets almost always carry. It is also
the smallest sample, so it is simultaneously the hardest sheet and the one
whose recall figure is least stable.

### Epoch trajectory

Recorded, never selected on. Reported at confidence 0.25:

| checkpoint | fold A r@5px | fold B r@5px | fold C r@5px |
|---|---:|---:|---:|
| epoch 25 | 0.789 | 0.941 | 0.333 |
| epoch 50 | 0.664 | 0.912 | 0.333 |
| epoch 75 | 0.891 | 0.941 | 0.333 |
| epoch 100 | 0.805 | 0.941 | 0.333 |
| epoch 125 | 0.766 | 1.000 | 0.333 |
| last (150) | 0.812 | 1.000 | 0.389 |

Two things follow.

**Fold A swings ±11 points between checkpoints with no trend.** That spread is
as wide as its confidence interval, so its reported figure carries checkpoint
noise on top of sampling noise. Selecting `best.pt` would have reported 0.891
instead of 0.812 — which is exactly the leak the no-selection rule avoids, and
exactly the size of the error it would have introduced. This is the detector's
version of the late-training decision drift v0.4 traced in the decoder, and it
is the second concrete argument for step 8's permanent validation split.

**Fold C is flat from epoch 25.** It does not degrade and it does not improve.
Whatever is wrong there is not a training-length or overfitting problem, and
more epochs will not touch it.

Per-fold detail: `analysis/detector/`.

---

## What this does and does not establish

It is **within-series detection on three sheets of one Soviet 1:50k series**,
evaluated leave-one-sheet-out. As with v0.1–v0.4, none of it is a test result:
there is no frozen test split, so all of it is development evidence.

Specifically not established:

- **That recall@5px of 0.81–1.00 generalizes.** Two of three sheets reach it.
  The third does not, and there is no fourth sheet to say which is typical.
- **That the height mark is the cause of fold C's collapse.** It is confounded
  with road crossing at n=18.
- **Any false-positive rate at sheet scale.** The 12 clips were selected for
  annotation. FP per window and per megapixel are reported as measured on
  annotation-selected clips and should not be extrapolated to a full sheet
  until one exists.
- **That YOLO26-s is the right architecture.** It is the first one tried, and
  it cleared the localization target immediately. Nothing here compares it to
  Faster R-CNN + FPN or to a centre-heatmap head, and the flat recall curve
  suggests the remaining problem is not one a different box regressor solves.

---

## What was not done

**Step 1 (more sheets)** — unchanged and still the binding constraint on every
claim above.

**Step 5 (end-to-end through MapSAM)** — implemented in
`src/archeo_topia/analysis/mapsam_end_to_end.py`, not run. `segment-anything`
and `scipy` are absent from the project virtualenv; the v0.4 fold checkpoints
and the base SAM ViT-B checkpoint are all present. It runs two prompt arms — a
fixed median-sized box, which isolates localization error, and the detector's
own box, which is what deployment does — because v0.4's jitter sweep translated
the ground-truth box rigidly and explicitly left box *scale* untested.

**Step 6 (second-stage classifier)** — not triggered. At confidence 0.25 the
detector emits 0.00–0.21 false positives per window. There is no false-positive
problem for a filter to solve yet.

**Step 7 (prompt-jitter augmentation)** — not triggered, and now positively
contraindicated. Its condition was p90 outside 5 px; p90 is 2.0–3.7 px on every
fold.

**Step 8 (permanent grouped splits)** — still blocked on more sheets, and the
fold A trajectory above is a second argument for it.

---

## Reproduce

```bash
python -m archeo_topia.datasets.build_detection_windows \
  --coco-json annotation/cvat/v0.0.1/instances_default.json \
  --images-root data/curated/datasets/mapsam_v02/images \
  --output-dir data/curated/datasets/mapsam_det_v0 --window 512 --stride 384

python -m archeo_topia.analysis.detect_template_baseline \
  --dataset data/curated/datasets/mapsam_det_v0 \
  --images-root data/curated/datasets/mapsam_v02/images \
  --fold all --output-dir artifacts/detection/v0_5_template \
  --thresholds 0.4 0.5 0.6 0.7 0.8 0.9

python -m archeo_topia.training.train_mound_detector \
  --dataset data/curated/datasets/mapsam_det_v0 \
  --output-dir artifacts/detection/v0_5_yolo26s \
  --fold all --model yolo26s.pt --imgsz 512 --epochs 150 --batch 16 --save-period 25
```

Environment: torch 2.14.0+cu130, ultralytics 8.4.150, one RTX 5090. Note that
v0.1–v0.4 ran on torch 2.12; the detector work did not re-verify the decoder
results under 2.14.
