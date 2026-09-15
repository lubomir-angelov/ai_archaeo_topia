# MapSAM v0.5 — first detector, and a ground-truth error it exposed

Executes steps 2, 3, 4 and 5 of `PLAN.md`. Steps 1, 6, 7 and 8 were not
attempted; 6 and 7 did not trigger. See *What was not done* at the end.

**This document reports two experiments, not one.** They differ only in the
annotations they were trained and evaluated against:

| | annotations | detector run | baseline run |
|---|---|---|---|
| **Experiment 1** | `annotation/cvat/v0.0.1` | `v0_5_yolo26s` | `v0_5_template` |
| **Experiment 2** | `annotation/cvat/v0.0.2` | `v0_5_yolo26s_gtfix` | `v0_5_template_gtfix` |

Model, hyperparameters, seed, folds and code are identical between them. Only
the labels changed, and they changed because experiment 1's failures turned out
to be annotation errors rather than model errors.

Experiment 1's numbers are reported exactly as they were measured. They are not
superseded and they have not been regenerated. The model declining to learn
from incorrect ground truth, and thereby surfacing the error, is a result worth
keeping — deleting it would destroy the evidence for it.

---

## Summary

1. **Localization was solved on the first attempt and stayed solved.** p90
   centre error is 2.0–3.7 source px in both experiments, against a target of
   ≤5. The recall curve is flat in radius on every fold in both: recall@5px
   equals recall@25px. The detector emits correct points or nothing, so there
   is no near-miss population and nothing for further localization work to fix.
2. **Experiment 1's one failing fold was a ground-truth error.** Recall@5px was
   1.000 / 0.859 / 0.389 by sheet. The failing sheet, K-34-35-B-g, is the one
   whose annotations were wrong: twelve symbols labelled `mound` are mills,
   confirmed with a second annotator, and one real mound was unlabelled.
3. **Eleven of the twelve mislabels are exactly the annotations the detector
   refused to fire on**, and the unlabelled mound is one the detector *did*
   find — at 1.8 px, confidence 0.586, scored as a false positive. On that
   sheet the evaluation penalized the model twice for being right.
4. **With the labels corrected, recall@5px is 1.000 / 0.898 / 1.000**, macro
   0.966 against 0.749. Fold C goes 0.389 → 1.000 (7/7).
5. **The error was not YOLO-specific.** The template-matching baseline,
   re-run unchanged on the corrected labels, moves on the same sheet from
   0.389 to 0.857. Two unrelated methods were penalized by the same twelve
   annotations in the same place.

**So the v0.4 handoff specification is met, and the binding constraint has
moved off the model entirely** — onto the quantity and correctness of the
annotated data.

---

## Step 2 — the detection dataset

`src/archeo_topia/datasets/build_detection_windows.py`. Window 512, stride 384,
clamped flush at the right and bottom edges; one class; hard negatives carried
in metadata rather than trained as a class; `uncertain_ignore` excluded in both
directions.

| | v0.0.1 | v0.0.2 |
|---|---:|---:|
| windows with ≥1 mound | 129 | 123 |
| windows with hard negatives, no mound | 270 | 276 |
| windows empty of both | 81 | 81 |
| **total windows** | **480** | **480** |
| mound annotations | 180 | 169 |
| target boxes written | 344 | 317 |
| merged components | 8 | 8 |
| mounds without geometry | 1 | **0** |

Every annotation appears in at least one window, and every one has at least one
window holding its box entire, so no miss below is attributable to tiling. All
untruncated targets round-trip from window to source coordinates exactly.

The 270-plus hard-negative-only windows are the point of the design: teaching
the detector that these symbols are background needs no mining or sampling
scheme. It falls out of tiling.

Note the last row. v0.0.1 had one mound annotation carrying a bounding box but
no mask — a symbol truncated at a tile's top edge — which is why the
segmentation manifest held 171 samples for 180 annotations and why end-to-end
scoring needed a 179-mound denominator. That annotation is one of the twelve
now relabelled, so under v0.0.2 every mound has geometry and the caveat is
gone.

---

# Experiment 1 — the v0.0.1 labels

## Step 3 — template-matching baseline

`src/archeo_topia/analysis/detect_template_baseline.py`. Eight k-means template
centroids over 32 px intensity-normalized crops from the fold's *training*
sheets, matched with `cv2.matchTemplate` / `TM_CCOEFF_NORMED`, local maxima,
merged at 10 px.

| fold | eval sheet | best recall@5px | at FP/window | recall@5px at ~2 FP/window |
|---|---|---:|---:|---:|
| A | K-35-51-B-a | 0.781 | 85.0 | 0.070 |
| B | K-35-8-G-a | 0.882 | 72.6 | 0.176 |
| C | K-34-35-B-g | 0.389 | 54.3 | 0.111 |

Normalized cross-correlation generates proposals with essentially no
discrimination: precision is around 1% wherever recall is useful.

**Scope.** This is a cheap reference, not the best achievable classical method.
It has no rotation handling, no multi-scale search and a single global
threshold. A stronger hand-built matcher is not ruled out by this; what is
ruled out is the hope that the obvious cheap version is competitive.

## Step 4 — YOLO26-s

`src/archeo_topia/training/train_mound_detector.py`. `yolo26s.pt`, imgsz 512
native, 150 epochs, batch 16, seed 42, one class, three leave-one-sheet-out
folds. Candidates merged across overlapping windows at 10 px on centre distance
and confidence.

**No checkpoint selection.** The only validation split available is the
held-out sheet, so Ultralytics' `best.pt` would select on the evaluation.
Everything reported is `last.pt`.

At confidence 0.25:

| fold | eval sheet | n | recall@5px | 95% CI | p90 error | FP/window |
|---|---|---:|---:|---|---:|---:|
| A | K-35-51-B-a | 128 | 0.812 | 0.736–0.871 | 3.2 px | 0.21 |
| B | K-35-8-G-a | 34 | 1.000 | 0.898–1.000 | 3.7 px | 0.00 |
| C | K-34-35-B-g | 18 | 0.389 | 0.203–0.614 | 2.0 px | 0.02 |

At the recall-oriented operating point, confidence 0.05: 0.859, 1.000, 0.389.
Macro mean 0.749; instance-weighted 151/180 = 0.839.

### The recall curve is flat in radius

| fold | recall@5px | @10px | @15px | @25px |
|---|---:|---:|---:|---:|
| A | 0.859 | 0.883 | 0.883 | 0.883 |
| B | 1.000 | 1.000 | 1.000 | 1.000 |
| C | 0.389 | 0.389 | 0.389 | 0.389 |

Widening the acceptance radius five-fold buys at most three instances across
180. This is the result that retires step 7: prompt-jitter augmentation was
made conditional on p90 landing outside 5 px, and it does not, on any fold.

### What experiment 1 concluded about fold C, and why it was wrong

The original write-up of this experiment reported that fold C's failures
concentrated on mounds with no `has_relative_height_mark` and `crossed_by_road`
— 0 of 9 on that intersection — and offered, as a hypothesis, that the detector
had learned "mound symbol plus companion height glyph" rather than the symbol
alone. That reading drew support from v0.4's unexplained directional
anisotropy, which had speculated about exactly such an adjacent element.

**That hypothesis is retracted.** The eleven missed annotations were not mounds.
The attribute correlation was real but incidental: mills sit along roads and do
not carry mound height marks, so "no height mark and crossed by a road" was a
description of the mislabelled class, not of a hard visual case. No mechanism in
the model needs explaining. v0.4's anisotropy remains unexplained and is now
unsupported by this evidence.

---

# The ground-truth error, and how it surfaced

Reviewing experiment 1's fold C failures, the annotator found that twelve
symbols labelled `mound` on K-34-35-B-g are mills, confirmed with a second
project annotator, and that one genuine mound on the same sheet had never been
labelled. `annotation/cvat/v0.0.2` corrects both.

## The correction lines up with the model's behaviour almost exactly

Experiment 1's fold C had **18 annotations and 7 detections**. Of the 11 it
missed:

| relabelled to `hard_negative_symbol` | detector behaviour in experiment 1 |
|---|---|
| 8 symbols on `_3` in a corridor at x 1500–1651, y 574–1103 | all missed |
| 1 on `_1` at (1454, 1739) | missed |
| 1 on `_1` at (733, 1596) | missed |
| 1 on `_2` at (2315, 446) | missed |
| 1 on `_3` at (1275, 52) — the box-without-mask edge symbol | **detected** |

Eleven of the twelve mislabels are precisely the annotations the detector
declined to fire on. The eight on `_3` lie in a narrow corridor running roughly
north–south, which is what a line of mills along a road looks like and is not
what a scatter of burial mounds looks like.

The twelfth is an honest exception and is recorded as one: the detector fired
on the truncated edge symbol at (1275, 52), agreeing with the incorrect label.
Under v0.0.2 that detection becomes a false positive.

## The unlabelled mound

The newly annotated mound sits at (1771, 813) on `K-34-35-B-g_3`. In experiment
1 the detector had already placed a candidate at **(1772.5, 812.0) — 1.8 px
away, confidence 0.586**, the second-strongest detection on the sheet. With no
annotation there, it was scored as a false positive.

So on this sheet experiment 1 charged the model 11 false negatives for correctly
rejecting mills and 1 false positive for correctly finding a mound.

## Why this is a model result and not only a data-cleaning anecdote

Two things make it more than a lucky catch.

**It was localized to one sheet by the metric design.** Recall and localization
were reported separately, per fold, with subsets. Fold C's p90 centre error was
2.00 px — the *best* of the three folds — while its recall was 0.389. A detector
that had genuinely failed on that sheet would be expected to degrade on both. A
single aggregate score, or mAP alone, would have shown one mediocre number and
given no reason to look at the annotations.

**It reproduces under a completely different method.** The template baseline,
unchanged, moves from 0.389 to 0.857 on that sheet when the labels are
corrected. Normalized cross-correlation has no capacity to overfit and shares
no code path with YOLO. Two independent methods were penalized by the same
twelve annotations.

---

# Experiment 2 — the v0.0.2 labels

Identical code, model, seed, hyperparameters and folds. Only the annotations
differ.

## Step 3 — template-matching baseline, corrected labels

| fold | eval sheet | best recall@5px | at FP/window | recall@5px at ≲2 FP/window |
|---|---|---:|---:|---:|
| A | K-35-51-B-a | 0.828 | 73.7 | 0.453 (at 0.6) |
| B | K-35-8-G-a | 0.882 | 58.6 | 0.147 (at 1.7) |
| C | K-34-35-B-g | 0.857 | 54.4 | 0.286 (at 2.4) |

Still far behind the learned detector, and still with ~1% precision where
recall is useful. Fold A's usable-precision figure improves substantially
(0.070 → 0.453) because its templates are fitted on training sheets that no
longer include mills labelled as mounds.

## Step 4 — YOLO26-s, corrected labels

At confidence 0.25:

| fold | eval sheet | n | recall@5px | 95% CI | p90 error | FP/window |
|---|---|---:|---:|---|---:|---:|
| A | K-35-51-B-a | 128 | 0.883 | 0.816–0.928 | 3.4 px | 0.36 |
| B | K-35-8-G-a | 34 | 1.000 | 0.898–1.000 | 3.7 px | 0.00 |
| C | K-34-35-B-g | 7 | 1.000 | 0.646–1.000 | 2.0 px | 0.02 |

At confidence 0.05: 0.898, 1.000, 1.000. Macro mean 0.966; instance-weighted
156/169 = 0.923.

The recall curve is flat in radius on all three folds here too.

## Side by side, at confidence 0.05

| fold | eval sheet | exp. 1 recall@5px | exp. 2 recall@5px | exp. 1 p90 | exp. 2 p90 |
|---|---|---:|---:|---:|---:|
| A | K-35-51-B-a | 0.859 (110/128) | 0.898 (115/128) | 3.30 px | 3.52 px |
| B | K-35-8-G-a | 1.000 (34/34) | 1.000 (34/34) | 3.73 px | 3.66 px |
| C | K-34-35-B-g | 0.389 (7/18) | **1.000 (7/7)** | 2.00 px | 2.00 px |
| | macro | 0.749 | **0.966** | | |

**Fold C's evaluation set is 7 mounds.** Its 95% interval is 0.646–1.000. The
figure is directionally clear and is not something to lean on; the cross-fold
picture rests on folds A and B.

### Fold A is a controlled measurement of training-label noise

Fold A evaluates K-35-51-B-a, whose annotations are byte-identical between the
two experiments. Only its *training* data changed, because K-34-35-B-g is one of
its two training sheets. Recall@5px moves 0.859 → 0.898.

That is the right shape for a measurement of what twelve mislabelled training
positives cost on an unrelated sheet — but it is **not separable from checkpoint
noise**. Fold A's recall across saved checkpoints spans 0.664–0.891 in
experiment 1 and 0.734–0.891 in experiment 2, both far wider than the 3.9-point
difference. Directionally consistent, not established.

### Epoch trajectory

Recorded, never selected on, at confidence 0.25:

| checkpoint | A exp.1 | A exp.2 | B exp.1 | B exp.2 | C exp.1 | C exp.2 |
|---|---:|---:|---:|---:|---:|---:|
| epoch 25 | 0.789 | 0.734 | 0.941 | 0.912 | 0.333 | 0.857 |
| epoch 50 | 0.664 | 0.875 | 0.912 | 0.912 | 0.333 | 0.857 |
| epoch 75 | 0.891 | 0.820 | 0.941 | 0.971 | 0.333 | 0.857 |
| epoch 100 | 0.805 | 0.844 | 0.941 | 1.000 | 0.333 | 0.857 |
| epoch 125 | 0.766 | 0.891 | 1.000 | 1.000 | 0.333 | 1.000 |
| last (150) | 0.812 | 0.883 | 1.000 | 1.000 | 0.389 | 1.000 |

Fold A swings ±11 points between checkpoints with no trend in both experiments,
as wide as its confidence interval. Selecting `best.pt` in experiment 1 would
have reported 0.891 instead of 0.812 — the leak the no-selection rule avoids,
and a measurement of how large it would have been. This is the detector's
version of the late-training drift v0.4 traced in the decoder, and the second
concrete argument for step 8's permanent validation split.

Fold C in experiment 1 is flat from epoch 25 at 0.333 and never improves —
consistent with a target that cannot be learned because it is not there to
learn.

Per-fold detail: `analysis/detector/` and `analysis/template/`.

---

# Step 5 — end-to-end, detector to mask

The first measurement in this project that is not conditional on already
knowing where the mound is. Detector candidates at confidence 0.25 from
experiment 2, prompts derived from them, masks from the **v0.4r** decoder —
the 512 px LOSO checkpoints retrained on the corrected annotations, see
`../v004/RESULTS_REVISION.md` — scored against each annotation's own geometry.

Two prompt arms separate two error sources. `fixed` puts a median-sized box
(25×23 px, 4 px padding) on the candidate centre, so only localization error
reaches the segmenter. `detector` uses the candidate's own box, which is what
deployment does. v0.4's jitter sweep translated the ground-truth box rigidly
and left box *scale* untested; this is that measurement.

| fold | sheet | arm | n | IoU≥0.5 | IoU≥0.75 | mean IoU | missed | false masks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | K-35-51-B-a | `fixed` | 128 | 0.875 | 0.586 | 0.6820 | 0.109 | 60 |
| A | K-35-51-B-a | `detector` | 128 | 0.883 | 0.570 | 0.6776 | 0.109 | 60 |
| B | K-35-8-G-a | `fixed` | 34 | 1.000 | 0.529 | 0.7482 | 0.000 | 0 |
| B | K-35-8-G-a | `detector` | 34 | 1.000 | 0.471 | 0.7455 | 0.000 | 0 |
| C | K-34-35-B-g | `fixed` | 7 | 1.000 | 0.857 | 0.7717 | 0.000 | 3 |
| C | K-34-35-B-g | `detector` | 7 | 1.000 | 0.571 | 0.7779 | 0.000 | 3 |

**Macro IoU≥0.5 is 0.958** on the `fixed` arm. Every mound the detector finds
on the two smaller sheets yields a usable mask.

## The end-to-end loss is detector recall, not segmentation

This is what the step was built to test, and it separates cleanly:

| fold | mean IoU, all annotations | mean IoU, found only | v0.4r conditional (GT prompt) | prompt cost |
|---|---:|---:|---:|---:|
| A | 0.6820 | 0.7657 | 0.7913 | −0.026 |
| B | 0.7482 | 0.7482 | 0.8288 | −0.081 |
| C | 0.7717 | 0.7717 | 0.8137 | −0.042 |

Fold A's missed rate is 0.109 against a detector miss rate of 0.117 at the same
threshold: **the pipeline loses what the detector misses and essentially
nothing else.** Fold B is the cleanest case — 34 of 34 found, so no recall loss
at all — and it isolates the cost of a detector-derived prompt against a
ground-truth one at 0.081 IoU, with a measured p90 centre error of 3.66 px.

That cost is real but modest, and it is in the direction v0.4's tolerance curve
predicts. It is also not purely localization: the `fixed` arm's box is the
dataset median, and symbol boxes run 8–33 px, so part of the 0.081 is box-scale
mismatch rather than centre error.

## Box scale costs mask precision, not mask detection

The `fixed` arm beats the `detector` arm at IoU≥0.75 on all three folds:

| fold | fixed | detector | Δ |
|---|---:|---:|---:|
| A (n=128) | 0.586 | 0.570 | +0.016 |
| B (n=34) | 0.529 | 0.471 | +0.059 |
| C (n=7) | 0.857 | 0.571 | +0.286 |

At IoU≥0.5 and in mean IoU the two arms are within 0.006 of each other. So the
detector's own box costs *precision* of the mask, not whether a usable mask is
produced at all.

The direction is consistent on 3 of 3 folds, but the magnitudes are not strong
evidence: fold A's +0.016 is two instances out of 128, and fold C's +0.286 is
6/7 against 4/7. Treat it as a cheap default worth adopting — prompt with a
canonical box on the detector's point rather than the detector's box — and not
as an established effect. A proper test would size the box from the predicted
symbol rather than from the dataset median.

## False masks

60 on fold A, 0 on fold B, 3 on fold C, at confidence 0.25. These are the
detector's false positives carried through segmentation, and as noted below
they are a review queue rather than purely an error count.

---

## What this does and does not establish

It is **within-series detection on three sheets of one Bulgarian 1:25k series**,
evaluated leave-one-sheet-out. There is no frozen test split, so all of it is
development evidence, not a test result.

Specifically not established:

- **That recall@5px near 0.90–1.00 generalizes.** Three sheets of one series,
  and one of them now contributes 7 evaluation mounds.
- **That experiment 2's improvement is attributable only to the correction.**
  For fold C it plainly is — the removed annotations were the misses. For fold
  A the 3.9-point gain sits inside checkpoint noise.
- **That the corrected labels are now complete.** The same review that found
  twelve errors on one sheet found an unlabelled mound on it as well. The other
  two sheets have not had an equivalent adversarial review, and the detector's
  remaining false positives are the obvious place to start one.
- **Any false-positive rate at sheet scale.** The 12 clips were selected for
  annotation. FP per window and per megapixel are reported as measured on
  annotation-selected clips.
- **That YOLO26-s is the right architecture.** It is the first one tried and it
  cleared the localization target immediately. The flat recall curve suggests
  the remaining headroom is not in the box regressor.

## A method note worth carrying forward

The detector's false positives are now a **review queue, not only an error
rate**. One of experiment 1's fold C false positives was a real mound. Before
treating a high-confidence false positive as a model defect, it is worth
checking the map. Fold A currently reports 103 false positives at confidence
0.05, of which 34 land on `road` and 8 on `decorative_symbol`; those are
probably genuine errors, but the 61 on `background` have not been inspected.

---

## What was not done

**Step 1 (more sheets)** — unchanged, and now the only thing standing between
this and a defensible robustness claim.

**Step 6 (second-stage classifier)** — not triggered. 0.00–0.36 false positives
per window at confidence 0.25.

**Step 7 (prompt-jitter augmentation)** — not triggered and contraindicated.
Its condition was p90 outside 5 px; p90 is 2.0–3.7 px on every fold of both
experiments.

**Step 8 (permanent grouped splits)** — blocked on more sheets; the fold A
trajectory is a second argument for it.

**v0.4 revision** — done, in `../v004/RESULTS_REVISION.md`. Its conclusions
survive the correction: the two folds with unchanged evaluation sets move by
under 0.003 IoU.

## Known data notes

**The `negative_type` patch.** One of the twelve relabelled symbols,
`K-34-35-B-g_3` at (1566, 1103), exported with `negative_type`
`__undefined__` where the other eleven carry `other`. It was corrected
mechanically in the repository copy rather than by re-exporting from CVAT, so
`annotation/cvat/v0.0.2/instances_default.json` differs from the CVAT export by
exactly that one field. No figure in this document changes: `negative_type` is
used only to attribute false positives, and none were attributed to it.

**A domain rule that predicts the error.** A mound cannot be crossed by a
*surface* watercourse — water runs along terrain lows and a mound is a raised
feature — so `crossed_by_water_line` on a `mound` is a review signal. **All
twelve** relabelled mills carry it, probably for a reason worth stating: water
mills sit *on* watercourses, so the attribute was true and it is mounds, not
the attribute, that the water excludes.

Applying the rule to the corrected export left two mounds flagged. Both have
now been reviewed, and they resolved differently:

| annotation | detector verdict | resolved |
|---|---|---|
| `K-35-51-B-a_3` (314, 15.5) | not detected; nearest candidate 253 px | **real mound, attribute wrong.** It sits beside a lake with no water line crossing it, and is heavily obscured by pencil marks and border/road lines |
| `K-35-8-G-a_1` (955.5, 627.5), COCO id 22 | detected at 0.8 px, conf 0.838 | **real mound, attribute right.** A genuine water line does cross it — an *underground* one |

**The second case forces a qualifier onto the rule.** An underground water line
is a pipe and can run beneath anything, including a mound; only a surface
watercourse carries the archaeological constraint. The `crossed_by_water_line`
attribute does not distinguish the two, so it cannot be used as a mechanical
check. A hit means one of three things — a mislabelled symbol, a wrongly set
attribute, or a legitimate underground crossing — and only a person looking at
the map can say which.

If the annotation protocol is revised, splitting the attribute into surface and
underground variants would restore it to a mechanical check. That is the
cheapest schema change available with the largest validation payoff.

The rule is one-way: it says nothing about symbols *not* crossed by a water
line. And the other correlations seen on the mills — no relative height mark,
crossed by a road — described that class incidentally and should not be
generalized at all.

**The `K-35-51-B-a_3` case is a confirmed detector miss on a real mound**, and
the reason given for it — pencil and border/road obfuscation — is consistent
with fold A's `blurred_or_bad_print` subset recall of 0.750 against 0.898
overall. That is the detector's real failure mode on this data, as distinct
from the mislabels that dominated the first experiment. Its incorrect attribute
is still in `v0.0.2`; no figure here depends on it.

## A method note worth carrying forward

The detector's false positives are now a **review queue, not only an error
rate**. One of experiment 1's fold C false positives was a real mound. Before
treating a high-confidence false positive as a model defect, it is worth
checking the map. Fold A currently reports 103 false positives at confidence
0.05, of which 34 land on `road` and 8 on `decorative_symbol`; those are
probably genuine errors, but the 61 on `background` have not been inspected.

---

## What was not done

**Step 1 (more sheets)** — unchanged, and now the only thing standing between
this and a defensible robustness claim.

**Step 6 (second-stage classifier)** — not triggered. 0.00–0.36 false positives
per window at confidence 0.25.

**Step 7 (prompt-jitter augmentation)** — not triggered and contraindicated.
Its condition was p90 outside 5 px; p90 is 2.0–3.7 px on every fold of both
experiments.

**Step 8 (permanent grouped splits)** — blocked on more sheets; the fold A
trajectory is a second argument for it.

**v0.4 revision** — done, in `../v004/RESULTS_REVISION.md`. Its conclusions
survive the correction: the two folds with unchanged evaluation sets move by
under 0.003 IoU.

## Known data notes

**The `negative_type` patch.** One of the twelve relabelled symbols,
`K-34-35-B-g_3` at (1566, 1103), exported with `negative_type`
`__undefined__` where the other eleven carry `other`. It was corrected
mechanically in the repository copy rather than by re-exporting from CVAT, so
`annotation/cvat/v0.0.2/instances_default.json` differs from the CVAT export by
exactly that one field. No figure in this document changes: `negative_type` is
used only to attribute false positives, and none were attributed to it.

**A domain rule that predicts the error.** A mound cannot be crossed by a water
line — water runs along terrain lows and a mound is a raised feature — so
`crossed_by_water_line` on a `mound` is a data-error signal. **All twelve**
relabelled mills carry it. Applying the rule to the corrected export leaves two
mounds still flagged, on sheets that have had no adversarial review:

| annotation | detector verdict | resolved |
|---|---|---|
| `K-35-51-B-a_3` (314, 15.5) | not detected; nearest candidate 253 px | **a real mound; the attribute is wrong.** Confirmed by the annotator: it sits beside a lake with no water line crossing it, and is heavily obscured by pencil marks and border/road lines |
| `K-35-8-G-a_1` (955.5, 627.5) | detected at 0.8 px, conf 0.838 | unreviewed; carries a trig point and both elevation marks, so most likely the same kind of attribute error |

**A violation does not say which field is wrong.** It means either the symbol
is mislabelled or the attribute was set in error, and only a person looking at
the map can tell them apart. On this data the split is clean: among the mills
the label was wrong, and among the two survivors the attribute is. The
constraint itself has produced no counter-example — no annotation yet examined
is a mound genuinely crossed by a water line.

The rule is one-way: it says nothing about symbols *not* crossed by a water
line. And the other correlations seen on the mills — no relative height mark,
crossed by a road — turned out to describe that class incidentally and should
not be generalized the same way.

The `K-35-51-B-a_3` annotation still carries the incorrect attribute in
`v0.0.2`; it is not used by any figure in this document, and correcting it is
left to the next export. It is, however, a **confirmed detector miss on a real
mound**, and the reason given for it — pencil and border/road obfuscation — is
consistent with fold A's `blurred_or_bad_print` subset recall of 0.750 against
0.898 overall. That is the detector's real failure mode on this data, as
distinct from the mislabels that dominated the first experiment.

---

## Reproduce

```bash
# Experiment 1 uses annotation/cvat/v0.0.1 and output dirs without the suffix.
python -m archeo_topia.datasets.build_detection_windows \
  --coco-json annotation/cvat/v0.0.2/instances_default.json \
  --images-root data/curated/datasets/mapsam_v02/images \
  --output-dir data/curated/datasets/mapsam_det_v1 --window 512 --stride 384

python -m archeo_topia.analysis.detect_template_baseline \
  --dataset data/curated/datasets/mapsam_det_v1 \
  --images-root data/curated/datasets/mapsam_v02/images \
  --fold all --output-dir artifacts/detection/v0_5_template_gtfix \
  --thresholds 0.4 0.5 0.6 0.7 0.8 0.9

python -m archeo_topia.training.train_mound_detector \
  --dataset data/curated/datasets/mapsam_det_v1 \
  --output-dir artifacts/detection/v0_5_yolo26s_gtfix \
  --fold all --model yolo26s.pt --imgsz 512 --epochs 150 --batch 16 --save-period 25

python -m archeo_topia.analysis.mapsam_end_to_end \
  --dataset data/curated/datasets/mapsam_det_v1 \
  --detector-run artifacts/detection/v0_5_yolo26s_gtfix \
  --images-root data/curated/datasets/mapsam_v02/images \
  --coco-json annotation/cvat/v0.0.2/instances_default.json \
  --mapsam-prefix v0_4r_loso512 --fold all --confidence 0.25 \
  --output-dir artifacts/detection/v0_5_end_to_end
```

Step 5 needs both optional extras together, `.[sam,detect]`: it spans the
detector and the decoder.

Environment: torch 2.14.0+cu130, ultralytics 8.4.150, one RTX 5090. v0.1–v0.4
ran on torch 2.12; the detector work did not re-verify the decoder results
under 2.14.
