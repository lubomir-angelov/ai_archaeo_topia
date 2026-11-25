# MapSAM v0.6 — the blind sweep over the 56 working sheets

Executes step 1 of `PLAN.md`. The v0.5 detector was run over every sheet in the
working pool before any of them was annotated and before any proposal was shown
to a reviewer. Raw artifacts are in `analysis/sweep/`; candidate files and crops
are in `artifacts/detection/v0_6_sweep/`.

**What this document is not.** The new sheets carry no ground truth, so nothing
here is a recall, a precision or a false-positive rate. Every number below is a
*candidate* count. That is the quantity that governs the annotation review
budget, and it bounds the false-positive rate from above; it cannot be turned
into a detection score without the blind test set, and the point of the blind
set is that it must not be.

---

## Summary

1. **The domain transferred.** Corpus-wide candidate density at confidence 0.25
   is 1.31–2.10 per megapixel depending on checkpoint, against 2.55 mounds per
   megapixel measured on the annotated clips. Same order of magnitude, and
   lower — which is what whole sheets containing uninformative terrain should
   do to a density measured on clips selected for annotation. The decision gate
   in `PLAN.md` step 1 reads *proceed*.
2. **The review burden is 0.17–0.30 candidates per window**, not the ~0.36–0.61
   the plan projected from fold A. Over 8,671 windows that is **1,613–2,593
   candidates at confidence 0.25** and 2,511–4,504 at 0.05, against projections
   of ~3,300 and ~5,700. The projection was pessimistic, in the direction that
   costs reviewer time, and it is now replaced by a measurement.
3. **One sheet is an outlier by an order of magnitude, and it is Sofia.**
   `K-34-47-G-v` covers the centre of the capital. It returns 957 / 140 / 418
   candidates from the three checkpoints where every other sheet returns under
   110, and the candidates land almost entirely inside dense urban blocks. It
   alone contributes 37% of fold A's corpus total. **Excluding it, the three
   checkpoints agree to within 0.14 candidates per megapixel.**
4. **Dense urban fabric is the detector's real unknown on this corpus.** The
   annotated corpus is three rural sheets. Nothing in it teaches the detector
   what a city block looks like, and the three checkpoints disagree most
   violently exactly where none of them has a basis for a decision.
5. **The alpha filter is a measured no-op.** The plan expected roughly 9% of
   windows to fall outside the printed frame. Zero windows were dropped, and on
   a sampled three sheets no window is below 50% valid. The clipped frame is
   irregular enough to intersect a quarter of the windows and never enough to
   empty one, so the candidates-per-window denominator needs no correction.

**The blind test set can be sized and the model-assisted pass can start.**
Neither is blocked by anything in this sweep.

---

## What was run

The v0.5 detector, unchanged, at `artifacts/detection/v0_5_yolo26s_gtfix`, over
the 56 sheets left in the working pool after the four frozen sheets were moved
to `_frozen/`. Window 512, stride 384, confidence 0.05, candidates merged across
overlapping windows at 10 px on centre distance.

**All three fold checkpoints were run, not one.** Every fold is equally
out-of-domain on a sheet none of them has seen, so there is no principled single
choice. Running three costs minutes and buys a cross-checkpoint agreement rate,
which is a stability signal available with no labels at all. Where one number is
wanted, fold C is the one to quote — *as a judgement about training-set size,
not as evidence that fold C generalizes best*: fold C holds out `K-34-35-B-g`
and therefore trains on 162 of the 169 mounds, the largest training set of the
three.

| | value |
|---|---:|
| sheets swept | 56 |
| windows tiled | 8,671 |
| windows dropped as transparent | 0 |
| area | 1,232.4 Mpx |

8,671 is exactly the 9,271 windows `INPUT_INVENTORY.md` projects for all 60
sheets, less the 600 belonging to the four frozen ones.

## Candidate density

| | fold A | fold B | fold C | plan projection |
|---|---:|---:|---:|---:|
| candidates at 0.05 | 4,504 | 2,511 | 4,100 | ~5,655 |
| candidates at 0.10 | 3,671 | 2,035 | 3,009 | — |
| candidates at 0.25 | 2,593 | 1,613 | 2,062 | ~3,338 |
| per window at 0.25 | 0.299 | 0.186 | 0.238 | 0.36 |
| per window at 0.05 | 0.519 | 0.290 | 0.473 | 0.61 |
| per Mpx at 0.25 | 2.10 | 1.31 | 1.67 | 2.55 (mounds) |
| median sheet, per Mpx at 0.25 | 1.04 | 0.75 | 1.03 | — |
| worst sheet, per Mpx at 0.25 | 41.42 | 6.06 | 18.09 | — |

The gap between the mean and the median per-sheet density is the outlier. With
`K-34-47-G-v` removed:

| excluding `K-34-47-G-v` | fold A | fold B | fold C |
|---|---:|---:|---:|
| candidates at 0.25 | 1,636 | 1,473 | 1,644 |
| per Mpx at 0.25 | 1.35 | 1.22 | 1.36 |
| per window at 0.25 | 0.192 | 0.173 | 0.193 |

Three independently trained checkpoints landing within 0.14 candidates per
megapixel of each other, over 55 unseen sheets, is the strongest evidence in
this document that the transfer is real rather than an artifact of one
checkpoint.

## Are the candidates plausible?

A density says the detector fires; only a picture says what it fires on. Crops
were rendered around the highest-confidence candidates on six working sheets
(`artifacts/detection/v0_6_sweep/*/<sheet>/crops/`).

They are centred on the mound starburst, and the large majority carry an
adjacent relative-height mark — `+1,7`, `+2,4`, `+3,1` — which is the convention
the annotated corpus records on 158 of its 169 mounds. This is the symbol the
detector was trained on, appearing on sheets it has never seen.

Two qualifications, both visible in the same crops and both untested here:

- Several candidates sit on a **triangle glyph rather than a starburst**, which
  is what a trig point looks like. `trig_point` is the second-largest hard
  negative class at 136 instances and `has_trig_point` is true on 41 mounds, so
  the same printed element is a negative alone and an attribute of a positive
  when it sits on a mound. v0.5 called this out as context-dependent by nature.
  Whether these are errors cannot be settled from a crop.
- Print appearance varies within the series more than the three annotated
  sheets show. `K-34-10-A-g` is visibly denser and more heavily coloured, and
  its candidates are less uniformly the familiar symbol.

## Cross-checkpoint agreement

At 10 px and confidence 0.25, per sheet, over 56 sheets:

| | median |
|---|---:|
| fold C candidates also found by fold A | 0.766 |
| fold C candidates also found by fold B | 0.813 |
| fold A candidates also found by fold C | 0.904 |

27 of 56 sheets have over 75% of fold C's candidates confirmed by both other
checkpoints. The low tail is dominated by sparse sheets where two or three
candidates decide the whole ratio — `K-34-23-A-g` scores 0.00 on 4 and 6
candidates — so the per-sheet figure is noisy by construction on quiet sheets
and should not be read as instability there.

Per sheet: `analysis/sweep/cross_checkpoint_agreement.csv`.

## The Sofia sheet

`K-34-47-G-v` is the centre of Sofia. Plotting fold A's 957 candidates over the
sheet puts essentially all of them inside the built-up blocks; the rural
margins of the same sheet behave like every other sheet in the corpus.

| | fold A | fold B | fold C |
|---|---:|---:|---:|
| candidates at 0.25 | 957 | 140 | 418 |
| share of that checkpoint's corpus total | 37% | 9% | 20% |
| fold C candidates confirmed by this checkpoint | 0.25 | 0.13 | — |

**This is a finding to report, not yet a defect to fix.** The annotated corpus
is three rural sheets and contains no urban fabric at all, so the detector has
no basis for a decision inside a city block — and the three checkpoints
disagreeing by a factor of seven is what "no basis" looks like from outside.
Whether these are false positives is not knowable from this sweep: it needs
labels, and there are none on this sheet.

Two sheets share `K-34-47`'s parent (`K-34-47-A-a`, `K-34-47-G-g`) and are not
outliers, so this is about what the sheet depicts and not about its scan.

**Recommendation, as a judgement and not as evidence:** hold `K-34-47-G-v` out
of the first model-assisted annotation pass. At 0.19 candidates per window the
rest of the corpus is a reviewable queue; at 6 per window this sheet is not, and
reviewing it first would spend the scarcest resource on the least representative
sheet. It is worth annotating eventually — urban sheets are exactly where a
deployed system would need to be trusted — but after there is a test result.

## The alpha filter

The sheets are clipped to an irregular map frame and carry a genuine alpha band
at 90–92% opaque, so `PLAN.md` required transparent windows to be dropped before
they inflated the per-window denominator.

Measured: **zero windows dropped across all 56 sheets**, at a 5%-valid
threshold. On three sampled sheets the distribution of window validity is:

| | windows below |
|---|---:|
| 5% valid | 0 |
| 50% valid | 0 |
| 90% valid | 73 of 455 |
| 99.9% valid | 119 of 455 |

The frame is diagonal within the raster, so it clips about a quarter of the
windows and empties none of them. The concern was correct in principle and
measures to zero here; the filter stays in the code because it costs nothing and
a differently-cut batch of sheets would need it.

Note the transparent collar reaches the detector as opaque black, since dropping
the alpha channel leaves RGB zeros. No clustering of candidates at the sheet
corners was observed, but this was not measured and is worth a glance if a later
sheet behaves oddly at its margins.

---

## Scope

This is a **candidate-density measurement over 56 unannotated sheets of one
Bulgarian 1:25,000 archival series**, produced by detectors trained on three
sheets of that same series. It says the detector fires at a plausible rate on
plausible-looking symbols. It does not establish recall, precision, or that the
pipeline works on a genuinely different cartographic source — 63 sheets of one
series cannot test that, and the caveat stays open.

## What this changes in the plan

- Step 1's three questions: **transfer confirmed**, **review burden measured**
  at 0.17–0.30 candidates per window, and **sheet selection was already spent**
  — the four blind sheets were frozen in commit `b8a35b8` before this sweep ran,
  and all four are going to annotation regardless.
- The plan's projected false-positive counts were **over-estimates by 25–35%**.
  They were recorded to be falsified and they have been.
- One new item, not in the plan: **urban sheets are an untested regime**, and
  `K-34-47-G-v` is the evidence for it.
