# MapSAM v0.4 — prompt robustness, failure anatomy, and 512-window LOSO

Follows `../v003/RESULTS.md` and executes steps 1–3 of `PLAN.md`. Same model
(SAM ViT-B, frozen image and prompt encoders, mask decoder only), same dataset
(`data/curated/datasets/mapsam_v02`, 3 sheets / 12 tiles / 171 samples), same
optimizer and loss, `pw20` frozen as the plan required.

Steps 4–6 were not executed; see *What was not done* at the end.

---

## Summary

1. **The 512 px window generalizes across all three sheets, not just the easy
   one.** Source-pixel disagreement area falls 64–69% on every fold, and the
   cross-sheet spread falls from 119.9 to 24.7 source px². This revises v0.3's
   central diagnosis rather than extending it: the dominant term in what v0.3
   measured as domain variation was the representation itself. Sheets differed
   most where the mound fell furthest below one useful encoder token, and
   normalizing the representation around the prompt removes most of that
   difference. A smaller residual sheet effect remains; it is no longer the
   dominant failure mode.
2. **Fold A's "prediction collapse" is mostly metric quantization, not
   collapse.** 17 of its 19 zero-IoU samples land within 40 source px of the
   target — inside or just outside a ~21–30 px symbol. At 512 all 19 are gone,
   as are folds B and C's.
3. **A displaced window is nearly free; a displaced prompt is not.** The window
   can be off by 250 source px with no measurable cost. The prompt has a
   tolerance of about 15 px, and that is the same for the full-tile model as
   for the 512 one.
4. **So window size is not coupled to localization accuracy** the way `PLAN.md`
   assumed. Adopting 512 does not impose a tighter detector requirement. It
   pays substantially better inside a tolerance band both representations
   need anyway.

Combined: **the plan's Outcome D fires.** Segmentation given a correct prompt
is no longer the bottleneck on this data. The unsolved problem is producing
the prompt — and the jitter curves now quantify how accurate that has to be.

---

## Methodology notes

**Epoch selection.** Unchanged from v0.3. Every table reports the final epoch
(50) as the primary figure; peaks are labelled *oracle* and are upper bounds,
not results.

**Scale-invariant reporting.** Every cross-window comparison leads with
`abs_error_source_px` — the symmetric difference |pred XOR gt| converted to
source-tile pixels. Decoder IoU rises under cropping by construction.

The plan asked for the IoU≥0.5 rate alongside it. That rate inherits the same
problem: it is a threshold on decoder IoU, so it is not comparable between a
full-tile arm and a 512 arm. It is used here only *within* one arm, as a
degradation curve against that arm's own zero-offset value.

**Jitter semantics.** The offset displaces the prompt after the target
instance has been selected, so the target is always the mound the annotation
points at. Three arms are run, because a localization error costs two things
at once in a windowed model — a worse prompt and a displaced input — and they
have to be separated. The box translates rigidly with the point, which models
a detector that sizes correctly and centres wrongly; box *scale* error is
untested.

**Re-scoring the v0.3 baselines.** The v0.3 LOSO runs trained on cached
embeddings, and the embedding dataset does not carry `window_xyxy`, so they
never recorded a source-pixel column. Each v0.3 checkpoint was re-scored
through the non-cached dataset to recover it. The re-scored decoder IoUs
reproduce the v0.3 report exactly (0.3498 / 0.6021 / 0.3714), which is the
check that the pass is faithful.

---

## Step 1 — prompt-jitter robustness

No training. Three arms over 8 compass directions × offsets 0–300 source px,
on the 34-sample `test` split (`K-35-8-G-a`). Zero-offset rows reproduce the
v0.3 resolution table exactly: 253.9 source px² / 0.6021 for full tile, 91.8 /
0.8261 for 512.

**Units.** Two different quantities appear below and both are in source-tile
pixels, but one is an area and the other a distance. Disagreement area is
**px²**; prompt offsets and centroid displacements are **px**. The per-sample
field is named `abs_error_source_px`, inherited from v0.3, but the quantity it
holds is an area.

| offset | full tile err | 512 err | 512 advantage | control err (window only) | control ≥0.5 |
|---:|---:|---:|---:|---:|---:|
| 0 | 253.9 | **91.8** | +162.1 | 91.8 | 1.00 |
| 5 | 287.4 | **140.5** | +147.0 | 96.8 | 1.00 |
| 10 | 365.0 | **255.0** | +110.0 | 95.5 | 1.00 |
| 15 | 479.0 | **444.6** | +34.5 | 96.8 | 1.00 |
| 25 | 702.4 | 734.7 | −32.3 | 98.4 | 1.00 |
| 50 | 767.1 | 744.2 | +22.8 | 106.2 | 1.00 |
| 100 | 704.2 | 734.2 | −30.0 | 100.6 | 1.00 |
| 200 | 666.8 | 718.6 | −51.9 | 100.9 | 1.00 |
| 250 † | 684.1 | 670.1 | +14.0 | 94.2 | 1.00 |
| 300 † | 685.6 | 519.3 | +166.3 | 91.3 | 0.62 |

† Past 250 px the 512 window starts cutting the mound off — mean GT area in
the window falls 470.4 → 432.2 → 291.0 — so the windowed rows there are scored
against a partly missing target. The plan's suggested maximum of 200 px never
reached this regime, which is why the sweep was extended.

**The control arm is the finding.** With the window displaced but the prompt
left on the truth, error moves 91.8 → 94.2 px² across 250 px of displacement and
every sample still clears IoU 0.5. The 512 window has roughly 250 px of
placement slack. Its prompt has about 15.

**Both representations die at the same scale, and it is the scale of the
symbol.** Full tile is not the robust alternative: it degrades 253.9 → 702.4
over the same 25 px, with its IoU≥0.5 rate falling 0.71 → 0.06. By 50 px both
arms are at zero mean IoU. This is simply what a prompted segmenter does — it
segments what is at the prompt, and 25 px is most of a mound symbol's width.

**512's advantage survives to about 18 px**, decaying +162 → +147 → +110 →
+34 → −32. Past 25 px the sign of the difference is noise between two
saturated failure modes.

The plan's branch — *"if 512 degrades much faster than full tile, the window
size must be chosen jointly with the achievable localization accuracy"* — is
answered no. 512 degrades faster in relative terms only because it starts so
much lower. Both need the prompt within ~15 px; inside that band 512 is
better everywhere.

Directional anisotropy, averaged over offsets 5–25 on the full-tile arm:
northward offsets cost about 1.4× less than west or southwest (355 vs 531
source px²). Not investigated. Recorded because a single-direction sweep would
have drawn a curve 30% optimistic or pessimistic depending on its choice.

Detail: `analysis/jitter/jitter_analysis.md`.

## Step 2 — anatomy of fold A's zero-IoU cases

No training. Fold A at epoch 50 has 19 of 120 samples at IoU exactly zero, 10
predicting nothing at all.

Measured by displacement rather than overlap:

| | n | median distance to target |
|---|---:|---:|
| Hits (IoU > 0) | 101 | 5.0 source px |
| Near misses (zero IoU, within 40 source px) | **17** | 19.0 source px |
| Wrong object (zero IoU, beyond 40 source px) | **2** | 868 / 1861 source px |

So it is largely not collapse. At full-tile resolution a mound covers about
five pixels on the 256×256 decoder grid, so a prediction displaced by one or
two pixels has no overlap and scores exactly zero while sitting on the right
symbol. This is v0.3's Question 2 quantization artifact in its extreme form.

The ten zero-pixel cases are a **threshold** problem, not a localization one:
nine of ten have their peak logit within 6.5–39.6 source px of the target,
with logit maxima from −13.2 to −0.3. Mean logit max is 0.99 over the 19
zero-IoU samples against 19.6 over the 101 that score anything.

Worth stating precisely, because it changes what the fix would be: the
activation is correctly *placed* but **negative**, not strong-and-mis-
thresholded. The model puts its peak on the right symbol and then declines to
commit to it. So these are recoverable by a lower decision threshold, but the
underlying quantity is a decoder that has learned to suppress on the held-out
sheet, not a miscalibrated cut point on a confident output.

And it is **acquired during training**: 2 zero-IoU at epoch 5 against 19 at
epoch 50; zero-pixel predictions first appear at epoch 20. Fold A's oracle
peak is at epoch 5 (0.4911) against 0.3498 at epoch 50 — the same fact seen
through the mean.

Target size does not correlate at source resolution (median component area 454
vs 449 px). Crowding correlates weakly (median nearest-neighbour distance 70
vs 98 px). Tiles differ 8/9/18/21%.

Detail: `analysis/collapse/collapse_analysis.md`. Overlays:
`artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/debug_predictions_v004_collapse/`.

## Step 3 — 512-window leave-one-sheet-out

Three training runs, protocol identical to the v0.3 full-tile LOSO except for
the window (512 px), the loss-crop margin (150, holding supervised ground area
constant) and the embedding cache (off, forced by the window).

| Fold | eval sheet | n | train | full tile err | **512 err** | reduction | full tile IoU | 512 IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | K-35-51-B-a | 120 | 51 | 373.8 | **116.4** | 69% | 0.3498 | 0.7930 |
| B | K-35-8-G-a | 34 | 137 | 253.9 | **91.8** | 64% | 0.6021 | 0.8261 |
| C | K-34-35-B-g | 17 | 154 | 324.5 | **103.1** | 68% | 0.3714 | 0.7574 |

| | full tile | 512 |
|---|---:|---:|
| Macro mean error | 317.4 | **103.8** (−67%) |
| Sample-weighted mean error | 345.1 | **110.2** (−68%) |
| Macro mean IoU | 0.4411 | 0.7922 |
| Sample-weighted mean IoU | 0.4021 | 0.7961 |
| **Sheet spread, error** | **119.9** | **24.7** |
| Sheet spread, IoU | 0.2522 | 0.0688 |

| Fold | 512 ≥0.5 | 512 ≥0.75 | full tile ≥0.5 | full tile ≥0.75 | 512 zero-IoU | full tile zero-IoU | 512 final | oracle peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.99 | 0.78 | 0.30 | 0.05 | 0 / 120 | 19 / 120 | 0.7930 | 0.8035 @ e5 |
| B | 1.00 | 0.91 | 0.71 | 0.15 | 0 / 34 | 0 / 34 | 0.8261 | 0.8360 @ e15 |
| C | 1.00 | 0.53 | 0.29 | 0.12 | 0 / 17 | 2 / 17 | 0.7574 | 0.8104 @ e5 |

This is the plan's first branch: **all three sheets improve substantially**.
The 47% spread across sheets at full tile becomes a 13% spread at 512.

Fold B reproduces the v0.3 resolution arm C to four decimals (0.8261, 91.8
source px²), which is the configuration check — arm C's split-based selection
partitions the data exactly as fold B's sheet-based selection does.

**Step 2's prediction holds.** Fold A at 512 has 0 of 120 zero-IoU, 0
predicting nothing, and median hit displacement 0.99 source px against 5.01 at
full tile. Both wrong-object cases are fixed too, by a different mechanism
worth naming: a 512 px window physically excludes a distractor 868 or 1861
source px away, so the model can no longer choose it.

Detail: `analysis/loso512/loso512_analysis.md`.

---

## What this means for the roadmap

The v0.4 plan set step 3 up as a branch point between *keep improving the
representation* and *segmentation is good enough; the unsolved problem is
detection*. The data points at the second.

Across all three sheets, with a ground-truth-derived prompt, **99–100% of
instances clear IoU 0.5** and the macro mean is 0.79. The ≥0.75 rate is less
uniform (0.78 / 0.91 / 0.53) but fold C carries 17 samples, where one sample
moves the rate by 6%.

Meanwhile the jitter curves show the whole result rests on a prompt accurate
to roughly 15 source px. There is no detector. Every number in v0.1–v0.4
assumes one that does not exist, and step 1 now puts a number on how good it
must be: **the same order as the symbol it is finding.**

That makes candidate generation the binding constraint, and it is what the 530
`hard_negative_symbol` annotations are for.

**What this claim is scoped to.** *Consistent within-series conditional
segmentation.* All three sheets are one Bulgarian 1:25k series, so
cross-cartographic generalization is entirely untested and the residual
24.7 px² spread may understate it badly. Domain robustness is not solved; what
is shown is that the dominant term in v0.3's sheet variation was
representational, and that what remains within this series is small. And
nothing here is a test result: every number in v0.1–v0.4, LOSO included, is
development evidence.

### Ranked next steps

1. **Build the candidate generator.** It is the binding constraint and the
   only thing standing between a 0.79 macro IoU and an end-to-end system. Its
   accuracy target is set by step 1: ~15 source px, with useful slack in where
   the window lands but none in where the prompt points.
2. **Measure the detector's localization error before changing the
   segmenter.** Prompt-jitter augmentation is available — `MapSamDataset` now
   takes `prompt_jitter_xy` — but widening the segmenter's tolerance and
   tightening the detector's accuracy are two ways to close the same gap, and
   only one of them is known to be needed. If candidates routinely land within
   5–8 px there is no reason to train MapSAM to accept 25–30 px prompts, and a
   reason not to: step 2 found the zero-IoU samples sit closer to their
   neighbours than average (median nearest-neighbour distance 70 px against
   98), so deliberate tolerance to large offsets risks trading mound-vs-
   neighbour discrimination for robustness nobody needs.
3. **Add sheets, and prefer diverse cartography.** Still the binding
   constraint on any claim about domain robustness, and unchanged by v0.4.
4. **Establish train / validation / frozen-test splits** once there are enough
   sheets. The epoch-50 drift that step 2 traced is exactly what a validation
   split is for; fold C still loses 6.5% between its peak and epoch 50.
5. **Leave 256 and 384 px in the backlog.** They are deprioritized on *value*,
   not on evidence: nothing here shows a tighter window cannot help. At zero
   prompt error more token coverage might well reduce the ~92–116 px² residual
   further, since even 512 puts only 2.76 tokens on a mound. But that is an
   optimization of something already clearing IoU 0.5 on 99–100% of instances,
   while the pipeline has no way to produce the prompt at all. Reopen it if
   the detector becomes good and segmentation becomes the bottleneck again.

Still not recommended: calibration, new losses, encoder variants, unfreezing
the encoder, or feeding hard negatives into the decoder loss.

## What was not done

**Step 4 (locate the resolution optimum)** was conditional on step 3 showing
broad improvement, which it does. It is nonetheless deprioritized, for the
reason given above: the case for spending runs on 256/384 is weaker than the
case for building the detector. This is a judgment about value, not a finding
— step 1 says nothing about whether a tighter window helps at zero prompt
error, and it might.

**Steps 5 and 6 (more sheets, permanent grouped splits)** are blocked on data,
not deferred by choice. `data/maps`, `data/georeferenced` and
`data/cvat_exports` are empty, and `data/curated/datasets` holds only
`mapsam_v0` and `mapsam_v02`. No additional sheets exist in the repository to
add. Both steps stay at the top of the v0.5 plan.

**Known debt, unchanged from v0.3**: seven pre-existing `sam2_mcp` /
`sam2_backend` test failures, and ruff errors outside `src/archeo_topia`
(`make lint` is scoped to `src/services tests`).

---

## Reproduce

```bash
# Step 1 — prompt jitter (no training, ~15 min)
OFF=0,5,10,15,25,50,100,200,250,300
python -m archeo_topia.analysis.mapsam_prompt_jitter \
  --config configs/mapsam/mapsam_v0_3_res_a_fulltile_pw20.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_res_a_fulltile_pw20/checkpoints/final.pt \
  --label fulltile --offsets $OFF --output-dir docs/mapsam/v004/analysis/jitter
# ...and the window512 / window512_gtprompt arms; see analysis/jitter/jitter_analysis.md

# Step 2 — collapse anatomy (no training, ~1 min)
python -m archeo_topia.analysis.mapsam_collapse_analysis \
  --config configs/mapsam/mapsam_v0_3_loso_foldA_pw20_cropon.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/checkpoints/final.pt \
  --name fold_a_fulltile_e50 --output-dir docs/mapsam/v004/analysis/collapse

# Step 3 — 512-window LOSO (~45 min)
for f in A B C; do
  python -m archeo_topia.training.train_mapsam_v0 \
    --config configs/mapsam/mapsam_v0_4_loso512_fold${f}_pw20.yaml
done
python -m archeo_topia.analysis.mapsam_compare_runs --prefix v0_4_loso512 \
  --output-dir docs/mapsam/v004/analysis/loso512 --name cross_sheet_512_results

# Overlays. Training does not write these despite outputs.save_debug_predictions;
# they come from a separate pass, as in v0.3.
for f in A B C; do
  python -m archeo_topia.training.mapsam_predict_debug \
    --config configs/mapsam/mapsam_v0_4_loso512_fold${f}_pw20.yaml \
    --checkpoint artifacts/models/mapsam/v0_4_loso512_fold${f}_pw20/checkpoints/final.pt \
    --split test --max-samples 8 \
    --output-dir artifacts/models/mapsam/v0_4_loso512_fold${f}_pw20/debug_predictions
done
```

Full commands, including the source-pixel re-scoring pass, are in each
analysis note.

## Artifacts

Per-run outputs stay under `artifacts/models/mapsam/v0_4_loso512_*/`:
`metrics.json`, `config_resolved.json`, `checkpoints/`, `debug_predictions/`,
per-sample `prediction_stats_val_e*.jsonl`, and the `jitter_*_per_sample.jsonl`
rows behind the step 1 and step 3 tables. The jitter sweep writes its
per-sample rows to the evaluated checkpoint's run directory rather than beside
the aggregates, because a full sweep is tens of thousands of rows and the
project keeps per-sample outputs under `artifacts/`. Step 2's overlays are in
`artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/debug_predictions_v004_collapse/`,
written to a new directory so the v0.3 overlays are untouched.

Aggregated tables are in `docs/mapsam/v004/analysis/{jitter,collapse,loso512}/`.

v0.1–v0.3 artifacts are untouched; every v0.4 config writes to its own
`outputs.root`.

## Code added in v0.4

| | |
|---|---|
| `MapSamDataset(prompt_jitter_xy=, jitter_prompts=)` | Displaces the prompt after instance selection, so the target stays the annotated mound. `jitter_prompts=False` moves the window only. |
| `mapsam_window.apply_prompt_jitter` | The geometry, with clamping to the tile. |
| `analysis.mapsam_prompt_jitter` | The offset × direction sweep. At `--offsets 0` it is an ordinary evaluation that records the source-pixel columns. |
| `analysis.mapsam_collapse_analysis` | Displacement rather than overlap; classifies zero-IoU samples as near miss or wrong object. |
| `mapsam_predict_debug --sample-ids` | Restricts overlays to a named set, from a list or a file. |
