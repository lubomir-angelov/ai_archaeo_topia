# MapSAM v0.5 — plan

Written at the end of v0.4, to be executed in a fresh session. Read
`../v004/RESULTS.md` first; this plan assumes its findings and does not repeat
the evidence for them.

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
- **The window has ~250 px of placement slack. The prompt has ~15 px.** A
  displaced window is nearly free; a displaced prompt kills the full-tile and
  512 models alike, at the scale of the symbol.

### What that is and is not scoped to

It is **consistent within-series conditional segmentation**. All three sheets
are one Soviet 1:50k series. Cross-cartographic generalization is untested and
the 24.7 px² residual may understate it badly. Nothing in v0.1–v0.4 is a test
result; all of it, LOSO included, is development evidence.

## The requirement v0.4 handed to v0.5

The jitter control arm separated two things that had always arrived together.
The localization requirement belongs to the **prompted segmentation
mechanism**, not to the crop:

- the window may be off by up to ~250 source px at 512 with no measurable cost;
- the prompt must land within roughly **15 source px** of the mound;
- by ~25 px both the full-tile and 512 models are outside their useful range.

So the detector does not need to produce a well-centred crop. It needs to
identify the symbol closely enough that the point and box prompts land on it.
That is a measurable target rather than "build a reasonably accurate
detector", and it is the target this plan is written against.

## Constraints carried forward

- Image and prompt encoders stay **frozen**.
- **`pw20` and the loss crop stay frozen.** No further positive-weight search.
- **512 px is the default input representation**, with the 150 loss-crop
  margin at 256 logits.
- Split by sheet, never by sample, for detection as well as segmentation.
- **Report the scale-invariant metric.** Any cross-window comparison leads with
  disagreement area in source px². The IoU≥0.5 rate is a threshold on decoder
  IoU and is only readable within one window size.
- Treat the current 512 decoder as a **fixed downstream component** while the
  proposal stage is built. Changing both ends at once makes an end-to-end
  number uninterpretable.
- Do not overwrite v0.1–v0.4 artifacts.

---

## The system this plan is building toward

```
map sheet
  → candidate generator          (proposals; recall-oriented)
  → candidate filter/classifier  (hard negatives; precision-oriented)
  → prompt-centred 512 window    (v0.4's representation, ~250 px of slack)
  → MapSAM decoder               (frozen for now; 0.79 macro IoU given a good prompt)
  → mound mask
```

The 180 `mound` annotations provide the positives. The 530
`hard_negative_symbol` annotations — text, roads, decorative marks, grid
artifacts, scan noise, legends, map symbology — are exactly the confusing
negatives the second stage needs, and they have never been used. v0.1–v0.4
deliberately kept them out of the decoder loss because "given a valid prompt,
segment the mound" is not the task they address. This is the task they
address.

---

## Step 1 — Acquire and annotate more sheets (blocking, unchanged)

The binding constraint on every domain-robustness claim. `data/maps`,
`data/georeferenced` and `data/cvat_exports` are empty; `data/curated/datasets`
holds only `mapsam_v0` and `mapsam_v02`. There is nothing in the repository to
add.

Prefer several more sheets from this Soviet 1:50k series **and** several
genuinely different cartographic sources. All of v0.4's variation is within
one series.

Two data gaps to close while annotating: all three `uncertain_ignore`
annotations live on one sheet, so two of three folds have no ignore-mask
coverage in evaluation at all; and `K-34-35-B-g` contributes 17 samples, where
one sample moves a rate by 6%.

One data question to resolve first: `DATASET.md` records 180 `mound`
annotations (146 train / 34 test) but the manifest holds 171 training samples
(137 train / 34 test), and says "each mound annotation produces one training
sample". Nine train annotations are unaccounted for. Segmentation could ignore
this; detection cannot, because recall is measured against annotations, not
against samples.

Everything below except step 4 can proceed without more sheets, but nothing
below produces a defensible robustness claim until they land.

## Step 2 — Candidate generation

The main event. Propose mound locations on an unseen sheet accurately enough
that the existing decoder can segment them.

No architecture is prescribed. The evaluation protocol matters more, and it is
prescribed, because the wrong metric will select the wrong detector here: a
model with good conventional mAP but occasional 20–30 px localization errors
will underperform one with mediocre box IoU that lands within 10 px almost
always.

**Metric hierarchy**, in decreasing order of what it settles:

1. **Mound recall within radius R**, reported at R = 5, 10, 15 and 25 source
   px. The curve is the deliverable, not any single R — it is what composes
   with v0.4's tolerance curve.
2. **Median and p90 centre-location error** over matched mounds. The p90
   matters more than the median: the tail is what falls outside 15 px.
3. **Candidate precision**, and false positives per unit map area, which is
   the figure that scales to a whole sheet.
4. **Recall against `hard_negative_symbol` annotations** — how many proposals
   land on one. This is the measurement that distinguishes a model that has
   learned the mound symbol from one that has learned "ink".
5. **End-to-end rate of annotated mounds that yield a MapSAM mask at IoU ≥0.5
   and ≥0.75.** This is eventually the only metric that matters; the four
   above exist to diagnose it.

Split by sheet, using the same leave-one-sheet-out folds, so detection and
segmentation numbers stay commensurable.

## Step 3 — End-to-end evaluation

Run the full pipeline: proposals from step 2, prompts derived from proposals,
masks from the 512 decoder, per sheet.

This is the first number in the project that answers the actual question.
Expect it well below 0.79, and expect the gap to be the detector's
localization error read through v0.4's jitter curve. If it is not — if
end-to-end quality is worse than the jitter curve predicts from the measured
error distribution — then something outside localization is wrong, and that is
worth knowing early.

## Step 4 — Prompt-jitter augmentation, *conditional on step 2*

Deliberately **not** scheduled before the detector exists.

Widening the segmenter's tolerance and tightening the detector's accuracy are
two routes to the same gap, and only one of them is known to be needed. If
step 2's p90 centre error lands within 5–8 px, there is no reason to train
MapSAM to accept 25–30 px prompts — and a reason not to: v0.4 step 2 found the
zero-IoU samples sat closer to their neighbours than average (median
nearest-neighbour distance 70 px against 98). A decoder trained to be
indifferent to a 30 px offset is a decoder trained to be less certain which of
two adjacent symbols it was asked about.

So the trigger is step 2's measured error distribution: run this only if the
p90 lands outside the useful band, and then match the augmentation magnitude
to the measured distribution rather than to a round number.

The mechanism is already in place — `MapSamDataset` takes `prompt_jitter_xy`;
training needs a per-sample random draw rather than a fixed offset. Guard
against jittering the target out of the window: at 512 px clipping begins
around 240 px, and v0.4 measured GT area falling 470 → 291 source px² between
offsets 250 and 300. Log mean GT area per epoch.

## Step 5 — Permanent grouped splits

Once enough sheets exist: train / validation / frozen test, split by sheet.
Validation for checkpoint and hyperparameter selection; the frozen test
touched only at milestones.

v0.4 sharpened why. The late-training decision drift its step 2 traced — fold
A going from 2 zero-IoU at epoch 5 to 19 at epoch 50, with the peak logit
correctly placed but negative — is smaller at 512 but not absent; fold C still
loses 6.5% between its epoch-5 peak and epoch 50. It is not worth solving
directly in v0.5. It is worth recording, because early stopping against a real
validation split is the fix, and there is currently no leak-free way to choose
an epoch.

## Deferred

**Window sizes below 512 (256, 384).** Deprioritized on value, not evidence.
Nothing in v0.4 shows a tighter window cannot help — even 512 puts only 2.76
tokens on a mound, and at zero prompt error more coverage might reduce the
~92–116 px² residual further. But that optimizes something already clearing
IoU 0.5 on 99–100% of instances while the pipeline cannot produce a prompt at
all. Reopen if the detector becomes strong and segmentation becomes the
bottleneck again.

Also deferred: calibration; new losses; encoder variants; encoder unfreezing;
any further positive-weight search; box-scale jitter, as opposed to the
translation v0.4 measured.

## Known debt, unrelated to MapSAM

Seven pre-existing `sam2_mcp` / `sam2_backend` test failures, and
repository-wide ruff errors outside `src/archeo_topia` (`make lint` is scoped
to `src/services tests`, so `src/archeo_topia` is not covered by the Make
target). Neither affects the v0.4 conclusions; both are worth a separate pass.
