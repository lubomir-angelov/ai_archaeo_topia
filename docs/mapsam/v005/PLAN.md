# MapSAM v0.5 — plan

Written at the end of v0.4, to be executed in a fresh session. Read
`../v004/RESULTS.md` first; this plan assumes its findings and does not repeat
the evidence for them.

## Where v0.4 left things

- **512 px prompt-centred windows generalize.** All three leave-one-sheet-out
  folds improve 64–69% in source-pixel disagreement area, and the cross-sheet
  spread falls 119.9 → 24.7 source px. Macro mean 103.8 source px / 0.7922
  decoder IoU; 99–100% of instances clear IoU 0.5 on every sheet.
- **Fold A's zero-IoU cases were quantization, not collapse.** 17 of 19 were
  sub-symbol displacements. At 512 all of them are gone, on every fold.
- **The window has ~250 px of placement slack. The prompt has ~15 px.** A
  displaced window is nearly free; a displaced prompt kills the full-tile and
  512 models alike, at the scale of the symbol.

**Working conclusion:** segmentation given a correct prompt is no longer the
bottleneck on this data. The system has no way to produce that prompt, and
v0.4 measured how accurate it must be.

## The decision this plan starts from

v0.3's plan and v0.4's both ended by naming Outcome D — *stop optimizing
segmentation, build the candidate generator* — as a possible branch. v0.4's
step 3 fired it. This plan therefore shifts the centre of gravity from the
decoder to detection, and keeps only the segmentation work that directly
supports it.

Every number in v0.1–v0.4 answers *"given the correct mound location, can
MapSAM segment it?"* Nothing yet answers *"can the system find mounds?"*

## Constraints carried forward

- Image and prompt encoders stay **frozen**.
- **`pw20` and the loss crop stay frozen.** No further positive-weight search.
- **512 px is now the default input representation**, not a candidate. Keep the
  per-arm loss-crop margin scaling (150 at 256 logits for 512 px windows).
- Split by sheet, never by sample. `train_sheets` / `eval_sheets` must be set
  together and may not overlap.
- **Report the scale-invariant metric.** Any cross-window comparison leads with
  `abs_error_source_px`. The IoU≥0.5 rate is a threshold on decoder IoU and is
  only readable within one window size.
- Do not overwrite v0.1–v0.4 artifacts.

---

## Step 1 — Acquire and annotate more sheets (blocking)

The binding constraint on every domain-robustness claim, and now blocking two
steps rather than one. `data/maps`, `data/georeferenced` and
`data/cvat_exports` are empty; `data/curated/datasets` holds only `mapsam_v0`
and `mapsam_v02`. There is nothing in the repository to add.

Prefer several more sheets from this Soviet 1:50k series **and** several
genuinely different cartographic sources. All of v0.4's variation is within
one series, so the residual 24.7 source px cross-sheet spread is a
within-series figure and may understate cross-cartographic behaviour badly.

Two data gaps to close while annotating: all three `uncertain_ignore`
annotations live on one sheet, so two of three folds have no ignore-mask
coverage in evaluation at all; and `K-34-35-B-g` contributes 17 samples, where
one sample moves a rate by 6%.

This step is a data-acquisition task, not a modelling one. Everything below
except step 2 can proceed without it, but nothing below produces a defensible
robustness claim until it lands.

## Step 2 — Prompt-jitter augmentation

Cheap, and it directly widens the tolerance the detector in step 3 has to
meet. `MapSamDataset` already takes `prompt_jitter_xy`; training needs a
per-sample random draw rather than a fixed offset, so this is a small dataset
change plus a config field (suggest a magnitude sampled uniformly in 0–25
source px, direction uniform).

Train the three 512 LOSO folds with augmentation on, then re-run
`mapsam_prompt_jitter` on the resulting checkpoints.

**Read the result as:** the useful number is not the zero-offset score, which
may fall slightly, but the offset at which the 512 arm crosses its own
untrained baseline. If augmentation moves the 15 px tolerance to 30–40 px, the
detector's accuracy requirement halves, which is worth far more than a point
of IoU at zero offset.

Be careful not to jitter so hard that the target leaves the window: at 512 px
clipping begins around 240 px, and v0.4 measured GT area falling 470 → 291
source px between offsets 250 and 300. Log mean GT area per epoch as a guard.

## Step 3 — Candidate generation

The main event. The task is to propose mound locations on an unseen sheet with
enough accuracy — roughly 15 source px untrained, more if step 2 succeeds —
that the existing decoder can segment them.

The 530 `hard_negative_symbol` annotations exist for exactly this and have
never been used: v0.1–v0.4 deliberately kept them out of the decoder loss,
because "given a valid prompt, segment the mound" is not the task they
address. They belong here.

No approach is prescribed. Whatever is chosen, the evaluation protocol matters
more than the architecture:

- Score **detection**, not segmentation: precision and recall against
  annotated mounds, at a source-pixel matching radius justified by step 1's
  tolerance curve rather than chosen for convenience.
- Split by sheet. The same leave-one-sheet-out folds, so detection and
  segmentation numbers are commensurable.
- Report the hard negatives separately — how many proposals land on
  `hard_negative_symbol` annotations is the measurement that says whether the
  model has learned the symbol or learned "ink".

## Step 4 — End-to-end evaluation

Once step 3 produces candidates, run the full pipeline: proposals from the
detector, prompts derived from proposals, masks from the 512 decoder. Report
per-sheet precision, recall and mask quality.

This is the first number in the project that answers the actual question.
Expect it to be much worse than 0.79, and expect the gap between it and 0.79
to be the detector's localization error read through step 1's curve.

## Step 5 — Permanent grouped splits

Once enough sheets exist: train / validation / frozen test, split by sheet.
Validation for checkpoint and hyperparameter selection; the frozen test
touched only at milestones.

v0.4 sharpened why this matters. The epoch-50 drift that step 2 of v0.4 traced
on fold A is smaller at 512 but not absent — fold C still loses 6.5% between
its epoch-5 peak and epoch 50 — and there is still no leak-free way to choose
an epoch. Everything in v0.1–v0.4, LOSO included, is development evidence.

## Deferred

Window sizes below 512 (256, 384): untested, and the return is bounded by
prompt accuracy, so it is not worth runs until step 3 exists. Calibration;
new losses; encoder variants; encoder unfreezing; any further positive-weight
search. Box-scale jitter, as opposed to the translation v0.4 measured.

## Known debt, unrelated to MapSAM

Seven pre-existing `sam2_mcp` / `sam2_backend` test failures, and
repository-wide ruff errors outside `src/archeo_topia` (`make lint` is scoped
to `src/services tests`, so `src/archeo_topia` is not covered by the Make
target). Neither affects the v0.4 conclusions; both are worth a separate pass.
