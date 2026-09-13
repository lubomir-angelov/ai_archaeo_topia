# MapSAM v0.4 — plan

Written at the end of v0.3, to be executed in a fresh session. Read
`../v003/RESULTS.md` first; this plan assumes its findings and does not repeat
the evidence for them.

## Where v0.3 left things

v0.3 separated three phenomena that v0.2 had conflated:

- **Domain variation.** Leave-one-sheet-out gives 0.6021 / 0.3498 / 0.3714
  (macro 0.441, sample-weighted 0.402). v0.2's 0.6876 came from the easiest
  sheet. Sheet spread (0.130) is 2–7× configuration spread (0.018–0.070).
- **Metric quantization.** Mask disagreement area is flat (~2–3 px) across every
  target-size bucket. Small targets score worse IoU because the denominator is
  smaller, not because the model resolves them worse.
- **Encoder spatial resolution.** A ~21 px mound covers 0.61 of one ViT patch on
  a full tile. Windowing raises that to 1.36 (1024 px) and 2.76 (512 px), and
  disagreement area falls 253.9 → 135.5 → 91.8 source px² — a real 64%
  reduction, not the IoU inflation that cropping produces automatically.

**Working conclusion:** MapSAM learns the task. Performance is dominated by
map-sheet variation and by insufficient spatial representation of the mound in
SAM's image encoder. Decoder mask size and loss hyperparameters are secondary.

## The gap this plan closes

The resolution arms (A/B/C) were all evaluated on `K-35-8-G-a` — the easy sheet.
So v0.3 demonstrated that **512 improves segmentation on the easy sheet**. It
did *not* demonstrate that 512 reduces the cross-sheet gap. Treating 512 as the
settled default would repeat exactly the error v0.3 found in v0.2: promoting a
single-sheet result to a general conclusion.

512 is therefore the **candidate** default, not the default, until step 3 below.

## Constraints carried forward

- Image and prompt encoders stay **frozen**. Revisit only if steps 1–5 leave a
  persistent ceiling.
- **Freeze `pw20` + loss crop.** No further positive-weight sweep. With sheet
  variance 2–7× configuration variance, weight tuning is measuring noise.
- Keep the **per-arm loss-crop margin scaling** (32 / 75 / 150 at 256 logits),
  which holds the supervised ground area near-constant across window sizes.
  Without it, window size silently changes the loss extent too.
- Split by sheet, never by sample. `select_samples` enforces this;
  `train_sheets` / `eval_sheets` must be set together and may not overlap.
- **Report the scale-invariant metric.** Decoder-space IoU is not comparable
  across window sizes. Any windowed comparison must lead with disagreement area
  in source pixels.
- Hard negatives stay out of the decoder loss. They belong to candidate
  generation, a different task.
- Do not overwrite v0.1–v0.3 artifacts.

---

## Step 1 — Prompt-jitter robustness (no training)

Runs against existing checkpoints. Cheapest experiment available and it can
invalidate the 512 recommendation before any compute is spent on it.

The window is prompt-centred, so localization error displaces the window itself.
A 100 px offset is 4% of a 2300 px tile but **20%** of a 512 px window, and at
some offset the mound leaves the window entirely. Every v0.3 number uses
ground-truth-derived prompts; a real detector will not.

Sweep centre offsets (suggest 0, 10, 25, 50, 100, 200 source px, several
directions per magnitude) against the full-tile and 512 checkpoints. Report
disagreement area and IoU≥0.5 rate versus offset for each.

**Read the result as:** if 512 degrades much faster than full tile, the window
size must be chosen jointly with the achievable localization accuracy, not
purely on segmentation quality. That changes step 2's target.

## Step 2 — Diagnose fold A's collapse cases (no training)

Fold A scores 0.3498 with **19/120 zero-IoU**, and 10 of those predict *zero
pixels*. That is prediction collapse, not boundary degradation, and mean IoU
hides it. It is not a small-target effect (zero-IoU GT sizes 2–7 px against a
non-zero mean of 5.61) and it is spread across all four tiles (8/9/18/21%).

Generate overlays for those 19 samples and characterize the failure. Resolution
and collapse are different problems; windowing may or may not touch this one.

## Step 3 — 512-window leave-one-sheet-out (the joining experiment)

Three uncached runs, ~45 min total. Same protocol as the v0.3 full-tile LOSO:
`pw20`, fixed epoch count, final-epoch as the primary figure, oracle peak
reported separately.

Compare against the v0.3 full-tile LOSO table, per sheet, in **source-pixel
disagreement area** as well as IoU.

**Read the result as:**

- *All three sheets improve substantially* → spatial representation was
  responsible for much of what looked like domain shift. Proceed to step 4.
- *Only the easy sheet improves* → two largely independent problems, encoder
  resolution **and** genuine domain shift. Deprioritize window tuning; go
  straight to more sheets (step 5).

Also record macro and sample-weighted means separately from here on.

## Step 4 — Locate the resolution optimum (conditional on step 3)

Only if step 3 shows broad improvement. Test 256 px (~5.3 tokens) and 384 px,
bounded by what step 1 says about localization tolerance — a tighter window is
worth less if it is more fragile to prompt error. Two or three runs, not a grid.

## Step 5 — More sheets

The binding constraint regardless of how steps 1–4 resolve. Prefer several
sheets from this Soviet 1:50k series **and** several genuinely different
cartographic sources; all of v0.3's variation is within one series, so
cross-cartographic behaviour is entirely untested.

Note two current data gaps: all three `uncertain_ignore` annotations live on one
sheet, so two of three folds have no ignore-mask coverage in evaluation at all;
and `K-34-35-B-g` contributes only 17 samples, where one sample moves the score
by ~6%.

## Step 6 — Permanent grouped splits

Once enough sheets exist: train / validation / frozen test, split by sheet.
Validation for checkpoint and hyperparameter selection; the frozen test touched
only at milestones. Everything in v0.1–v0.3, LOSO included, is development
evidence — none of it is an untouched test result.

---

## The decision this plan is really making

Every v0.3 number answers *"given the correct mound location, can MapSAM segment
it?"* — prompts are ground-truth-derived. It does not answer *"can the system
find mounds?"*

On the easy sheet at 512, **100% of instances clear IoU 0.5 and 91% clear 0.75**.
If step 3 shows that holding across sheets, the v0.3 plan's **Outcome D** fires:
stop optimizing segmentation and build the candidate generator — which is what
the 530 `hard_negative_symbol` annotations are for.

So step 3 is not only "how much domain shift survives windowing". It is the
branch point between *keep improving the representation* and *segmentation is
good enough; the unsolved problem is detection*. Step 1 informs it, because a
detector's localization error is what the window has to tolerate.

## Deferred

Calibration; new losses; encoder variants; encoder unfreezing; hard negatives in
the decoder loss; any further positive-weight search.

## Known debt, unrelated to MapSAM

Seven pre-existing `sam2_mcp` / `sam2_backend` test failures, and 223
repository-wide ruff errors outside `src/archeo_topia` (`make lint` is scoped to
`src/services tests`, so `src/archeo_topia` is not covered by the Make target).
Neither affects the v0.3 conclusions; both are worth a separate pass.
