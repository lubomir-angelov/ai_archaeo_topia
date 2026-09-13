# Prompt-jitter robustness (v0.4 step 1)

No training. Three arms run existing v0.3 checkpoints over a grid of prompt
offsets: 8 compass directions per magnitude, diagonals unit-scaled so every
direction at one magnitude displaces the prompt by the same distance. The
evaluation set is the `test` split, `K-35-8-G-a`, 34 samples — the same sheet
the v0.3 resolution arms were trained and evaluated against.

| Arm | checkpoint | what the offset moves |
|---|---|---|
| `fulltile` | `v0_3_res_a_fulltile_pw20/final.pt` | the point and box prompts |
| `window512` | `v0_3_res_c_window512_pw20/final.pt` | the prompts **and** the input window |
| `window512_gtprompt` | `v0_3_res_c_window512_pw20/final.pt` | the input window only |

The third arm is the control. Both costs of a localization error arrive
together in the second arm — a worse prompt and a displaced input — and they
have to be separated before "512 is fragile" can mean anything.

Lead column is `mean_abs_error_source_px`: symmetric difference converted to
source-tile pixels. It is an **area**, so the unit is px²; prompt offsets in
the same table are distances, in px. (The field name says `_px` because it is
inherited from v0.3.) Decoder IoU is not comparable between window sizes, and
the IoU≥0.5 rate inherits that, so the IoU columns are only readable down a
column, not across.

## Result

| offset | full tile err | 512 err | 512 advantage | full tile IoU | 512 IoU | full tile ≥0.5 | 512 ≥0.5 | control err | control ≥0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 253.9 | **91.8** | +162.1 | 0.6021 | 0.8261 | 0.71 | 1.00 | 91.8 | 1.00 |
| 5 | 287.4 | **140.5** | +147.0 | 0.5699 | 0.7396 | 0.71 | 1.00 | 96.8 | 1.00 |
| 10 | 365.0 | **255.0** | +110.0 | 0.4824 | 0.5460 | 0.55 | 0.69 | 95.5 | 1.00 |
| 15 | 479.0 | **444.6** | +34.5 | 0.3650 | 0.2880 | 0.34 | 0.02 | 96.8 | 1.00 |
| 25 | 702.4 | 734.7 | −32.3 | 0.1222 | 0.0089 | 0.06 | 0.00 | 98.4 | 1.00 |
| 50 | 767.1 | 744.2 | +22.8 | 0.0012 | 0.0000 | 0.00 | 0.00 | 106.2 | 1.00 |
| 100 | 704.2 | 734.2 | −30.0 | 0.0000 | 0.0000 | 0.00 | 0.00 | 100.6 | 1.00 |
| 200 | 666.8 | 718.6 | −51.9 | 0.0000 | 0.0000 | 0.00 | 0.00 | 100.9 | 1.00 |
| 250 † | 684.1 | 670.1 | +14.0 | 0.0000 | 0.0000 | 0.00 | 0.00 | 94.2 | 1.00 |
| 300 † | 685.6 | 519.3 | +166.3 | 0.0000 | 0.0000 | 0.00 | 0.00 | 91.3 | 0.62 |

† Past 250 px the 512 px window starts cutting the mound off: mean GT area in
the window falls 470.4 → 432.2 (250 px) → 291.0 (300 px). Error figures for
the windowed arms at those offsets are measured against a partly missing
target and are not comparable with the rows above them. The full-tile arm is
unaffected; its GT area is 479.0 at every offset.

## Three findings

**1. A displaced window is almost free. A displaced prompt is not.**

The control arm is flat: from 0 to 250 px of window displacement the error
moves 91.8 → 94.2 px² and every sample still clears IoU 0.5. The window can be
pushed until the mound is nearly at its edge with no measurable cost. The
only failure is at 300 px, where the mound is leaving the window entirely
(GT area down 38%) and 38% of samples fall below IoU 0.5.

So the v0.4 plan's framing — that the window size must be chosen jointly with
the achievable localization accuracy, because the window moves with the prompt
— is not what the data says. Window placement has roughly 250 px of slack at
512. Prompt accuracy has about 15.

**2. Both arms die at the same scale, and it is the scale of the symbol.**

Full tile is not the robust alternative. It degrades from 253.9 to 702.4 over
the same 25 px, and its IoU≥0.5 rate falls 0.71 → 0.06. By 50 px both arms are
at zero mean IoU. This is what a prompted segmenter does: it segments what is
at the prompt, and 25 px is already most of a mound symbol's ~21–30 px width.

**3. The 512 advantage survives to about 15–20 px of prompt error.**

512 is better at 0, 5, 10 and 15 px and worse at 25, so the crossover is near
18 px. The advantage decays roughly linearly: +162 → +147 → +110 → +34 → −32.
Past 25 px both arms are equally useless and the sign of the difference is
noise between two saturated failure modes.

Read together: adopting 512 does not impose a *tighter* localization
requirement than full tile does. Both need the prompt within roughly 15 px.
What 512 buys is a 64% lower error when that requirement is met, and it keeps
most of that advantage across the tolerance band the full-tile arm needs
anyway.

## Directional anisotropy

Averaged over offsets 5–25 on the full-tile arm, error by direction:

| NE | N | E | NW | SE | S | SW | W |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 355 | 370 | 432 | 447 | 490 | 520 | 522 | 531 |

Northward displacement costs about 1.4× less than west or southwest. Not
investigated; the plausible cause is that mound symbols on this series carry
an associated element (an elevation label, a neighbouring symbol) on one side,
so the model has more to latch onto in some directions than others. Recorded
because a single-direction jitter experiment would have drawn a curve 30%
optimistic or pessimistic depending on which direction it picked.

## What this does not cover

- Box *scale* error. The jitter translates the box rigidly, which models a
  detector that sizes correctly and centres wrongly. A detector that proposes
  boxes 50% too large is untested.
- Jitter-augmented training. Both checkpoints were trained with
  ground-truth-derived prompts only, so this measures a jitter-naive model.
  Whether training with jitter widens the 15 px tolerance is the obvious next
  question and is untested.
- One sheet. The resolution arms both evaluate on `K-35-8-G-a`, the easiest of
  the three. The relative comparison between arms is sound, but the absolute
  tolerance on a harder sheet is unknown.

## Reproduce

```bash
OFF=0,5,10,15,25,50,100,200,250,300
python -m archeo_topia.analysis.mapsam_prompt_jitter \
  --config configs/mapsam/mapsam_v0_3_res_a_fulltile_pw20.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_res_a_fulltile_pw20/checkpoints/final.pt \
  --label fulltile --offsets $OFF --output-dir docs/mapsam/v004/analysis/jitter

python -m archeo_topia.analysis.mapsam_prompt_jitter \
  --config configs/mapsam/mapsam_v0_3_res_c_window512_pw20.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_res_c_window512_pw20/checkpoints/final.pt \
  --label window512 --offsets $OFF --output-dir docs/mapsam/v004/analysis/jitter

python -m archeo_topia.analysis.mapsam_prompt_jitter \
  --config configs/mapsam/mapsam_v0_3_res_c_window512_pw20.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_res_c_window512_pw20/checkpoints/final.pt \
  --label window512_gtprompt --no-jitter-prompts --offsets $OFF \
  --output-dir docs/mapsam/v004/analysis/jitter
```

Each arm writes `jitter_<label>_by_direction.csv` (one row per offset and
direction), `jitter_<label>_by_offset.csv` (averaged over directions, plus the
worst direction) and `jitter_<label>_summary.json` to `--output-dir`. The
per-sample rows — tens of thousands for a full sweep — go to the evaluated
checkpoint's own run directory under `artifacts/`, following the project
convention that docs hold aggregates and artifacts hold per-sample outputs.
`--per-sample-dir` overrides that.

The zero-offset rows reproduce the v0.3 resolution table exactly: 253.9 source
px / 0.6021 IoU for full tile, 91.8 / 0.8261 for 512.
