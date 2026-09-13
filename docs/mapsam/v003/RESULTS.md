# MapSAM v0.3 — cross-sheet generalization and input resolution

Follows `../v002/RESULTS.md`. Same model (SAM ViT-B, frozen image and prompt
encoders, mask decoder only), same dataset (`data/curated/datasets/mapsam_v02`,
3 sheets / 12 tiles / 171 samples), same optimizer and loss.

v0.3 asked four questions. Three have clear answers and one of them
invalidates a number in the v0.2 report.

---

## Summary

1. **v0.2 does not generalize across sheets.** The 0.6876 in `../v002/RESULTS.md`
   came from the easiest of the three sheets. The other two score 0.35–0.42
   with the same recipe.
2. **Target size does not limit accuracy — the metric made it look like it did.**
   Absolute boundary error is flat across every target-size bucket. Small
   targets score worse IoU only because the denominator is smaller.
3. **Prompt-centred windows nevertheless help, and substantially.** Absolute
   error per mound falls 64% from full tile to a 512 px window. This is a real
   gain, measured on the source-tile grid, not the IoU inflation that cropping
   produces automatically.
4. **Sheet-to-sheet variance is 2–7× configuration variance.** Which sheet you
   evaluate on matters far more than which of the v0.2 configurations you pick.

Combined: the next bottleneck is **map-domain variation**, and the resolution
fix is already in hand.

---

## Methodology notes

**Epoch selection.** With three sheets and no validation split there is no
leak-free way to choose an epoch. Every table below reports the **final epoch**
(50) as the primary figure. The peak is shown separately and labelled *oracle*,
because it was chosen by looking at the evaluation sheet and is an upper bound,
not a result.

`metrics.json`'s `best_val_iou` field is the IoU at the best-val-*loss* epoch,
not the peak. All peaks here are recomputed from the epoch history.

**Oracle-prompt evaluation.** The prompt encoder is fed a ground-truth-derived
box and centre point for every sample, in training and evaluation alike. These
numbers describe "given a correct mound prompt, how well is the mound
segmented" — not a deployable detector. Candidate generation is a separate
problem and the 530 `hard_negative_symbol` annotations belong to it.

**Reproducibility check.** LOSO fold B is the v0.2 split. It reproduces v0.2
exactly: `pw20_cropon` final 0.6021 / peak 0.6689 @ e5, `pw200_cropoff` peak
0.6876 @ e25. Both match `../v002/RESULTS.md`.

---

## Question 1 — does v0.2 generalize across the three sheets?

**No.** Leave-one-sheet-out, `pw20_cropon`, final epoch:

| Eval sheet | n | train samples | final IoU | oracle peak |
|---|---:|---:|---:|---:|
| K-35-8-G-a  | 34  | 137 | 0.6021 | 0.6689 @ e5 |
| K-35-51-B-a | 120 | 51  | 0.3498 | 0.4911 @ e5 |
| K-34-35-B-g | 17  | 154 | 0.3714 | 0.5234 @ e15 |

Mean across folds: **0.441**, against the 0.6689 v0.2 reported from
`K-35-8-G-a` alone. That sheet is the easiest of the three.

The three folds train on 51 / 137 / 154 samples, so sheet difficulty is
confounded with training-set size. The size-matched arm pins every fold to 51:

| Eval sheet | n | final IoU (51 train) | oracle peak |
|---|---:|---:|---:|
| K-35-8-G-a  | 34  | 0.4567 | 0.5947 |
| K-35-51-B-a | 120 | 0.3498 | 0.4911 |
| K-34-35-B-g | 17  | 0.3263 | 0.4033 |

Training-set size is worth a lot on its own — holding the sheet fixed,
`K-35-8-G-a` drops 0.6021 → 0.4567 when training data goes 137 → 51. But it
does not explain the ranking: fold C trains on the **most** data (154) and
still scores 0.3714. Sheet identity dominates.

Caveat: all three sheets are the same Soviet 1:50k series (K-34/K-35), so this
is variation *within* one map series. Genuinely different cartography is
untested. Fold C evaluates on 17 samples, where one sample moves the score by
~6%.

## Question 2 — does performance correlate with target size?

**Not in absolute terms.** On the 137-sample train split, bucketing by GT
pixels at the 256 decoder looks damning at first:

| GT size | n | mean IoU | **mean abs error** |
|---|---:|---:|---:|
| 1–3 px | 16 | 0.508 | **2.88 px** |
| 4–5 px | 56 | 0.756 | **2.00 px** |
| 6–8 px | 57 | 0.736 | **2.39 px** |
| 9+ px  | 8  | 0.727 | **3.25 px** |

The last column is the finding. Symmetric difference — the count of pixels
where prediction and truth disagree — is **flat**, and if anything grows with
target size. The model makes the same ~2 px boundary error on a 3 px mound as
on a 12 px one. IoU falls on small targets purely because that constant error
is divided by a smaller denominator. Size-vs-absolute-error Spearman across the
four v0.2 runs: −0.005 to +0.20.

This also means Pearson correlation, which v0.3 originally specified, is the
wrong statistic: it reads 0.07–0.20 and would have been reported as "weak size
effect", missing both the real threshold below ~4 px and the artifact.

The held-out sheet cannot answer this question at all — 33 of its 34 samples
fall in 4–8 px, so its correlations swing from −0.03 to +0.59 across configs on
pure range restriction.

Artifacts: `docs/mapsam/v003/analysis/train/`, `.../heldout/`.

## Question 3 — does prompt-centred cropping improve performance?

**Yes, and the improvement is real.** Arms differ only in the input window;
optimizer, learning rate, epochs, loss and positive weight are identical. The
loss-crop margin is scaled per arm (32 / 75 / 150 at 256 logits) so the
supervised ground area stays at ~300 source px in all three.

| Arm | window | decoder IoU | **abs error (source px²)** | GT area (source px²) | encoder tokens |
|---|---|---:|---:|---:|---:|
| A full tile | ~2300 px | 0.6021 | **253.9** | 479.0 | 0.61 |
| B medium | 1024 px | 0.7567 | **135.5** | 470.6 | 1.36 |
| C tight | 512 px | 0.8261 | **91.8** | 470.4 | 2.76 |

Both columns matter and they say different things.

Decoder IoU alone would be misleading. Windowing magnifies the target from ~6
to ~118 pixels on the decoder grid, so the same absolute error scores a far
better IoU automatically — a **single epoch** of arm C reached 0.797, already
past the v0.2 25-epoch best. Any crop experiment scored on decoder IoU
succeeds by construction.

Converting the error to source-tile pixels removes that. It falls 253.9 → 135.5
→ 91.8, a **64% reduction**, on a GT area that is constant across arms
(470–479, confirming all three see the same mounds). Zero-IoU samples: 0/34 in
every arm.

The mechanism is encoder token coverage, not decoder resolution. The 256×256
logits are upsampled from the ViT's 64×64 token grid; a ~21 px mound on a
2400 px tile covers 0.61 of one 16 px patch. Windowing raises that to 1.36 and
2.76 tokens — it is the only lever that moves it while the encoder is frozen.
Even the tight arm is under 3 tokens, so this is probably not exhausted.

Overlays for the same 8 samples in each arm:
`artifacts/models/mapsam/v0_3_res_{a_fulltile,b_window1024,c_window512}_pw20/debug_predictions/`.

## Question 4 — sheet variance vs configuration variance

**Sheet variance is 2–7× larger.**

- Configuration (pw20 vs pw200, same fold): 0.018 / 0.052 / 0.070
- Sheet (same config, size-matched): 0.3263 → 0.4567, spread **0.130**

Choosing between the v0.2 configurations is noise next to which sheet the model
is asked to generalize to. The v0.2 exercise of separating 0.684 from 0.688 was
measuring nothing.

## Probability saturation (diagnostic)

v0.2 reported max predicted probability of 1.0000 for every held-out sample.
That is float32 saturation, not confidence: sigmoid pins to exactly 1.0 for any
logit past ~16, and the stored value was rounded to 6 places on top.

| Run | mean logit max | mean logit min | mean prob in GT | mean prob in background |
|---|---:|---:|---:|---:|
| pw20_cropon  | 25.3 | −63.0 | 0.771 | 2.6e−05 |
| pw200_cropoff| 36.3 | −80.1 | 0.742 | 2.2e−05 |
| window 1024  | 61.0 | −124.8 | 0.876 | 7.0e−05 |
| window 512   | 88.5 | −91.3 | 0.931 | 2.2e−04 |

Saturation is present at both positive weights, so it is not caused by the high
weight — `pw20` saturates too (31/34 samples hit exactly 1.0). Windowing makes
it worse (34/34, logits to 88).

**Do not use max probability as confidence.** Mean probability inside the GT
region (0.74–0.93) does vary usefully and is the better candidate if a
confidence signal is needed later. No calibration attempted.

---

## Recommendation

The plan's decision rules point at Outcome A *and* Outcome B together: cropping
substantially improves all arms, and one sheet is much worse than another.

Ranked next steps:

1. **Adopt the prompt-centred window as the standard representation.** 512 px
   is the best of the two tested and is not obviously the limit — token
   coverage is still only 2.76. Worth testing 256 px before settling.
2. **Add map sheets, and prefer diverse cartography.** This is now the binding
   constraint. Three sheets from one Soviet series cannot support a conclusion
   about map-domain robustness, and two of the three folds have no
   ignore-mask coverage at all in their evaluation set.
3. **Establish train / validation / frozen-test splits once there are enough
   sheets.** Everything in this report is development evidence. Nothing here is
   an untouched test result, including the LOSO folds.
4. **Do not unfreeze the encoder yet.** The frozen encoder is not what limits
   accuracy at 512 px, and the cross-sheet gap will not be fixed by capacity.

Not recommended yet: calibration, new losses, encoder variants, or feeding the
hard negatives into the decoder loss. The task the decoder is trained on
("given a valid prompt, segment the mound") is not the task hard negatives
address.

---

## Reproduce

```bash
# cross-sheet folds and resolution arms (12 runs)
for cfg in configs/mapsam/mapsam_v0_3_loso_*.yaml configs/mapsam/mapsam_v0_3_res_*.yaml; do
  python -m archeo_topia.training.train_mapsam_v0 --config "$cfg"
done

# tables
python -m archeo_topia.analysis.mapsam_compare_runs --prefix v0_3_loso \
  --output-dir docs/mapsam/v003/analysis --name cross_sheet_results
python -m archeo_topia.analysis.mapsam_compare_runs --prefix v0_3_res \
  --output-dir docs/mapsam/v003/analysis --name resolution_results

# size analysis (no training needed; reads existing runs)
python -m archeo_topia.analysis.mapsam_size_analysis \
  --run-dir artifacts/models/mapsam/v0_2_decoder_only_pw200_cropoff \
  --split train --epoch 20 --output-dir docs/mapsam/v003/analysis/train

# overlays
python -m archeo_topia.training.mapsam_predict_debug \
  --config configs/mapsam/mapsam_v0_3_res_c_window512_pw20.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_res_c_window512_pw20/checkpoints/final.pt \
  --split test --output-dir artifacts/models/mapsam/v0_3_res_c_window512_pw20/debug_predictions
```

## Artifacts

Per-run outputs stay where they are written, under
`artifacts/models/mapsam/v0_3_*/`: `metrics.json`, `config_resolved.json`,
`checkpoints/`, `debug_predictions/`, and per-sample
`prediction_stats_val_e*.jsonl`. Aggregated tables are in
`docs/mapsam/v003/analysis/`.

v0.1 and v0.2 artifacts are untouched; every v0.3 config writes to its own
`outputs.root`.
