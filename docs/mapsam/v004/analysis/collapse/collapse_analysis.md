# Fold A's zero-IoU cases (v0.4 step 2)

No training. The v0.3 leave-one-sheet-out fold A — train on `K-34-35-B-g` +
`K-35-8-G-a`, evaluate on `K-35-51-B-a` — scores 0.3498 at epoch 50 with 19 of
its 120 samples at IoU exactly zero, 10 of which predict no pixels at all.

The v0.4 plan called this "prediction collapse, not boundary degradation".
Measured, it is mostly neither. It is the same metric quantization v0.3 found
in its Question 2, now in its most extreme form: at full-tile resolution a
mound covers about five pixels on the 256×256 decoder grid, so a prediction
displaced by one or two pixels has *no* overlap with the truth and scores
exactly zero while still sitting on the right symbol.

## How the 19 break down

`mapsam_collapse_analysis` measures displacement instead of overlap: the
distance from the target centroid to the predicted mask's centroid, and — for
samples that predict nothing — to the peak logit, which is defined even when
nothing crosses the threshold. Both converted to source-tile pixels.

| | n | median distance to target |
|---|---:|---:|
| Hits (IoU > 0) | 101 | 5.0 source px |
| **Near misses** (zero IoU, within 40 source px) | **17** | **19.0 source px** |
| **Wrong object** (zero IoU, beyond 40 source px) | **2** | 868 / 1861 source px |

The 40 px threshold is about twice a mound symbol's width (median prompt box
32×30 px). Every near miss lands between 4 and 40 source px of the target —
inside or just outside the symbol itself. Only two samples are genuine
recognition failures, and both are gross: the model segments something 40–90
symbol-widths away.

Per-sample detail, sorted by class, is in `fold_a_fulltile_e50_collapse.csv`.
Overlays for all 19 are in
`artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/debug_predictions_v004_collapse/`.

## The zero-pixel cases are a threshold problem

Ten samples predict nothing. Nine of those ten have their peak logit within
6.5–39.6 source px of the target — they are looking in the right place and
never crossing 0.5. Their logit maxima run from −13.2 to −0.3: the model is
not confused about where the mound is, it is unwilling to commit.

Mean logit max over the 19 zero-IoU samples is 0.99, against 19.6 over the 101
that score anything. That is the cleanest separator in the data, and it is a
calibration signal rather than a localization one.

## It is acquired during training, not intrinsic to those samples

Rerunning the same analysis on fold A's epoch-5 checkpoint:

| epoch | zero-IoU | predicting nothing | near miss | wrong object |
|---:|---:|---:|---:|---:|
| 5 | 2 | 0 | 2 | 0 |
| 50 | 19 | 10 | 17 | 2 |

The per-epoch validation statistics fill in the trajectory: 2 zero-IoU at e5,
6 at e10, 7 at e20, 5 at e30, 15 at e40, 19 at e50; zero-pixel predictions
first appear at e20 and reach 10 by e50.

So it is not that 19 samples are intrinsically hard. Between epoch 5 and 50
the model progressively withdraws from the held-out sheet, and the withdrawal
shows up first as near misses and then as silence. Fold A's oracle peak is at
epoch 5 (0.4911) against 0.3498 at epoch 50, which is the same fact seen
through the mean.

## Weak and absent correlates

- **Target size: absent.** Median source-resolution component area is 454 px
  for the zero-IoU samples against 449 for the rest. The apparent size effect
  in the decoder-grid figures (4.95 vs 5.61 px) is the v0.3 quantization
  artifact again.
- **Tile: present but small.** 8% / 9% / 18% / 21% across the sheet's four
  tiles, on 25 / 11 / 56 / 28 samples.
- **Crowding: weak.** Median nearest-neighbour distance is 70 px for zero-IoU
  samples against 98 px for the rest; 18 of 19 have a neighbour within 256 px,
  against 80 of 101. Suggestive, not established on 19 samples.

## What this predicts for step 3

If 17 of 19 failures are sub-symbol displacements scored against a 5-pixel
target, then magnifying the target should remove most of them. A 512 px window
puts roughly 118 pixels on the decoder grid instead of 5, so a 19 source px
displacement — 2 decoder pixels at full tile — becomes 9.5 decoder pixels
against a target 11 pixels across, which still overlaps.

Step 3 tests this directly. Prediction: fold A's zero-IoU count falls sharply
at 512, and what remains is dominated by the wrong-object class rather than
near misses. If instead the near misses persist, the displacement is real and
resolution is not the explanation.

## Reproduce

```bash
python -m archeo_topia.analysis.mapsam_collapse_analysis \
  --config configs/mapsam/mapsam_v0_3_loso_foldA_pw20_cropon.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/checkpoints/final.pt \
  --name fold_a_fulltile_e50 --output-dir docs/mapsam/v004/analysis/collapse

python -m archeo_topia.analysis.mapsam_collapse_analysis \
  --config configs/mapsam/mapsam_v0_3_loso_foldA_pw20_cropon.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/checkpoints/epoch_5.pt \
  --name fold_a_fulltile_e5 --output-dir docs/mapsam/v004/analysis/collapse

python -m archeo_topia.training.mapsam_predict_debug \
  --config configs/mapsam/mapsam_v0_3_loso_foldA_pw20_cropon.yaml \
  --checkpoint artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/checkpoints/final.pt \
  --split test --sample-ids docs/mapsam/v004/analysis/collapse/fold_a_zero_iou_sample_ids.txt \
  --max-samples 200 \
  --output-dir artifacts/models/mapsam/v0_3_loso_foldA_pw20_cropon/debug_predictions_v004_collapse
```

The e50 run reproduces fold A's v0.3 figure exactly: mean IoU 0.349843 over
120 samples.
