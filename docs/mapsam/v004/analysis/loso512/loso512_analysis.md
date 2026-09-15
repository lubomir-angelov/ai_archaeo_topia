# 512-window leave-one-sheet-out (v0.4 step 3)

Three training runs, one per fold, identical to the v0.3 full-tile LOSO
protocol except for the input window. The v0.3 resolution arms established
that a 512 px prompt-centred window improves segmentation *on the easy sheet*;
this asks whether it narrows the cross-sheet gap.

Differences from `mapsam_v0_3_loso_fold{A,B,C}_pw20_cropon.yaml`:

| | v0.3 | v0.4 |
|---|---|---|
| `dataset.window_px` | unset (full tile) | 512 |
| `loss.bbox_loss_crop_margin` | 32 | 150 |
| `model.use_cached_embeddings` | true | false |

The margin change holds the supervised ground area near-constant at ~300
source px across window sizes, as v0.3 established. The cache change is
forced: the embedding cache holds one tensor per image tile, and a
prompt-centred window makes the encoder input per sample.

Everything else — `pw20`, 50 epochs, lr 1e-4, seed 42, the fold sheet
assignments — is unchanged. Final epoch is the primary figure; the peak is
reported separately and labelled oracle, because it was chosen by looking at
the evaluation sheet.

## Result

Lead column is source-pixel disagreement area, in **px²**. Decoder IoU is not
comparable across window sizes; it is shown because the v0.3 tables are in it.
Centroid displacements later in this note are distances, in px.

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

Quality distribution and oracle peaks:

| Fold | 512 ≥0.5 | 512 ≥0.75 | full tile ≥0.5 | full tile ≥0.75 | 512 zero-IoU | full tile zero-IoU | 512 final | 512 oracle peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.99 | 0.78 | 0.30 | 0.05 | 0 / 120 | 19 / 120 | 0.7930 | 0.8035 @ e5 |
| B | 1.00 | 0.91 | 0.71 | 0.15 | 0 / 34 | 0 / 34 | 0.8261 | 0.8360 @ e15 |
| C | 1.00 | 0.53 | 0.29 | 0.12 | 0 / 17 | 2 / 17 | 0.7574 | 0.8104 @ e5 |

Fold B reproduces the v0.3 resolution arm C exactly — 0.8261 decoder IoU, 91.8
source px² — as it must, since arm C's split-based selection partitions the
data the same way fold B's sheet-based selection does. That is the
configuration check for these runs.

## Reading

This is unambiguously the plan's first branch: **all three sheets improve
substantially**, by 64–69% each, and the cross-sheet spread falls by a factor
of five (119.9 → 24.7 source px²).

Much of what v0.3 measured as domain shift was insufficient spatial
representation of the mound in a frozen encoder. v0.3's four-fifths ranking
survives in the sense that fold B is still the best fold, but the gap that
motivated "map-domain variation is the next bottleneck" has largely closed:
103.8 px² macro against 91.8 px² on the easiest sheet is a 13% spread, where
full tile was a 47% spread.

Two things this does **not** show.

It does not show that domain robustness is solved. What it establishes is
*within-series* consistency: all three sheets are the same Bulgarian 1:25k
series, so this revises v0.3's diagnosis of what caused the variation it saw —
the representation, mostly — without establishing anything about a different
cartographic source. Cross-cartographic behaviour remains entirely untested,
and the residual 24.7 px² spread within one series may understate what a
different series would cost.

It does not show that the epoch-50 degradation is gone. Fold C still peaks at
epoch 5 (0.8104) and ends at 0.7574, a 6.5% relative drop; fold A drops 1.3%
and fold B 1.2%. The withdrawal that step 2 traced on the full-tile fold A is
smaller here but not absent, and with 17 evaluation samples on fold C it is
also not well measured.

## Step 2's prediction, tested

Step 2 predicted that if 17 of fold A's 19 zero-IoU samples were sub-symbol
displacements scored against a five-pixel target, magnifying the target would
remove most of them.

Fold A at 512: **0 of 120 zero-IoU**, 0 predicting nothing, and median hit
displacement 0.99 source px against 5.01 at full tile. Folds B and C are also
at zero. The two samples step 2 classified as wrong-object failures are fixed
too — which is a different mechanism worth naming: a 512 px window physically
excludes a distractor 868 or 1861 source px away, so the model is no longer
able to choose it.

## Reproduce

```bash
for f in A B C; do
  python -m archeo_topia.training.train_mapsam_v0 \
    --config configs/mapsam/mapsam_v0_4_loso512_fold${f}_pw20.yaml
done

python -m archeo_topia.analysis.mapsam_compare_runs --prefix v0_4_loso512 \
  --output-dir docs/mapsam/v004/analysis/loso512 --name cross_sheet_512_results
```

The source-pixel columns need a separate evaluation pass. The v0.3 LOSO runs
trained on cached embeddings, and the embedding dataset does not carry
`window_xyxy`, so those runs never recorded `abs_error_source_px`. Re-scoring
each checkpoint through the non-cached dataset recovers it; the prompt-jitter
tool at zero offset is exactly that evaluation, so it is reused rather than
duplicated:

```bash
for f in A B C; do
  python -m archeo_topia.analysis.mapsam_prompt_jitter \
    --config configs/mapsam/mapsam_v0_3_loso_fold${f}_pw20_cropon.yaml \
    --checkpoint artifacts/models/mapsam/v0_3_loso_fold${f}_pw20_cropon/checkpoints/final.pt \
    --label fold${f}_fulltile --offsets 0 --output-dir docs/mapsam/v004/analysis/loso512
  python -m archeo_topia.analysis.mapsam_prompt_jitter \
    --config configs/mapsam/mapsam_v0_4_loso512_fold${f}_pw20.yaml \
    --checkpoint artifacts/models/mapsam/v0_4_loso512_fold${f}_pw20/checkpoints/final.pt \
    --label fold${f}_window512 --offsets 0 --output-dir docs/mapsam/v004/analysis/loso512
done
```

Hence the `jitter_fold*_by_offset.csv` files in this directory: each holds a
single zero-offset row, which is an ordinary evaluation, not a jitter sweep.
The full-tile rows reproduce the v0.3 decoder IoUs exactly (0.3498 / 0.6021 /
0.3714), which is the check that the re-scoring pass is faithful.
