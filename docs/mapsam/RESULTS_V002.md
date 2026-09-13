# MapSAM v0.2 — Training Results

Second training iteration. The only substantive change from v0.1 is the
**target mask**: each sample is now supervised on the single mound instance its
prompt points at, rather than on every mound present on the map sheet. The
architecture, the frozen encoder, the optimiser and the source annotations are
unchanged.

The v0.1 results in `RESULTS.md` are left untouched. Read them as a record of a
different objective — see the comparability caveats below before putting the
two sets of numbers side by side.

Background and evidence for the change: `TRAINING_V001_FINDINGS.md`.

## What changed

| | v0.1 | v0.2 |
|---|---|---|
| Target mask | Whole sheet, every mound | Single prompted instance |
| Component labelling | n/a | Original resolution, before resize |
| Mean target foreground | ~30x inflated | 5.4 px @ 256x256 logits |
| Ignore masks | Empty in all 12 files | 2 files populated (600, 892 px) |
| Dataset root | `data/curated/datasets/mapsam_v0` | `data/curated/datasets/mapsam_v02` |

## Dataset

Rebuilt from the CVAT COCO export
(`data_lake/curated/datasets/cvat/v0.0.1`) with the current pipeline. Source
annotations are identical to v0.1; only the derived dataset is new.

| Metric | Value |
|--------|-------|
| Source images | 12 (3 map sheets) |
| COCO annotations | 180 mound, 530 hard_negative_symbol, 3 uncertain_ignore |
| Training samples | 171 (137 train, 34 test, 0 val) |
| Split strategy | By sheet — train `K-34-35-B-g`, `K-35-51-B-a`; test `K-35-8-G-a` |
| Skipped components | 1 (area < 20 px) |
| Component area | mean 458.9 px, median 445, range 116-1037 |
| Bbox size | mean 31.9 x 30.9 px |

### Target verification

Checked across all 137 training samples after the fix:

| Check | Result |
|-------|--------|
| Targets with more than one connected component | 0 |
| Targets empty at 1024 x 1024 | 0 |
| Targets empty at 256 x 256 (loss resolution) | 0 |
| Mean target foreground @1024 | 85.9 px (range 19-190) |
| Mean target foreground @256 | 5.4 px (range 1-12) |

The concern that downsampling would erase thin mound symbols did not
materialise: no instance vanished at either resolution. 49 of 137 targets are
under 5 px at logit resolution, which matters for interpreting the scores.

## Experiment 1 — Overfit debug

**Goal:** confirm the training loop can fit a tiny sample set under correct
labels. 5 unique samples, repeat factor 200 (1000 effective), 200 epochs,
validation on the same samples. Identical protocol to v0.1.

Run as a 2x2 over `foreground_bce_pos_weight` and `use_bbox_loss_crop`.

| Run | Peak val IoU | @ep | Final val IoU | Final val Dice | GT px | Pred px |
|-----|-------------:|----:|--------------:|---------------:|------:|--------:|
| pw20_cropon | 1.0000 | 10 | 1.0000 | 1.0000 | 5.2 | 5.2 |
| pw20_cropoff | 1.0000 | 10 | 1.0000 | 1.0000 | 5.2 | 5.2 |
| pw200_cropoff | 1.0000 | 10 | 1.0000 | 1.0000 | 5.2 | 5.2 |
| pw200_cropon | 0.9000 | 5 | 0.6875 | 0.7763 | 5.2 | 7.6 |

Three of four configurations reach a perfect overfit by epoch 10 and hold it to
epoch 200, predicting exactly as many positive pixels as the ground truth
contains. Final training loss is 0.000000 in all four.

`pw200_cropon` is the exception and never converges: it oscillates between 0.44
and 0.90 and ends at 0.688, consistently over-predicting (7.6 px against a 5.2 px
target). A 200x positive weight combined with a loss crop that already
concentrates the loss on the bbox neighbourhood is too much foreground pressure
for a 5 px target.

For reference, the v0.1 overfit run reached 0.650 after the same 200 epochs.

## Experiment 2 — Decoder-only, held-out sheet

**Goal:** measure generalisation. Full train split (137 samples), validated on
the held-out `K-35-8-G-a` sheet (34 samples) which contributes nothing to
training. 50 epochs. Same 2x2.

| Run | Peak val IoU | @ep | Final val IoU | Final val Dice | Final train loss |
|-----|-------------:|----:|--------------:|---------------:|-----------------:|
| pw200_cropoff | 0.6876 | 25 | 0.5845 | 0.7234 | 0.3545 |
| pw200_cropon | 0.6838 | 25 | 0.5698 | 0.7119 | 0.1572 |
| pw20_cropon | 0.6689 | 5 | 0.6021 | 0.7372 | 0.2584 |
| pw20_cropoff | 0.6172 | 30 | 0.5219 | 0.6757 | 0.5545 |

Validation IoU by epoch:

| Epoch | pw20_cropon | pw200_cropon | pw20_cropoff | pw200_cropoff |
|------:|------------:|-------------:|-------------:|--------------:|
| 5 | 0.669 | 0.645 | 0.397 | 0.450 |
| 10 | 0.610 | 0.641 | 0.421 | 0.551 |
| 15 | 0.602 | 0.646 | 0.442 | 0.620 |
| 20 | 0.664 | 0.665 | 0.583 | 0.667 |
| 25 | 0.620 | 0.684 | 0.548 | 0.688 |
| 30 | 0.576 | 0.661 | 0.617 | 0.632 |
| 40 | 0.589 | 0.642 | 0.458 | 0.610 |
| 50 | 0.602 | 0.570 | 0.522 | 0.585 |

Per-sample statistics at the best epoch (`pw200_cropoff`, epoch 25, 34 held-out
samples):

| Metric | Mean | Min | Max |
|--------|-----:|----:|----:|
| IoU | 0.6876 | 0.3750 | 1.0000 |
| Dice | 0.8048 | 0.5455 | 1.0000 |
| GT positive px | 5.88 | 3 | 9 |
| Predicted positive px | 6.35 | 4 | 11 |
| Max predicted probability | 1.0000 | 1.0000 | 1.0000 |

**No held-out sample scored zero IoU.** Every mound on the unseen sheet was at
least partially segmented, and predicted area tracks ground-truth area closely
(6.35 px predicted against 5.88 px actual).

## Interaction between the loss crop and the positive weight

The two mechanisms both exist to counter the class imbalance, and the 2x2 shows
they partly substitute for each other. Using peak validation IoU on the held-out
sheet:

| Positive weight | Crop on | Crop off | Effect of crop |
|-----------------|--------:|---------:|---------------:|
| 20 | 0.6689 | 0.6172 | **+0.0517** |
| 200 | 0.6838 | 0.6876 | **-0.0038** |

At a low positive weight the crop clearly helps. At a high positive weight its
effect disappears into the noise — the weight alone already supplies enough
foreground signal. The crop also accelerates early convergence markedly: at
epoch 5 the crop-on runs are at 0.645-0.669 while the crop-off runs are at
0.397-0.450.

In the overfit setting the same interaction appears as instability rather than
gain: `pw200_cropon`, the only cell with both mechanisms at full strength, is
also the only cell that fails to converge.

This is why the ablation was run as a full 2x2 rather than as two crop runs at a
single fixed weight: a two-run experiment would have reported whatever the
chosen weight happened to imply about the crop.

## Comparison with v0.1

| | v0.1 | v0.2 |
|---|---|---|
| Overfit, final val IoU (200 epochs) | 0.650 | **1.000** (3 of 4 configs) |
| Decoder-only, peak val IoU | 0.097 (epoch 3) | **0.688** (epoch 25) |
| Decoder-only, val IoU at epoch 10 | 0.025 | 0.551-0.641 |
| Held-out samples with zero IoU | not reported | 0 of 34 |

### Comparability caveats

These numbers are **not** a like-for-like measurement, and the absolute gap
should not be quoted as a speedup or an improvement factor:

1. **Different denominators.** v0.1 scored predictions against the all-mounds
   target over the whole image. v0.2 scores against the single prompted
   instance. The two IoUs answer different questions.
2. **Different dataset build.** v0.2 was rebuilt from the same COCO export, and
   its ignore masks are populated where v0.1's were empty. Source annotations
   are identical.
3. **Coarse quantisation.** Targets are 3-9 px at logit resolution, so IoU moves
   in large discrete steps. At 5 px ground truth and 6 px predicted with 5
   overlapping, IoU is 0.833 — one pixel changes the score by roughly 0.15.
4. **One test sheet.** Generalisation is measured on 34 samples from a single
   held-out sheet. That is enough to falsify "cannot generalise" but not enough
   to estimate a reliable operating score.

What the comparison does support is the qualitative conclusion. v0.1 concluded
that the model "cannot generalize on current data" with root cause "extreme
class imbalance". With the same architecture, the same frozen encoder, the same
137 training samples and the same annotations, the corrected labels produce a
model that segments every mound on an unseen sheet. The blocker was the
supervision signal, not the data volume or the class balance.

## Limitations

- **Resolution floor.** A mound is ~5 px at the decoder's 256x256 output. This
  is the dominant constraint on achievable IoU and is not addressed by anything
  in v0.2. Tiling the sheets, or upsampling before the decoder, would raise the
  ceiling more than any loss tuning.
- **Mild overfitting remains.** All four decoder runs peak around epoch 25-30
  and decline modestly by epoch 50. Nothing like the v0.1 collapse, but early
  stopping around epoch 25 is warranted.
- **530 hard negatives are still unused.** The `hard_negative_symbol` class
  outnumbers mounds 3:1 and currently contributes nothing to training.
- **No validation split.** Splitting by sheet across 3 sheets leaves
  train/test only. Model selection currently reads the test split, which is
  methodologically unsound for any published figure.
- **Sample selection in the overfit runs** is the first 5 samples of the
  manifest, all from one sheet.

## Next steps

1. **Tile the sheets** so mounds occupy a usable pixel fraction at decoder
   resolution. This is the highest-leverage change available.
2. **Early stop at ~epoch 25**, or add a proper validation split and select on
   it rather than on the test sheet.
3. **Use the hard negatives** as explicit negative prompts or as an additional
   loss term.
4. **More sheets.** Three sheets cannot support a trustworthy generalisation
   estimate however good the labels are.
5. **Re-tune the positive weight** now that the crop's contribution is
   understood; the useful range appears to sit between 20 and 200.

## Artifacts

Training outputs are gitignored and live under `artifacts/models/mapsam/`:

- `v0_2_overfit_debug_{pw20,pw200}_{cropon,cropoff}/`
- `v0_2_decoder_only_{pw20,pw200}_{cropon,cropoff}/`

Each contains `metrics.json`, `config_resolved.json`, `checkpoints/`
(`best.pt`, `final.pt`, periodic epochs) and per-epoch
`prediction_stats_{train,val}_e*.jsonl`.

Debug overlays for the best decoder run are in
`v0_2_decoder_only_pw200_cropoff/debug_predictions/` (8 held-out samples,
aggregate IoU 0.6850 / Dice 0.8090). Each overlay labels the ground-truth panel
with its connected-component count, which reads `n=1` throughout — a direct
visual confirmation that targets are single-instance.

Reproduce with:

```bash
python -m archeo_topia.datasets.prepare_mapsam_coco \
    --coco-json <data_lake>/curated/datasets/cvat/v0.0.1/annotations/instances_default.json \
    --images-dir <data_lake>/curated/datasets/cvat/v0.0.1/images/default \
    --output-dir data/curated/datasets/mapsam_v02 --split-by sheet

python -m archeo_topia.datasets.generate_mapsam_prompts \
    --dataset-root data/curated/datasets/mapsam_v02 \
    --output-path data/curated/datasets/mapsam_v02/metadata/training_samples.jsonl

python -m archeo_topia.training.cache_sam_embeddings \
    --config configs/mapsam/mapsam_v0_2_decoder_only_pw200_cropoff.yaml --split train
python -m archeo_topia.training.cache_sam_embeddings \
    --config configs/mapsam/mapsam_v0_2_decoder_only_pw200_cropoff.yaml --split test

python -m archeo_topia.training.train_mapsam_v0 \
    --config configs/mapsam/mapsam_v0_2_decoder_only_pw200_cropoff.yaml
```
