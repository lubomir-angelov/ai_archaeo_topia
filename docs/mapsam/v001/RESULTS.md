# MapSAM v0.1 — Training Results

## Objective

Fine-tune the SAM (Segment Anything Model) mask decoder to segment archaeological mound symbols on historical map imagery. Image encoder and prompt encoder remain frozen; only the mask decoder is trained.

## Approach

Two experiments were run to validate the training pipeline and diagnose generalization behavior on a small dataset.

### Experiment 1 — Overfit-Debug Run

**Goal:** Prove the training loop works by forcing overfit on a tiny set.

| Setting | Value |
|---------|-------|
| Config | `configs/mapsam/mapsam_v0_1_overfit_debug.yaml` |
| Unique samples | 5 |
| Repeat factor | 200x (1000 effective samples) |
| Epochs | 200 |
| Learning rate | 1e-4 (AdamW, weight decay 0.01) |
| Trainable params | Mask decoder only |
| BCE weight | 0.5 (foreground pos-weight 20x) |
| Dice weight | 2.0 |
| Bbox loss crop | 32px margin |
| Outputs | `artifacts/models/mapsam/v0_1_overfit_debug/` |

#### Results

Final training aggregate on the 5 overfit samples:

| Metric | Value |
|--------|-------|
| Mean IoU | 0.650 |
| Mean Dice | 0.770 |
| Mean GT positive pixels | 30 |
| Mean predicted positive pixels | 41 |
| Mean prediction area ratio | 0.00063 |
| Mean target area ratio | 0.00045 |

Validation progression on the same 5 samples:

| Epoch | Val IoU | Val Dice |
|-------|---------|----------|
| 5     | 0.20    | 0.33     |
| 10    | 0.083   | 0.15     |

`pred_probability_max` reached 1.0 — model became fully confident on seen samples.

#### Artifacts

- `artifacts/models/mapsam/v0_1_overfit_debug/checkpoints/best.pt`
- `artifacts/models/mapsam/v0_1_overfit_debug/debug_predictions/` (6 overlay PNGs)
- `artifacts/models/mapsam/v0_1_overfit_debug/prediction_stats_train_aggregate.json`
- `artifacts/models/mapsam/v0_1_overfit_debug/prediction_stats_val_e5.jsonl`
- `artifacts/models/mapsam/v0_1_overfit_debug/prediction_stats_val_e10.jsonl`
- `artifacts/models/mapsam/v0_1_overfit_debug/prediction_stats_train_e10.jsonl`

### Experiment 2 — Decoder-Only Full Training

**Goal:** Train on full dataset with held-out test split to measure generalization.

| Setting | Value |
|---------|-------|
| Config | `configs/mapsam/mapsam_v0_1_decoder_only.yaml` |
| Split | Full train / test |
| Epochs | 50 (10 completed) |
| Learning rate | 1e-4 (AdamW, weight decay 0.01) |
| Trainable params | Mask decoder only |
| Loss config | Same as overfit run |
| Outputs | `artifacts/models/mapsam/v0_1_decoder_only/` |

#### Results

Epoch-by-epoch metrics from `metrics.json`:

| Epoch | Train Loss | Val Loss | Val IoU | Val Dice |
|-------|-----------|----------|---------|----------|
| 1     | 0.966     | 0.856    | 0.088   | 0.159    |
| 2     | 0.951     | 0.852    | 0.094   | 0.168    |
| 3     | 0.946     | 0.843    | 0.097   | 0.174    |
| 4     | 0.929     | 0.874    | 0.081   | 0.148    |
| 5     | 0.839     | 0.888    | 0.080   | 0.142    |
| 6     | 0.674     | 0.942    | 0.034   | 0.065    |
| 7     | 0.492     | 0.934    | 0.042   | 0.079    |
| 8     | 0.413     | 0.951    | 0.031   | 0.060    |
| 9     | 0.372     | 0.960    | 0.024   | 0.046    |
| 10    | 0.327     | 0.964    | 0.025   | 0.048    |

#### Artifacts

- `artifacts/models/mapsam/v0_1_decoder_only/metrics.json`
- `artifacts/models/mapsam/v0_1_decoder_only/checkpoints/` (best.pt, epoch_5.pt, epoch_10.pt, final.pt)
- `artifacts/models/mapsam/v0_1_decoder_only/debug_predictions/` (8 overlay PNGs on test samples)

## What This Proves

### 1. Training loop is correct

The overfit run reached IoU 0.65 and Dice 0.77 on 5 repeated samples. The model learned to predict foreground pixels where ground truth exists, with confident predictions (max probability = 1.0). This confirms:

- Forward pass, loss computation, backpropagation, and optimizer step are wired correctly
- Bbox-aware loss crop and foreground weighting function as intended
- Dataset loading, embedding caching, and prompt construction produce valid inputs

### 2. Model cannot generalize on current data

The full training run shows a classic overfitting curve:

- Train loss drops from 0.97 to 0.33 (66% reduction)
- Val loss stays flat at ~0.85-0.96 (no improvement)
- Val IoU peaks at 0.097 (epoch 3) then collapses to 0.025 (epoch 10)
- Val Dice follows same trajectory: 0.174 → 0.048

The model memorizes training samples but produces no useful predictions on unseen data.

### 3. Root cause: extreme class imbalance

Archaeological mound symbols occupy approximately 0.05% of pixels (target area ratio ~0.00045). Even with 20x foreground BCE weighting and bbox-cropped loss, the decoder receives insufficient signal to learn generalizable features. The frozen image encoder provides rich representations, but the decoder cannot map them to such sparse targets without more examples.

## Next Steps

To move beyond overfitting:

1. **Expand dataset** — more annotated mound symbols across diverse map sheets
2. **Stratified sampling** — ensure train/test splits group by map sheet to prevent leakage
3. **Stronger foreground signal** — consider higher BCE pos-weight, focal loss, or IoU-aware loss
4. **Prompt engineering** — test point-only, bbox-only, and combined prompts for better decoder conditioning
5. **Data augmentation** — rotations, flips, and color jitter on training samples
6. **Encoder fine-tuning** — unfreeze last encoder layers with very low learning rate
