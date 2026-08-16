# MapSAM Training Guide

The key point: SAM is prompt-based. So your next step is to generate prompts from your masks:

```
image + generated bbox/point prompt → SAM → predicted mound mask
```

SAM is built around an image encoder, prompt encoder, and mask decoder, and it predicts masks from prompts such as points or boxes. MapSAM's repo also expects a prepared dataset path and trains from a dataset root, so your `mapsam_v0/` folder is the right starting point.

## What to do now

### 0. Stop if masks are still fully white

Before training, verify:

- `masks` = mostly black with small white mound blobs
- `ignore_masks` = mostly black, sometimes with white uncertain regions

If positive masks are still full white, fix Step 1 first.

### 1. Create prompt-generation from masks

From each mound mask, generate training prompts:

```
binary mound mask
  ↓
connected components
  ↓
one training sample per mound blob
  ↓
bbox prompt + center point prompt
```

For each connected component:

- `bbox` = `[x_min, y_min, x_max, y_max]`
- `point` = center of component
- `target` = mask for that one mound instance

Even though your exported mask is binary semantic, for SAM training it is better to treat each separate mound blob as one prompted object.

Output a metadata file like:

```
data/curated/datasets/mapsam_v0/metadata/training_samples.jsonl
```

Each row:

```json
{
  "split": "train",
  "image": "images/train/K-35-8-G-a_1.png",
  "mask": "masks/train/K-35-8-G-a_1.png",
  "ignore_mask": "ignore_masks/train/K-35-8-G-a_1.png",
  "bbox": [622, 256, 648, 284],
  "point": [635, 270],
  "sheet_id": "K-35-8-G-a"
}
```

### 2. Implement a MapSamDataset

Create something like:

```
src/archeo_topia/datasets/mapsam_dataset.py
```

It should return:

```python
{
    "image": image_tensor,
    "target_mask": mask_tensor,
    "ignore_mask": ignore_mask_tensor,
    "box_prompt": box_tensor,
    "point_prompt": point_tensor,
    "image_path": image_path,
}
```

The first version should use only:

- positive mound components

Hard negatives stay in the image as background.

### 3. Create the first training script

Create:

```
src/archeo_topia/training/train_mapsam_v0.py
```

Training behavior:

1. Load SAM checkpoint
2. Freeze image encoder
3. Train prompt encoder and/or mask decoder
4. Use bbox or center point prompts
5. Compute loss between predicted mask and mound target mask
6. Ignore uncertain/ignore pixels in the loss

For v0.1, I would do:

```
freeze image_encoder = true
train prompt_encoder = false or true
train mask_decoder = true
```

Start with training only the mask decoder. If that underfits, unfreeze the prompt encoder too.

### 4. Use a simple loss

Use:

- `BCEWithLogitsLoss` + Dice loss

Apply the ignore mask:

```
loss is computed only where ignore_mask == 0
```

So uncertain regions do not punish the model.

### 5. Add a config file

Example:

```yaml
experiment_name: mapsam_v0_1_decoder_only

dataset:
  root: data/curated/datasets/mapsam_v0
  samples: metadata/training_samples.jsonl
  image_size: 1024

model:
  sam_checkpoint: checkpoints/sam_vit_b_01ec64.pth
  model_type: vit_b
  freeze_image_encoder: true
  train_prompt_encoder: false
  train_mask_decoder: true

training:
  epochs: 50
  batch_size: 1
  learning_rate: 0.0001
  weight_decay: 0.01
  device: cuda

loss:
  bce_weight: 1.0
  dice_weight: 1.0
```

Start with `vit_b` if possible. It is much easier to debug than `vit_h`.

### 6. Training command

Target command:

```bash
python -m src.archeo_topia.training.train_mapsam_v0 \
  --config configs/mapsam/mapsam_v0_1_decoder_only.yaml
```

Outputs:

```
artifacts/models/mapsam/v0_1_decoder_only/
  checkpoints/
  logs/
  predictions/
  metrics.json
```

### 7. Validate visually before caring about metrics

After each epoch or every few epochs, save overlays:

- image
- ground-truth mound mask
- predicted mask
- overlay

For your current phase, visual validation matters more than a polished metric table.

## Minimal next checklist

Do these in order:

1. Verify masks are correct.
2. Generate bbox/point prompts from mound masks.
3. Build `training_samples.jsonl`.
4. Implement `MapSamDataset`.
5. Load SAM checkpoint.
6. Freeze image encoder.
7. Train mask decoder with bbox/point prompts.
8. Save prediction overlays.
9. Evaluate on val/test sheets.

Your `mapsam_v0/` folder is enough for the first training run as long as the mound masks are correct. You do not need YOLO, SAM2, or CVAT anymore for this step.
