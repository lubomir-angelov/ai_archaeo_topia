# MapSAM v0.1 — Training Findings

Analysis of the v0.1 training pipeline carried out before starting v0.2. The
headline result is that the v0.1 target masks did not encode the task the model
was prompted to solve, and that the reported v0.1 metrics were computed against
a different objective than the one the loss optimised.

This document records what was found and why it matters. It does not modify any
v0.1 result; `RESULTS.md` and `../DATASET.md` are left exactly as they were so the
historical record stays accurate.

## Summary

| Finding | Severity | Status in v0.2 |
|---------|----------|----------------|
| Target mask contains every mound on the sheet, not the prompted one | Critical | Fixed |
| Connected components labelled after downsampling | Moderate | Fixed |
| `foreground_bce_pos_weight` tuned against an inflated target | Moderate | Re-tuned, swept |
| Bbox loss crop silently load-bearing | Moderate | Ablated |
| Reported metrics scored against the all-mounds target | Moderate | Documented |
| Ignore masks are entirely empty | Low | Documented |

## Finding 1 — The target mask is the whole sheet, not the prompted instance

`generate_mapsam_prompts.py` creates one training sample per connected component,
each with its own bounding box and centre point. Every one of those samples,
however, points at the same whole-sheet mask file:

```python
"mask_path": f"masks/{split}/{filename}",   # identical for all components
```

`MapSamDataset.__getitem__` then returned that file verbatim as `target_mask`.
The consequence is that all samples derived from one map sheet receive a
byte-identical target containing every mound on that sheet, and differ only in
their prompt.

Measured on the v0.1 mask files:

| Mask file | Components | Foreground px |
|---|---:|---:|
| `train/K-35-51-B-a_3.png` | 56 | 28,228 |
| `train/K-35-51-B-a_4.png` | 28 | 12,578 |
| `train/K-35-51-B-a_1.png` | 25 | 10,331 |
| `test/K-35-8-G-a_3.png` | 14 | 6,469 |
| `test/K-35-8-G-a_4.png` | 12 | 5,327 |
| `train/K-35-51-B-a_2.png` | 11 | 4,844 |
| `train/K-34-35-B-g_3.png` | 9 | 2,707 |
| `train/K-34-35-B-g_2.png` | 6 | 3,138 |
| `test/K-35-8-G-a_2.png` | 5 | 2,224 |
| `test/K-35-8-G-a_1.png` | 4 | 1,993 |
| `train/K-34-35-B-g_1.png` | 2 | 640 |
| `train/K-34-35-B-g_4.png` | 0 | 0 |
| **Total** | **172** | |

172 components across 12 files, consistent with the 171 samples reported in
`../DATASET.md` once the single sub-20px component is skipped.

Weighting by sample count, the average sample's target carried **roughly 30x
more foreground than the single mound its prompt indicated**
(sum of n squared over sum of n, 5168/172 = 30.0). For the worst sheet, all 56
samples shared one 56-mound target.

This is a contradiction in the supervision signal rather than merely noisy
labels. SAM is a promptable architecture whose entire premise is that the prompt
selects which object to segment. A label set that returns the same mask
regardless of prompt teaches the model to ignore the prompt.

`../DATASET.md` already described the intended behaviour — "Binary target mask
(single connected component)" — so the documentation and the code had diverged.

## Finding 2 — Components were labelled after downsampling

The fix for Finding 1 selects a connected component, which requires labelling
the mask. Doing that on the resized mask is unsafe: mean component area is
458.9 px at roughly 2474x2242, which becomes about 79 px at 1024 and about 5 px
at the 256x256 logit resolution. Nearest-neighbour downsampling can fragment or
erase thin, ring-shaped mound symbols before labelling ever happens, which would
make the component selection unstable in exactly the cases that matter.

In v0.2 the component is selected on the original-resolution mask and the
selected instance is then resized, so labelling always sees the full-resolution
symbol.

## Finding 3 — The bbox loss crop was quietly load-bearing

`combined_bce_dice_loss` supports `use_bbox_loss_crop`, which marks everything
outside the prompt bounding box plus a margin as ignored. With
`bbox_loss_crop_margin: 32` applied in logit space, the loss window is roughly
67x67 within a 256x256 map, about 7% of the image.

That crop is what kept v0.1 from collapsing outright: most of the other mounds
on a sheet fell outside the window and contributed no gradient. It is a
mitigation, not a fix, because mounds *inside* the window still appeared as
positive target and still penalised the model for not firing on unprompted
neighbours.

The important consequence is that the crop was doing correctness work rather
than the optimisation work it was introduced for. With correct instance targets
the crop becomes a genuine choice, so v0.2 ablates it on and off rather than
assuming it.

## Finding 4 — Reported metrics scored a different task than the loss

`compute_prediction_stats` derives its valid-pixel mask from the dataset's
ignore mask, not from the bbox crop:

```python
valid = (ignore_mask == 0).float()
```

The v0.1 ignore masks are empty in every file (`ignore_fg = 0` for all twelve),
so `valid` was the entire image. The reported IoU and Dice were therefore
computed over the whole image against the all-mounds target, while the loss was
optimised over a small cropped window against the same target.

The numbers corroborate this. `RESULTS.md` reports a mean GT positive count of
30 px at 256x256 logit resolution. A single mound at that resolution is about
5-6 px; a whole sheet's worth is about 30 (for example `K-34-35-B-g_2`,
3138 px scaled by (256/2474)(256/2242), gives about 34). The reported ground
truth was the union of all mounds, not the prompted instance.

The v0.1 mean IoU of 0.650 therefore includes credit for predicting mounds the
prompt never asked about, and is not comparable to an instance-level score.

## Finding 5 — Class balance, and what it implies for the loss

Because the v0.2 target drops roughly 30x in foreground area, the loss
hyperparameters tuned against the inflated target no longer apply.
`foreground_bce_pos_weight: 20.0` and `dice_weight: 2.0` were chosen when the
positive class was far larger than it will now be, so the positive weight is
expected to need raising.

v0.2 treats this as an experiment rather than a guess and sweeps the positive
weight instead of silently changing it, so that one run stays directly
comparable to v0.1.

## Finding 6 — Ignore masks carry no signal

All twelve v0.1 ignore masks are empty. The COCO export contains only 3
`uncertain_ignore` annotations against 180 `mound` and 530
`hard_negative_symbol` annotations, so the ignore pathway is effectively
untested. This is recorded rather than fixed; the 530 hard negatives in
particular are currently unused by the training pipeline and represent an
obvious source of signal for a later iteration.

## Expected effect on results

Correcting the target is expected to **lower** the reported scores, and that
lowering is the correct outcome rather than a regression. The v0.1 numbers were
partly credit for segmenting unprompted objects. After the fix:

- IoU and Dice measure whether the prompted mound was segmented, which is the
  deployment question and is comparable to instance-segmentation literature.
- The prompt-independence pathology disappears. Under v0.1 supervision the
  loss-minimising behaviour inside the crop window was to fire on anything
  mound-like nearby, which at inference produces merged blobs where mounds
  cluster and makes instance counting and localisation impossible.
- Full-image supervision becomes viable, because disabling the crop no longer
  trains the model to paint the whole sheet.

`RESULTS.md` from v0.1 is not comparable to v0.2 results at the instance level
and should be read as a record of the earlier objective.
