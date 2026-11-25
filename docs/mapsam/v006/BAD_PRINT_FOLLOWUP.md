# MapSAM v0.6 — follow-up proposal for pale and blurred prints

**Status: proposal only. Nothing in this document has been implemented, trained or scored.** The only thing run for it was a read-only count of existing annotation files (section 4). Measured facts and suggestions are kept apart: sections 1 and 4 hold what was read from files; sections 2, 3 and 5 hold what is suggested.

---

## 1. Observation (measured, read from source files)

### Where the refit still misses

Source: `artifacts/detection/v0_6_refit/opening/detector/refit/metrics.json` (opening 2, refit `epoch125.pt`, recall radius 5 px). `per_annotation_distance_px` holds `None` for an annotation with no candidate within the association radius.

- 24 of 249 test annotations have no hit. 23 are on `K-35-39-V-g` (36/59 hit, recall 0.610); the other one is on `K-35-22-A-v`.
- Of the 23 misses on `K-35-39-V-g`, all 23 are on the clip `K-35-39-V-g_4.png` (56 annotations on that clip, 33 hit). The other clip with annotations from this sheet, `K-35-39-V-g_3.png`, holds 3 annotations and has no miss.
- Recall at confidence 0.05 equals recall at 0.25: 225/249 in both, and the same 24 annotations are unmatched. The misses are therefore not detections suppressed by the confidence threshold. At 0.05 the detector emits no candidate within 25 px of them.

### The same sheet under the baseline

Source: `artifacts/detection/v0_6_test_baseline/detector/fold{A,B,C}/metrics.json` (opening 1), confidence 0.25, recall@5px on `K-35-39-V-g`: foldA 27/59 (0.458), foldB 25/59 (0.424), foldC 29/59 (0.492). The refit gained 7 to 11 hits on this sheet and still misses 23.

### What the attributes say about the misses

Source: `data/curated/datasets/mapsam_det_frozen_v001/metadata/annotations.jsonl` (the frozen ground truth as carried into the test dataset; `blurred_or_bad_print` is an annotator attribute, `component_annotation_count` is the number of mounds sharing a connected component) joined to the distances above. This was read for the diagnosis only; nothing was chosen from it.

| | annotations | hit by refit | missed |
|---|---:|---:|---:|
| `K-35-39-V-g`, all | 59 | 36 | 23 |
| `blurred_or_bad_print` = true | 35 | 31 | 4 |
| `blurred_or_bad_print` = true, single-mound component | 26 | 26 | 0 |
| `blurred_or_bad_print` = true, merged component (2 or more) | 9 | 5 | 4 |
| merged component (2 or more), regardless of blur | 32 | 9 | 23 |

- `K-35-39-V-g` carries 35 of the 55 `blurred_or_bad_print` annotations in the whole test set (`K-35-22-A-v` 11, `K-35-39-G-v` 6, `L-35-139-V-v` 3). Refit recall on that attribute is 0.886 on this sheet (31/35); baseline 0.629, 0.600, 0.657.
- All 23 misses on this sheet are in merged components of 2 to 5 mounds. Only 4 of the 23 are also tagged blurred. Every one of the 26 blurred, single-mound annotations on the sheet is hit.
- The pooled `merged_component` subset (38 annotations) is hit 15 times by the refit; its 23 misses are the 23 on this sheet.

**What this means for the premise.** The annotator reported that most annotated symbols on `K-35-39-V-g` are "blurred or bad print", and the sheet is visually pale. The attribute counts agree with the first part (35 of 59). But in the refit's misses, blur is not what separates hit from miss on this sheet: merged components are. Pale or blurred print may still be the reason the symbols merge, but the files above do not show that, and nothing here measures it. Treat the interventions below as candidates for bad prints in general, not as a demonstrated fix for these 23 misses. A separate question for the same misses is whether touching or merged mounds need a different detector target or decoder step; that is out of scope here and is not proposed.

Sources for visual inspection (qualitative viewing only, see section 3): raw `data_lake/raw/mound_test_20260915/_frozen/02_maps_test/K-35-39-V-g_clipped.tif`; clips `data_lake/cleaned/map_clips/_frozen/K-35-39-V-g/`.

---

## 2. Candidate interventions (suggested, in order)

None has been tried.

**a) Deterministic contrast normalization, identical in training and inference.** CLAHE on the L channel of LAB (`cv2.createCLAHE`), or a per-clip percentile stretch. It must be applied by one shared function in both the dataset build and the inference path; enhancing only at inference on a model trained on unenhanced images risks a distribution shift. Cost of adopting it: a retrain.

**b) Bad-print augmentation in training.** Stronger fade and low contrast, blur and washout applied to training windows. Ultralytics' `hsv_v` (default 0.4) already varies brightness, so the addition would be contrast, blur and washout specifically. This makes the model robust to degraded prints without altering clean ones, and is often better than enhancement for degraded scans. Cost: a retrain.

**c) Unsharp masking or sharpening, last.** On a blurred print it also amplifies noise and hatching and can add false positives, so it ranks below (a) and (b).

---

## 3. Constraint: the frozen test set is closed

Per `TEST_PROTOCOL.md`, the frozen test set is closed after two openings, and both are used. `K-35-39-V-g` is a test sheet.

- It may be viewed qualitatively.
- No parameter, threshold, augmentation strength or choice between (a), (b) and (c) may be chosen from it, and nothing may be scored on it.
- Any claim that an intervention improves bad prints needs a new blind test set (the v0.7 test).

Section 1 was assembled from the already-published opening results and the frozen annotations; it is a diagnosis and does not select anything.

---

## 4. First cheap step (read-only, run)

Count `blurred_or_bad_print` mounds per sheet to check there is enough data to build a development subset without spending annotator time.

**Where the attribute is stored:** on each annotation in `annotation/cvat/v0.0.3/instances_default.json` and `annotation/cvat/v0.0.5/instances_default.json` (COCO, `annotations[].attributes.blurred_or_bad_print`). Counted for `category_id` 1 (`mound`); sheet taken from the image file name; split from `configs/splits/v0_6_splits.json`. No sheet is in both exports, and no test-split sheet is in either (`K-35-39-A-g` is excluded from `v0.0.5`).

| split | sheets with mounds | mounds | blurred_or_bad_print |
|---|---:|---:|---:|
| train | 42 | 1412 | 129 |
| val | 10 | 308 | 60 |

(`v0.0.3` contributes 3 train sheets, 169 mounds, 33 blurred; `v0.0.5` the rest, 1551 mounds, 156 blurred.)

Sheets with at least 4 blurred mounds:

| sheet | export | split | mounds | blurred | share |
|---|---|---|---:|---:|---:|
| K-35-32-A-a | v0.0.5 | train | 51 | 29 | 0.57 |
| K-34-47-G-g | v0.0.5 | val | 37 | 25 | 0.68 |
| K-35-51-B-a | v0.0.3 | train | 128 | 20 | 0.16 |
| K-35-27-B-b | v0.0.5 | val | 79 | 18 | 0.23 |
| K-35-38-A-a | v0.0.5 | val | 80 | 9 | 0.11 |
| K-35-19-V-a | v0.0.5 | train | 39 | 8 | 0.21 |
| K-35-20-B-g | v0.0.5 | train | 110 | 8 | 0.07 |
| K-35-28-A-g | v0.0.5 | train | 30 | 8 | 0.27 |
| K-35-49-V-g | v0.0.5 | train | 57 | 8 | 0.14 |
| K-35-8-G-a | v0.0.3 | train | 34 | 7 | 0.21 |
| K-34-35-B-g | v0.0.3 | train | 7 | 6 | 0.86 |
| K-34-24-G-a | v0.0.5 | train | 12 | 5 | 0.42 |
| K-35-51-B-g | v0.0.5 | train | 19 | 5 | 0.26 |
| K-34-60-V-b | v0.0.5 | train | 36 | 4 | 0.11 |
| K-34-96-B-v | v0.0.5 | train | 48 | 4 | 0.08 |
| K-35-67-B-v | v0.0.5 | val | 44 | 4 | 0.09 |
| K-35-77-A-g | v0.0.5 | train | 51 | 4 | 0.08 |

Reading the table:

- There is data: 189 blurred mounds over 52 sheets, with 9 sheets at 8 or more. The two val sheets `K-34-47-G-g` and `K-35-27-B-b` hold 43 of the 60 val blurred mounds.
- Caveats on what the attribute is. In `v0.0.5` it was filled in during the model-assisted review, so its meaning may be applied less uniformly than the blind annotation of the test set. Whether the train/val blurred mounds look like the `K-35-39-V-g` ones has not been checked.
- The refit trained on the train sheets, so its recall there is not an out-of-sample baseline (section 5).

---

## 5. Honest evaluation plan (suggested)

1. **Build a development subset** from train/val sheets with many blurred mounds, using the attribute already in the labels. No annotator time. Use the table above as the starting inventory.
2. **Baseline: the current refit on that subset.** Because the refit was fitted on the train sheets, measure out-of-sample either on the val sheets only (only 10 sheets, 60 blurred mounds, and these sheets were also used to select the checkpoint, so selection is slightly optimistic there) or by retraining with whole sheets held out (grouped on the 1:100k parent as in `configs/splits/v0_6_splits.json`, to avoid leakage). Decide this before any run.
3. **Compare** a CLAHE-trained refit with an augmentation-trained refit against the baseline, on the same subset and with the same metric (recall@5px at 0.25, plus precision). Each is a retrain of similar cost to the `v0_6_refit` (150 epochs, one GPU, about 2.7 hours in `train_record.json`: 11:12 to 13:57 UTC) and needs approval before it is started.
4. **Pre-register the winner as a v0.7 candidate**, with the settings frozen before the v0.7 blind test is opened. Only that test can support a claim of improvement on bad prints.
5. **Report merged components separately.** Section 1 suggests they, not blur, account for the refit's misses on `K-35-39-V-g`. Score the development subset with and without merged components, so an effect on blur is not hidden by, or credited for, the merged-component rate.

---

## Status

Proposal only. Nothing is implemented, no model was trained, no test-set number was produced or used for selection. The read-only counts in sections 1 and 4 can be regenerated from the files named there.
