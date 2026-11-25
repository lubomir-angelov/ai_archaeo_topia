# MapSAM v0.6 — frozen test results

Written before opening 1, as `TEST_PROTOCOL.md` requires. It holds only what
exists before the frozen test set is touched: the C1 reproduction outcome, the
build input that was restored, and the enumerated C3/C4 differences. No test-set
number exists yet. Each opening adds its own section below this one.

---

## Pre-opening-1: C1 reproduction check

**C1 passed.** The generalized scorer on v0.5 inputs (V0) equals v0.5 exactly on
all three folds (A, B, C), for both detection and end to end. Every
`V0 against v0.5` comparison reports 0 mismatches.

The check was run twice.

- **First run, failed:** `artifacts/detection/v0_6_c1_reproduction`. Fold A
  disagreed on end to end. The protocol's pre-registered "missed" rule (no
  candidate within 25 px) differed from v0.5's rule (best IoU over candidates is
  exactly 0) on two mounds on `K-35-51-B-a_3.png`, #543 and #552, where a 25 px
  neighbour's candidate segmented the neighbour, not the mound. The artifact is
  kept as evidence.
- **Amendment 2026-11-23**, made after that failure and before any test number
  existed: the end-to-end missed rule was changed to the v0.5 rule, so that C1
  reproduces v0.5 exactly with C3 the only difference. False masks and the
  detector-level metrics were unchanged. The scorer change is commit `176fd5d`.
- **Second run, passed:** `artifacts/detection/v0_6_c1_reproduction_r2`, after
  the amendment. Report: `reproduction_report.md` and `reproduction_report.json`
  in that directory.

Found-only mean IoU on the fixed arm, computed by V0 (the report marks this as a
reviewer comparison against `docs/mapsam/v005/RESULTS.md`, not machine-checked):

| fold | computed (V0) |
|---|---:|
| foldA | 0.7657 |
| foldB | 0.7482 |
| foldC | 0.7717 |

## Restored build input

The rebuild started from the CVAT export `mapsam_det_v1` was built from, not the
repository copy. One annotation differs in `negative_type` between the two:

| annotation_id | repository value | built value |
|---:|---|---|
| 235 | `other` | `__undefined__` |

This is the documented v0.5 repository patch. It is an input restoration, not a
C3/C4 difference. The restored export is
`artifacts/detection/v0_6_c1_build_input/instances_default.json`, derived from
`annotation/cvat/v0.0.2/instances_default.json`. The dataset rebuilt from it is
`data/curated/datasets/mapsam_det_v1_c3`.

## C3/C4 differences

**Regions.** Both reconstructions hold 545 regions: the v0.5 window-clipped
reconstruction and the rebuild from source coordinates.

**C3, region boxes.** 22 `hard_negative_symbol` boxes differ between the v0.5
window-clipped reconstruction and the source-coordinate record. In each, exactly one coordinate
differs materially, always a top (`y1`) or left (`x1`) edge: the v0.5 value equals a
window boundary of that image (taken from `metadata/windows.jsonl` of
`data/curated/datasets/mapsam_det_v1_c3`, whose window grid equals the original's),
and the source value lies beyond it, so the v0.5 box is the source box clipped at a
window edge. Three rows also carry float-representation-only differences in other
coordinates (266, 400 and 122: e.g. 731.5799999999999 against 731.58); those
coordinates are not clips, and each of the three still has one material edge
difference (266: `y1` 1780 against 1779.81). Boxes
are `[x1, y1, x2, y2]` in pixels of the named sheet image, copied from the report
without rounding.

| image | annotation_id | v0.5 box | source box |
|---|---:|---|---|
| `K-34-35-B-g_2.png` | 185 | [768, 367.13, 782.53, 383.53] | [763.23, 367.13, 782.53, 383.53] |
| `K-34-35-B-g_3.png` | 232 | [1570.0, 768, 1589.0, 784.0] | [1570.0, 764.0, 1589.0, 784.0] |
| `K-34-35-B-g_4.png` | 244 | [1986, 748.7, 1998.39, 762.37] | [1979.49, 748.7, 1998.39, 762.37] |
| `K-34-35-B-g_4.png` | 256 | [507.29, 768, 518.87, 784.39] | [507.29, 765.1, 518.87, 784.39] |
| `K-34-35-B-g_4.png` | 266 | [731.5799999999999, 1780, 750.0699999999999, 1796.3] | [731.58, 1779.81, 750.07, 1796.3] |
| `K-34-35-B-g_4.png` | 270 | [384, 1744.52, 403.11, 1771.97] | [381.05, 1744.52, 403.11, 1771.97] |
| `K-34-35-B-g_4.png` | 277 | [1538.5, 1152, 1560.03, 1167.06] | [1538.5, 1149.3, 1560.03, 1167.06] |
| `K-35-51-B-a_1.png` | 315 | [1979.78, 1536, 2003.66, 1555.56] | [1979.78, 1535.66, 2003.66, 1555.56] |
| `K-35-51-B-a_1.png` | 316 | [2168.84, 1536, 2190.73, 1560.54] | [2168.84, 1535.17, 2190.73, 1560.54] |
| `K-35-51-B-a_1.png` | 326 | [1961, 883.69, 1981.2, 904.13] | [1958.65, 883.69, 1981.2, 904.13] |
| `K-35-51-B-a_1.png` | 382 | [384, 397.54, 395.21, 417.55] | [380.2, 397.54, 395.21, 417.55] |
| `K-35-51-B-a_1.png` | 400 | [768, 988.3199999999999, 778.85, 998.6] | [766.52, 988.32, 778.85, 998.6] |
| `K-35-51-B-a_1.png` | 421 | [1014.41, 1152, 1037.16, 1170.06] | [1014.41, 1150.95, 1037.16, 1170.06] |
| `K-35-51-B-a_2.png` | 482 | [546.05, 1152, 568.89, 1170.03] | [546.05, 1142.11, 568.89, 1170.03] |
| `K-35-51-B-a_2.png` | 492 | [1152, 1669.97, 1163.36, 1699.15] | [1144.33, 1669.97, 1163.36, 1699.15] |
| `K-35-51-B-a_2.png` | 504 | [1536, 2084.89, 1564.33, 2115.35] | [1533.25, 2084.89, 1564.33, 2115.35] |
| `K-35-51-B-a_2.png` | 525 | [271.36, 1152, 302.21, 1173.93] | [271.36, 1144.08, 302.21, 1173.93] |
| `K-35-51-B-a_3.png` | 619 | [1920, 1878.05, 1939.58, 1895.38] | [1916.83, 1878.05, 1939.58, 1895.38] |
| `K-35-51-B-a_4.png` | 675 | [1410.74, 1536, 1428.58, 1550.73] | [1410.74, 1530.22, 1428.58, 1550.73] |
| `K-35-51-B-a_4.png` | 701 | [768, 2117.48, 802.65, 2141.29] | [736.45, 2117.48, 802.65, 2141.29] |
| `K-35-8-G-a_4.png` | 100 | [998.44, 1536, 1011.39, 1546.11] | [998.44, 1531.71, 1011.39, 1546.11] |
| `K-35-8-G-a_4.png` | 122 | [1051.33, 768, 1069.3899999999999, 780.48] | [1051.33, 764.14, 1069.39, 780.48] |

**Effect on v0.5 figures: none.** V1 (rebuilt, source-coordinate regions) minus
V0 is 0 differences on every fold (A, B, C), for detection and for end to end.
The C3 fix moves no v0.5 figure.

**C4.** The report has no C4-specific section. Its end-to-end comparison is
labelled `c3_c4_differences` and is empty on all three folds, so C4 contributes no
differences on the v0.5 inputs.

Source: `artifacts/detection/v0_6_c1_reproduction_r2/reproduction_report.md`.

---

## Opening 1 — baseline (`v0_6_test_baseline`)

Opening 1 of the frozen blind test: the v0.5 detectors with the v0.4r decoders, unchanged, scored on the frozen test set. Source: `artifacts/detection/v0_6_test_baseline/` (`run_record.json`, `detector/`, `end_to_end/`). Every figure below is copied from those files. Rates are shown to the three decimals the run reports use. The 95% intervals are Wilson intervals that treat instances as independent and understate the uncertainty.

### Opening record

- Started (UTC): 2026-11-23T09:34:39.598434+00:00
- Finished (UTC): 2026-11-23T09:36:32.166579+00:00
- Manifest: `docs/mapsam/v006/analysis/frozen_manifest.json`, sha256 `5d111b38a52da5ca265cd653faba928b9e124a7b15513c26ef58b3724140fd0c`
- Environment: python 3.13.5, torch 2.14.0, ultralytics 8.4.150, segment_anything 1.0; device `cuda:0` (NVIDIA GeForce RTX 5090)
- Test set: dataset `data/curated/datasets/mapsam_det_frozen_v001`, listing `test.txt`, sheets K-35-22-A-v, K-35-39-G-v, K-35-39-V-g, L-35-139-V-v; 249 annotations
- Settings: imgsz 512, minimum confidence 0.05, merge radius 10.0 px, association radius 25.0 px (one-to-one), recall radii [5.0, 10.0, 15.0, 25.0], attribution radius 15.0 px, thresholds [0.05, 0.1, 0.25, 0.5, 0.7], headline confidence 0.25
- Pairs (detector, decoder):
  - foldA: `artifacts/detection/v0_5_yolo26s_gtfix/foldA/weights/last.pt`, `artifacts/models/mapsam/v0_4r_loso512_foldA_pw20/checkpoints/final.pt`
  - foldB: `artifacts/detection/v0_5_yolo26s_gtfix/foldB/weights/last.pt`, `artifacts/models/mapsam/v0_4r_loso512_foldB_pw20/checkpoints/final.pt`
  - foldC: `artifacts/detection/v0_5_yolo26s_gtfix/foldC/weights/last.pt`, `artifacts/models/mapsam/v0_4r_loso512_foldC_pw20/checkpoints/final.pt`
- SAM checkpoint: `models/checkpoints/sam/sam_vit_b_01ec64.pth`; end-to-end box modes ['fixed', 'detector']

The three folds are three detector/decoder pairs scored on the same test set; each fold row below covers all 249 annotations.

### Detector, headline confidence 0.25, pooled

FP/MP is false positives per megapixel; its area is the full clip rectangle, including any transparent margin (Fork 2 A).

| fold | candidates | recall@5px (primary) hits/n, 95% CI | recall@10px | recall@15px | recall@25px |
|---|---:|---|---|---|---|
| foldA | 203 | 0.795 (198/249, 0.741–0.841) | 0.803 (200/249, 0.749–0.848) | 0.803 (200/249, 0.749–0.848) | 0.803 (200/249, 0.749–0.848) |
| foldB | 201 | 0.803 (200/249, 0.749–0.848) | 0.807 (201/249, 0.754–0.851) | 0.807 (201/249, 0.754–0.851) | 0.807 (201/249, 0.754–0.851) |
| foldC | 216 | 0.819 (204/249, 0.767–0.862) | 0.827 (206/249, 0.775–0.869) | 0.827 (206/249, 0.775–0.869) | 0.827 (206/249, 0.775–0.869) |

| fold | precision hits/n, 95% CI | F1 | FP | FP per 512 px window | FP per megapixel | localization p50 px | localization p90 px | neutralized by ignore |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| foldA | 0.985 (200/203, 0.957–0.995) | 0.885 | 3 | 0.0048 | 0.0349 | 1.64 | 2.94 | 0 |
| foldB | 1.000 (201/201, 0.981–1.000) | 0.893 | 0 | 0.0000 | 0.0000 | 1.42 | 2.82 | 0 |
| foldC | 0.954 (206/216, 0.917–0.975) | 0.886 | 10 | 0.0160 | 0.1164 | 1.56 | 2.82 | 0 |

False positives by what they land on:

| fold | by negative_type | hard negative without a negative_type | background |
|---|---|---:|---:|
| foldA | `decorative_symbol` 1, `other` 1 | 0 | 1 |
| foldB | none | 0 | 0 |
| foldC | `text` 2, `decorative_symbol` 1 | 0 | 7 |

### Across checkpoints (confidence 0.25)

Mean, min and max over the three folds, from `detector/across_checkpoints.json`.

| quantity | foldA | foldB | foldC | mean | min | max |
|---|---:|---:|---:|---:|---:|---:|
| recall_5px | 0.795 | 0.803 | 0.819 | 0.806 | 0.795 | 0.819 |
| recall_10px | 0.803 | 0.807 | 0.827 | 0.813 | 0.803 | 0.827 |
| recall_15px | 0.803 | 0.807 | 0.827 | 0.813 | 0.803 | 0.827 |
| recall_25px | 0.803 | 0.807 | 0.827 | 0.813 | 0.803 | 0.827 |
| precision | 0.985 | 1.000 | 0.954 | 0.980 | 0.954 | 1.000 |
| f1_at_association_radius | 0.885 | 0.893 | 0.886 | 0.888 | 0.885 | 0.893 |
| p90_error_px | 2.941 | 2.819 | 2.819 | 2.860 | 2.819 | 2.941 |
| fp_per_window | 0.005 | 0.000 | 0.016 | 0.007 | 0.000 | 0.016 |
| fp_per_megapixel | 0.035 | 0.000 | 0.116 | 0.050 | 0.000 | 0.116 |

### Per sheet and macro (confidence 0.25)

† fewer than 50 annotations on the sheet (the rule used by the run reports).

| fold | sheet | n | recall@5px hits/n, 95% CI | recall@25px | precision | FP | FP/window | p90 px |
|---|---|---:|---|---:|---:|---:|---:|---:|
| foldA | K-35-22-A-v | 82 | 0.866 (71/82, 0.776–0.923) | 0.878 | 0.986 | 1 | 0.0069 | 3.24 |
| foldA | K-35-39-G-v | 98 | 0.929 (91/98, 0.860–0.965) | 0.939 | 0.989 | 1 | 0.0060 | 2.55 |
| foldA | K-35-39-V-g | 59 | 0.458 (27/59, 0.337–0.583) | 0.458 | 0.964 | 1 | 0.0060 | 2.55 |
| foldA | L-35-139-V-v † | 10 | 0.900 (9/10, 0.596–0.982) | 0.900 | 1.000 | 0 | 0.0000 | 2.27 |
| foldB | K-35-22-A-v | 82 | 0.854 (70/82, 0.761–0.914) | 0.854 | 1.000 | 0 | 0.0000 | 3.44 |
| foldB | K-35-39-G-v | 98 | 0.969 (95/98, 0.914–0.990) | 0.980 | 1.000 | 0 | 0.0000 | 2.69 |
| foldB | K-35-39-V-g | 59 | 0.424 (25/59, 0.306–0.551) | 0.424 | 1.000 | 0 | 0.0000 | 1.91 |
| foldB | L-35-139-V-v † | 10 | 1.000 (10/10, 0.722–1.000) | 1.000 | 1.000 | 0 | 0.0000 | 2.24 |
| foldC | K-35-22-A-v | 82 | 0.829 (68/82, 0.734–0.895) | 0.841 | 1.000 | 0 | 0.0000 | 3.23 |
| foldC | K-35-39-G-v | 98 | 0.990 (97/98, 0.944–0.998) | 1.000 | 0.980 | 2 | 0.0119 | 2.81 |
| foldC | K-35-39-V-g | 59 | 0.492 (29/59, 0.368–0.616) | 0.492 | 0.806 | 7 | 0.0417 | 2.65 |
| foldC | L-35-139-V-v † | 10 | 1.000 (10/10, 0.722–1.000) | 1.000 | 0.909 | 1 | 0.0069 | 2.38 |

Macro over sheets (unweighted mean of the four sheets, with min and max):

| fold | recall@5px | recall@25px | precision | F1 | p90 px | FP/window | FP/MP |
|---|---|---|---|---|---|---|---|
| foldA | 0.788 (0.458–0.929) | 0.794 (0.458–0.939) | 0.985 (0.964–1.000) | 0.865 (0.621–0.963) | 2.65 (2.27–3.24) | 0.0047 (0.0000–0.0069) | 0.0343 (0.0000–0.0463) |
| foldB | 0.812 (0.424–1.000) | 0.814 (0.424–1.000) | 1.000 (1.000–1.000) | 0.876 (0.595–1.000) | 2.57 (1.91–3.44) | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) |
| foldC | 0.828 (0.492–1.000) | 0.833 (0.492–1.000) | 0.924 (0.806–1.000) | 0.867 (0.611–0.990) | 2.77 (2.38–3.23) | 0.0151 (0.0000–0.0417) | 0.1147 (0.0000–0.3191) |

### Weak subsets (attribution, confidence 0.25)

Recall of the pooled 249 annotations within each subset, as reported. The attribution radius is 15 px.

| fold | subset | n | recall@5px hits/n, 95% CI | recall@15px |
|---|---|---:|---|---:|
| foldA | merged_component | 38 | 0.368 (14/38, 0.234–0.527) | 0.368 |
| foldA | has_trig_point | 38 | 0.632 (24/38, 0.473–0.766) | 0.658 |
| foldA | blurred_or_bad_print | 55 | 0.709 (39/55, 0.579–0.812) | 0.709 |
| foldA | crossed_by_contour | 69 | 0.841 (58/69, 0.737–0.909) | 0.841 |
| foldB | merged_component | 38 | 0.395 (15/38, 0.256–0.553) | 0.395 |
| foldB | has_trig_point | 38 | 0.605 (23/38, 0.447–0.744) | 0.605 |
| foldB | blurred_or_bad_print | 55 | 0.709 (39/55, 0.579–0.812) | 0.709 |
| foldB | crossed_by_contour | 69 | 0.855 (59/69, 0.753–0.919) | 0.855 |
| foldC | merged_component | 38 | 0.395 (15/38, 0.256–0.553) | 0.395 |
| foldC | has_trig_point | 38 | 0.579 (22/38, 0.422–0.721) | 0.605 |
| foldC | blurred_or_bad_print | 55 | 0.745 (41/55, 0.617–0.842) | 0.745 |
| foldC | crossed_by_contour | 69 | 0.899 (62/69, 0.805–0.950) | 0.899 |

### End to end (confidence 0.25, frozen 512 decoder)

Denominator is annotations with geometry. Missed follows the v0.5 rule: best IoU over candidates exactly 0 (amendment 2026-11-23). `fixed` and `detector` are the two box modes. Found-only mean IoU is as reported; found-only median IoU is not a field of `end_to_end.json` and is computed here as the median of the nonzero entries of `per_annotation_iou` (the nonzero count equals the reported found n and their mean equals the reported found mean, both asserted when this section was generated).

| fold | arm | n | candidates | IoU>=0.5 hits/n, 95% CI | IoU>=0.75 hits/n, 95% CI | found n | found mean IoU | found median IoU | missed | missed rate | false masks | neutralized by ignore |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| foldA | fixed | 249 | 203 | 0.863 (215/249, 0.815–0.901) | 0.538 (134/249, 0.476–0.599) | 224 | 0.7507 | 0.7725 | 25 | 0.100 | 3 | 0 |
| foldA | detector | 249 | 203 | 0.863 (215/249, 0.815–0.901) | 0.518 (129/249, 0.456–0.579) | 224 | 0.7485 | 0.7655 | 25 | 0.100 | 3 | 0 |
| foldB | fixed | 249 | 201 | 0.863 (215/249, 0.815–0.901) | 0.602 (150/249, 0.541–0.661) | 224 | 0.7611 | 0.7812 | 25 | 0.100 | 0 | 0 |
| foldB | detector | 249 | 201 | 0.863 (215/249, 0.815–0.901) | 0.606 (151/249, 0.545–0.665) | 224 | 0.7659 | 0.7877 | 25 | 0.100 | 0 | 0 |
| foldC | fixed | 249 | 216 | 0.880 (219/249, 0.833–0.914) | 0.622 (155/249, 0.561–0.680) | 229 | 0.7582 | 0.7727 | 20 | 0.080 | 10 | 0 |
| foldC | detector | 249 | 216 | 0.876 (218/249, 0.829–0.911) | 0.602 (150/249, 0.541–0.661) | 229 | 0.7612 | 0.7800 | 20 | 0.080 | 10 | 0 |

Per sheet, end to end:

| fold | arm | sheet | n | candidates | IoU>=0.5 | IoU>=0.75 | mean IoU | found mean IoU | missed | missed rate | false masks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| foldA | fixed | K-35-22-A-v | 82 | 73 | 0.878 | 0.500 | 0.6687 | 0.7616 | 10 | 0.122 | 1 |
| foldA | fixed | K-35-39-G-v | 98 | 93 | 0.939 | 0.551 | 0.7150 | 0.7617 | 6 | 0.061 | 1 |
| foldA | fixed | K-35-39-V-g | 59 | 28 | 0.712 | 0.576 | 0.6166 | 0.7276 | 9 | 0.153 | 1 |
| foldA | fixed | L-35-139-V-v † | 10 | 9 | 0.900 | 0.500 | 0.6875 | 0.6875 | 0 | 0.000 | 0 |
| foldA | detector | K-35-22-A-v | 82 | 73 | 0.878 | 0.476 | 0.6632 | 0.7553 | 10 | 0.122 | 1 |
| foldA | detector | K-35-39-G-v | 98 | 93 | 0.939 | 0.500 | 0.7107 | 0.7570 | 6 | 0.061 | 1 |
| foldA | detector | K-35-39-V-g | 59 | 28 | 0.712 | 0.610 | 0.6218 | 0.7338 | 9 | 0.153 | 1 |
| foldA | detector | L-35-139-V-v † | 10 | 9 | 0.900 | 0.500 | 0.6950 | 0.6950 | 0 | 0.000 | 0 |
| foldB | fixed | K-35-22-A-v | 82 | 70 | 0.854 | 0.537 | 0.6519 | 0.7636 | 12 | 0.146 | 0 |
| foldB | fixed | K-35-39-G-v | 98 | 96 | 0.980 | 0.704 | 0.7650 | 0.7809 | 2 | 0.020 | 0 |
| foldB | fixed | K-35-39-V-g | 59 | 25 | 0.661 | 0.542 | 0.5820 | 0.7154 | 11 | 0.186 | 0 |
| foldB | fixed | L-35-139-V-v † | 10 | 10 | 1.000 | 0.500 | 0.7719 | 0.7719 | 0 | 0.000 | 0 |
| foldB | detector | K-35-22-A-v | 82 | 70 | 0.854 | 0.573 | 0.6580 | 0.7708 | 12 | 0.146 | 0 |
| foldB | detector | K-35-39-G-v | 98 | 96 | 0.980 | 0.663 | 0.7654 | 0.7813 | 2 | 0.020 | 0 |
| foldB | detector | K-35-39-V-g | 59 | 25 | 0.661 | 0.576 | 0.5924 | 0.7282 | 11 | 0.186 | 0 |
| foldB | detector | L-35-139-V-v † | 10 | 10 | 1.000 | 0.500 | 0.7638 | 0.7638 | 0 | 0.000 | 0 |
| foldC | fixed | K-35-22-A-v | 82 | 69 | 0.841 | 0.646 | 0.6576 | 0.7815 | 13 | 0.159 | 0 |
| foldC | fixed | K-35-39-G-v | 98 | 100 | 1.000 | 0.704 | 0.7794 | 0.7794 | 0 | 0.000 | 2 |
| foldC | fixed | K-35-39-V-g | 59 | 36 | 0.712 | 0.458 | 0.6050 | 0.6865 | 7 | 0.119 | 7 |
| foldC | fixed | L-35-139-V-v † | 10 | 11 | 1.000 | 0.600 | 0.7642 | 0.7642 | 0 | 0.000 | 1 |
| foldC | detector | K-35-22-A-v | 82 | 69 | 0.829 | 0.598 | 0.6559 | 0.7795 | 13 | 0.159 | 0 |
| foldC | detector | K-35-39-G-v | 98 | 100 | 1.000 | 0.653 | 0.7773 | 0.7773 | 0 | 0.000 | 2 |
| foldC | detector | K-35-39-V-g | 59 | 36 | 0.712 | 0.525 | 0.6221 | 0.7059 | 7 | 0.119 | 7 |
| foldC | detector | L-35-139-V-v † | 10 | 11 | 1.000 | 0.600 | 0.7657 | 0.7657 | 0 | 0.000 | 1 |

End-to-end macro over sheets (mean, min–max):

| fold | arm | IoU>=0.5 | IoU>=0.75 | mean IoU | found mean IoU | missed rate |
|---|---|---|---|---|---|---|
| foldA | fixed | 0.857 (0.712–0.939) | 0.532 (0.500–0.576) | 0.672 (0.617–0.715) | 0.735 (0.688–0.762) | 0.084 (0.000–0.153) |
| foldA | detector | 0.857 (0.712–0.939) | 0.521 (0.476–0.610) | 0.673 (0.622–0.711) | 0.735 (0.695–0.757) | 0.084 (0.000–0.153) |
| foldB | fixed | 0.874 (0.661–1.000) | 0.571 (0.500–0.704) | 0.693 (0.582–0.772) | 0.758 (0.715–0.781) | 0.088 (0.000–0.186) |
| foldB | detector | 0.874 (0.661–1.000) | 0.578 (0.500–0.663) | 0.695 (0.592–0.765) | 0.761 (0.728–0.781) | 0.088 (0.000–0.186) |
| foldC | fixed | 0.888 (0.712–1.000) | 0.602 (0.458–0.704) | 0.702 (0.605–0.779) | 0.753 (0.686–0.781) | 0.069 (0.000–0.159) |
| foldC | detector | 0.885 (0.712–1.000) | 0.594 (0.525–0.653) | 0.705 (0.622–0.777) | 0.757 (0.706–0.780) | 0.069 (0.000–0.159) |

### Observations

Facts only, read off the tables above; no causes are asserted.

- Pooled recall@5px at confidence 0.25 is 0.795, 0.803 and 0.819 for foldA, foldB and foldC (mean 0.806). Precision is 0.985, 1.000 and 0.954.
- K-35-39-V-g is the hardest sheet on every fold: recall@5px 0.458, 0.424 and 0.492 against 0.829 to 0.990 on K-35-22-A-v and K-35-39-G-v. The detectors return 28, 25 and 36 candidates for its 59 annotations. Its end-to-end IoU>=0.5 rate is 0.712, 0.661 and 0.712 (`fixed`), and its missed rate is 0.153, 0.186 and 0.119.
- K-35-39-G-v is the easiest sheet with at least 50 annotations: recall@5px 0.929, 0.969 and 0.990.
- foldC has 10 false positives against 3 (foldA) and 0 (foldB); 7 of the 10 are on K-35-39-V-g, and its precision on that sheet is 0.806.
- By subset, `merged_component` has the lowest recall@5px on every fold (0.368, 0.395, 0.395; 38 annotations); `has_trig_point` is second lowest on every fold, and `crossed_by_contour` the highest (0.841, 0.855, 0.899).
- End-to-end IoU>=0.5 is 0.863, 0.863 and 0.880 (`fixed`), higher than the detector recall@25px (0.803, 0.807, 0.827) on each fold, under the two different scoring definitions the tables use. IoU>=0.75 is 0.538, 0.602 and 0.622 (`fixed`).
- The `fixed` and `detector` arms differ by at most 0.004 in pooled IoU>=0.5 (foldC, 0.880 against 0.876) and at most 0.020 in IoU>=0.75 (foldA, 0.538 against 0.518; foldC, 0.622 against 0.602). Missed count is identical across arms on every fold (25, 25, 20).
- No candidate was neutralized by an ignore region (0 on every fold, detector and end to end). L-35-139-V-v has 10 annotations, so its rates carry wide intervals.

---

## Before opening 2 — the refit

Written before opening 2, as `TEST_PROTOCOL.md` ("Opening 2 — the refit") requires. Opening 2 has not run; no refit test number exists. Every figure is copied from the named source file. Sheet names appear, no coordinates.

### Refit configuration

`configs/refit/v0_6_refit.json`, sha256 `db627b9ade994eaf6fd9a36fdc157975eed4662b8ee4ec880f17f53f942fef3f` (equal to `refit.config.sha256` in `data/curated/datasets/mapsam_det_refit_v001/metadata/build_report.json`, and to the hash recorded in `train_record.json`).

| key | value |
|---|---|
| name | `v0_6_refit` |
| model | `yolo26s.pt` |
| window / stride | 512 / 384 |
| epochs, batch, seed | 150, 16, 42 |
| save_period | 25 |
| development_source | `v0.0.3` |
| label source `v0.0.3` | `annotation/cvat/v0.0.3/instances_default.json`, mound_rule `all` (input sha256 `3a84c53d874a4efe0a544b2f37cf5d6960fd9ef640d3c2d06b0cd5ecdae46a97`) |
| label source `v0.0.5` | `annotation/cvat/v0.0.5/instances_default.json`, mound_rule `reviewed` (input sha256 `781d19ed01eb45d67bcabc7a5395c226c3b4c554da8fc11046e2dfa003812218`) |

Training used imgsz 512 (`train_record.json`); early stopping is disabled by `patience=epochs + 1` in `src/archeo_topia/training/train_mound_detector.py`.

### Label sources

- **`annotation/cvat/v0.0.5`** (`PROVENANCE.json`): 55 sheets, cut from `artifacts/review/v0_7` against `configs/splits/v0_6_splits.json` (sha256 `5dada05532d3be114cbe87db183a4c503ada07790f4d16b88b0edda18c5a5ac5`). Counts by category: `mound` 1551, `hard_negative_symbol` 984, `uncertain_ignore` 0. Review status: `mound` confirmed 1366, corrected 140, added 45; `hard_negative_symbol` rejected 984.
- **Excluded from the export:** `K-35-39-A-g`, reason `test_split`. It is a frozen-test sheet and never enters a label source used for training or selection.
- **`annotation/cvat/v0.0.3`:** the corrected ground truth of record for the three development sheets `K-34-35-B-g`, `K-35-51-B-a`, `K-35-8-G-a`. No sheet is in both sources (`sheets_by_source` in the build report: 3 in `v0.0.3`, 55 in `v0.0.5`).

### Training and selection sheets

From `build_report.json`, `listing.listings` (plan `train_val`):

- **train** (`train.txt`): 46 sheets, 7325 windows, the three development sheets included. Sheets: K-34-22-V-b, K-34-23-A-g, K-34-24-G-a, K-34-35-A-v, K-34-35-B-g, K-34-36-V-b, K-34-48-G-b, K-34-58-B-a, K-34-60-V-b, K-34-70-B-v, K-34-84-A-a, K-34-96-B-v, K-35-13-G-v, K-35-14-B-v, K-35-18-A-b, K-35-19-V-a, K-35-20-B-g, K-35-21-G-a, K-35-28-A-g, K-35-29-G-b, K-35-31-V-g, K-35-32-A-a, K-35-40-B-v, K-35-41-G-a, K-35-42-A-g, K-35-43-V-g, K-35-49-V-g, K-35-5-G-b, K-35-50-B-b, K-35-50-V-b, K-35-51-A-b, K-35-51-B-a, K-35-51-B-g, K-35-52-G-v, K-35-55-A-g, K-35-62-G-v, K-35-63-B-g, K-35-66-A-a, K-35-73-A-v, K-35-75-B-a, K-35-75-B-g, K-35-77-A-g, K-35-8-G-a, K-35-8-V-g, K-35-86-A-b, K-35-88-G-g.
- **val / selection** (`val.txt`): 12 sheets, 1908 windows. Sheets: K-34-10-A-g, K-34-47-A-a, K-34-47-G-g, K-34-83-V-g, K-35-16-A-a, K-35-16-B-a, K-35-27-B-b, K-35-38-A-a, K-35-44-B-a, K-35-53-A-g, K-35-67-B-v, K-35-87-G-a.
- Dataset totals: 9240 windows, 1756 with a mound, 1612 hard-negative-only, 5872 empty; 1720 mound annotations, 3370 mound targets written.

### Excluded sheets

Copied from `refit.excluded_sheets` in the build report:

| sheet | reason |
|---|---|
| `K-34-47-G-v` | `absent_from_label_source` |

`K-34-47-G-v` is a `val`-split sheet of `configs/splits/v0_6_splits.json` (13 sheets) with no annotations in its label source, so it contributes no windows: 12 of the 13 val sheets are selection sheets. This is exclusion under the pre-registered *reviewed sheet* rule ("a sheet that is present in its label source and has no `unreviewed` mound"), not a deviation. No other sheet was excluded, and no test-split sheet is in the dataset.

### Fork 6 B: uncertain regions and rejected mounds

- `uncertain_dropped_train_windows`: count 7, all on `K-34-35-B-g` (development sheet, `v0.0.3`): `K-34-35-B-g_3_x01152_y01152`, `_3_x01152_y01536`, `_3_x01536_y01152`, `_3_x01536_y01536`, `_4_x00000_y00768`, `_4_x00000_y01152`, `_4_x00384_y01152`. `mound_targets_in_dropped`: 0, so no mound target was lost with the dropped windows.
- Regions written: `hard_negative_symbol` 1526, `uncertain_ignore` 3.
- `rejected_mounds_as_background`: count 0 (by sheet: none). The 984 rejected annotations in `v0.0.5` are `hard_negative_symbol`, which enter as background by rule.

### `human_added` share on the selection sheets

From `refit.selection_human_added`: 308 mound targets on the 12 selection sheets, 11 of them `human_added`; pooled share 0.0357 (11/308).

| sheet | mound targets | human_added |
|---|---:|---:|
| K-34-10-A-g | 0 | 0 |
| K-34-47-A-a | 2 | 0 |
| K-34-47-G-g | 37 | 2 |
| K-34-83-V-g | 7 | 0 |
| K-35-16-A-a | 14 | 2 |
| K-35-16-B-a | 6 | 0 |
| K-35-27-B-b | 79 | 2 |
| K-35-38-A-a | 80 | 3 |
| K-35-44-B-a | 4 | 0 |
| K-35-53-A-g | 35 | 1 |
| K-35-67-B-v | 44 | 1 |
| K-35-87-G-a | 0 | 0 |

Framing, as the protocol sets it. The review of these sheets was thorough and included a deliberate search for mounds the model had missed, so the labels are intended to be on par with the frozen-set annotation. It was model-assisted, not blind: it started from v0.5 proposals. The `human_added` share measures how much that model-assisted start mattered; it is not a recall floor, and review recall is not claimed. Checkpoint selection on these sheets may slightly favour checkpoints that behave like v0.5. The frozen test set, annotated blind, is the judge.

### Training

Source: `artifacts/detection/v0_6_refit/train/train_record.json`.

- Model `yolo26s.pt`, imgsz 512, epochs 150, batch 16, seed 42, save_period 25.
- Environment: python 3.13.5, torch 2.14.0, ultralytics 8.4.150, segment_anything 1.0; device `cuda:0` (NVIDIA GeForce RTX 5090).
- Started (UTC) 2026-11-23T11:12:46.348008+00:00, finished 2026-11-23T13:57:44.850962+00:00.
- Dataset `data/curated/datasets/mapsam_det_refit_v001`, build report sha256 `2d705ad30ce8e17df147523b33557fedf967ad04382a5a841201f870bc4ea2b7`.

### Checkpoint selection

Source: `artifacts/detection/v0_6_refit/selection/selection.json` (listing `val.txt`, 12 sheets, 1908 windows, confidence 0.25, imgsz 512, merge radius 10.0 px). Rule, in order: highest pooled recall@5px hits; then fewer false positives; then the later epoch. `best.pt` (mAP fitness) and `epoch0.pt` are ignored. Ultralytics names periodic saves by the 0-based epoch index, so `epochN.pt` holds the state after N+1 epochs; `last.pt` is the epoch-150 state.

| candidate | epochs completed | hits/n | recall@5px | FP | FP/window |
|---|---:|---:|---:|---:|---:|
| `epoch25.pt` | 26 | 297/308 | 0.9643 | 11 | 0.0058 |
| `epoch50.pt` | 51 | 302/308 | 0.9805 | 4 | 0.0021 |
| `epoch75.pt` | 76 | 303/308 | 0.9838 | 6 | 0.0031 |
| `epoch100.pt` | 101 | 302/308 | 0.9805 | 4 | 0.0021 |
| **`epoch125.pt` (chosen)** | 126 | 304/308 | 0.9870 | 2 | 0.0010 |
| `last.pt` | 150 | 303/308 | 0.9838 | 5 | 0.0026 |

`epoch125.pt` has the highest hit count outright, so neither tie-break applies. sha256 `8612f1247e3bc112316f5ef2e34e3f4783d2c97231b64420f26ba8c6ad3e2aff`. These are validation figures on model-assisted labels, not test results.

### Frozen manifest

`docs/mapsam/v006/analysis/frozen_manifest.json`: base of 29 entries, sha256 `5d111b38a52da5ca265cd653faba928b9e124a7b15513c26ef58b3724140fd0c`, equal to the manifest sha256 in opening 1's `run_record.json`. Extended to 31 entries, new sha256 `e93885a28326c99589f67ec8007f64287479b58e77151ee46c5f46df59dcb6c9` (recomputed with `sha256sum`). The two appended entries:

| role | path | sha256 |
|---|---|---|
| `refit_detector` (checkpoint `epoch125.pt`) | `artifacts/detection/v0_6_refit/train/refit/weights/epoch125.pt` | `8612f1247e3bc112316f5ef2e34e3f4783d2c97231b64420f26ba8c6ad3e2aff` |
| `refit_selection` | `artifacts/detection/v0_6_refit/selection/selection.json` | `f2822f674217a01a0f350e8d6919457cbd07a56a8852d0c3c2de97d5aeeb4007` |

### Deviations from the pre-registered refit

None found. Compared against the protocol table: model `yolo26s.pt`, imgsz 512, epochs 150, batch 16, seed 42, early stopping disabled, windows 512/384, exactly two label sources with no shared sheet, the `all` and `reviewed` mound rules, train and selection sheets from the `train` and `val` splits, no test-split sheet, six checkpoint candidates scored at confidence 0.25 with `best.pt` unused, and artifacts under `artifacts/detection/v0_6_refit/`. The one excluded sheet (`K-34-47-G-v`) falls under the reviewed-sheet rule and is not a deviation. The `rejected` mound rule did not trigger (0 rejected mounds as background).
---

## Opening 2 — the refit (`v0_6_refit`)

Opening 2 of the frozen blind test: the refit detector (`epoch125.pt`, chosen on the validation sheets before this opening) with the three frozen v0.4r decoders, scored on the same frozen test set as opening 1. Source: `artifacts/detection/v0_6_refit/` (`opening/run_record.json`, `opening/detector/refit/metrics.json`, `opening/end_to_end/fold{A,B,C}/end_to_end.json`, `paired.json`). Every figure below is copied or computed from those files and from the opening-1 files named above. Rates are shown to the three decimals the run reports use. The 95% intervals are Wilson intervals that treat instances as independent and understate the uncertainty. Both openings of the frozen test are now used; nothing further may be scored on it.

### Opening record

- Started (UTC): 2026-11-23T14:04:00.400764+00:00
- Finished (UTC): 2026-11-23T14:05:47.741120+00:00
- Manifest: `docs/mapsam/v006/analysis/frozen_manifest.json`, sha256 `e93885a28326c99589f67ec8007f64287479b58e77151ee46c5f46df59dcb6c9` (base sha256 `5d111b38a52da5ca265cd653faba928b9e124a7b15513c26ef58b3724140fd0c`, equal to the manifest sha256 of opening 1)
- Environment: python 3.13.5, torch 2.14.0, ultralytics 8.4.150, segment_anything 1.0; device `cuda:0` (NVIDIA GeForce RTX 5090)
- Selection record: `artifacts/detection/v0_6_refit/selection/selection.json`, sha256 `f2822f674217a01a0f350e8d6919457cbd07a56a8852d0c3c2de97d5aeeb4007`
- Test set: dataset `data/curated/datasets/mapsam_det_frozen_v001`, listing `test.txt`, sheets K-35-22-A-v, K-35-39-G-v, K-35-39-V-g, L-35-139-V-v; 249 annotations
- Settings: imgsz 512, minimum confidence 0.05, merge radius 10.0 px, association radius 25.0 px (one-to-one), recall radii [5.0, 10.0, 15.0, 25.0], attribution radius 15.0 px, thresholds [0.05, 0.1, 0.25, 0.5, 0.7], headline confidence 0.25
- Detector label `refit`; pairs (detector, decoder):
  - foldA: `artifacts/detection/v0_6_refit/train/refit/weights/epoch125.pt`, `artifacts/models/mapsam/v0_4r_loso512_foldA_pw20/checkpoints/final.pt`
  - foldB: `artifacts/detection/v0_6_refit/train/refit/weights/epoch125.pt`, `artifacts/models/mapsam/v0_4r_loso512_foldB_pw20/checkpoints/final.pt`
  - foldC: `artifacts/detection/v0_6_refit/train/refit/weights/epoch125.pt`, `artifacts/models/mapsam/v0_4r_loso512_foldC_pw20/checkpoints/final.pt`
- SAM checkpoint: `models/checkpoints/sam/sam_vit_b_01ec64.pth`; end-to-end box modes ['fixed', 'detector']

One refit detector is paired with three frozen decoders. The detector-level figures are therefore one set of numbers (the three fold rows differ only end to end). The decoder is not refit.

### Detector, headline confidence 0.25, pooled

FP/MP is false positives per megapixel; its area is the full clip rectangle, including any transparent margin (Fork 2 A).

| candidates | recall@5px (primary) hits/n, 95% CI | recall@10px | recall@15px | recall@25px |
|---:|---|---|---|---|
| 227 | 0.904 (225/249, 0.861–0.934) | 0.904 (225/249, 0.861–0.934) | 0.904 (225/249, 0.861–0.934) | 0.904 (225/249, 0.861–0.934) |

| precision hits/n, 95% CI | F1 | FP | FP per 512 px window | FP per megapixel | localization p50 px | localization p90 px | neutralized by ignore |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.991 (225/227, 0.968–0.998) | 0.945 | 2 | 0.0032 | 0.0233 | 1.41 | 2.51 | 0 |

False positives by what they land on: by negative_type none; hard negative without a negative_type 0; background 2.

### Detector, confidence 0.05 (recall-oriented, second), pooled

FP/MP is false positives per megapixel; its area is the full clip rectangle, including any transparent margin (Fork 2 A).

| candidates | recall@5px (primary) hits/n, 95% CI | recall@10px | recall@15px | recall@25px |
|---:|---|---|---|---|
| 235 | 0.904 (225/249, 0.861–0.934) | 0.904 (225/249, 0.861–0.934) | 0.904 (225/249, 0.861–0.934) | 0.904 (225/249, 0.861–0.934) |

| precision hits/n, 95% CI | F1 | FP | FP per 512 px window | FP per megapixel | localization p50 px | localization p90 px | neutralized by ignore |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.957 (225/235, 0.923–0.977) | 0.930 | 10 | 0.0160 | 0.1164 | 1.41 | 2.51 | 0 |

False positives by what they land on: by negative_type `road` 1, `decorative_symbol` 1; hard negative without a negative_type 0; background 8.

### Per sheet and macro (confidence 0.25)

† fewer than 50 annotations on the sheet (the rule used by the run reports).

| sheet | n | recall@5px hits/n, 95% CI | recall@25px | precision | FP | FP/window | p90 px |
|---|---:|---|---:|---:|---:|---:|---:|
| K-35-22-A-v | 82 | 0.988 (81/82, 0.934–0.998) | 0.988 | 0.988 | 1 | 0.0069 | 2.70 |
| K-35-39-G-v | 98 | 1.000 (98/98, 0.962–1.000) | 1.000 | 0.990 | 1 | 0.0060 | 2.30 |
| K-35-39-V-g | 59 | 0.610 (36/59, 0.483–0.724) | 0.610 | 1.000 | 0 | 0.0000 | 2.02 |
| L-35-139-V-v † | 10 | 1.000 (10/10, 0.722–1.000) | 1.000 | 1.000 | 0 | 0.0000 | 1.75 |

Macro over sheets (unweighted mean of the four sheets, with min and max):

| recall@5px | recall@25px | precision | F1 | p90 px | FP/window | FP/MP |
|---|---|---|---|---|---|---|
| 0.899 (0.610–1.000) | 0.899 (0.610–1.000) | 0.994 (0.988–1.000) | 0.935 (0.758–1.000) | 2.19 (1.75–2.70) | 0.0032 (0.0000–0.0069) | 0.0229 (0.0000–0.0463) |

### Weak subsets (attribution, confidence 0.25)

Recall of the pooled 249 annotations within each subset, as reported. The attribution radius is 15 px.

| subset | n | recall@5px hits/n, 95% CI | recall@15px |
|---|---:|---|---:|
| merged_component | 38 | 0.395 (15/38, 0.256–0.553) | 0.395 |
| has_trig_point | 38 | 0.974 (37/38, 0.865–0.995) | 0.974 |
| blurred_or_bad_print | 55 | 0.909 (50/55, 0.804–0.961) | 0.909 |
| crossed_by_contour | 69 | 1.000 (69/69, 0.947–1.000) | 1.000 |

### End to end (confidence 0.25, frozen 512 decoder)

Denominator is annotations with geometry. Missed follows the v0.5 rule: best IoU over candidates exactly 0 (amendment 2026-11-23). `fixed` and `detector` are the two box modes. Found-only mean IoU is as reported; found-only median IoU is computed as the median of the nonzero entries of `per_annotation_iou`, as in opening 1 (the nonzero count equals the reported found n and their mean equals the reported found mean, both asserted when this section was generated).

| fold | arm | n | candidates | IoU>=0.5 hits/n, 95% CI | IoU>=0.75 hits/n, 95% CI | found n | found mean IoU | found median IoU | missed | missed rate | false masks | neutralized by ignore |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| foldA | fixed | 249 | 227 | 0.944 (235/249, 0.908–0.966) | 0.570 (142/249, 0.508–0.630) | 248 | 0.7458 | 0.7688 | 1 | 0.004 | 1 | 0 |
| foldA | detector | 249 | 227 | 0.944 (235/249, 0.908–0.966) | 0.578 (144/249, 0.516–0.638) | 248 | 0.7456 | 0.7643 | 1 | 0.004 | 1 | 0 |
| foldB | fixed | 249 | 227 | 0.944 (235/249, 0.908–0.966) | 0.671 (167/249, 0.610–0.726) | 248 | 0.7563 | 0.7838 | 1 | 0.004 | 1 | 0 |
| foldB | detector | 249 | 227 | 0.944 (235/249, 0.908–0.966) | 0.679 (169/249, 0.618–0.734) | 248 | 0.7571 | 0.7831 | 1 | 0.004 | 1 | 0 |
| foldC | fixed | 249 | 227 | 0.944 (235/249, 0.908–0.966) | 0.679 (169/249, 0.618–0.734) | 248 | 0.7554 | 0.7793 | 1 | 0.004 | 1 | 0 |
| foldC | detector | 249 | 227 | 0.944 (235/249, 0.908–0.966) | 0.667 (166/249, 0.606–0.722) | 248 | 0.7554 | 0.7770 | 1 | 0.004 | 1 | 0 |

Per sheet, end to end:

| fold | arm | sheet | n | candidates | IoU>=0.5 | IoU>=0.75 | mean IoU | found mean IoU | missed | missed rate | false masks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| foldA | fixed | K-35-22-A-v | 82 | 82 | 0.988 | 0.512 | 0.7465 | 0.7557 | 1 | 0.012 | 0 |
| foldA | fixed | K-35-39-G-v | 98 | 99 | 1.000 | 0.571 | 0.7630 | 0.7630 | 0 | 0.000 | 1 |
| foldA | fixed | K-35-39-V-g | 59 | 36 | 0.780 | 0.661 | 0.7008 | 0.7008 | 0 | 0.000 | 0 |
| foldA | fixed | L-35-139-V-v † | 10 | 10 | 1.000 | 0.500 | 0.7626 | 0.7626 | 0 | 0.000 | 0 |
| foldA | detector | K-35-22-A-v | 82 | 82 | 0.988 | 0.500 | 0.7405 | 0.7496 | 1 | 0.012 | 0 |
| foldA | detector | K-35-39-G-v | 98 | 99 | 1.000 | 0.582 | 0.7640 | 0.7640 | 0 | 0.000 | 1 |
| foldA | detector | K-35-39-V-g | 59 | 36 | 0.780 | 0.695 | 0.7065 | 0.7065 | 0 | 0.000 | 0 |
| foldA | detector | L-35-139-V-v † | 10 | 10 | 1.000 | 0.500 | 0.7633 | 0.7633 | 0 | 0.000 | 0 |
| foldB | fixed | K-35-22-A-v | 82 | 82 | 0.988 | 0.634 | 0.7561 | 0.7655 | 1 | 0.012 | 0 |
| foldB | fixed | K-35-39-G-v | 98 | 99 | 1.000 | 0.735 | 0.7822 | 0.7822 | 0 | 0.000 | 1 |
| foldB | fixed | K-35-39-V-g | 59 | 36 | 0.780 | 0.644 | 0.6970 | 0.6970 | 0 | 0.000 | 0 |
| foldB | fixed | L-35-139-V-v † | 10 | 10 | 1.000 | 0.500 | 0.7790 | 0.7790 | 0 | 0.000 | 0 |
| foldB | detector | K-35-22-A-v | 82 | 82 | 0.988 | 0.634 | 0.7528 | 0.7621 | 1 | 0.012 | 0 |
| foldB | detector | K-35-39-G-v | 98 | 99 | 1.000 | 0.735 | 0.7828 | 0.7828 | 0 | 0.000 | 1 |
| foldB | detector | K-35-39-V-g | 59 | 36 | 0.780 | 0.678 | 0.7045 | 0.7045 | 0 | 0.000 | 0 |
| foldB | detector | L-35-139-V-v † | 10 | 10 | 1.000 | 0.500 | 0.7737 | 0.7737 | 0 | 0.000 | 0 |
| foldC | fixed | K-35-22-A-v | 82 | 82 | 0.988 | 0.732 | 0.7668 | 0.7763 | 1 | 0.012 | 0 |
| foldC | fixed | K-35-39-G-v | 98 | 99 | 1.000 | 0.684 | 0.7792 | 0.7792 | 0 | 0.000 | 1 |
| foldC | fixed | K-35-39-V-g | 59 | 36 | 0.780 | 0.593 | 0.6840 | 0.6840 | 0 | 0.000 | 0 |
| foldC | fixed | L-35-139-V-v † | 10 | 10 | 1.000 | 0.700 | 0.7754 | 0.7754 | 0 | 0.000 | 0 |
| foldC | detector | K-35-22-A-v | 82 | 82 | 0.988 | 0.683 | 0.7618 | 0.7712 | 1 | 0.012 | 0 |
| foldC | detector | K-35-39-G-v | 98 | 99 | 1.000 | 0.673 | 0.7798 | 0.7798 | 0 | 0.000 | 1 |
| foldC | detector | K-35-39-V-g | 59 | 36 | 0.780 | 0.627 | 0.6907 | 0.6907 | 0 | 0.000 | 0 |
| foldC | detector | L-35-139-V-v † | 10 | 10 | 1.000 | 0.700 | 0.7689 | 0.7689 | 0 | 0.000 | 0 |

End-to-end macro over sheets (mean, min–max):

| fold | arm | IoU>=0.5 | IoU>=0.75 | mean IoU | found mean IoU | missed rate |
|---|---|---|---|---|---|---|
| foldA | fixed | 0.942 (0.780–1.000) | 0.561 (0.500–0.661) | 0.743 (0.701–0.763) | 0.746 (0.701–0.763) | 0.003 (0.000–0.012) |
| foldA | detector | 0.942 (0.780–1.000) | 0.569 (0.500–0.695) | 0.744 (0.707–0.764) | 0.746 (0.707–0.764) | 0.003 (0.000–0.012) |
| foldB | fixed | 0.942 (0.780–1.000) | 0.628 (0.500–0.735) | 0.754 (0.697–0.782) | 0.756 (0.697–0.782) | 0.003 (0.000–0.012) |
| foldB | detector | 0.942 (0.780–1.000) | 0.637 (0.500–0.735) | 0.753 (0.704–0.783) | 0.756 (0.704–0.783) | 0.003 (0.000–0.012) |
| foldC | fixed | 0.942 (0.780–1.000) | 0.677 (0.593–0.732) | 0.751 (0.684–0.779) | 0.754 (0.684–0.779) | 0.003 (0.000–0.012) |
| foldC | detector | 0.942 (0.780–1.000) | 0.671 (0.627–0.700) | 0.750 (0.691–0.780) | 0.753 (0.691–0.780) | 0.003 (0.000–0.012) |

### Pre-registered paired comparison (recall@5px, confidence 0.25)

This is the only comparative claim the protocol allows about the refit's detection recall (`TEST_PROTOCOL.md`, "Comparing the refit against the baseline"). Notation as defined there: R is the refit's pooled recall@5px, B<sub>max</sub> and B<sub>min</sub> the highest and lowest of the three baseline checkpoints, S = B<sub>max</sub> − B<sub>min</sub>. The discordant counts are mounds found by one detector and missed by the other; the p-value is the exact two-sided binomial (McNemar) value. Source: `paired.json`, `claim`.

- Confidence 0.25, hit radius 5 px, 249 annotations, alpha 0.05
- R = 0.9036 (225/249)
- Baseline hits: foldA 198/249 (0.7952), foldB 200/249 (0.8032), foldC 204/249 (0.8193)
- B<sub>max</sub> = 204 hits, B<sub>min</sub> = 198 hits, S = 6 hits (0.0241 in recall)
- R − B<sub>max</sub> = 21 hits (S = 6)

| baseline | refit only (found by refit, missed by baseline) | baseline only | both | neither | n | exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| foldA | 29 | 2 | 196 | 22 | 249 | 4.629e-07 |
| foldB | 28 | 3 | 197 | 21 | 249 | 4.649e-06 |
| foldC | 22 | 1 | 203 | 23 | 249 | 5.722e-06 |

Case: `improves`. Rule 1 holds: R > B<sub>max</sub> (225 > 204), p < 0.05 against all three checkpoints, and R − B<sub>max</sub> = 21 ≥ S = 6.

Wording, verbatim from `paired.json`: **The refit improves recall.**

The wording is exactly the string the protocol fixes for case 1 ("**The refit improves recall.**"), and no other comparative wording about the refit's detection recall is used in this document.

### Descriptive paired comparisons

Descriptive only, not a claim and not described as an effect of refitting (`TEST_PROTOCOL.md`: everything other than the claim above is descriptive). First = baseline, second = refit.

Detector, recall@5px at confidence 0.05:

| baseline | refit only | baseline only | both | neither | n | exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| foldA | 21 | 2 | 204 | 22 | 249 | 6.604e-05 |
| foldB | 21 | 3 | 204 | 21 | 249 | 2.772e-04 |
| foldC | 18 | 1 | 207 | 23 | 249 | 7.629e-05 |

End to end, paired per decoder (refit into decoder X against baseline fold X into decoder X):

| fold | arm | metric | refit only | baseline only | both | neither | n | exact two-sided p |
|---|---|---|---:|---:|---:|---:|---:|---:|
| foldA | fixed | iou_0.5 | 20 | 0 | 215 | 14 | 249 | 1.907e-06 |
| foldA | fixed | iou_0.75 | 15 | 7 | 127 | 100 | 249 | 1.338e-01 |
| foldA | detector | iou_0.5 | 20 | 0 | 215 | 14 | 249 | 1.907e-06 |
| foldA | detector | iou_0.75 | 23 | 8 | 121 | 97 | 249 | 1.067e-02 |
| foldB | fixed | iou_0.5 | 20 | 0 | 215 | 14 | 249 | 1.907e-06 |
| foldB | fixed | iou_0.75 | 23 | 6 | 144 | 76 | 249 | 2.316e-03 |
| foldB | detector | iou_0.5 | 20 | 0 | 215 | 14 | 249 | 1.907e-06 |
| foldB | detector | iou_0.75 | 26 | 8 | 143 | 72 | 249 | 2.935e-03 |
| foldC | fixed | iou_0.5 | 16 | 0 | 219 | 14 | 249 | 3.052e-05 |
| foldC | fixed | iou_0.75 | 22 | 8 | 147 | 72 | 249 | 1.612e-02 |
| foldC | detector | iou_0.5 | 17 | 0 | 218 | 14 | 249 | 1.526e-05 |
| foldC | detector | iou_0.75 | 25 | 9 | 141 | 74 | 249 | 9.041e-03 |

### Side by side: opening 1 against opening 2

Opening 1 rows are the three baseline folds (v0.5 detector fold X with decoder X); the opening 2 row is the refit detector (one detector; its end-to-end rows use decoder X). Detector columns at confidence 0.25.

| run | recall@5px | recall@25px | precision | FP | FP/MP | p90 px |
|---|---|---|---|---:|---:|---:|
| opening 1 foldA | 0.795 (198/249, 0.741–0.841) | 0.803 | 0.985 | 3 | 0.0349 | 2.94 |
| opening 1 foldB | 0.803 (200/249, 0.749–0.848) | 0.807 | 1.000 | 0 | 0.0000 | 2.82 |
| opening 1 foldC | 0.819 (204/249, 0.767–0.862) | 0.827 | 0.954 | 10 | 0.1164 | 2.82 |
| opening 2 refit | 0.904 (225/249, 0.861–0.934) | 0.904 | 0.991 | 2 | 0.0233 | 2.51 |

Detector recall@5px at confidence 0.05:

| run | recall@5px | precision | FP |
|---|---|---|---:|
| opening 1 foldA | 0.827 (206/249, 0.775–0.869) | 0.950 | 11 |
| opening 1 foldB | 0.831 (207/249, 0.780–0.873) | 0.981 | 4 |
| opening 1 foldC | 0.835 (208/249, 0.784–0.876) | 0.821 | 46 |
| opening 2 refit | 0.904 (225/249, 0.861–0.934) | 0.957 | 10 |

Recall@5px per sheet, confidence 0.25 (hits/n):

| run | K-35-22-A-v | K-35-39-G-v | K-35-39-V-g | L-35-139-V-v |
|---|---:|---:|---:|---:|
| opening 1 foldA | 0.866 (71/82) | 0.929 (91/98) | 0.458 (27/59) | 0.900 (9/10) |
| opening 1 foldB | 0.854 (70/82) | 0.969 (95/98) | 0.424 (25/59) | 1.000 (10/10) |
| opening 1 foldC | 0.829 (68/82) | 0.990 (97/98) | 0.492 (29/59) | 1.000 (10/10) |
| opening 2 refit | 0.988 (81/82) | 1.000 (98/98) | 0.610 (36/59) | 1.000 (10/10) |

End to end, `fixed` arm, IoU>=0.5 / IoU>=0.75 / found mean IoU / missed / false masks:

| fold | opening 1 | opening 2 |
|---|---|---|
| foldA | 0.863 / 0.538 / 0.7507 / 25 / 3 | 0.944 / 0.570 / 0.7458 / 1 / 1 |
| foldB | 0.863 / 0.602 / 0.7611 / 25 / 0 | 0.944 / 0.671 / 0.7563 / 1 / 1 |
| foldC | 0.880 / 0.622 / 0.7582 / 20 / 10 | 0.944 / 0.679 / 0.7554 / 1 / 1 |

Same, `detector` arm:

| fold | opening 1 | opening 2 |
|---|---|---|
| foldA | 0.863 / 0.518 / 0.7485 / 25 / 3 | 0.944 / 0.578 / 0.7456 / 1 / 1 |
| foldB | 0.863 / 0.606 / 0.7659 / 25 / 0 | 0.944 / 0.679 / 0.7571 / 1 / 1 |
| foldC | 0.876 / 0.602 / 0.7612 / 20 / 10 | 0.944 / 0.667 / 0.7554 / 1 / 1 |

### Observations

Facts only, read off the tables above; no causes are asserted.

- Pooled recall@5px at confidence 0.25 is 0.904 (225/249) for the refit, against 0.795, 0.803 and 0.819 (198, 200, 204 of 249) for the three baseline folds. Recall is the same at 5, 10, 15 and 25 px. Precision is 0.991 with 2 false positives, both background (one on K-35-22-A-v, one on K-35-39-G-v).
- 24 of the 249 annotations are not hit within 5 px at 0.25. By sheet: 23 on K-35-39-V-g (36 of 59 hit), 1 on K-35-22-A-v (81 of 82), none on K-35-39-G-v or L-35-139-V-v.
- K-35-39-V-g, the hardest sheet on every baseline fold (recall@5px 0.458, 0.424, 0.492), is also the hardest for the refit (0.610). The refit returns 36 candidates for its 59 annotations (baseline 28, 25, 36), none of them false.
- By subset, `merged_component` has the lowest recall@5px (0.395, 15/38), the same value as baseline folds B and C (0.395) and above fold A (0.368). `has_trig_point` is 0.974 (37/38), against 0.579 to 0.632 on the baseline folds. `crossed_by_contour` is 1.000 (69/69).
- Lowering the confidence from 0.25 to 0.05 adds 8 candidates and 8 false positives and no hit at the refit (recall@5px 0.904 at both).
- End-to-end IoU>=0.5 is 0.944 (235/249) on all three decoders and both arms; IoU>=0.75 is 0.570 to 0.679. One annotation is missed (best IoU exactly 0) on every decoder, against 25, 25 and 20 at baseline. Found-only mean IoU is 0.7458, 0.7563 and 0.7554 (`fixed`), against 0.7507, 0.7611 and 0.7582 at baseline: the found-only mean is not higher on any decoder, while IoU>=0.5 and the number of annotations counted as found (248 against 224 to 229) are higher.
- The end-to-end "missed" definition (best IoU over candidates exactly 0) differs from the detector's 5 px hit, so K-35-39-V-g shows 0 missed end to end while 23 of its annotations are not hit at 5 px; its end-to-end IoU>=0.5 is 0.780.
- The end-to-end paired comparison at IoU>=0.5 has no annotation that the baseline scored and the refit did not (baseline only 0) on any decoder or arm. At IoU>=0.75 the discordance runs both ways (baseline only 6 to 9; refit only 15 to 26). These are descriptive.
- No candidate was neutralized by an ignore region (0, detector and end to end). L-35-139-V-v has 10 annotations, so its rates carry wide intervals.

### Validation-selection bias and the blind test

The refit checkpoint (`epoch125.pt`) was chosen on 12 validation sheets whose labels come from a model-assisted review: it started from v0.5 proposals and was not blind, although the review was thorough and included a deliberate search for missed mounds. The `human_added` share on those sheets is 11/308 (0.0357; see "Before opening 2"). The protocol records that selection on them may slightly favour checkpoints that behave like v0.5. The frozen test set was annotated blind and has now been opened twice, so it served as the judge for both the baseline and the refit; no checkpoint, threshold or radius was chosen on it.
