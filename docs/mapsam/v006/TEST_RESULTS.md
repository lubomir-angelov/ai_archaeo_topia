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
