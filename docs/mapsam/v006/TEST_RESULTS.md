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
