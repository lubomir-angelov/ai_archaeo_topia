## YOLO26-s — foldC (eval sheet K-34-35-B-g)

30 candidates against 18 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.389 | 7/18 | 0.203–0.614 |
| 10 px | 0.389 | 7/18 | 0.203–0.614 |
| 15 px | 0.389 | 7/18 | 0.203–0.614 |
| 25 px | 0.389 | 7/18 | 0.203–0.614 |

| localization error | px |
|---|---:|
| median | 1.51 |
| **p90** | **2.00** |
| p99 | 2.08 |
| max | 2.09 |

False positives: 23 (0.14 per 512 px window, 1.00 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 21 |
| `trig_point` | 2 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 2 | 1.000 | 1.000 |
| has_trig_point | 2 | 0.000 | 0.000 |
| blurred_or_bad_print | 9 | 0.778 | 0.778 |
| crossed_by_contour | 9 | 0.333 | 0.333 |

### Operating curve

| conf | candidates | recall@5px | recall@10px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 30 | 0.389 | 0.389 | 0.389 | 2.0 | 23 | 0.14 |
| 0.10 | 19 | 0.389 | 0.389 | 0.389 | 2.0 | 12 | 0.07 |
| 0.25 | 10 | 0.389 | 0.389 | 0.389 | 2.0 | 3 | 0.02 |
| 0.50 | 8 | 0.333 | 0.333 | 0.333 | 2.0 | 2 | 0.01 |
| 0.70 | 7 | 0.333 | 0.333 | 0.333 | 2.0 | 1 | 0.01 |
