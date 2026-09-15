## YOLO26-s — foldC (eval sheet K-34-35-B-g)

30 candidates against 7 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 1.000 | 7/7 | 0.646–1.000 |
| 10 px | 1.000 | 7/7 | 0.646–1.000 |
| 15 px | 1.000 | 7/7 | 0.646–1.000 |
| 25 px | 1.000 | 7/7 | 0.646–1.000 |

| localization error | px |
|---|---:|
| median | 1.78 |
| **p90** | **2.00** |
| p99 | 2.08 |
| max | 2.09 |

False positives: 23 (0.14 per 512 px window, 1.00 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 20 |
| `trig_point` | 2 |
| `other` | 1 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 2 | 1.000 | 1.000 |
| has_trig_point | 1 | 1.000 | 1.000 |
| blurred_or_bad_print | 6 | 1.000 | 1.000 |
| crossed_by_contour | 4 | 1.000 | 1.000 |

### Operating curve

| conf | candidates | recall@5px | recall@10px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 30 | 1.000 | 1.000 | 1.000 | 2.0 | 23 | 0.14 |
| 0.10 | 19 | 1.000 | 1.000 | 1.000 | 2.0 | 12 | 0.07 |
| 0.25 | 10 | 1.000 | 1.000 | 1.000 | 2.0 | 3 | 0.02 |
| 0.50 | 8 | 1.000 | 1.000 | 1.000 | 2.0 | 1 | 0.01 |
| 0.70 | 7 | 0.857 | 0.857 | 0.857 | 2.0 | 1 | 0.01 |
