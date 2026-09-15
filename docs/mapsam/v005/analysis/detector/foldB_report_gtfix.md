## YOLO26-s — foldB (eval sheet K-35-8-G-a)

36 candidates against 34 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 1.000 | 34/34 | 0.898–1.000 |
| 10 px | 1.000 | 34/34 | 0.898–1.000 |
| 15 px | 1.000 | 34/34 | 0.898–1.000 |
| 25 px | 1.000 | 34/34 | 0.898–1.000 |

| localization error | px |
|---|---:|
| median | 1.95 |
| **p90** | **3.66** |
| p99 | 4.36 |
| max | 4.40 |

False positives: 2 (0.01 per 512 px window, 0.09 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 2 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| has_trig_point | 12 | 1.000 | 1.000 |
| blurred_or_bad_print | 7 | 1.000 | 1.000 |
| crossed_by_contour | 9 | 1.000 | 1.000 |

### Operating curve

| conf | candidates | recall@5px | recall@10px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 36 | 1.000 | 1.000 | 1.000 | 3.7 | 2 | 0.01 |
| 0.10 | 34 | 1.000 | 1.000 | 1.000 | 3.7 | 0 | 0.00 |
| 0.25 | 34 | 1.000 | 1.000 | 1.000 | 3.7 | 0 | 0.00 |
| 0.50 | 34 | 1.000 | 1.000 | 1.000 | 3.7 | 0 | 0.00 |
| 0.70 | 34 | 1.000 | 1.000 | 1.000 | 3.7 | 0 | 0.00 |
