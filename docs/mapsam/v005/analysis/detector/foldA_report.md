## YOLO26-s — foldA (eval sheet K-35-51-B-a)

200 candidates against 128 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.859 | 110/128 | 0.789–0.909 |
| 10 px | 0.883 | 113/128 | 0.816–0.928 |
| 15 px | 0.883 | 113/128 | 0.816–0.928 |
| 25 px | 0.883 | 113/128 | 0.816–0.928 |

| localization error | px |
|---|---:|
| median | 1.65 |
| **p90** | **3.30** |
| p99 | 5.91 |
| max | 6.55 |

False positives: 87 (0.52 per 512 px window, 3.92 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 55 |
| `road` | 24 |
| `decorative_symbol` | 8 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 15 | 0.667 | 0.800 |
| has_trig_point | 28 | 0.893 | 0.929 |
| blurred_or_bad_print | 20 | 0.700 | 0.700 |
| crossed_by_contour | 50 | 0.780 | 0.840 |

### Operating curve

| conf | candidates | recall@5px | recall@10px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 200 | 0.859 | 0.883 | 0.883 | 3.3 | 87 | 0.52 |
| 0.10 | 176 | 0.844 | 0.859 | 0.859 | 3.1 | 66 | 0.39 |
| 0.25 | 142 | 0.812 | 0.828 | 0.828 | 3.2 | 36 | 0.21 |
| 0.50 | 114 | 0.742 | 0.750 | 0.750 | 3.2 | 18 | 0.11 |
| 0.70 | 81 | 0.617 | 0.617 | 0.617 | 3.1 | 2 | 0.01 |
