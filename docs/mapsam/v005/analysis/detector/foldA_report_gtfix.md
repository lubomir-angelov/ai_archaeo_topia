## YOLO26-s — foldA (eval sheet K-35-51-B-a)

218 candidates against 128 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.898 | 115/128 | 0.834–0.940 |
| 10 px | 0.898 | 115/128 | 0.834–0.940 |
| 15 px | 0.898 | 115/128 | 0.834–0.940 |
| 25 px | 0.898 | 115/128 | 0.834–0.940 |

| localization error | px |
|---|---:|
| median | 1.72 |
| **p90** | **3.52** |
| p99 | 4.92 |
| max | 4.94 |

False positives: 103 (0.61 per 512 px window, 4.65 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 61 |
| `road` | 34 |
| `decorative_symbol` | 8 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 15 | 0.800 | 0.800 |
| has_trig_point | 28 | 0.893 | 0.893 |
| blurred_or_bad_print | 20 | 0.750 | 0.750 |
| crossed_by_contour | 50 | 0.860 | 0.860 |

### Operating curve

| conf | candidates | recall@5px | recall@10px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 218 | 0.898 | 0.898 | 0.898 | 3.5 | 103 | 0.61 |
| 0.10 | 198 | 0.898 | 0.898 | 0.898 | 3.5 | 83 | 0.49 |
| 0.25 | 173 | 0.883 | 0.883 | 0.883 | 3.4 | 60 | 0.36 |
| 0.50 | 150 | 0.867 | 0.867 | 0.867 | 3.5 | 39 | 0.23 |
| 0.70 | 116 | 0.812 | 0.812 | 0.812 | 3.4 | 12 | 0.07 |
