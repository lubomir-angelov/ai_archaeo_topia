## Template baseline — foldA (eval sheet K-35-51-B-a)

12501 candidates against 128 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.828 | 106/128 | 0.753–0.884 |
| 10 px | 0.883 | 113/128 | 0.816–0.928 |
| 15 px | 0.898 | 115/128 | 0.834–0.940 |
| 25 px | 0.992 | 127/128 | 0.957–0.999 |

| localization error | px |
|---|---:|
| median | 2.50 |
| **p90** | **11.65** |
| p99 | 23.71 |
| max | 25.00 |

False positives: 12374 (73.65 per 512 px window, 558.19 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 11757 |
| `decorative_symbol` | 266 |
| `road` | 231 |
| `trig_point` | 78 |
| `text` | 40 |
| `__undefined__` | 2 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 15 | 0.667 | 0.667 |
| has_trig_point | 28 | 0.857 | 0.929 |
| blurred_or_bad_print | 20 | 0.550 | 0.700 |
| crossed_by_contour | 50 | 0.760 | 0.880 |

### Operating curve

| threshold | candidates | recall@5px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 12501 | 0.828 | 0.898 | 11.6 | 12374 | 73.7 |
| 0.50 | 4707 | 0.766 | 0.828 | 5.1 | 4595 | 27.4 |
| 0.60 | 1692 | 0.680 | 0.703 | 4.0 | 1602 | 9.5 |
| 0.70 | 155 | 0.453 | 0.469 | 4.1 | 95 | 0.6 |
| 0.80 | 12 | 0.086 | 0.086 | 3.0 | 1 | 0.0 |
| 0.90 | 0 | 0.000 | 0.000 | nan | 0 | 0.0 |
