## Template baseline — foldB (eval sheet K-35-8-G-a)

10487 candidates against 34 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.882 | 30/34 | 0.734–0.953 |
| 10 px | 1.000 | 34/34 | 0.898–1.000 |
| 15 px | 1.000 | 34/34 | 0.898–1.000 |
| 25 px | 1.000 | 34/34 | 0.898–1.000 |

| localization error | px |
|---|---:|
| median | 2.18 |
| **p90** | **5.18** |
| p99 | 8.35 |
| max | 8.73 |

False positives: 10453 (72.59 per 512 px window, 490.54 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 10243 |
| `trig_point` | 86 |
| `decorative_symbol` | 82 |
| `other` | 11 |
| `road` | 10 |
| `grid` | 10 |
| `text` | 6 |
| `colored_pencil` | 3 |
| `__undefined__` | 2 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| has_trig_point | 12 | 0.833 | 1.000 |
| blurred_or_bad_print | 7 | 0.571 | 1.000 |
| crossed_by_contour | 9 | 0.667 | 1.000 |

### Operating curve

| threshold | candidates | recall@5px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 10487 | 0.882 | 1.000 | 5.2 | 10453 | 72.6 |
| 0.50 | 3827 | 0.882 | 1.000 | 5.2 | 3793 | 26.3 |
| 0.60 | 1354 | 0.824 | 0.882 | 4.0 | 1324 | 9.2 |
| 0.70 | 388 | 0.559 | 0.559 | 3.0 | 369 | 2.6 |
| 0.80 | 219 | 0.176 | 0.176 | 2.2 | 213 | 1.5 |
| 0.90 | 0 | 0.000 | 0.000 | nan | 0 | 0.0 |
