## Template baseline — foldB (eval sheet K-35-8-G-a)

8478 candidates against 34 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.882 | 30/34 | 0.734–0.953 |
| 10 px | 0.941 | 32/34 | 0.809–0.984 |
| 15 px | 0.971 | 33/34 | 0.851–0.995 |
| 25 px | 1.000 | 34/34 | 0.898–1.000 |

| localization error | px |
|---|---:|
| median | 2.09 |
| **p90** | **5.33** |
| p99 | 19.31 |
| max | 21.52 |

False positives: 8444 (58.64 per 512 px window, 396.26 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 8271 |
| `trig_point` | 78 |
| `decorative_symbol` | 64 |
| `road` | 8 |
| `other` | 8 |
| `grid` | 6 |
| `colored_pencil` | 4 |
| `text` | 3 |
| `__undefined__` | 2 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| has_trig_point | 12 | 0.833 | 0.917 |
| blurred_or_bad_print | 7 | 0.714 | 0.857 |
| crossed_by_contour | 9 | 0.778 | 0.889 |

### Operating curve

| threshold | candidates | recall@5px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 8478 | 0.882 | 0.971 | 5.3 | 8444 | 58.6 |
| 0.50 | 3939 | 0.882 | 0.971 | 5.3 | 3905 | 27.1 |
| 0.60 | 1483 | 0.853 | 0.912 | 4.0 | 1452 | 10.1 |
| 0.70 | 506 | 0.647 | 0.676 | 3.8 | 483 | 3.4 |
| 0.80 | 251 | 0.147 | 0.147 | 2.4 | 246 | 1.7 |
| 0.90 | 0 | 0.000 | 0.000 | nan | 0 | 0.0 |
