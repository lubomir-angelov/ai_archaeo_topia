## Template baseline — foldC (eval sheet K-34-35-B-g)

9148 candidates against 7 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.857 | 6/7 | 0.487–0.974 |
| 10 px | 0.857 | 6/7 | 0.487–0.974 |
| 15 px | 0.857 | 6/7 | 0.487–0.974 |
| 25 px | 1.000 | 7/7 | 0.646–1.000 |

| localization error | px |
|---|---:|
| median | 2.69 |
| **p90** | **11.56** |
| p99 | 21.77 |
| max | 22.90 |

False positives: 9140 (54.40 per 512 px window, 398.96 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 8889 |
| `trig_point` | 93 |
| `decorative_symbol` | 91 |
| `other` | 31 |
| `road` | 19 |
| `colored_pencil` | 7 |
| `grid` | 5 |
| `text` | 2 |
| `contour` | 2 |
| `__undefined__` | 1 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 2 | 1.000 | 1.000 |
| has_trig_point | 1 | 1.000 | 1.000 |
| blurred_or_bad_print | 6 | 0.833 | 0.833 |
| crossed_by_contour | 4 | 1.000 | 1.000 |

### Operating curve

| threshold | candidates | recall@5px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 9148 | 0.857 | 0.857 | 11.6 | 9140 | 54.4 |
| 0.50 | 3215 | 0.857 | 0.857 | 11.6 | 3207 | 19.1 |
| 0.60 | 1463 | 0.857 | 0.857 | 3.6 | 1457 | 8.7 |
| 0.70 | 861 | 0.714 | 0.714 | 3.5 | 856 | 5.1 |
| 0.80 | 404 | 0.286 | 0.286 | 2.6 | 402 | 2.4 |
| 0.90 | 0 | 0.000 | 0.000 | nan | 0 | 0.0 |
