## Template baseline — foldC (eval sheet K-34-35-B-g)

9148 candidates against 18 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.389 | 7/18 | 0.203–0.614 |
| 10 px | 0.389 | 7/18 | 0.203–0.614 |
| 15 px | 0.556 | 10/18 | 0.337–0.754 |
| 25 px | 1.000 | 18/18 | 0.824–1.000 |

| localization error | px |
|---|---:|
| median | 11.88 |
| **p90** | **23.52** |
| p99 | 23.95 |
| max | 23.99 |

False positives: 9129 (54.34 per 512 px window, 398.48 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 8910 |
| `trig_point` | 93 |
| `decorative_symbol` | 91 |
| `road` | 19 |
| `colored_pencil` | 7 |
| `grid` | 5 |
| `text` | 2 |
| `contour` | 2 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 2 | 1.000 | 1.000 |
| has_trig_point | 2 | 0.000 | 0.500 |
| blurred_or_bad_print | 9 | 0.667 | 0.667 |
| crossed_by_contour | 9 | 0.444 | 0.667 |

### Operating curve

| threshold | candidates | recall@5px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 9148 | 0.389 | 0.556 | 23.5 | 9129 | 54.3 |
| 0.50 | 3215 | 0.389 | 0.500 | 23.2 | 3199 | 19.0 |
| 0.60 | 1463 | 0.278 | 0.278 | 19.4 | 1456 | 8.7 |
| 0.70 | 861 | 0.278 | 0.278 | 13.1 | 855 | 5.1 |
| 0.80 | 404 | 0.111 | 0.111 | 2.6 | 402 | 2.4 |
| 0.90 | 0 | 0.000 | 0.000 | nan | 0 | 0.0 |
