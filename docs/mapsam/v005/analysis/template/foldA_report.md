## Template baseline — foldA (eval sheet K-35-51-B-a)

14401 candidates against 128 annotations (association radius 25 px, one-to-one).

| radius | recall | hits | 95% CI |
|---:|---:|---:|---|
| 5 px **(primary)** | 0.781 | 100/128 | 0.702–0.844 |
| 10 px | 0.836 | 107/128 | 0.762–0.890 |
| 15 px | 0.875 | 112/128 | 0.807–0.922 |
| 25 px | 0.977 | 125/128 | 0.933–0.992 |

| localization error | px |
|---|---:|
| median | 2.50 |
| **p90** | **15.26** |
| p99 | 24.35 |
| max | 24.87 |

False positives: 14276 (84.98 per 512 px window, 643.99 per megapixel)

| false positive lands on | n |
|---|---:|
| `background` | 13515 |
| `decorative_symbol` | 351 |
| `road` | 239 |
| `trig_point` | 111 |
| `text` | 56 |
| `__undefined__` | 4 |

| subset | n | recall@5px | recall@15px |
|---|---:|---:|---:|
| merged_component | 15 | 0.667 | 0.933 |
| has_trig_point | 28 | 0.679 | 0.786 |
| blurred_or_bad_print | 20 | 0.450 | 0.550 |
| crossed_by_contour | 50 | 0.700 | 0.840 |

### Operating curve

| threshold | candidates | recall@5px | recall@15px | p90 err | FP | FP/window |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 14401 | 0.781 | 0.875 | 15.3 | 14276 | 85.0 |
| 0.50 | 6545 | 0.750 | 0.805 | 10.9 | 6432 | 38.3 |
| 0.60 | 2532 | 0.641 | 0.656 | 4.6 | 2443 | 14.5 |
| 0.70 | 814 | 0.391 | 0.398 | 4.6 | 761 | 4.5 |
| 0.80 | 357 | 0.070 | 0.070 | 4.4 | 348 | 2.1 |
| 0.90 | 157 | 0.000 | 0.000 | nan | 157 | 0.9 |
