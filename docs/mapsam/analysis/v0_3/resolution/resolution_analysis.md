# MapSAM target size vs segmentation quality

### v0_3_res_a_fulltile_pw20

Source: `artifacts/models/mapsam/v0_3_res_a_fulltile_pw20/prediction_stats_val_e50.jsonl`

Target size vs IoU correlation: **0.0166** (Pearson), **0.0911** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.0644**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-9 px | 34 | 0.6021 | 0.5982 | 0.7372 | 0.0 | 0.7059 | 5.88 | 6.0 | 3.12 | 0.606 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-35-8-G-a | 34 | 0.6021 | 0.5982 | 0.7372 | 0.0 | 0.7059 | 5.88 | 6.0 | 3.12 | 0.606 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 0.6021 | 0.5982 | 0.7372 | 0.0 | 0.7059 | 5.88 | 6.0 | 3.12 | 0.606 |

### v0_3_res_b_window1024_pw20

Source: `artifacts/models/mapsam/v0_3_res_b_window1024_pw20/prediction_stats_val_e50.jsonl`

Target size vs IoU correlation: **0.1501** (Pearson), **0.0887** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.1302**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 10-39 px | 34 | 0.7567 | 0.7422 | 0.8584 | 0.0 | 1.0 | 29.41 | 30.24 | 8.47 | 1.356 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-35-8-G-a | 34 | 0.7567 | 0.7422 | 0.8584 | 0.0 | 1.0 | 29.41 | 30.24 | 8.47 | 1.356 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 0.7567 | 0.7422 | 0.8584 | 0.0 | 1.0 | 29.41 | 30.24 | 8.47 | 1.356 |

### v0_3_res_c_window512_pw20

Source: `artifacts/models/mapsam/v0_3_res_c_window512_pw20/prediction_stats_val_e50.jsonl`

Target size vs IoU correlation: **0.2008** (Pearson), **0.2025** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.1145**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 40-99 px | 5 | 0.8008 | 0.8077 | 0.8876 | 0.0 | 1.0 | 90.6 | 107.4 | 22.0 | 2.38 |
| 100+ px | 29 | 0.8305 | 0.8496 | 0.9065 | 0.0 | 1.0 | 122.24 | 126.1 | 23.1 | 2.764 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-35-8-G-a | 34 | 0.8261 | 0.844 | 0.9038 | 0.0 | 1.0 | 117.59 | 123.35 | 22.94 | 2.711 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 0.8261 | 0.844 | 0.9038 | 0.0 | 1.0 | 117.59 | 123.35 | 22.94 | 2.711 |
