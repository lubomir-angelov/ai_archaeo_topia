# MapSAM target size vs segmentation quality

### v0_2_decoder_only_pw200_cropoff

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw200_cropoff/prediction_stats_val_e25.jsonl`

Target size vs IoU correlation: **0.16** (Pearson), **0.169** (Spearman)

Target size vs *absolute* boundary error (Spearman): **-0.029**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 1 | 0.5 | 0.5 | 0.6667 | 0.0 | 1.0 | 3.0 | 6.0 | 3.0 | 0.433 |
| 4-5 px | 9 | 0.6635 | 0.625 | 0.7824 | 0.0 | 0.8889 | 4.56 | 6.11 | 2.44 | 0.534 |
| 6-8 px | 23 | 0.7013 | 0.7 | 0.8165 | 0.0 | 0.9565 | 6.39 | 6.43 | 2.39 | 0.632 |
| 9+ px | 1 | 0.7778 | 0.7778 | 0.875 | 0.0 | 1.0 | 9.0 | 7.0 | 2.0 | 0.75 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-35-8-G-a | 34 | 0.6876 | 0.6833 | 0.8048 | 0.0 | 0.9412 | 5.88 | 6.35 | 2.41 | 0.606 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 0.6876 | 0.6833 | 0.8048 | 0.0 | 0.9412 | 5.88 | 6.35 | 2.41 | 0.606 |

### v0_2_decoder_only_pw200_cropon

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw200_cropon/prediction_stats_val_e25.jsonl`

Target size vs IoU correlation: **-0.0319** (Pearson), **0.1392** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.137**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 1 | 0.6 | 0.6 | 0.75 | 0.0 | 1.0 | 3.0 | 5.0 | 2.0 | 0.433 |
| 4-5 px | 9 | 0.6705 | 0.625 | 0.7928 | 0.0 | 1.0 | 4.56 | 5.33 | 2.11 | 0.534 |
| 6-8 px | 23 | 0.6982 | 0.7 | 0.818 | 0.0 | 1.0 | 6.39 | 6.26 | 2.3 | 0.632 |
| 9+ px | 1 | 0.5556 | 0.5556 | 0.7143 | 0.0 | 1.0 | 9.0 | 5.0 | 4.0 | 0.75 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-35-8-G-a | 34 | 0.6838 | 0.6667 | 0.8063 | 0.0 | 1.0 | 5.88 | 5.94 | 2.29 | 0.606 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 0.6838 | 0.6667 | 0.8063 | 0.0 | 1.0 | 5.88 | 5.94 | 2.29 | 0.606 |

### v0_2_decoder_only_pw20_cropon

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw20_cropon/prediction_stats_val_e25.jsonl`

Target size vs IoU correlation: **0.1544** (Pearson), **0.1574** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.0305**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 1 | 0.5 | 0.5 | 0.6667 | 0.0 | 1.0 | 3.0 | 6.0 | 3.0 | 0.433 |
| 4-5 px | 9 | 0.5624 | 0.5 | 0.7065 | 0.0 | 0.6667 | 4.56 | 6.22 | 3.22 | 0.534 |
| 6-8 px | 23 | 0.6462 | 0.6667 | 0.7716 | 0.0 | 0.7826 | 6.39 | 7.0 | 3.13 | 0.632 |
| 9+ px | 1 | 0.6667 | 0.6667 | 0.8 | 0.0 | 1.0 | 9.0 | 6.0 | 3.0 | 0.75 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-35-8-G-a | 34 | 0.6203 | 0.6125 | 0.7521 | 0.0 | 0.7647 | 5.88 | 6.74 | 3.15 | 0.606 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 0.6203 | 0.6125 | 0.7521 | 0.0 | 0.7647 | 5.88 | 6.74 | 3.15 | 0.606 |

### v0_2_decoder_only_pw20_cropoff

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw20_cropoff/prediction_stats_val_e25.jsonl`

Target size vs IoU correlation: **0.5531** (Pearson), **0.5925** (Spearman)

Target size vs *absolute* boundary error (Spearman): **-0.4229**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 1 | 0.25 | 0.25 | 0.4 | 0.0 | 0.0 | 3.0 | 7.0 | 6.0 | 0.433 |
| 4-5 px | 9 | 0.4471 | 0.4286 | 0.6087 | 0.0 | 0.4444 | 4.56 | 8.44 | 5.22 | 0.534 |
| 6-8 px | 23 | 0.5911 | 0.6 | 0.7213 | 0.0 | 0.7391 | 6.39 | 8.78 | 4.3 | 0.632 |
| 9+ px | 1 | 0.7778 | 0.7778 | 0.875 | 0.0 | 1.0 | 9.0 | 7.0 | 2.0 | 0.75 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-35-8-G-a | 34 | 0.5484 | 0.5556 | 0.6865 | 0.0 | 0.6471 | 5.88 | 8.59 | 4.53 | 0.606 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 0.5484 | 0.5556 | 0.6865 | 0.0 | 0.6471 | 5.88 | 8.59 | 4.53 | 0.606 |
