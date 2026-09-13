# MapSAM target size vs segmentation quality

### v0_2_decoder_only_pw200_cropoff

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw200_cropoff/prediction_stats_train_e20.jsonl`

Target size vs IoU correlation: **0.1173** (Pearson), **0.0348** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.1414**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 16 | 0.5078 | 0.45 | 0.6145 | 0.0625 | 0.5 | 2.31 | 4.44 | 2.88 | 0.38 |
| 4-5 px | 56 | 0.7561 | 0.8 | 0.8304 | 0.0179 | 0.8571 | 4.41 | 5.52 | 2.0 | 0.525 |
| 6-8 px | 57 | 0.736 | 0.8333 | 0.8237 | 0.0 | 0.8596 | 6.42 | 6.63 | 2.39 | 0.633 |
| 9+ px | 8 | 0.7268 | 0.7596 | 0.8386 | 0.0 | 1.0 | 10.38 | 9.62 | 3.25 | 0.805 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-34-35-B-g | 17 | 0.6909 | 0.75 | 0.7776 | 0.0 | 0.7647 | 4.24 | 5.29 | 1.76 | 0.514 |
| K-35-51-B-a | 120 | 0.7207 | 0.7596 | 0.8065 | 0.0167 | 0.8333 | 5.51 | 6.21 | 2.42 | 0.587 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 137 | 0.717 | 0.75 | 0.8029 | 0.0146 | 0.8248 | 5.35 | 6.09 | 2.34 | 0.578 |

### v0_2_decoder_only_pw200_cropon

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw200_cropon/prediction_stats_train_e20.jsonl`

Target size vs IoU correlation: **0.0792** (Pearson), **0.0755** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.0929**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 16 | 0.7324 | 0.6667 | 0.8199 | 0.0 | 0.875 | 2.31 | 3.31 | 1.13 | 0.38 |
| 4-5 px | 56 | 0.7843 | 0.8 | 0.8622 | 0.0 | 0.9107 | 4.41 | 5.36 | 1.48 | 0.525 |
| 6-8 px | 57 | 0.8338 | 0.8571 | 0.8979 | 0.0 | 0.9649 | 6.42 | 6.88 | 1.4 | 0.633 |
| 9+ px | 8 | 0.7629 | 0.7679 | 0.8585 | 0.0 | 1.0 | 10.38 | 11.25 | 3.12 | 0.805 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-34-35-B-g | 17 | 0.8412 | 0.8 | 0.8986 | 0.0 | 0.9412 | 4.24 | 5.12 | 0.88 | 0.514 |
| K-35-51-B-a | 120 | 0.7914 | 0.8167 | 0.8681 | 0.0 | 0.9333 | 5.51 | 6.23 | 1.59 | 0.587 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 137 | 0.7976 | 0.8 | 0.8719 | 0.0 | 0.9343 | 5.35 | 6.09 | 1.5 | 0.578 |

### v0_2_decoder_only_pw20_cropon

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw20_cropon/prediction_stats_train_e20.jsonl`

Target size vs IoU correlation: **0.0681** (Pearson), **-0.0047** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.1963**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 16 | 0.6493 | 0.7083 | 0.7459 | 0.0 | 0.75 | 2.31 | 3.56 | 1.63 | 0.38 |
| 4-5 px | 56 | 0.7571 | 0.8 | 0.8402 | 0.0 | 0.8571 | 4.41 | 5.14 | 1.66 | 0.525 |
| 6-8 px | 57 | 0.7445 | 0.75 | 0.8298 | 0.0175 | 0.8772 | 6.42 | 6.33 | 2.23 | 0.633 |
| 9+ px | 8 | 0.8138 | 0.8444 | 0.8933 | 0.0 | 1.0 | 10.38 | 12.0 | 2.63 | 0.805 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-34-35-B-g | 17 | 0.7732 | 0.8 | 0.8489 | 0.0 | 0.8235 | 4.24 | 4.59 | 1.18 | 0.514 |
| K-35-51-B-a | 120 | 0.7383 | 0.75 | 0.825 | 0.0083 | 0.8667 | 5.51 | 6.03 | 2.06 | 0.587 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 137 | 0.7426 | 0.75 | 0.828 | 0.0073 | 0.8613 | 5.35 | 5.85 | 1.95 | 0.578 |

### v0_2_decoder_only_pw20_cropoff

Source: `artifacts/models/mapsam/v0_2_decoder_only_pw20_cropoff/prediction_stats_train_e20.jsonl`

Target size vs IoU correlation: **0.1955** (Pearson), **0.1885** (Spearman)

Target size vs *absolute* boundary error (Spearman): **0.1438**

By target size:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 px | 16 | 0.2607 | 0.3095 | 0.3621 | 0.375 | 0.25 | 2.31 | 4.31 | 4.13 | 0.38 |
| 4-5 px | 56 | 0.4815 | 0.5278 | 0.5942 | 0.125 | 0.5714 | 4.41 | 5.52 | 3.93 | 0.525 |
| 6-8 px | 57 | 0.526 | 0.5714 | 0.646 | 0.0702 | 0.6491 | 6.42 | 6.12 | 4.3 | 0.633 |
| 9+ px | 8 | 0.5605 | 0.6154 | 0.6976 | 0.0 | 0.75 | 10.38 | 8.0 | 5.37 | 0.805 |

By sheet:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| K-34-35-B-g | 17 | 0.3832 | 0.5 | 0.4961 | 0.2353 | 0.5294 | 4.24 | 4.18 | 3.82 | 0.514 |
| K-35-51-B-a | 120 | 0.4924 | 0.5 | 0.6086 | 0.1083 | 0.5833 | 5.51 | 6.0 | 4.24 | 0.587 |

Overall:

| Group | N | mean IoU | median IoU | mean Dice | zero-IoU | IoU>=0.5 | GT px | pred px | abs err px | tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 137 | 0.4788 | 0.5 | 0.5947 | 0.1241 | 0.5766 | 5.35 | 5.77 | 4.19 | 0.578 |
