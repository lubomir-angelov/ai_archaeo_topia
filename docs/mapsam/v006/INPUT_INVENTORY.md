# MapSAM v0.6 — input inventory for the 60 new sheets

Measured with `gdalinfo` over every file at
`data_lake/raw/mound_test_20260915`, before any model was run against them.
`PLAN.md` carries the conclusions; this document is the raw record, kept so a
later session can check a claim without re-reading 60 rasters.

## Aggregates

| | value |
|---|---|
| sheets | 60 — 20 in `01_maps_test`, 40 in `02_maps_test` |
| width px | 4682–5112, median 4920 |
| height px | 4328–4635, median 4464 |
| area | 20.3–23.4 Mpx per sheet, **1318 Mpx total** |
| pixel size | 2.1061–2.1566 m/px, median 2.1192 (2.4% spread) |
| implied scan DPI at 1:25,000 | 294–302, median 300 |
| ground extent | ~10.4 × 9.5 km per sheet |
| CRS | EPSG:25835 (ETRS89 / UTM 35N) on all 60, **embedded in the GeoTIFF** |
| bands | RGBA, Byte, DEFLATE, 512×512 blocks, on all 60 |
| windows at 512/384 | 132–156 per sheet, **9271 total** |
| windows per Mpx | 7.03 (current annotated corpus: 7.23) |

Comparison with the annotated corpus: 12 clips, 66.4 Mpx, 169 mounds, 480
windows. The new sheets are a **20-fold** area expansion.

## Derived figures used in the plan

| | value | basis |
|---|---:|---|
| projected false positives at conf 0.25 | ~3,338 | fold A's 0.36 FP/window |
| projected false positives at conf 0.05 | ~5,655 | fold A's 0.61 FP/window |
| projected mounds (upper bound) | ~3,362 | 2.55 mounds/Mpx, measured on annotation-selected clips |

The mound figure is an upper bound by construction: it applies a density
measured on clips chosen for annotation to whole sheets that contain far more
uninformative terrain. Step 1 of the plan replaces all three with
measurements. They are recorded to be falsified, not believed.

## Sheets sharing a 1:100k parent with an annotated sheet

No sheet id collides with the three annotated sheets, but four share a parent:

| new sheet | parent | annotated sheet under the same parent |
|---|---|---|
| `K-34-35-A-v` | `K-34-35` | `K-34-35-B-g` |
| `K-35-51-A-b` | `K-35-51` | `K-35-51-B-a` |
| `K-35-51-B-g` | `K-35-51` | `K-35-51-B-a` — **adjacent quadrant** |
| `K-35-8-V-g` | `K-35-8` | `K-35-8-G-a` |

`K-35-51-B-g` and the annotated `K-35-51-B-a` are neighbouring 1:25k cells of
the same 1:50k sheet `K-35-51-B`. They share terrain, survey campaign, print
run and scan batch, so splits should group on the 1:100k parent rather than
the 1:25k sheet id — see step 4 of `PLAN.md`.

Parent distribution across the 60: 53 distinct parents, largest 3 sheets
(`K-34-47`), so the corpus is spread rather than clustered.

Geographic spread by 1:1M zone: `K-34` 16, `K-35` 43, `L-35` 1.
The annotated corpus sits entirely in `K-34` and `K-35`; `L-35-139-V-v` is the
only sheet from a different 1:1M zone, and it is also the smallest and the only
one that tiles to 132 windows rather than 156.

## Per-sheet detail

| set | sheet | width | height | m/px | Mpx | windows | parent |
|---|---|---:|---:|---:|---:|---:|---|
| 01 | `K-34-10-A-g` | 4971 | 4618 | 2.1178 | 23.0 | 156 | `K-34-10` |
| 01 | `K-34-36-V-b` | 4916 | 4506 | 2.1428 | 22.2 | 156 | `K-34-36` |
| 01 | `K-34-60-V-b` | 5051 | 4584 | 2.1061 | 23.2 | 156 | `K-34-60` |
| 01 | `K-34-70-B-v` | 5075 | 4598 | 2.1231 | 23.3 | 156 | `K-34-70` |
| 01 | `K-34-83-V-g` | 5087 | 4571 | 2.1252 | 23.3 | 156 | `K-34-83` |
| 01 | `K-34-96-B-v` | 5112 | 4575 | 2.1064 | 23.4 | 156 | `K-34-96` |
| 01 | `K-35-14-B-v` | 4904 | 4515 | 2.1084 | 22.1 | 156 | `K-35-14` |
| 01 | `K-35-21-G-a` | 4864 | 4463 | 2.1087 | 21.7 | 156 | `K-35-21` |
| 01 | `K-35-22-A-v` | 4845 | 4455 | 2.1198 | 21.6 | 156 | `K-35-22` |
| 01 | `K-35-28-A-g` | 4776 | 4365 | 2.1564 | 20.8 | 156 | `K-35-28` |
| 01 | `K-35-31-V-g` | 4805 | 4369 | 2.1228 | 21.0 | 156 | `K-35-31` |
| 01 | `K-35-42-A-g` | 4831 | 4383 | 2.1197 | 21.2 | 156 | `K-35-42` |
| 01 | `K-35-5-G-b` | 4791 | 4415 | 2.1117 | 21.2 | 156 | `K-35-5` |
| 01 | `K-35-50-B-b` | 4942 | 4484 | 2.1193 | 22.2 | 156 | `K-35-50` |
| 01 | `K-35-51-B-g` | 4935 | 4466 | 2.1141 | 22.0 | 156 | `K-35-51` |
| 01 | `K-35-53-A-g` | 4887 | 4415 | 2.1180 | 21.6 | 156 | `K-35-53` |
| 01 | `K-35-63-B-g` | 4965 | 4471 | 2.1118 | 22.2 | 156 | `K-35-63` |
| 01 | `K-35-67-B-v` | 4865 | 4366 | 2.1275 | 21.2 | 156 | `K-35-67` |
| 01 | `K-35-75-B-a` | 4969 | 4458 | 2.1209 | 22.2 | 156 | `K-35-75` |
| 01 | `K-35-88-G-g` | 4965 | 4408 | 2.1276 | 21.9 | 156 | `K-35-88` |
| 02 | `K-34-22-V-b` | 5020 | 4635 | 2.1102 | 23.3 | 156 | `K-34-22` |
| 02 | `K-34-23-A-g` | 4923 | 4547 | 2.1374 | 22.4 | 156 | `K-34-23` |
| 02 | `K-34-24-G-a` | 4873 | 4488 | 2.1480 | 21.9 | 156 | `K-34-24` |
| 02 | `K-34-35-A-v` | 4994 | 4590 | 2.1203 | 22.9 | 156 | `K-34-35` |
| 02 | `K-34-47-A-a` | 5023 | 4599 | 2.1163 | 23.1 | 156 | `K-34-47` |
| 02 | `K-34-47-G-g` | 5035 | 4588 | 2.1110 | 23.1 | 156 | `K-34-47` |
| 02 | `K-34-47-G-v` | 5035 | 4589 | 2.1135 | 23.1 | 156 | `K-34-47` |
| 02 | `K-34-48-G-b` | 4984 | 4543 | 2.1188 | 22.6 | 156 | `K-34-48` |
| 02 | `K-34-58-B-a` | 5058 | 4610 | 2.1176 | 23.3 | 156 | `K-34-58` |
| 02 | `K-34-84-A-a` | 5075 | 4573 | 2.1145 | 23.2 | 156 | `K-34-84` |
| 02 | `K-35-13-G-v` | 4900 | 4503 | 2.1274 | 22.1 | 156 | `K-35-13` |
| 02 | `K-35-16-A-a` | 4843 | 4457 | 2.1152 | 21.6 | 156 | `K-35-16` |
| 02 | `K-35-16-B-a` | 4841 | 4454 | 2.1102 | 21.6 | 156 | `K-35-16` |
| 02 | `K-35-18-A-b` | 4797 | 4406 | 2.1091 | 21.1 | 156 | `K-35-18` |
| 02 | `K-35-19-V-a` | 4753 | 4351 | 2.1285 | 20.7 | 143 | `K-35-19` |
| 02 | `K-35-20-B-g` | 4841 | 4446 | 2.1070 | 21.5 | 156 | `K-35-20` |
| 02 | `K-35-27-B-b` | 4781 | 4378 | 2.1566 | 20.9 | 156 | `K-35-27` |
| 02 | `K-35-29-G-b` | 4799 | 4373 | 2.1311 | 21.0 | 156 | `K-35-29` |
| 02 | `K-35-32-A-a` | 4737 | 4328 | 2.1534 | 20.5 | 143 | `K-35-32` |
| 02 | `K-35-38-A-a` | 4935 | 4504 | 2.1200 | 22.2 | 156 | `K-35-38` |
| 02 | `K-35-39-A-g` | 4918 | 4476 | 2.1163 | 22.0 | 156 | `K-35-39` |
| 02 | `K-35-39-G-v` | 4933 | 4477 | 2.1124 | 22.1 | 156 | `K-35-39` |
| 02 | `K-35-39-V-g` | 4914 | 4461 | 2.1234 | 21.9 | 156 | `K-35-39` |
| 02 | `K-35-40-B-v` | 4817 | 4379 | 2.1461 | 21.1 | 156 | `K-35-40` |
| 02 | `K-35-41-G-a` | 4846 | 4394 | 2.1248 | 21.3 | 156 | `K-35-41` |
| 02 | `K-35-43-V-g` | 4793 | 4334 | 2.1396 | 20.8 | 143 | `K-35-43` |
| 02 | `K-35-44-B-a` | 4776 | 4343 | 2.1531 | 20.7 | 143 | `K-35-44` |
| 02 | `K-35-49-V-g` | 5022 | 4546 | 2.1102 | 22.8 | 156 | `K-35-49` |
| 02 | `K-35-50-V-b` | 4988 | 4516 | 2.1108 | 22.5 | 156 | `K-35-50` |
| 02 | `K-35-51-A-b` | 4946 | 4483 | 2.1125 | 22.2 | 156 | `K-35-51` |
| 02 | `K-35-52-G-v` | 4923 | 4438 | 2.1167 | 21.8 | 156 | `K-35-52` |
| 02 | `K-35-55-A-g` | 4815 | 4343 | 2.1353 | 20.9 | 143 | `K-35-55` |
| 02 | `K-35-62-G-v` | 4989 | 4487 | 2.1210 | 22.4 | 156 | `K-35-62` |
| 02 | `K-35-66-A-a` | 4865 | 4373 | 2.1274 | 21.3 | 156 | `K-35-66` |
| 02 | `K-35-73-A-v` | 5048 | 4537 | 2.1173 | 22.9 | 156 | `K-35-73` |
| 02 | `K-35-77-A-g` | 4960 | 4433 | 2.1084 | 22.0 | 156 | `K-35-77` |
| 02 | `K-35-8-V-g` | 4793 | 4411 | 2.1164 | 21.1 | 156 | `K-35-8` |
| 02 | `K-35-86-A-b` | 5031 | 4497 | 2.1192 | 22.6 | 156 | `K-35-86` |
| 02 | `K-35-87-G-a` | 5005 | 4456 | 2.1219 | 22.3 | 156 | `K-35-87` |
| 02 | `L-35-139-V-v` | 4682 | 4328 | 2.1401 | 20.3 | 132 | `L-35-139` |

## Notes

- Files are named `<sheet>_clipped.tif` and are already cut to the map frame,
  so no collar or margin removal is needed before tiling.
- The 20 `.tif.aux.xml` files in `01_maps_test` hold only GDAL PAM band
  statistics (min, max, mean, stddev, valid percent). They are irrelevant to
  georeferencing, which is embedded in all 60 rasters.
- The alpha band is present on all 60; `STATISTICS_VALID_PERCENT` around 91%
  on the sampled sheet suggests a non-rectangular valid region, so tiling
  should expect transparent or nodata corners.
