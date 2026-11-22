# Frozen blind-test ground truth, v0.0.1

`instances_Test.json` is the CVAT COCO 1.0 annotation file of the frozen blind-test
set, extracted unmodified (CVAT's original file name kept) from
`ai_archaeo_topia_frozen_v1.zip`, member `annotations/instances_Test.json`.

This is the consensus ground truth of the frozen blind test. It is **test-only**:
never use it for training, tuning, threshold selection or prompt design.

## Provenance

| Item | Value |
|---|---|
| COCO export zip | `ai_archaeo_topia_frozen_v1.zip` |
| COCO zip sha256 | `5779581677b0befbb7ac131a404ed25cefadfb2ac1d6c62704ebd9608c68cdf3` |
| CVAT 1.1 export zip | `ai_archaeo_topia_frozen_v1_cvat.zip` |
| CVAT 1.1 zip sha256 | `4ae7d050d25e76f1e1c789e7012e0e61a67db191ab8baa95db742423c65c4d2f` |
| `instances_Test.json` sha256 | `0d13665231e45c24bf6b3e78e4c057c9b7842bcb7d2f5c219766dd1cd829b259` |
| Export date | 2026-11-22 (CVAT project `updated` 18:12 UTC; zip members stamped 18:13; zip files on disk 20:14 local) |
| Zip location | `data_lake/raw/mound_test_20260915/_frozen/cvat/` |

## Images

Images are not stored in git. They live inside the two raw zips (`images/Test/`)
and are byte-identical, once decoded, to the frozen clips in
`data_lake/cleaned/map_clips/_frozen/<sheet>/`. `file_name` values in the JSON are
bare clip names (`<sheet>_<n>.png`) with no subset folder prefix.

Label schema: `annotation/cvat/labels.json` (v0.0.4).
