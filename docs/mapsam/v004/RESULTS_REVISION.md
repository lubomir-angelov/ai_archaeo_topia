# MapSAM v0.4 — revision against corrected annotations (v0.4r)

`RESULTS.md` in this folder is unchanged and remains the record of what was
measured in v0.4. This document reports a re-run of its step 3 arm — the
512 px leave-one-sheet-out segmentation — against the corrected annotations in
`annotation/cvat/v0.0.2`, and says which of v0.4's conclusions survive.

**Why the re-run was needed.** v0.5's detector failed on `K-34-35-B-g` in a way
that turned out to be an annotation error: twelve symbols labelled `mound` on
that sheet are mills, confirmed with a second annotator, and one real mound was
unlabelled. See `../v005/RESULTS.md`. v0.4 trained and evaluated on those
twelve, so its numbers were computed partly against incorrect ground truth.

Configs are identical to v0.4's except the dataset root, the experiment name
and the output root: `configs/mapsam/mapsam_v0_4r_loso512_fold*_pw20.yaml`
against `data/curated/datasets/mapsam_v03`. Same model, seed, optimizer, loss,
`pw20`, 512 px window, 150 loss-crop margin, 50 epochs.

---

## Summary

**v0.4's segmentation conclusions survive the correction.** The two folds whose
evaluation sets are unchanged move by less than 0.003 decoder IoU. The macro
figure rises from 0.7922 to 0.8113, and essentially all of that rise is fold C
no longer being asked to segment eleven small mill symbols.

The deeper point is about which stage could see the error at all.

> **MapSAM is a conditional segmenter and does not decide semantics.** Given a
> prompt on a printed symbol it outlines that symbol, and a mill is a printed
> symbol. So the mislabels cost v0.1–v0.4 almost nothing and were invisible to
> every metric those versions reported. It took the detector — the first stage
> that has to decide *whether* something is a mound — to expose them.

That is a structural property of the pipeline, not an accident of this dataset.
A conditional segmenter cannot audit its own labels. A detector can.

---

## The fold structure makes this unusually clean

The correction touches only `K-34-35-B-g`, and that sheet is a training sheet
for folds A and B and the evaluation sheet for fold C. So the three folds
isolate two different questions:

| fold | train samples | eval samples | what changed |
|---|---|---|---|
| A | 51 → **40** | 120 → **120** | training data only |
| B | 137 → **126** | 34 → **34** | training data only |
| C | 154 → **154** | 17 → **6** | evaluation data only |

Fold C's training set is *identical* between the two runs — it trains on
`K-35-51-B-a` and `K-35-8-G-a`, neither of which changed.

## Results

Final-epoch numbers, the convention v0.4 used. Lead column is disagreement area
in source px², which is the scale-invariant metric; decoder IoU is only
readable within one window size.

| fold | eval sheet | n | err px² v0.4 | err px² v0.4r | IoU v0.4 | IoU v0.4r | Δ IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| A | K-35-51-B-a | 120 | 116.4 | 110.6 | 0.7930 | 0.7913 | −0.0017 |
| B | K-35-8-G-a | 34 | 91.8 | 91.1 | 0.8261 | 0.8288 | +0.0027 |
| C | K-34-35-B-g | 17 → 6 | 103.1 | 106.0 | 0.7574 | 0.8137 | +0.0563 |
| | **macro** | | **103.8** | **102.6** | **0.7922** | **0.8113** | +0.0191 |

Cross-sheet spread in disagreement area: 24.6 → 19.5 source px².

Supporting rates:

| fold | IoU≥0.5 v0.4 → v0.4r | IoU≥0.75 v0.4 → v0.4r | zero-IoU | mean GT area px² |
|---|---|---|---|---|
| A | 0.99 → 0.99 | 0.78 → 0.77 | 0 → 0 | 465.9 → 465.9 |
| B | 1.00 → 1.00 | 0.91 → 0.97 | 0 → 0 | 470.4 → 470.4 |
| C | 1.00 → 1.00 | 0.53 → 0.83 | 0 → 0 | **381.2 → 550.0** |

## Reading the three folds

**Folds A and B: training-label noise cost segmentation nothing measurable.**
Both lost mislabelled training samples (11 of 51, and 11 of 137) with their
evaluation sets untouched. IoU moves −0.0017 and +0.0027. Those are far inside
the run-to-run and epoch-to-epoch variation v0.4 documented, so the honest
statement is *no measurable effect*, not *a small improvement*. Twelve
incorrect positives in a training set of 51 did not detectably degrade a
conditional segmenter — which is consistent with the mechanism above, since
the decoder was learning "outline the symbol under the prompt" either way.

**Fold C: the improvement is an evaluation-population change, not better
segmentation.** Its training set is byte-identical between runs, so nothing
about the model changed. What changed is that eleven small symbols left the
evaluation set, and the mean ground-truth area on that sheet rose 381.2 → 550.0
source px². v0.3 established that smaller symbols segment worse, so a 0.056 IoU
rise on a sheet whose mean symbol grew 44% is largely explained without any
claim about model quality. The IoU≥0.75 rate moving 0.53 → 0.83 is the same
effect seen at a stricter threshold.

**Fold C now evaluates six samples.** Nothing derived from it alone should be
leaned on. It is reported for completeness and because its *training*-side
invariance is what makes the A/B comparison interpretable.

## What this changes in v0.4's claims

| v0.4 claim | status |
|---|---|
| 512 px windows generalize across all three sheets | **stands** — 64–69% error reduction was measured against the full-tile arm, which this revision does not touch |
| Macro 0.7922 IoU / 103.8 px² disagreement area | **superseded in value, unchanged in substance** — 0.8113 / 102.6 px² on corrected labels |
| Cross-sheet spread 119.9 → 24.7 px² | **stands**, and narrows further to 19.5 |
| 99–100% of instances clear IoU 0.5 on every sheet | **stands** — unchanged on all three folds |
| Prompt tolerance, window slack, zero-IoU anatomy | **untouched** — those arms used v0.3 checkpoints and the `K-35-8-G-a` sheet, which the correction does not affect |
| The directional anisotropy hypothesis | **weakened** — v0.5 withdrew the companion-glyph reading that appeared to support it; the anisotropy itself is unexplained again |

Nothing in v0.4's argument depended on the twelve mislabelled symbols.

## Scope, unchanged

Still three sheets of one Soviet 1:50k series, still leave-one-sheet-out, still
no frozen test split, so still development evidence rather than a test result.
The correction removes a known error; it does not add data or broaden the
series. And the same review that found twelve errors on one sheet found an
omission on it too — the other two sheets have not had an equivalent
adversarial pass.

## Reproduce

```bash
python -m archeo_topia.datasets.prepare_mapsam_coco \
  --coco-json annotation/cvat/v0.0.2/instances_default.json \
  --images-dir <flat dir of the 12 clips> \
  --output-dir data/curated/datasets/mapsam_v03 \
  --positive-label mound --ignore-label uncertain_ignore --split-by sheet

python -m archeo_topia.datasets.generate_mapsam_prompts \
  --dataset-root data/curated/datasets/mapsam_v03 \
  --output-path data/curated/datasets/mapsam_v03/metadata/training_samples.jsonl \
  --bbox-padding 4

for f in foldA foldB foldC; do
  python -m archeo_topia.training.train_mapsam_v0 \
    --config configs/mapsam/mapsam_v0_4r_loso512_${f}_pw20.yaml
done

python -m archeo_topia.analysis.mapsam_compare_runs \
  --runs-dir artifacts/models/mapsam --prefix v0_4r_loso512
```

`prepare_mapsam_coco` expects a flat image directory; the curated dataset
stores clips under `images/train` and `images/test`, so stage symlinks first.

Environment: torch 2.14.0+cu130, segment-anything 1.0, one RTX 5090. v0.4 ran
on torch 2.12, so the two runs differ in torch version as well as in labels —
a confound that is small relative to the effects discussed but is not zero.
