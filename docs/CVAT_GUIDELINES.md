Yes, use **CVAT during all three phases**, but do **not** make CVAT the system of record for the experimental results.

Use CVAT as:

```text
1. Ground-truth annotation tool
2. Assisted annotation tool
3. Visual inspection / qualitative comparison tool
4. Import/export hub for COCO/YOLO masks and boxes
```

Use your own experiment scripts/files as:

```text
1. Baseline runner
2. Few-shot training runner
3. Metrics calculator
4. Reproducibility log
5. Paper tables/figures generator
```

CVAT supports serverless model integration through Nuclio, including Segment Anything, and the official tutorial explicitly describes wrapping DL models as serverless functions callable by CVAT. ([docs.cvat.ai][1]) It also supports COCO import/export for boxes, polygons, and masks, which makes it a good bridge between annotation and ML experiments. ([docs.cvat.ai][2])

## Recommended split of responsibility

| Phase                      | Use CVAT? | What CVAT should do                                     | What should stay outside CVAT                  |
| -------------------------- | --------: | ------------------------------------------------------- | ---------------------------------------------- |
| Days 3–4 annotation sprint |       Yes | Create ground truth boxes/masks, hard negatives, review | Dataset split logic, versioning                |
| Days 5–6 SAM baselines     |    Partly | Run SAM interactively, visualize/import predictions     | Reproducible prompting, metrics, tables        |
| Days 7–8 MapSAM/few-shot   | Mostly no | Export GT, inspect/import predictions                   | Training, adaptation, evaluation, run tracking |

So the practical answer is:

> **Use CVAT for annotation and visual review. Use Python scripts for zero-shot/few-shot experiments and metrics. Import model outputs back into CVAT only for inspection and qualitative figures.**

## Days 3–4: CVAT is the right primary tool

Use CVAT here fully.

Create one project:

```text
archeo_mound_detection_v0_1
```

Labels:

```text
mound
hard_negative_symbol
uncertain_ignore
```

Attributes:

```text
has_trig_point
has_elevation_mark
crossed_by_grid
crossed_by_contour
crossed_by_road
crossed_by_powerline
affected_by_colored_pencil
blurred_or_bad_print
overlaps_other_mound
negative_type
```

Tasks should be grouped by **map sheet**, not random tile:

```text
task_K-35-50-B-b
task_K-35-51-A-b
task_K-35-8-G-a
```

This helps avoid leakage later.

Use CVAT’s review/quality features for annotation QA. CVAT supports validation sets, ground-truth jobs, review mode, and quality analytics for annotation quality control. ([docs.cvat.ai][3])

## Days 5–6: Use CVAT for SAM interaction, but not as the full benchmark runner

CVAT can run SAM as an interactor. The SAM integration in CVAT Community is described as an interactive tool using positive and negative points. ([CVAT.ai][4])

That is useful for annotation, but it is not ideal for reproducible paper metrics because the manual click history is not the same thing as a controlled experiment protocol.

### For the article, run SAM baselines outside CVAT

For reproducibility, generate prompts programmatically from ground truth.

Example protocols:

```text
SAM-zero-point:
  prompt = center point of each ground-truth mound box

SAM-box:
  prompt = ground-truth bounding box

SAM-auto-proposal:
  prompt = automatically generated candidate points/boxes from image-processing proposals
```

Then store:

```text
predicted masks
predicted boxes
confidence/score if available
prompt type
model checkpoint
runtime
IoU / Dice / AP / recall / precision
qualitative examples
```

You can then import the prediction masks/boxes into CVAT as a **separate task copy** for visual inspection. Do not upload them into the original GT task, because CVAT warns that uploading annotations to a task removes the existing annotations. ([docs.cvat.ai][5])

Use task copies like:

```text
GT_task_K-35-50-B-b
SAM_zero_point_task_K-35-50-B-b
SAM_box_prompt_task_K-35-50-B-b
SAM_auto_proposal_task_K-35-50-B-b
```

That lets colleagues visually compare model outputs without corrupting the ground truth.

## Days 7–8: Use CVAT as dataset source and visualizer, not as MapSAM training environment

MapSAM-style adaptation should run outside CVAT.

MapSAM is specifically relevant to your paper idea because it adapts SAM for automated feature detection in historical maps, and the paper describes parameter-efficient fine-tuning with limited data, including very small training sets such as 10-shot settings. ([arXiv][6])

Use CVAT to export the data:

```text
CVAT GT → COCO export → MapSAM/few-shot training script
```

Then after training:

```text
MapSAM predictions → COCO predictions/masks → import into separate CVAT task for inspection
```

But keep metrics outside CVAT.

CVAT’s Python SDK can help here because it includes a PyTorch adapter and auto-annotation API for connecting CVAT datasets to Python ML workflows. ([docs.cvat.ai][7])

## Can CVAT “record the results” for zero/few-shot?

Only partially.

CVAT can store:

```text
model-produced annotations
manual corrections
visual overlays
QA/review status
exported COCO masks/boxes
```

CVAT is not the right place to store the full experimental matrix:

```text
model checkpoint
SAM variant
prompting protocol
5-shot / 10-shot / 25-shot split
random seed
training config
per-class AP
per-sheet metrics
IoU distribution
runtime
failure categories
paper-ready aggregate tables
```

For that, create an experiment folder outside CVAT.

Recommended structure:

```text
experiments/
  sam_zero_point/
    config.json
    predictions_coco.json
    metrics_by_instance.csv
    metrics_by_sheet.csv
    summary.json
    qualitative_examples/
  sam_box_prompt/
    config.json
    predictions_coco.json
    metrics_by_instance.csv
    metrics_by_sheet.csv
    summary.json
    qualitative_examples/
  sam_auto_proposal/
    config.json
    predictions_coco.json
    metrics_by_instance.csv
    metrics_by_sheet.csv
    summary.json
    qualitative_examples/
  mapsam_5shot/
    split.json
    config.json
    predictions_coco.json
    metrics_by_instance.csv
    summary.json
  mapsam_10shot/
  mapsam_25shot/
```

## Best workflow for your plan

Use this version:

```text
Days 3–4: Annotation sprint
  Use CVAT fully.
  Create ground-truth boxes/masks.
  Annotate mound, hard_negative_symbol, uncertain_ignore.
  Review difficult samples.
  Export GT as COCO.
  Freeze this as dataset_v0.1.

Days 5–6: SAM baselines
  Use Python scripts for reproducible SAM runs.
  Run:
    - zero-shot point prompt
    - box prompt
    - automated proposal prompt
  Save masks, metrics, and configs outside CVAT.
  Import predictions into duplicate CVAT tasks only for visual review.

Days 7–8: MapSAM / few-shot adaptation
  Train/adapt outside CVAT.
  Use CVAT-exported COCO as input.
  Use fixed 5-shot, 10-shot, 25-shot splits by map sheet.
  Compute metrics outside CVAT.
  Import selected outputs back to CVAT for qualitative comparison.
```

## Important caution

Do not mix these into one CVAT task:

```text
human ground truth
SAM zero-shot predictions
SAM box-prompt predictions
MapSAM 5-shot predictions
MapSAM 10-shot predictions
MapSAM 25-shot predictions
```

That will become unmanageable and risks overwriting or contaminating the ground truth.

Keep one immutable GT project/task set, then create separate prediction/evaluation task copies.

## Final recommendation

Use CVAT as the **annotation and visualization workbench**, not as the **experiment tracker**.

The clean setup is:

```text
CVAT = source of human ground truth + visual review
COCO files = interchange format
Python evaluation scripts = source of benchmark results
experiment folders = source of paper tables
duplicate CVAT tasks = qualitative comparison screenshots
```

That gives you a defensible article workflow: expert GT in CVAT, reproducible SAM/MapSAM experiments outside CVAT, and CVAT overlays for qualitative examples.

[1]: https://docs.cvat.ai/docs/guides/serverless-tutorial/ "Serverless tutorial | CVAT"
[2]: https://docs.cvat.ai/docs/dataset_management/formats/format-coco/ "COCO | CVAT"
[3]: https://docs.cvat.ai/docs/qa-analytics/quality-control/ "Quality control | CVAT"
[4]: https://www.cvat.ai/resources/changelog/segment-anything-model-in-cvat "Meta's Segment Anything Model is now available in CVAT Community | CVAT Changelog"
[5]: https://docs.cvat.ai/docs/dataset_management/import-datasets/ "Import annotations and data to CVAT | CVAT"
[6]: https://arxiv.org/abs/2411.06971 "[2411.06971] MapSAM: Adapting Segment Anything Model for Automated Feature Detection in Historical Maps"
[7]: https://docs.cvat.ai/docs/api_sdk/sdk/ "CVAT Python SDK | CVAT"
