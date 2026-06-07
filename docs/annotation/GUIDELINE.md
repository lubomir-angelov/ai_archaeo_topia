## Recommended training roadmap
### Phase 1 — annotation

Use CVAT like this:

```bash
mound:
  polygon/mask preferred

hard_negative_symbol:
  box or polygon; useful for analysis and false-positive mining

uncertain_ignore:
  annotate but exclude from training
```

Keep the attributes:

```bash
has_relative_height_mark
has_absolute_elevation_mark
crossed_by_grid
crossed_by_contour
blurred_or_bad_print
```

These are useful for stratified evaluation later.

### Phase 2 — export master dataset

Export the whole project:

Project export → COCO 1.0 → with images

This becomes the canonical dataset.

### Phase 3 — create training subsets

Split by map sheet/task, not random image/object:

```bash
train:
  task_K-35-50-B-b
  task_K-35-51-A-b

val:
  task_K-35-8-G-a

test:
  completely held-out map sheet
```

This avoids leakage.

### Phase 4 — train/evaluate methods

Run in this order:

```bash
1. Zero-shot SAM
2. Prompted SAM using boxes/points from annotations
3. YOLO detection baseline
4. YOLO segmentation baseline, if masks are good
5. MapSAM-like adaptation
```

What to train first in practice

If you want the fastest engineering feedback:

Train YOLO detection first.

Reason: it is quick, exposes dataset issues, and gives you a baseline.

But if you ask what is better aligned with your real goal:

Prepare the dataset for MapSAM from day one.

That means masks/polygons + COCO export + sheet-level splits.

### Recommendation

Do not switch your dataset design to YOLO-first.

Use this as your policy:

```bash
Annotation target:
  segmentation-quality mound polygons/masks

Master export:
  COCO 1.0 with images

Main model:
  MapSAM-like SAM adaptation

Baseline:
  YOLO detection/segmentation
```

## Dataset goals  

For MapSAM-style training, count annotated mound instances, not just images/tasks.


```bash
Minimum viable experiment:
  100–150 mounds
  100–200 hard_negative_symbol
  uncertain_ignore as needed, but keep it below ~10–15% of reviewed symbols

Good v0.1 dataset:
  300–500 mounds
  300–600 hard_negative_symbol
  50–100 uncertain_ignore

Stronger paper-grade dataset:
  800–1,500+ mounds
  800–2,000 hard_negative_symbol
  uncertain_ignore still controlled, not allowed to dominate
```

MapSAM is relevant here because it targets historical-map segmentation with limited training data and reports adaptation even in very low-shot settings, including 10-shot experiments. But for your own archaeology-symbol task, I would not stop at 10 or 50 examples except for a prototype. You need enough held-out map-sheet data to show that the method generalizes.

### Split ratio

Split by map sheet/task, not by individual symbol.

Use:

```bash
Train:      ~70% of map sheets
Validation: ~15% of map sheets
Test:       ~15% of map sheets
```

Or, if you only have a few sheets:

```bash
Train: several sheets
Validation: one sheet
Test: one completely held-out sheet
```

Do not put symbols from the same original map sheet into both train and test.

### Label ratio

For the full annotation set, aim roughly for:

```bash
mound:                 50–60%
hard_negative_symbol:  35–45%
uncertain_ignore:       5–10%
```

A practical target for v0.1:

```bash
mound:                 400
hard_negative_symbol:  300
uncertain_ignore:       50
Total reviewed symbols: ~750
```

This is enough to start meaningful MapSAM-style adaptation, build few-shot subsets, and measure false positives.

What each label should mean

### mound


Use this for clear mound symbols, including mounds with nearby complications:

```bash
mound + relative height mark
mound + absolute elevation mark
mound crossed by grid
mound near contour
mound affected by colored pencil
mound partly touching text
```

The mask/polygon should cover only the mound symbol.

### hard_negative_symbol

Use this aggressively. These are not training positives, but they are very important for evaluating false positives.

Examples:

```bash
relative height mark without mound
absolute elevation mark without mound
trig point
text fragment
contour crossing
grid intersection
road/powerline crossing
colored pencil artifact
decorative cartographic symbol
print noise that resembles a mound
```

For MapSAM-style semantic/instance segmentation, these usually become background, but keeping them annotated lets you measure whether the model falsely segments them.

### uncertain_ignore

Use this only when you genuinely cannot decide.

Examples:

```bash
symbol is too degraded
mound/text overlap is inseparable
print is too blurry
partial symbol at image edge
possible mound but not enough context
```

Do not overuse it. Too many uncertain_ignore labels means the dataset becomes hard to train and hard to evaluate.

Few-shot subsets to create later

Even if you annotate 300–500 mounds, you can still run few-shot experiments by sampling from the training sheets only:

```bash
5-shot
10-shot
25-shot
50-shot
100-shot
full-train
```

Then always evaluate on the same held-out validation/test sheets.

This gives you a strong experimental story:

How much annotation is needed for archaeological mound segmentation?
What I would annotate first

Start with this first batch:

```bash
100 clear mounds
100 hard negatives
20–30 uncertain ignores maximum
```

Then export COCO and run a sanity check. CVAT’s COCO export supports polygons, masks, boxes, and attributes, so it is appropriate as your master format for MapSAM-style experiments.

After that, expand toward:

```bash
300–500 mounds
300–600 hard negatives
```

That is the point where your results will start to be credible beyond a toy demo.