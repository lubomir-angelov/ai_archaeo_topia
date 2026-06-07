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

## Direction:  

MapSAM-like directly for the research path, but train YOLO early as a sanity-check and baseline.