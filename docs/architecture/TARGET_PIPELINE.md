# Target pipeline: from a scanned map sheet to a reviewable archaeological observation

This is the high-level target state for the detection and segmentation side of
the project. It is a design document, not a status report — most of what it
describes does not exist yet. Each stage is marked with what is built today.

The organizing idea is a change of role for MapSAM:

> **MapSAM is not the model that finds mounds.** In the complete system it is a
> geometry-refinement service downstream of candidate detection. The detector
> does semantic localization; MapSAM does spatial delineation.

That separation is what `docs/mapsam/v004/RESULTS.md` made concrete. The
prompt-jitter control arm showed that the input window can be misplaced by up
to ~250 source px at no measurable cost, while the *prompt* must land on the
symbol. So the detector does not need to produce a well-centred crop, or a good
mask, or a tight box. It needs to point at the right symbol.

That is a writable specification, and having it is the main reason the MapSAM
work was worth doing before the detector exists.

**Read the prompt tolerance carefully.** v0.4 summarized it as "about 15 px",
but that is where the segmenter *fails*, not where it works: the IoU≥0.5 rate
runs 1.00 at 0 px, 1.00 at 5 px, 0.69 at 10 px and **0.02 at 15 px**. The
working band is ≤5 px, 5–10 px is marginal, and 15 px is effectively a miss.
The detector target is therefore **p90 centre error ≤ 5 source px**.
`docs/mapsam/v005/PLAN.md` states the correction in full.

---

## The pipeline

```text
Georeferenced historical map sheet
          │
          ▼
   Tile / pyramid reader                        [not built]
          │
          ▼
   Candidate generator                          [not built]
    high-recall detector
          │
          ▼
 Mound / hard-negative classifier                [not built]
          │
          ▼
 Candidate point (p90 localization error ≤ 5 px)
          │
          ▼
 512 px prompt-centred crop                      [built, v0.4]
          │
          ▼
        MapSAM decoder                           [built, v0.4]
          │
          ▼
  Symbol segmentation mask
          │
          ├────────► duplicate suppression       [not built]
          │
          ├────────► attribute extraction        [labels exist, model not built]
          │
          ├────────► quality estimation          [signal identified, unvalidated]
          │
          ▼
 mask centroid + source geometry
          │
          ▼
  pixel → map coordinates                        [partially built, 30.5% pass rate]
          │
          ▼
       GIS object
          │
          ▼
GeoPackage / GeoJSON / PostGIS                   [not built]
          │
          ▼
 archaeological review                           [annotation-side tooling exists]
```

---

## Stage 0 — Tile / pyramid reader

**Status: not built, and the gap is larger than it looks.**

The current dataset is 12 prepared image clips of roughly 2400×2200 px, four
per sheet, produced upstream of `prepare_mapsam_coco` — which does no tiling of
its own. A detector has to run over a whole sheet, which needs a sliding-window
reader and a stitching/deduplication layer across window seams.

This matters for evaluation, not just for engineering. Any "false positives per
unit map area" figure measured on these clips extrapolates to a full sheet only
if the clips are representative of sheet content, including the large empty or
uninformative regions a real sheet contains. They were prepared for annotation,
so that is unverified and should be checked before an FP-rate is quoted at
sheet scale.

## Stage 1 — Candidate generation: "where might a mound be?"

**Status: not built. This is the v0.5 problem.**

Run over the sheet and emit, per proposal:

```text
candidate_id
sheet_id
source_image
center_x, center_y        # source-tile pixels
bbox_xyxy
mound_probability
```

The metric that matters is **not** conventional detector mAP. It is recall at a
localization radius, because that is what composes with MapSAM's tolerance
curve. A detector with good mAP but occasional 20–30 px centre errors will
underperform one with mediocre box IoU that lands within 10 px almost always.

Report recall at R = 5, 10, 15 and 25 source px, and the p90 of centre error
rather than the median — the tail is what falls outside the tolerance band.
**Recall@5px is the primary figure**; the rest of the curve is reported for
composition with v0.4's tolerance curve, not for acceptance decisions. At
R = 25 px, matching must be strictly one-to-one: 17 of 180 mounds have a
neighbour within 25 px, so greedy matching inflates the figure.
`docs/mapsam/v005/PLAN.md` carries the full metric hierarchy.

**Supervision available today**, verified against
`annotation/cvat/v0.0.1/instances_default.json` (713 annotations):

| Category | n | Role |
|---|---:|---|
| `mound` | 180 | positives |
| `hard_negative_symbol` | 530 | confusable negatives |
| `uncertain_ignore` | 3 | excluded from training and evaluation |

Note the ratio: nearly 3:1 negatives to positives, already hand-selected for
confusability. That is an unusually good starting position for a
recall-oriented detector, and it is the asset v0.1–v0.4 deliberately never
touched.

## Stage 2 — Candidate filtering: "is this actually a mound symbol?"

**Status: not built.**

Keep this conceptually distinct from stage 1 even if the two are eventually one
model. The design priority differs: stage 1 must not miss mounds, stage 2 must
not pass junk.

```text
500 candidates  →  [filter]  →  80 plausible mounds
```

False positives are cheap early, because the filter and the reviewer can reject
them. False negatives are not recoverable downstream.

### The negatives are typed, and the filter should use that

`negative_type` is populated on all 530 hard negatives:

| `negative_type` | n |
|---|---:|
| `decorative_symbol` | 246 |
| `trig_point` | 136 |
| `road` | 97 |
| `text` | 29 |
| `colored_pencil` | 7 |
| `grid` | 6 |
| `other` | 5 |
| `__undefined__` | 3 |
| `contour` | 1 |

So the filter can be multi-class rather than binary, and — more importantly —
errors can be reported *by negative type*. "Precision 0.86" says much less than
"we reject decorative symbols reliably and confuse trig points 40% of the
time."

### The trig-point ambiguity is the sharpest design problem here

`trig_point` is the second-largest negative class at 136 instances. But
`has_trig_point` is also `true` on **42 of the 180 mounds (23%)**.

The same printed element is a negative when it stands alone and an *attribute
of a positive* when it sits on a mound. A filter trained on appearance alone
will be pulled in both directions by it, and no amount of extra data fixes a
label geometry that is genuinely context-dependent. Whatever is built should be
evaluated on this subset specifically, because aggregate precision will hide
it.

## Stage 3 — MapSAM: "exactly which pixels belong to this symbol?"

**Status: built (v0.4).**

For each surviving candidate:

```text
candidate centre
      ↓  crop 512×512 source px, prompt-centred, clamped inside the tile
      ↓  map centre and box into window coordinates
      ↓  MapSAM (SAM ViT-B, frozen encoders, fine-tuned mask decoder)
      ↓
binary mound-symbol mask
```

Measured performance given a correct prompt, leave-one-sheet-out across the
three available sheets: 99–100% of instances clear IoU 0.5, macro mean 0.7922
decoder IoU, macro disagreement area 103.8 source px².

The division of labour:

```text
Detector:  [ roughly here ]

MapSAM:        ███
             ██████
              █████
                ██
```

**Scope caveat that must travel with those numbers:** all three sheets are one
Soviet 1:50k series. This is *within-series conditional* segmentation.
Cross-cartographic behaviour is untested.

## Stage 4 — Mask back into source-map coordinates

**Status: the transform exists (`window_xyxy` is carried through the dataset
and prediction stats); the instance record does not.**

The window's position in the sheet is known, so the mask maps back directly.
Per instance, derive `source_mask`, `source_bbox`, `pixel_centroid`,
`mask_area_px`, `width_px`, `height_px`.

**Keep both the detector's point and the mask centroid.** Their difference,

```text
detector_to_mask_centroid_distance_px
```

is a production monitor for detector localization quality that needs no ground
truth. Given v0.4's tolerance curve, it is directly interpretable: values
drifting past ~5 px predict segmentation degradation before anyone notices it
in the output, and ~10 px is already a third of instances lost.

## Stage 5 — Pixel coordinates to GIS geometry

**Status: partially built, and it is a second blocking dependency, not a
formality.**

`src/georeference/georeference.py` produces GeoTIFFs via GCP fitting and
`gdal.Warp` (default EPSG:25835). But `docs/GEOREF_IMPROVEMENTS.md` records
**61 of 200 matched images passing quality checks — 30.5%**, with frame-detection
aspect-ratio error the dominant failure at 56%. Stage 5 is not "apply the
affine transform"; it depends on a pipeline that currently rejects seven sheets
in ten.

**The mitigation is to keep georeferencing off the critical path.** Every stage
above operates in source-tile pixel space and needs no map projection. The
instance record should therefore be complete and useful *without* a geotransform,
carrying the pixel geometry as primary and the projected geometry as an
optional enrichment with its own quality flag. Detection work is then not
blocked on georeferencing improving, and georeferencing can be fixed
independently without reprocessing detections.

### Point versus polygon: a semantic distinction that belongs in the schema

> **The MapSAM polygon is the printed cartographic mound symbol. It is not the
> physical footprint of the archaeological mound.**

Do not export it as though it were a site boundary. The convention should be:

- **point** — the archaeological location;
- **polygon** — source-symbol geometry, i.e. provenance and evidence.

This is not pedantry. Symbol component areas run 116–1037 px with a mean of 459
(`docs/mapsam/DATASET.md`), on symbols roughly 21 px across — that spread is
cartographic drafting variation and scan condition, not mound size. A consumer
who reads `mask_area_px` as ground extent will be wrong by an unbounded factor.
Name the fields so that misreading takes effort.

## Stage 6 — Attribute extraction

**Status: labels exist for every annotation; no model built.**

This is where segmentation earns more than a prettier mask. Once the exact
symbol mask is known, a controlled ROI follows:

```text
MapSAM mask → dilate by N px → attribute ROI
```

Downstream rules or models can then ask whether a contour intersects the
symbol, whether a grid line crosses it, whether a trig point sits inside it —
against a tight anchor rather than a crude detector box full of irrelevant map.

**Stronger than it first appears: these attributes are already annotated**, on
all 713 objects, so this is a supervised task with labels in hand rather than
only a rule-based query. Rates on the 180 mounds:

| Attribute | n | rate |
|---|---:|---:|
| `has_relative_height_mark` | 158 | 88% |
| `crossed_by_contour` | 68 | 38% |
| `crossed_by_road` | 52 | 29% |
| `has_trig_point` | 42 | 23% |
| `crossed_by_forestation_line` | 40 | 22% |
| `has_absolute_elevation_mark` | 37 | 21% |
| `blurred_or_bad_print` | 36 | 20% |
| `crossed_by_grid` | 33 | 18% |
| `crossed_by_water_line` | 14 | 8% |
| `overlaps_other_mound` | 13 | 7% |
| `affected_by_colored_pencil` | 10 | 6% |
| `crossed_by_powerline` | 4 | 2% |

Two corrections to note if working from earlier drafts: there is no
`has_elevation_mark` — the schema distinguishes `has_absolute_elevation_mark`
from `has_relative_height_mark` — and `crossed_by_forestation_line` and
`crossed_by_water_line` exist and are frequently true.

Realistic expectations: with 180 positives, the frequent attributes are
learnable and the rare ones are not. `crossed_by_powerline` at 4 instances and
`affected_by_colored_pencil` at 10 cannot support a per-attribute model yet;
they are rule-or-review territory until more sheets land.

One hypothesis these rates make testable: v0.4 found prompt jitter to be
*directionally* anisotropic, with northward offsets costing ~1.4× less than
westward. With 88% of mounds carrying a relative height mark, a consistently
placed adjacent glyph is the obvious candidate explanation, and the attribute
labels make it checkable rather than speculative.

## Stage 7 — Duplicate suppression

**Status: not built.**

A high-recall detector will emit several proposals per symbol:

```text
A: (1020, 830)   B: (1027, 826)   C: (1015, 838)
```

Segment them and they resolve onto essentially the same mask, so deduplicate on
**mask IoU plus centroid distance** rather than detector-box NMS alone. With
symbols this small, box NMS is noisy in a way mask agreement is not.

**One ordering caveat.** Segmenting every proposal to deduplicate it is not
free: each is a ViT-B encoder pass, measured at roughly 100 ms per sample
end-to-end including data loading on an RTX 5090. At 500 proposals that is ~50 s
per sheet — fine for one sheet, hours across a corpus of 200. Cheap box-level
deduplication first, mask-level arbitration only for survivors and ambiguous
clusters, is the same result at a fraction of the cost.

## Stage 8 — Segmentation-quality gate

**Status: signal identified, never validated.**

Do **not** use `max(sigmoid(mask_logits))`. v0.3 established that it is float32
saturation, not confidence: sigmoid pins to exactly 1.0 past a logit of ~16,
and the windowed runs reach logits of 88.

Two candidate signals, in order of how cheaply they can be checked:

**1. SAM's own mask-quality head.** `sam.mask_decoder` returns
`iou_predictions` as its second value, currently discarded at
`src/archeo_topia/training/train_mapsam_v0.py:372` and `:431`. Validate before
relying on it: that head has never been in the loss while the mask decoder
around it has been fine-tuned, so it is predicting the quality of *stock* SAM's
notion of the mask for a decoder that has moved away from it. The validation is
cheap — 171 samples where true IoU is known.

**2. Mask stability under prompt perturbation.** v0.4's jitter arms measured
exactly the asymmetry that makes this promising: the model is near-invariant to
window displacement and highly sensitive to prompt displacement. So resegmenting
with a few small prompt offsets and measuring mask agreement is a proxy for
*prompt correctness* — which is the dominant failure mode this gate needs to
catch. It costs one encoder pass per perturbation, so it is the more expensive
option and should be tried only if `iou_predictions` fails validation.

The eventual review policy:

```text
high detector + high segmentation quality   → auto-accept
high detector + low segmentation quality    → human review
low detector                                → reject or review, per recall policy
```

## Stage 9 — Human review and the active-learning loop

**Status: annotation-side tooling exists; the production loop does not.**

Instead of asking an archaeologist to draw every mound, present the candidate
location, the predicted mask, the source crop and the predicted attributes. The
reviewer accepts, rejects, corrects, or classifies ambiguous cases.

```text
new maps → detector → MapSAM → automatic GIS layer
   ↑                                   │
   └──── corrected annotations ←── archaeologist reviews uncertain cases
```

**Relationship to the existing MCP stack.** `docs/automation/MCP_ARCHITECTURE.md`
and `docs/automation/PDF_SAM2_ANNOTATION_PIPELINE.md` describe an
agent-orchestrated annotation-assist path (a geospatial agent routing to OCR and
SAM 2 MCP services). That is a different concern from this pipeline and the two
should not be conflated: the MCP stack uses *stock* SAM 2 to help produce
annotations, while stage 3 here is a *fine-tuned* SAM ViT-B decoder doing
production inference. They meet at this stage — the review surface and the
corrected annotations that flow back — and nowhere else.

---

## The canonical output record

One mound instance, close to:

```json
{
  "mound_id": "...",
  "sheet_id": "K-35-51-B-a",
  "source_image": "images/train/K-35-51-B-a_1.png",

  "source_pixel": { "x": 1234.5, "y": 876.2 },
  "detector": {
    "score": 0.93,
    "localization": [1231, 873],
    "detector_to_mask_centroid_distance_px": 3.4
  },
  "classifier": {
    "mound_probability": 0.91,
    "top_negative_type": "trig_point",
    "top_negative_probability": 0.06
  },
  "segmentation": {
    "symbol_bbox_xyxy": [1218, 860, 1250, 891],
    "symbol_mask_area_px": 472,
    "symbol_polygon": "...",
    "quality_score": 0.88,
    "quality_source": "sam_iou_head | perturbation_stability"
  },
  "attributes": {
    "has_trig_point": false,
    "has_relative_height_mark": true,
    "has_absolute_elevation_mark": false,
    "crossed_by_contour": true,
    "crossed_by_grid": false
  },

  "geo": {
    "available": true,
    "epsg": 25835,
    "geometry": { "type": "Point", "coordinates": [0.0, 0.0] },
    "georeference_quality": "passed",
    "transform_source": "georeference.py GCP fit"
  },

  "review": { "status": "auto_accepted", "reviewer": null, "notes": null }
}
```

Three deliberate choices in that shape:

1. **The segmentation fields are named `symbol_*`.** The mask is evidence and
   geometry for a feature, not the feature itself, and the field names should
   make the printed-symbol semantics hard to misread as ground extent.
2. **`geo` is optional and self-describing.** The record is complete and useful
   in pixel space, so nothing downstream of stage 4 is blocked by the 30.5%
   georeferencing pass rate, and a record can be re-projected later without
   redoing detection.
3. **The classifier's runner-up negative type is retained.** It costs nothing
   and it is the field that will explain a systematic error class when one
   appears.

---

## What this architecture depends on that does not exist yet

Two hard dependencies, one soft:

1. **A detector with p90 centre error inside ~5 source px.** Everything from
   stage 3 onward is validated and waiting; nothing runs without this.
2. **More map sheets, and preferably a second cartographic series.** Three
   sheets of one Soviet 1:50k series cannot support a robustness claim for any
   stage. `data/maps`, `data/georeferenced` and `data/cvat_exports` are
   currently empty.
3. **Georeferencing above 30.5%** — soft, because stage 5 is designed to be
   optional, but the GIS output is one of the project's main practical
   deliverables and it is gated on this.

The 180-annotations-versus-171-samples question this section previously raised
is resolved: one mound annotation carries no geometry and eight pairs of
touching mounds share a connected component. `docs/mapsam/v005/PLAN.md` holds
the accounting. Detection recall is measured against the 180 annotations; the
end-to-end denominator is the 179 that have geometry.

## Why the MapSAM work was worth doing first

The expensive, uncertain, research-heavy stage is done and validated: given a
prompt within ~5 px, the pipeline turns it into a precise symbol mask at
99–100% IoU≥0.5 across every sheet available.

More usefully, v0.4 produced the *specification* for the stage that does not
exist. Without the jitter tolerance curve there would be no way to say what
"good enough detection" means, and the detector would be built against
conventional metrics that do not predict end-to-end behaviour here. Stage 1 now
has a target with a number on it.

---

## References

| | |
|---|---|
| `docs/mapsam/v004/RESULTS.md` | Tolerance curve, 512-window LOSO, failure anatomy |
| `docs/mapsam/v005/PLAN.md` | The detector-first plan and its metric hierarchy |
| `docs/mapsam/DATASET.md` | Annotation counts, component-area statistics |
| `docs/annotation/PROTOCOL_BG.md` | Attribute definitions and annotation rules |
| `docs/GEOREF_IMPROVEMENTS.md` | Georeferencing failure analysis and pass rate |
| `docs/automation/MCP_ARCHITECTURE.md` | The annotation-assist stack, distinct from this pipeline |
