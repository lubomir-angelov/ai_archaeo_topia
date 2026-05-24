# Annotation Guide for Creating the Archaeological Map Symbol Dataset
# BG Version below

## 1. Purpose of the dataset

The goal is to create a consistent, reviewable, and machine-readable annotation dataset for training and evaluating ML models that detect **archaeological mound symbols** on historical topographic maps.

The dataset should include both clear and difficult cases:

* isolated mound symbols;
* mound necropoleis;
* mounds with elevation marks;
* mounds with triangulation points;
* mounds crossed by map grid lines, roads, contours, powerlines, or boundaries;
* overlapping or densely clustered mounds;
* blurred or poorly printed symbols;
* manually colored pencil marks;
* confusing non-mound symbols such as watermills, windmills, reservoirs, forest-belt symbols, elevation dots, buildings, roads, text, decorative marks, and grid artifacts.

The task is not only “detect mound symbols”, but also **detect them under cartographic noise, overprinting, visual ambiguity, and regional symbol variation**.

---

## 2. Main annotation unit

We work with two levels of annotation:

## 2.1. Sample / image crop

A **sample** is one cropped image from a historical map. It may contain:

* one mound;
* several mounds;
* a mound necropolis;
* hard negative symbols;
* uncertain symbols;
* surrounding cartographic context.

Each sample receives a unique `sample_id`.

Example:

```text
AG_013_K-35-50-B-b_K-35-50-029_045.png
```

## 2.2. Object / annotated instance

An **object** is one individual symbol marked with a bounding box or mask.

Examples:

* one mound;
* one watermill;
* one windmill;
* one reservoir-like symbol;
* one confusing circular symbol;
* one uncertain symbol.

For mound necropoleis, each mound should be annotated as a separate object whenever possible.

---

## 3. File naming convention

Each image crop should have a stable and readable file name.

Recommended format:

```text
<ANNOTATOR>_<NUMBER>_<SHEET_25K>_<SHEET_5K>.png
```

Examples:

```text
NK_001_K-34-35-B-g_K-34-35-080.png
AG_013_K-35-50-B-b_K-35-50-029_045.png
ICH_021_K-35-21-B-a_K-35-21-012_028.png
JTZ_006_K-34-58-B-b_K-34-58-013.png
```

Rules:

* Do not use spaces in file names.
* Do not overwrite original crops.
* If a crop is edited, save it as a separate derived file.
* If screenshots are embedded in a document, extract them first and assign each image its own `sample_id`.
* Keep the connection between the image and the original description.

---

## 4. Recommended dataset folder structure

```text
dataset/
  raw/
    images/
      NK/
      AG/
      ICH/
      JTZ/
  annotations/
    master_samples.csv
    objects_coco.json
    objects_yolo/
  docs/
    annotation_guidelines.md
    label_schema.md
  review/
    samples_for_review/
    reviewed_samples.csv
  splits/
    train.txt
    val.txt
    test.txt
```

`raw/images/` contains the original image crops.

`annotations/` contains the annotation exports.

`master_samples.csv` contains sample-level metadata.

`review/` contains samples and decisions that need expert review.

`splits/` contains the final train/validation/test split.

---

## 5. Master sample table

Each image crop should have one row in `master_samples.csv`.

Recommended columns:

```text
sample_id
annotator
image_path
sheet_25k
sheet_5k
province
position_on_25k_sheet
original_description
target_present
target_count_claimed
contains_necropolis
contains_single_mound
contains_hard_negative
relief_original
relief_normalized
difficulty
uncertainty
annotation_status
review_status
notes
```

Example row:

```text
AG_013,
AG,
raw/images/AG/AG_013_K-35-50-B-b_K-35-50-029_045.png,
K-35-50-B-b,
K-35-50-(29); K-35-50-(45),
Plovdiv,
western part,
"Mound necropolis of 13 mounds. One has a triangulation point. There are small circles that may confuse the model.",
true,
13,
true,
false,
true,
"slope of a mountain",
slope,
hard,
false,
annotated,
pending_review,
"Contains confusing circular symbols."
```

Important: keep `original_description` unchanged. Add normalized metadata in separate fields.

---

## 6. Relief normalization

The original relief description should be preserved exactly in `relief_original`.

Examples:

```text
plain
semi-mountainous
mountainous
hilly
slope
ridge
urban area
river valley
```

For ML/evaluation purposes, use a controlled set in `relief_normalized`:

```text
plain
hilly
mountain
slope
ridge
valley
urban
mixed
unknown
```

Example mapping:

| Original description     | Normalized value |
| ------------------------ | ---------------- |
| plain                    | `plain`          |
| hilly                    | `hilly`          |
| mountainous              | `mountain`       |
| mountain slope           | `slope`          |
| ridge of a hill          | `ridge`          |
| river valley             | `valley`         |
| urban environment        | `urban`          |
| plain / semi-mountainous | `mixed`          |

---

## 7. Label schema v0.1

For the first version, keep the label set small.

Recommended labels:

```text
mound
hard_negative_symbol
uncertain_ignore
```

## 7.1. `mound`

Use this label for a confirmed or highly probable mound symbol.

Includes:

* isolated mound;
* mound in a necropolis;
* mound with elevation mark;
* mound with triangulation point;
* mound with name label;
* mound crossed by grid, road, contour, powerline, boundary, or other map feature;
* partially blurred but still recognizable mound symbol.

## 7.2. `hard_negative_symbol`

Use this label for a symbol that is **not a mound** but may confuse the model.

Examples:

* watermill;
* windmill;
* reservoir or tank;
* forest-belt circle;
* elevation dot;
* building symbol;
* pit or negative terrain form;
* vineyard/orchard symbol;
* unknown circular cartographic symbol;
* decorative or repeated map symbol that resembles a mound.

## 7.3. `uncertain_ignore`

Use this label when the annotator cannot reliably decide whether the symbol is a mound or not.

These objects should not be used as positive or negative training examples until reviewed.

Use this class for:

* heavily damaged symbols;
* ambiguous mound/watermill cases;
* unclear circular signs;
* symbols that require checking another map scale;
* cases where the object cannot be separated from nearby symbols.

---

## 8. Object-level attributes

In addition to the class label, each annotated object should have attributes when possible.

Recommended object attributes:

```text
object_id
sample_id
class_name
bbox_xmin
bbox_ymin
bbox_xmax
bbox_ymax
has_trig_point
has_elevation_mark
has_name_label
crossed_by_grid
crossed_by_contour
crossed_by_road
crossed_by_powerline
crossed_by_boundary
affected_by_colored_pencil
blurred_or_bad_print
overlaps_other_mound
negative_type
is_uncertain
comment
```

For `negative_type`, use:

```text
watermill
windmill
reservoir_or_tank
forest_belt_symbol
elevation_dot
building
pit_or_negative_form
vineyard_or_orchard_symbol
unknown_round_symbol
text
road
grid_artifact
decorative_mark
other
```

---

## 9. Bounding box rules

## 9.1. Mound symbols

The bounding box should cover the visible mound symbol itself.

If the mound includes a triangulation point, elevation dot, or internal mark, include the combined symbol in the box.

Do not include large surrounding text unless it is visually inseparable from the symbol.

Correct:

```text
Box tightly around the mound symbol.
```

Incorrect:

```text
Box around the entire mound necropolis.
Box around the whole map area.
Box around nearby labels, roads, and unrelated context.
```

## 9.2. Mound necropoleis

Do not annotate a necropolis as one large object.

Annotate each visible mound separately.

If individual mounds overlap but are still separable, create one box per mound.

If they cannot be separated reliably, annotate the ambiguous area as `uncertain_ignore` and add a comment.

## 9.3. Grid, road, contour, powerline, or boundary crossing

If a mound is crossed by another map element, still annotate the mound.

Then set the relevant attribute:

```text
crossed_by_grid = true
crossed_by_road = true
crossed_by_contour = true
crossed_by_powerline = true
crossed_by_boundary = true
```

Do not skip mounds just because they are partially crossed.

## 9.4. Colored pencil or manual markings

If the mound is affected by colored pencil but is still recognizable, annotate it.

Set:

```text
affected_by_colored_pencil = true
```

If the colored marking makes the symbol too ambiguous, use `uncertain_ignore`.

## 9.5. Uncertain cases

If the annotator is not confident, do not force a positive or negative label.

Use:

```text
class_name = uncertain_ignore
is_uncertain = true
comment = "Reason for uncertainty..."
```

Examples:

```text
"Symbol resembles a watermill, but may be a mound according to another map scale."
"Two overlapping symbols cannot be separated reliably."
"Possible mound, but the print is too blurred."
```

---

## 10. What not to annotate

Do not annotate every map feature.

Do not annotate all:

* text;
* roads;
* contour lines;
* grid lines;
* buildings;
* administrative boundaries;
* vineyards;
* orchards;
* forest symbols;
* rivers;
* general decorative marks.

Only annotate these features if they are important hard negatives or directly interfere with a mound symbol.

Examples:

* A road crossing a mound: annotate the mound, not the road.
* A watermill that looks like a mound: annotate the watermill as `hard_negative_symbol`.
* A row of forest-belt circles that resembles mound symbols: annotate a few representative confusing symbols, not the entire row.
* A random text label: do not annotate unless it visually causes confusion with the target symbol.

---

## 11. Difficulty categories

Each sample should receive a difficulty label.

Use:

```text
easy
medium
hard
very_hard
```

## 11.1. `easy`

Clear mound symbol, little surrounding clutter.

## 11.2. `medium`

Some surrounding noise, but the mound is clearly visible.

## 11.3. `hard`

The mound is affected by one or more of:

* grid line;
* contour line;
* road;
* powerline;
* colored pencil;
* nearby text;
* dense map symbols;
* urban environment;
* overlapping symbols;
* blurred print.

## 11.4. `very_hard`

The symbol is highly ambiguous, heavily damaged, or very similar to a non-mound symbol.

---

## 12. Annotation status

Each sample should have an annotation status.

Use:

```text
not_started
in_progress
annotated
needs_review
reviewed
rejected
```

Meaning:

| Status         | Meaning                                    |
| -------------- | ------------------------------------------ |
| `not_started`  | Image exists but has not been annotated    |
| `in_progress`  | Someone is actively annotating it          |
| `annotated`    | First annotation pass is complete          |
| `needs_review` | Requires expert or second annotator review |
| `reviewed`     | Checked and accepted                       |
| `rejected`     | Image is unusable or unsuitable            |

---

## 13. Workflow for each annotator

## Step 1 — Select a sample

Choose the next unassigned image from `master_samples.csv`.

Set:

```text
annotation_status = in_progress
```

## Step 2 — Check sample metadata

Before annotating, check:

* `sample_id`;
* 1:25,000 map sheet;
* 1:5,000 map sheet;
* province;
* original description;
* expected number of mounds, if provided;
* notes about hard negatives or uncertainty.

## Step 3 — Open the image in the annotation tool

Recommended tool: CVAT.

Use one project with a shared label schema.

## Step 4 — Annotate all visible mounds

Each mound receives its own bounding box or mask.

Use `mound` for confirmed mounds.

Do not group multiple mounds into one box.

## Step 5 — Annotate important hard negatives

Annotate non-mound symbols only when they are likely to confuse the model.

Use `hard_negative_symbol`.

Set `negative_type` where possible.

## Step 6 — Add object attributes

For each object, fill attributes such as:

```text
has_trig_point
has_elevation_mark
crossed_by_grid
crossed_by_contour
crossed_by_road
affected_by_colored_pencil
blurred_or_bad_print
overlaps_other_mound
```

## Step 7 — Add comments for difficult cases

Use short, clear comments.

Examples:

```text
"Symbol is crossed by a vertical grid line."
"Possible mound, but resembles a watermill."
"Mound is partially covered by colored pencil."
"Two mounds overlap; separation is uncertain."
"Checked against 1:50,000 map according to source note."
```

## Step 8 — Set final status

After finishing the sample:

* use `annotated` if confident;
* use `needs_review` if there are uncertain cases;
* use `rejected` if the image is unusable.

---

## 14. Review rules

At least 20–30% of samples should be reviewed by a second person.

Mandatory review cases:

* all `uncertain_ignore` objects;
* mound symbols that resemble watermills;
* unclear mound counts;
* heavily blurred signs;
* colored pencil directly over the symbol;
* necropoleis with more than 10 mounds;
* cases where interpretation depends on another map scale;
* cases where the annotator is unsure.

Reviewer actions:

```text
accept
correct
mark_as_uncertain
reject
request_discussion
```

Reviewer comments should be kept short and specific.

---

## 15. Dataset split rules

Do not split the dataset randomly by image tile.

Split by map sheet or region to avoid leakage.

Recommended approach:

```text
train: selected map sheets
validation: different map sheets
test: completely held-out map sheets or provinces
```

The test set should answer:

```text
Can the model generalize to unseen map sheets and regional cartographic styles?
```

Keep a file for each split:

```text
splits/train.txt
splits/val.txt
splits/test.txt
```

Each file should contain `sample_id` or image path entries.

---

## 16. Minimum target for dataset v0.1

For the first dataset version, prioritize consistency over size.

Recommended target:

```text
100–150 image crops
500–1000 object annotations if possible
at least 100 hard negative symbols
at least 30 difficult or uncertain cases
```

The dataset should include variation across:

* province;
* map sheet;
* terrain type;
* isolated mounds;
* mound necropoleis;
* urban and rural contexts;
* flat, hilly, and mountainous terrain;
* clean and degraded print;
* hard negative symbol types.

---

## 17. Checklist before accepting a sample

Before a sample is marked as complete, check:

```text
[ ] Image file has a valid name
[ ] sample_id matches master_samples.csv
[ ] 1:25,000 sheet is recorded
[ ] 1:5,000 sheet is recorded
[ ] Province is recorded
[ ] Original description is preserved
[ ] All visible mounds are annotated
[ ] Important hard negatives are annotated
[ ] Bounding boxes are not too large
[ ] Necropoleis are not annotated as one large box
[ ] Difficult cases have comments
[ ] Uncertain cases are marked as uncertain_ignore
[ ] Attributes are filled where relevant
[ ] Annotation status is updated
[ ] Review status is updated if needed
```

---

## 18. Important annotation principle

The goal is not to create visually perfect annotations.

The goal is to create annotations that are:

```text
consistent
reviewable
machine-readable
useful for ML training and evaluation
archaeologically meaningful
```

When uncertain:

1. Do not guess.
2. Use `uncertain_ignore`.
3. Add a short comment.
4. Send the sample for review.

When a mound is clear, even if crossed or partially degraded:

1. Annotate it as `mound`.
2. Add the relevant attributes.
3. Mark the sample as `hard` if needed.

---

## 19. Recommended first iteration

The first working iteration should be:

```text
1. Extract all embedded screenshots as separate image files.
2. Assign a unique sample_id to every image.
3. Fill master_samples.csv with the original descriptions.
4. Annotate 20–30 diverse samples as a pilot.
5. Compare annotations between team members.
6. Refine the label schema and difficult-case rules.
7. Lock label schema v0.1.
8. Annotate the remaining samples.
9. Review difficult and uncertain cases.
10. Export the dataset to COCO and/or YOLO format.
11. Train the first baseline model.
```

---

## 20. Recommended v0.1 labels

Use this minimal label schema first:

```text
mound
hard_negative_symbol
uncertain_ignore
```

Only after the first version is stable, consider expanding to:

```text
mound_plain
mound_with_trig_point
mound_with_elevation_mark
watermill
windmill
reservoir_or_tank
forest_belt_symbol
other_confusing_symbol
```

Do not expand too early. A smaller, consistent label set is better than a large, noisy one.

---