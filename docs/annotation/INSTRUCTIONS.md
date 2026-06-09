## CVAT Annotation Instructions

### General rule

Annotate only the visual symbol that belongs to the selected label. Do not include nearby numbers, text, contour lines, grid lines, or other map features unless they are physically inseparable from the symbol.

For mound symbols with elevation values, draw the annotation around the mound symbol only. Store relative or absolute elevation information as attributes, not as part of the geometry.

---

## 1. `mound`

Preferred geometry: **polygon or mask**

Use this label for clear archaeological mound symbols. This includes clean mounds and mounds affected by nearby map features, as long as the mound symbol itself is recognizable.

### Option A: Manual polygon

1. Open the task/job in CVAT.
2. Select the `mound` label from the label list.
3. Choose the **Polygon** drawing tool.
4. Click around the visible outline of the mound symbol.
5. Keep the polygon tight around the mound symbol only.
6. Do not include nearby elevation numbers such as `+1,0` or `225,56`.
7. Finish the polygon by clicking **Done** or pressing `N`.
8. Set the relevant attributes, for example:

   * `has_relative_height_mark`
   * `has_absolute_elevation_mark`
   * `crossed_by_grid`
   * `crossed_by_contour`
   * `blurred_or_bad_print`

### Option B: SAM-assisted mask/polygon

1. Open the task/job in CVAT.
2. Open the **Magic Wand / AI tools** panel.
3. Select the SAM interactor.
4. Select the `mound` label.
5. Enable mask-to-polygon conversion if polygon export is preferred.
6. Add positive clicks inside the mound symbol.
7. Add negative clicks outside the mound symbol if SAM captures nearby text, grid lines, or elevation numbers.
8. Accept the result only when the mask/polygon covers the mound symbol correctly.
9. Manually correct the result if SAM includes nearby numbers or unrelated map features.
10. Set the relevant attributes.

### Annotation rule for `mound`

The geometry should represent the mound symbol itself. Elevation numbers and contextual map features are metadata, not part of the mound geometry.

---

## 2. `hard_negative_symbol`

Preferred geometry: **box or polygon**

Use this label for symbols or artifacts that look similar to mounds but are not mounds. These annotations are useful for false-positive mining and model evaluation.

Examples:

* elevation mark without a mound
* relative height mark without a mound
* trig point
* grid intersection
* contour crossing
* road or powerline crossing
* text fragment
* decorative map symbol
* colored pencil artifact
* print artifact that resembles a mound

### Bounding box workflow

1. Select the `hard_negative_symbol` label.
2. Choose the **Rectangle / Bounding box** tool.
3. Draw a tight box around the confusing symbol or artifact.
4. Do not include unrelated surrounding features unless they are part of the confusing object.
5. Set the `negative_type` attribute.
6. Set any relevant context attributes, for example:

   * `crossed_by_grid`
   * `crossed_by_contour`
   * `affected_by_colored_pencil`
   * `blurred_or_bad_print`

### Polygon workflow

Use a polygon instead of a box when the hard negative has an irregular shape or when a tight outline is useful.

1. Select the `hard_negative_symbol` label.
2. Choose the **Polygon** tool.
3. Click around the visible boundary of the confusing symbol.
4. Finish with **Done** or `N`.
5. Set `negative_type`.
6. Set any relevant context attributes.

### Annotation rule for `hard_negative_symbol`

These objects should not be treated as mound positives. They are explicitly annotated to help identify and analyze likely false positives.

---

## 3. `uncertain_ignore`

Preferred geometry: **box or polygon**

Use this label only when the object cannot be reliably classified as either a mound or a hard negative.

Examples:

* symbol is too blurred
* symbol is partially cut off at the image edge
* symbol overlaps text or elevation marks too strongly
* print quality is too poor
* possible mound, but not enough context
* ambiguous merged symbols

### Workflow

1. Select the `uncertain_ignore` label.
2. Choose either **Rectangle / Bounding box** or **Polygon**.
3. Annotate the ambiguous region tightly.
4. Set the uncertainty reason if available.
5. Do not use this label for objects that are clearly mounds or clearly hard negatives.

### Annotation rule for `uncertain_ignore`

These annotations are for review and exclusion from training. They should not be used as normal training targets. During dataset preparation, `uncertain_ignore` objects should be filtered out or used as ignored regions, depending on the training pipeline.


## Export CVAT Project as COCO

After finishing the annotation tasks, export the full CVAT project as a COCO dataset.

### Recommended export scope

Export from the **project**, not from individual tasks.

This keeps all task annotations together and preserves a consistent label mapping.

### UI export steps

1. Open CVAT.

2. Go to **Projects**.

3. Open the project:

   ```text
   ai_archeo_topia
   ```

4. Open the project actions menu.

5. Click **Export dataset**.

6. Select format:

   ```text
   COCO 1.0
   ```

7. Enable:

   ```text
   Save images
   ```

8. Start the export.

9. Download the generated `.zip` file.

10. Store it under the dataset artifacts folder, for example:

```text
artifacts/datasets/cvat_exports/archeo_mound_detection_v0_1_coco.zip
```

### Optional CLI export

Find the project ID:

```bash
cvat-cli project ls
```

Export the project as COCO with images:

```bash
PROJECT_ID="<your_project_id>"

cvat-cli project export-dataset \
  --format "COCO 1.0" \
  --with-images yes \
  "${PROJECT_ID}" \
  "archeo_mound_detection_v0_1_coco.zip"
```

### Notes

Use the COCO export as the canonical dataset export for MapSAM-style training.

The `uncertain_ignore` label should be filtered out during dataset preparation unless the training pipeline explicitly supports ignore regions.
