Here is a concise forwardable summary:

---

## ANNPR 2026 In-Progress Paper Plan: ArchaeoTopia, SAM, and MapSAM

We propose preparing an **in-progress study** for ANNPR 2026 based on the current ArchaeoTopia stage, where most historical/topographic maps are now georeferenced. The next step is to establish a reproducible computer vision baseline for extracting archaeologically relevant features from these maps.

### Main proposed title

**ArchaeoTopia Baselines: Few-Shot Segmentation of Archaeological Map Features from Georeferenced Historical Maps using SAM and MapSAM**

### Backup proposed title

**A CRS-Aware Evaluation of SAM Prompting Strategies for Archaeological Feature Extraction from Georeferenced Historical Maps**

## Core goal

The paper should not claim a finished ArchaeoTopia system. Instead, it should present a focused baseline study:

> Evaluate how well SAM and MapSAM-style few-shot adaptation can segment archaeological/cartographic features from georeferenced historical maps and convert the outputs into usable GIS vector layers.

## Main research questions

1. **Zero-shot transfer:**
   How well does baseline SAM perform on georeferenced archaeological/historical maps without training?

2. **Few-shot adaptation:**
   Does MapSAM-style adaptation improve segmentation with a small number of annotated examples?

3. **GIS usefulness:**
   Can predicted masks be converted into useful GeoJSON/vector layers for archaeological GIS workflows?

## Proposed experimental scope

The safest scope is to focus on **one primary feature class**:

* archaeological mound/site symbols or other recurring archaeological map symbols

A secondary feature class can be included only if time allows:

* isolines / contour-like linear features

If time becomes tight, the paper should focus only on archaeological symbol extraction.

## Baselines to compare

The minimum useful comparison should include:

| Method                                    | Purpose                                            |
| ----------------------------------------- | -------------------------------------------------- |
| Classical CV baseline                     | simple thresholding / morphology / color filtering |
| SAM automatic masks                       | zero-shot segmentation baseline                    |
| SAM with point/box prompts                | human-in-the-loop upper-bound                      |
| SAM with simple proposal prompts          | semi-automated practical baseline                  |
| MapSAM / MapSAM-style few-shot adaptation | main adapted model comparison                      |

## Evaluation

The paper should include both computer vision and GIS-oriented metrics:

| Level          | Metrics                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| Pixel-level    | IoU, Dice/F1                                                                  |
| Object-level   | precision, recall, F1                                                         |
| GIS-level      | centroid distance in meters, false positives per map sheet, GeoJSON usability |
| Runtime/effort | inference time, annotation effort, qualitative failure cases                  |

The GIS-aware evaluation is the main differentiator: the output should not just be segmentation masks, but candidate archaeological layers that can be inspected in QGIS or another GIS tool.

## Two-week work plan

### Days 1–2: Freeze scope and prepare data

* Select representative georeferenced map sheets.
* Tile the maps into manageable image patches.
* Create a manifest with sheet ID, tile ID, CRS, bounds, split, and labels.
* Decide whether the paper is symbol-only or symbol + isolines.

### Days 3–4: Annotation sprint

* Annotate the primary archaeological symbols.
* Include hard negative examples such as text, roads, decorative marks, and grid artifacts.
* Split data by map sheet, not by random tile, to avoid leakage.

### Days 5–6: Run SAM baselines

* Run zero-shot SAM.
* Run prompted SAM with points/boxes.
* Run SAM with simple automated proposal generation.
* Store masks, metrics, and qualitative examples.

### Days 7–8: Run MapSAM / few-shot adaptation

* Train or adapt MapSAM-style model on small labeled subsets.
* Compare against SAM baselines.
* Use few-shot settings such as 5-shot, 10-shot, and 25-shot if possible.

### Day 9: GIS vectorization

* Convert masks to polygons or centroids.
* Export GeoJSON.
* Evaluate spatial error in meters.
* Prepare QGIS-style overlay figures.

### Day 10: Results and figures

Prepare:

* pipeline diagram
* dataset examples
* qualitative comparison figure
* GIS overlay figure
* failure case figure
* metrics tables

### Days 11–12: Write paper

Suggested structure:

1. Introduction
2. Related Work
3. Dataset and Georeferencing Context
4. SAM and MapSAM Baselines
5. CRS-Aware Evaluation
6. Experiments
7. Results
8. Discussion and Limitations
9. Conclusion

### Final days: Review and submit

* Compress to LNCS/LNAI 12-page format.
* Ensure single-blind compliance.
* Finalize figures and citations.
* Submit before the deadline.

## Fallback strategy

If MapSAM training or adaptation does not work in time, the backup paper is still viable:

**A CRS-Aware Evaluation of SAM Prompting Strategies for Archaeological Feature Extraction from Georeferenced Historical Maps**

This would focus on SAM prompting strategies, GIS vectorization, and the limitations of zero-shot segmentation for archaeological maps.

## Recommended positioning

The paper should be submitted as an **in-progress applied pattern recognition study** for the archaeology-focused session. The strongest contribution is:

> georeferenced archaeological maps → segmentation baselines → CRS-aware evaluation → GIS-ready vector outputs.

This is realistic for the current stage of ArchaeoTopia and gives us a credible, focused ANNPR submission.
