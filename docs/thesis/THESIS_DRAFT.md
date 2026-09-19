# Model for identification of geographic and archaeological characteristics of burial mounds

**Master's thesis — annotated working draft (English)**

Lyubomir Angelov, fac. № 24346001 · Artificial Intelligence, MSc
Burgas Free University, Centre for Informatics and Technical Sciences

> **Status of this document.** This is a *skeleton with the evidence already in
> place*, not a finished thesis. Every section carries a target length, the
> repository documents it must be written from, and a writing brief. Chapters 2
> to 4 carry finished prose and every results table is already filled with the
> real measured numbers. Chapter 1 carries structure and a verification list
> rather than text, for the reason given in §0.5.
>
> Language: English first, as agreed; a Bulgarian translation for БСУ is a
> separate pass once the text settles.

---

## 0. How to read this document, and what has to be agreed first

### 0.1 The задание, verbatim

| field | value |
|---|---|
| Тема | Модел за идентификация на географски и археологически характеристики на надгробни могили |
| Изходни данни | сегменти от карти с изолинии и/или символи за селищни могили |
| Увод | Цел и задачи на дипломната работа |
| Глава 1 | Обзор на съществуващи решения |
| Глава 2 | Моделиране на софтуерната система, базирана на **SAM и FAISS** |
| Глава 3 | Проектиране и реализиране на системата |
| Глава 4 | Тестване на системата и резултати от изследването |
| Заключение | Изводи и направления за развитие |

### 0.2 Four points where the задание and the built system disagree

The БСУ указание explicitly permits this: *"Допуска се обявена тема да се
доуточни след консултация с научния ръководител, който я предлага."* Each of
the four below should be raised with the научен ръководител **before** the text
is written, because each changes what a chapter contains.

**1. FAISS was never used, and there is nothing for it to index.**
`grep -ri faiss` over the whole repository returns nothing: no import, no
dependency in `pyproject.toml`, no design note. Nor is there a latent need for
it. FAISS is an approximate nearest-neighbour index over dense vectors; it pays
off when a query embedding must be matched against a corpus too large for exact
search. The system's retrieval-shaped step — deciding which printed symbols on a
sheet are mounds — is solved by a trained detector over 8,671 sliding windows
per corpus pass, which is exact, takes about six minutes on one GPU, and needs
no index. Per your instruction, FAISS is dropped from the thesis rather than
retrofitted.

**Proposed amended title for Глава 2:** *"Моделиране на софтуерната система,
базирана на SAM и YOLO"* — or, closer to what is actually modelled, *"…
базирана на детектор YOLO и промптван сегментатор SAM (MapSAM)"*.

**2. The pre-YOLO baseline is not a CNN.** The reference method in experiment
v0.5 is **classical template matching**: eight k-means centroids over 32 px
intensity-normalised crops taken from the fold's *training* sheets, matched with
`cv2.matchTemplate` / `TM_CCOEFF_NORMED`, followed by local-maxima extraction
and merging at 10 px (`src/archeo_topia/analysis/detect_template_baseline.py`).
It contains no learned convolutional filters. It is a deliberate *non-neural*
reference, and the thesis is stronger for saying so plainly: the comparison it
supports is "learned detector vs. classical correlation", which is a cleaner
scientific statement than "one CNN vs. another".

There is a second, genuinely neural pre-YOLO attempt, and it is worth one
subsection of its own: **stock SAM 2 automatic proposal generation** over map
clips, orchestrated through an MCP service stack
(`docs/AUTO_ANNOTATION_RESULTS.md`, `docs/automation/`). It never produced
usable masks — the Nuclio SAM 2 function returned only embeddings, and CVAT's
upload path rejected the images — so the run fell back to heuristic placeholder
boxes. That is a negative result with a clear cause, and negative results with
clear causes belong in a thesis.

**3. "надгробни могили" (burial mounds) in the тема vs. "селищни могили"
(settlement mounds / tells) in the изходни данни.** These are different
archaeological objects. The project's own DMP says *"burial and settlement
mounds"*, and the annotation schema does not distinguish them: there is exactly
one positive class, `mound`, defined by the printed cartographic symbol rather
than by the site type beneath it. The thesis should state this once, early, and
then use "mound" throughout. Ask the научен ръководител which wording he wants
in the signed тема; the built system is agnostic.

**4. Isolines (изолинии) are an input condition, not an output.** The изходни
данни name *"сегменти от карти с изолинии и/или символи"*. The system does not
segment contour lines. Contours appear in the work in two places: as the
attribute `crossed_by_contour`, true on 68 of 180 mounds (38%), and as the
dominant source of visual clutter that a detector must learn to ignore. If the
supervisor expects contour extraction as a deliverable, that is a scope
expansion that does not currently exist in code and should be settled now.

### 0.3 What the thesis can honestly claim

This is the single most important framing decision in the document, and getting
it wrong is the fastest way to lose points with a рецензент.

> **Every number produced by this project to date is development evidence.
> There is no frozen test result.** All models were trained and evaluated under
> leave-one-sheet-out cross-validation over **three** annotated sheets of one
> Bulgarian 1:25,000 archival series. A permanent 45/13/5 split over 63 sheets
> now exists (`configs/splits/v0_6_splits.json`) and four sheets are frozen and
> untouched, but the blind annotation pass that would turn them into a test set
> is human work and has not been done.

The thesis should therefore be written as a **methodological study with a
working prototype**, not as a benchmark paper. That is not a weakness to be
hidden; it is the frame under which the project's most interesting results —
the ground-truth error in §5.5, the metric-design argument in §3.4, the
supervision-signal bug in §5.2 — are results at all. Chapter 4 states the
scoping caveat in its own subsection (§5.10) and every table in it is labelled.

### 0.4 Page budget

Master's theses at ЦИТН are **60–80 pages**. Mapping the задание onto the
указание's generic structure:

| part | указание target | plan here | source material |
|---|---|---|---|
| Увод | 1–2 pp | 2 pp | §1 below |
| Глава 1 — Обзор | 20–30 pp | 24 pp | external literature + `docs/ANPR_PAPER_PLAN.md` |
| Глава 2 — Моделиране | 5–10 pp | 10 pp | `architecture/TARGET_PIPELINE.md`, `v004/RESULTS.md` |
| Глава 3 — Проектиране и реализация | 5–15 pp | 14 pp | `src/`, `docs/mapsam/TRAINING.md`, `docs/automation/` |
| Глава 4 — Тестване и резултати | 10–25 pp | 24 pp | `docs/mapsam/v00{1..6}/`, `REPORT_2026-09.md` |
| Заключение | 2–4 pp | 4 pp | `REPORT_2026-09.md` §8 |
| **body total** | | **78 pp** | |
| Литература | — | 4 pp | ~45–60 entries |
| Приложения | — | 8 pp | schema, configs, listings |
| Съдържание | — | 2 pp | **placed last**, per указание |

Note the указание's ordering, which is unusual and is easy to get wrong:
**Литература → Приложения → Съдържание**, with the table of contents at the very
end of the document, and the signed задание inserted immediately after the title
page.

The two chapters that carry the master's-level requirement (*"теоретичен анализ
на предлаганите решения и методи за проектиране"* in ch. 2–3, *"по-задълбочен
характер"* in ch. 4) are Chapters 2 and 4, and both are budgeted at their
maximum.

### 0.5 Why Chapter 1 is structure only

Chapter 1 is 20–30 pages of literature review, and it is the one part of the
thesis that the repository cannot supply. I have no web access in this session,
so any citation I wrote out in full would be a plausible-looking guess at
authors, venue and year — exactly the kind of thing a рецензент checks. §2 below
therefore gives the full section structure, the argument each section has to
make, and a **verification list**: the works that must be looked up and
confirmed before they are cited. Names marked `[verify]` are ones I am
confident exist but whose exact reference details must be checked against the
publisher.

### 0.6 Conventions used throughout

- **Disagreement area** is the symmetric difference `|pred XOR gt|` expressed in
  **source-tile pixels²**. It is an *area*, not a boundary displacement. Prompt
  offsets and centroid distances are distances in **px**. The per-sample field
  name `abs_error_source_px` is historical and holds an area.
- **Decoder IoU** is measured on the 256×256 logit grid and is **not comparable
  between window sizes** — cropping inflates it by construction. Every
  cross-window comparison in Chapter 4 leads with disagreement area.
- **Macro** means averaged over the three leave-one-sheet-out folds;
  **sample-weighted** means averaged over instances. Macro is reported first
  because the question is robustness across map domains.
- **Oracle** labels any figure selected by looking at the evaluation split. It
  is an upper bound, never a result.
- Sheet ids follow Soviet-style nomenclature: `K-35-51-B-a` is the 1:25,000
  quadrant of the 1:50,000 sheet `K-35-51-B`, under the 1:100,000 parent
  `K-35-51`. Earlier drafts in this project called the series "Soviet 1:50k";
  that was a misreading of the nomenclature and is corrected throughout.

---

## 1. Увод / Introduction

**Target: 2 pages. Status: brief + opening prose drafted.**
**Sources:** `docs/AI_ARCHEO_TOPIA_DMP.md`, `docs/architecture/TARGET_PIPELINE.md`.

The указание asks the Увод for: the relevance of the problem, a short statement
of how it is solved here, and brief data on the methods and tools used.

### 1.1 Relevance

Draft prose:

> Bulgaria's burial and settlement mounds are among the most exposed categories
> of archaeological monument in the country. They stand in arable land, they are
> not individually protected, and modern agricultural machinery, road
> construction and looting remove them faster than they can be surveyed. A mound
> that has been ploughed flat leaves no surface trace, but it was drawn on a map
> before it was destroyed.
>
> The mid-20th-century Bulgarian archival topographic series at 1:25,000,
> compiled from the 1969 *Balance of Earth* survey, records these monuments at a
> moment when most of them were still intact. Each mound appears as a small
> printed cartographic symbol roughly 21 pixels across on a 300 dpi scan. A
> single sheet covers about 10.4 × 9.5 km and holds between a handful and
> several hundred of them. The national series runs to thousands of sheets.
> Reading them by hand is possible and is what archaeologists currently do; it
> is simply far slower than the rate at which the monuments are disappearing.
>
> This thesis is carried out inside the AI ArchaeoTopia project (NAIM-BAS,
> Bulgarian National Science Fund contract КП-06-Н100/5), whose object is
> precisely this: to convert an archival paper map series into a georeferenced
> digital inventory of mound locations, with enough accuracy and enough
> traceability that an archaeologist can act on it.

### 1.2 Purpose (Цел)

> To design, implement and evaluate a software system that locates archaeological
> mound symbols on scanned archival topographic map sheets, delineates each
> symbol precisely, extracts its geographic and archaeological characteristics,
> and delivers the result as a georeferenced GIS layer that a domain expert can
> review and correct.

### 1.3 Tasks (Задачи)

Number these; Chapter 4's conclusions and the Заключение both refer back by
number.

1. Analyse existing approaches to symbol extraction from historical maps, and to
   promptable and conventional segmentation and detection in remote sensing and
   archaeology.
2. Build an annotation protocol and a labelled corpus sufficient to train and
   evaluate a two-stage system, including typed hard negatives.
3. Adapt a promptable foundation segmenter (SAM) to the mound-symbol domain
   under the constraint of a very small training set.
4. Determine experimentally what input representation the segmenter requires,
   and derive from it a *quantitative specification* for the detector that must
   feed it.
5. Build a candidate detector meeting that specification, and compare it against
   a non-neural reference method.
6. Compose the two stages and measure the system end to end.
7. Transfer the system to an unseen corpus twenty times larger than the training
   data, and deliver the output in a form a domain team can review.
8. Establish the data-governance apparatus — versioned annotation exports, a
   machine-readable schema, permanent leak-free splits — that makes a future
   test result possible.

### 1.4 Methods and tools, in brief

SAM ViT-B with frozen image and prompt encoders and a fine-tuned mask decoder;
YOLO26-s as candidate detector; classical normalised cross-correlation template
matching as reference; CVAT for annotation; PyTorch 2.14 + CUDA 13.0 on a single
RTX 5090; GDAL 3.4.1 via its command-line tools for geospatial conversion;
COCO, GeoJSON and GeoPackage as interchange formats. Roughly 23,700 lines of
Python under `src/`, with 684 automated tests.

### 1.5 Structure of the thesis

One short paragraph per chapter. Write this last.

---

## 2. Глава 1 — Review of existing solutions

**Target: 24 pages. Status: structure and argument only — see §0.5.**

The указание asks this chapter to gather and process information from literature
sources, analyse existing solutions, and **close with conclusions that form the
transition into the substantive chapters**. That closing subsection (§2.7) is
the load-bearing one: it must end by stating the gap that Chapters 2–4 fill.

### 2.1 The application domain: archaeological prospection from historical maps (≈4 pp)

**Argument to make:** historical maps are an established archaeological source,
and their systematic machine reading is an active but young field. Position the
work against manual and semi-automated map digitisation practice, and against
the alternative evidence sources (LiDAR-derived DTM, satellite and aerial
imagery, geophysics) — noting that the archival map is unique in recording the
monument *before* modern destruction, which no present-day sensor can do.

**To verify and cite:** work on burial-mound detection from LiDAR in the Balkans
and Central Europe `[verify]`; the *Balance of Earth 1969* series provenance via
AGKK `[verify — the DMP is the internal source]`; digital-archaeology surveys by
A. Sobotkova and I. Berganzo-Besga, both named as project experts in the DMP,
whose own publications are the natural anchor for this section `[verify]`.

### 2.2 Symbol and feature extraction from scanned maps (≈4 pp)

**Argument:** classical pipelines — colour separation, morphological filtering,
template matching, Hough-type accumulators — are what the field used before deep
learning, and they are exactly what §4.6 benchmarks against. Explain *why* they
degrade on this material: a scanned archival sheet has no clean colour
separation, the symbol is small and drawn by hand-set drafting conventions that
vary between print runs, and the background is dense with contours, grid lines,
roads and text that share the symbol's stroke width.

**To verify and cite:** the map-processing and graphics-recognition literature
(GREC workshop series, IJDAR) `[verify]`; `cv2.matchTemplate` /
`TM_CCOEFF_NORMED` is documented in OpenCV and can be cited from the manual.

### 2.3 Deep learning for detection and segmentation (≈5 pp)

**Argument:** establish the vocabulary Chapters 2–4 use, and justify the eventual
architectural choice. Cover, in this order: fully convolutional semantic
segmentation and U-Net; two-stage detection (Faster R-CNN + FPN); one-stage and
anchor-free detection and the YOLO lineage up to YOLO26; transformer detectors
(DETR, RT-DETR). For each, state what it would require *here* — U-Net wants dense
masks over whole sheets, Faster R-CNN wants more positives than 180, RT-DETR
wants a long schedule — and carry that into §2.7.

**To verify and cite:** Ronneberger et al., U-Net `[verify]`; Ren et al., Faster
R-CNN `[verify]`; Lin et al., FPN `[verify]`; Redmon et al. and the subsequent
Ultralytics releases `[verify]`; Carion et al., DETR `[verify]`; Zhao et al.,
RT-DETR `[verify]`. **YOLO26 in particular must be verified** — cite the
Ultralytics release actually used, `ultralytics 8.4.150`, not a paper that may
not exist for this version. Note the AGPL-3.0 licence of Ultralytics explicitly;
the repository is MIT and the boundary is worth one sentence.

### 2.4 Foundation models for segmentation: SAM and its adaptations (≈6 pp)

**This is the most important section of the chapter** and should be the longest.

**Argument:** SAM changed what "segmentation" costs, by separating a heavy,
general image encoder from a light, promptable mask decoder. Explain the three
components — image encoder (ViT), prompt encoder, mask decoder — and then make
the point the whole thesis rests on:

> A promptable segmenter is *conditional*. It answers "what is the object at
> this location?" It does not answer "is there an object, and is it the kind of
> object I care about?" Those are different questions, they need different
> models, and conflating them is what §5.5 shows to be expensive.

Then survey domain adaptation of SAM: decoder-only fine-tuning, adapter and
LoRA-style approaches, and the specific line of work on **historical and
archival map segmentation that gives MapSAM its name**. Explain what is
inherited from it — the decoder-only regime, the prompt-centred crop, the
binary-mask dataset layout — and what this project adds: an experimentally
derived prompt-tolerance curve, and a detector built to satisfy it.

**To verify and cite:** Kirillov et al., *Segment Anything* (SAM) `[verify]`;
Ravi et al., *SAM 2* `[verify]`; the MapSAM paper on adapting SAM to historical
maps `[verify — this is the direct antecedent and MUST be cited correctly,
since the project's core module is named after it]`; LoRA `[verify]`; surveys of
SAM in remote sensing `[verify]`.

### 2.5 Annotation, weak supervision and human-in-the-loop (≈3 pp)

**Argument:** with a domain where annotation requires an archaeologist,
annotator time is the binding resource, and the system's design is shaped by it.
Cover CVAT and its serverless interactors, model-assisted pre-labelling, active
learning, and the failure mode this project actually hit: **label noise that is
invisible to the metric you are optimising**. That sets up §5.5 directly.

**To verify and cite:** CVAT `[verify]`; active-learning surveys `[verify]`;
work on learning with noisy labels `[verify]`; COCO format (Lin et al.)
`[verify]`.

### 2.6 Georeferencing and GIS delivery (≈2 pp)

**Argument:** a detection in pixel space is not yet an archaeological
observation. Cover GCP-based affine and polynomial fitting, `gdal.Warp`,
EPSG:25835 (ETRS89 / UTM 35N) as the Bulgarian national working projection, and
the interchange formats — with the concrete finding from §4.6 that Shapefile is
unusable here because all 14 `mound` attribute names exceed its ten-character
field limit.

**To verify and cite:** GDAL/OGR documentation; RFC 7946 (GeoJSON); OGC
GeoPackage; EPSG registry entry 25835.

### 2.7 Conclusions from the review (≈2 pp) — the transition

**This subsection is what the указание grades.** It must produce four statements
that Chapter 2 then acts on:

1. **No published system does this task on this material.** The closest work
   adapts SAM to historical maps, but for linear and areal features, not for
   small discrete archaeological point symbols with an attribute schema.
2. **A promptable segmenter cannot be the whole system**, because it is
   conditional by construction, and therefore the architecture must be at least
   two-stage.
3. **The composition constraint between the stages is not documented anywhere**
   in the literature. How accurate must a detector's point be before a promptable
   segmenter degrades? That number does not exist and must be measured — which
   is Chapter 4's step 1 and this thesis's first original contribution.
4. **Data volume, not model capacity, is the expected binding constraint**, so
   the design must be evaluable at small scale and the experimental protocol must
   be honest about what three sheets can and cannot establish.

---

## 3. Глава 2 — Modelling the software system

**Target: 10 pages. Status: prose drafted; needs figures.**
**Sources:** `docs/architecture/TARGET_PIPELINE.md`, `docs/mapsam/v004/RESULTS.md`,
`docs/mapsam/v005/PLAN.md`.

> **Chapter title.** The signed задание reads *"…базирана на SAM и FAISS"*.
> Subject to §0.2, this should become *"…базирана на SAM и YOLO"*.

The указание requires this chapter to select block diagrams, algorithms and
models, and — for a master's degree — to add a **theoretical analysis of the
proposed solutions**. §3.3 and §3.4 carry that analysis.

### 3.1 Problem decomposition

The organising decision of the whole system is a division of labour:

> **The detector does semantic localisation. The segmenter does spatial
> delineation. Neither does the other's job.**

Stated as questions:

| stage | question | model | why not the other model |
|---|---|---|---|
| detection | *where might a mound be?* | YOLO26-s | SAM is conditional; given a prompt on a mill it outlines the mill |
| segmentation | *exactly which pixels are this symbol?* | SAM ViT-B, fine-tuned decoder | a detector box is 512 px of map, most of it not the symbol |

This is not an implementation convenience. §5.5 shows it is a property with
consequences: a conditional segmenter **cannot audit its own labels**, because it
is never asked to decide whether the thing it was pointed at belongs to the
class. A detector must decide that on every window, and that is why the detector
— and not four prior versions of the segmenter — was the stage that exposed
twelve mislabelled annotations.

### 3.2 Target pipeline

```text
Georeferenced archival map sheet (≈4920 × 4464 px, 2.12 m/px, EPSG:25835)
          │
          ▼
   Sliding-window reader           512 px window, stride 384 → ~156 windows/sheet
          │
          ▼
   Candidate generator             YOLO26-s, one class
          │                        → candidate point, box, confidence
          ▼
   Candidate point (requirement: p90 centre error ≤ 5 source px)
          │
          ▼
   512 px prompt-centred crop      window follows the prompt
          │
          ▼
   MapSAM decoder                  SAM ViT-B, encoders frozen
          │
          ▼
   Symbol segmentation mask
          │
          ├──► duplicate suppression      (box-level first, mask-level for survivors)
          ├──► attribute extraction       (labels exist; model not built)
          ├──► quality estimation         (signal identified; not validated)
          ▼
   mask centroid + symbol geometry, in source pixels
          │
          ▼
   pixel → EPSG:25835                    (geotransform carried on the sheet)
          │
          ▼
   GeoPackage: mound_points + mound_symbols
          │
          ▼
   archaeological review → corrected annotations → retraining
```

Two design rules are visible in that diagram and both should be argued
explicitly in the text.

**Georeferencing is kept off the critical path.** Every stage above the
projection step operates in source-pixel space. The instance record is complete
and useful without a geotransform, which carries the pixel geometry as primary
and the projected geometry as an optional enrichment with its own quality flag.
This matters because the project's own frame-detection georeferencing pipeline
passes quality checks on only **61 of 200 sheets (30.5%)**; a design that
required it would be blocked on it. The 60 new sheets arrive already
georeferenced from the supplier, which is why the delivery in §5.8 was possible
at all.

**The polygon is evidence, not extent.**

> The MapSAM polygon is the *printed cartographic mound symbol*. It is not the
> physical footprint of the archaeological mound.

Measured symbol component areas run **116–1037 px** with a mean of 459, on a
symbol roughly 21 px across. That 9× spread is drafting variation and scan
condition, not mound size. A consumer reading `mask_area_px` as ground extent is
wrong by an unbounded factor, so the convention is: **point = the archaeological
location; polygon = source-symbol geometry, i.e. provenance**. The delivered GIS
layers enforce it by naming every polygon field with a `symbol_` prefix.

### 3.3 Theoretical analysis I — why the input representation dominates

*This subsection is the master's-level theoretical analysis the указание asks
for, and it is the part of the design that was derived rather than chosen.*

SAM's image encoder is a ViT operating on 16 px patches over a 1024×1024 input,
producing a 64×64 token grid; the decoder's 256×256 logits are upsampled from
it. The quantity that governs how much the encoder can say about an object is
therefore **how many tokens that object covers**.

For a mound symbol of ~21 source px:

| input window | scale to 1024 | symbol size at encoder | **encoder tokens on the symbol** |
|---|---:|---:|---:|
| full clip, ~2300 px | 0.445 | 9.3 px | **0.61** |
| 1024 px window | 1.0 | 21 px | **1.36** |
| 512 px window | 2.0 | 42 px | **2.76** |

At full-clip resolution the symbol occupies **less than one patch**. The encoder
physically cannot represent it as a distinct entity, and no amount of decoder
fine-tuning recovers information the encoder never encoded. This predicts, before
any experiment, that cropping around the prompt will dominate loss tuning — and
§5.3 measures a 64% reduction in absolute error that confirms it.

It also predicts the shape of the failure. If the symbol is sub-token, the
decoder's best available strategy is to respond to *context* rather than to the
symbol, which at inference means firing on whatever is mound-like nearby. §5.4
measures exactly that: 19 of 120 samples at IoU zero, of which 17 sit within
40 px of the correct target and 2 have locked onto a distractor 868 and 1861 px
away. Windowing removes both classes, the first by resolution and the second
because a 512 px window *physically excludes* a distractor 868 px away.

**Even the 512 px window puts under three tokens on a mound**, so this lever is
not exhausted. 256 and 384 px windows remain untested, and §6 records why.

### 3.4 Theoretical analysis II — the composition constraint

The two stages compose only if the detector's output lands inside the
segmenter's tolerance. Nothing in the literature says what that tolerance is, so
it was measured (§5.4) by displacing the prompt in 8 compass directions over
offsets 0–300 source px.

The deployment-relevant arm — prompt and window moving together, which is what a
detector-driven crop does — reads:

| prompt offset | disagreement area px² | IoU ≥ 0.5 rate |
|---:|---:|---:|
| 0 px | 91.8 | 1.00 |
| 5 px | 140.5 | 1.00 |
| 10 px | 255.0 | 0.69 |
| **15 px** | 444.6 | **0.02** |

> **This is the thesis's central engineering specification.** The working band is
> ≤ 5 px; 5–10 px is marginal; 15 px is a miss. Therefore the detector's target
> is **p90 centre error ≤ 5 source px, with recall@5px as the primary metric** —
> not mAP, and not recall at a loose radius.

Three consequences deserve separate paragraphs in the text.

**Detection and localisation must stay separate axes.** A single aggregate score
— mAP, or recall at a generous radius — averages "did you find it" together with
"did you point at it accurately". Keeping them apart is what made §5.5's
ground-truth error findable: the failing fold had the project's *best* p90
(2.00 px) alongside its worst recall (0.389). A model genuinely failing on a
sheet degrades on both axes; a model correctly refusing to fire on mislabelled
objects degrades on one. One number would have shown one mediocre figure and
given nobody a reason to look at the annotations.

**Window placement and prompt placement have wildly different tolerances.** The
control arm — window displaced, prompt left on truth — moves the error from 91.8
to 94.2 px² across **250 px** of displacement, with every sample still clearing
IoU 0.5. So the window has ~250 px of slack and the prompt has ~5. The detector
does not need to produce a well-centred crop, a good mask, or a tight box. **It
needs to point at the right symbol.** That is a writable specification, and
having it is the main reason the segmentation work was worth doing before the
detector existed.

**Both representations fail at the scale of the symbol.** The full-clip arm is
not the robust alternative: it degrades from 253.9 to 702.4 px² over the same
25 px, with its IoU ≥ 0.5 rate falling 0.71 → 0.06. By 50 px both arms are at
zero mean IoU. This is simply what a prompted segmenter does — it segments what
is at the prompt, and 25 px is most of a mound symbol's width.

### 3.5 Choice of models, with the alternatives considered

| decision | chosen | alternatives | reason |
|---|---|---|---|
| segmenter | SAM ViT-B, decoder-only fine-tuning | train from scratch; U-Net; SAM 2; ViT-L/H | 171 training samples; a frozen encoder plus ~4M trainable decoder parameters is the only regime this data supports. ViT-B is also what fits an embedding cache for full-clip arms. |
| detector | YOLO26-s | Faster R-CNN + FPN; RT-DETR; DETR | the `s` scale is chosen against 180 positives: extra capacity is extra variance, not extra accuracy. The others are benchmarked **only if a failure appears that they could address** — none has. |
| reference | `cv2.matchTemplate` NCC, k-means templates | SIFT/ORB matching; hand-tuned morphology | cheapest honest non-neural reference; establishes that the obvious classical method is not competitive, without claiming the best classical method is ruled out. |
| prompt | ground-truth box + centre point (training), detector point + canonical box (inference) | point only; box only | box scale is measured in §5.7: a canonical median-sized box on the detector's point beats the detector's own box at IoU ≥ 0.75 on all three folds. |
| negatives | 530 typed hard negatives carried by tiling | explicit negative mining; a negative class | 270+ windows contain hard negatives and no mound, so "these symbols are background" is taught by the tiling itself, with no sampling scheme. |

### 3.6 The canonical output record

One mound instance, in full, with the three deliberate choices in its shape
annotated. Reproduce the JSON from `docs/architecture/TARGET_PIPELINE.md` here;
the three choices to argue are: `symbol_*` field naming (§3.2), `geo` as an
optional self-describing block (§3.2), and retaining the classifier's **runner-up
negative type**, which costs nothing and is the field that will explain a
systematic error class when one appears.

### 3.7 Figures needed

| fig. | content | source |
|---|---|---|
| 2.1 | Pipeline block diagram | the ASCII diagram in §3.2, redrawn |
| 2.2 | SAM architecture: encoder / prompt encoder / mask decoder, with the frozen–trainable boundary marked | redraw |
| 2.3 | Token-coverage illustration: the same symbol at full clip / 1024 / 512, with the 16 px patch grid overlaid | render from a real clip |
| 2.4 | Prompt-tolerance curve: IoU ≥ 0.5 rate vs. prompt offset, both arms + the window-only control | `v004/analysis/jitter/` |
| 2.5 | Point vs. polygon semantics: one symbol, its mask, its centroid, with the "not a site boundary" caption | render |

---

## 4. Глава 3 — Design and implementation of the system

**Target: 14 pages. Status: prose drafted; listings to be selected.**
**Sources:** `src/`, `docs/mapsam/TRAINING.md`, `docs/mapsam/DATASET.md`,
`docs/annotation/PROTOCOL_EN.md`, `docs/automation/`, `Makefile`.

### 4.1 Repository and package layout (≈1 p)

Roughly 23,700 lines of Python under `src/`, with 29 test modules and 684
passing tests. The installed package is `archeo_topia`:

| package | responsibility |
|---|---|
| `archeo_topia.datasets` | COCO → training data; prompts; sliding windows; sheet clipping; schema migration |
| `archeo_topia.training` | SAM decoder fine-tuning, losses, embedding cache, detector training, checkpoint migration |
| `archeo_topia.analysis` | evaluation, jitter sweep, failure anatomy, cross-run comparison, corpus sweep, symbol decoding, the template baseline |
| `archeo_topia.formats` | `LabelSchema`, `CocoDocument`, `SheetReference`, `FeatureCollection`, GIS export, review ingest |
| `services.sam2_backend`, `services.sam2_mcp`, `services.annotation_pipeline` | the annotation-assist stack (§4.7) |

Two directories are deliberately *outside* the installed package and should be
described as such rather than quietly omitted: `src/georeference/` is orphaned
(it imports a repo-root `utils.py` and needs a `grid_25k.geojson` that is not
present, so its 30.5% pass rate cannot currently be reproduced), and
`src/data_lake/docx_mound_extractor/` serves a separate corpus-ingest task.

### 4.2 Data pipeline (≈3 pp)

```text
scanned sheet (GeoTIFF, RGBA, DEFLATE, EPSG:25835)
   │  sheet_clips.py split --grid 2x2
   ▼
clips (~2400 × 2200 px) + geo provenance per clip
   │  CVAT annotation (§4.3)
   ▼
COCO export  annotation/cvat/v0.0.{1,2,3}/instances_default.json
   │
   ├─ prepare_mapsam_coco.py ──► mapsam_v0{,2,3}/   binary masks + ignore masks, split by sheet
   │       │  generate_mapsam_prompts.py
   │       ▼  training_samples.jsonl — one sample per connected component
   │
   └─ build_detection_windows.py ─► mapsam_det_v{0,1}/  512/384 windows, YOLO labels
```

Three implementation points are worth a paragraph each because each was a bug
that cost a version.

**One sample per connected component, and the component is labelled at full
resolution.** Mean component area is 458.9 px on a ~2474×2242 clip, which is
~79 px at 1024 and ~5 px at the 256 logit grid. Nearest-neighbour downsampling
can fragment or erase a thin ring-shaped symbol before labelling happens, so the
component is selected on the original-resolution mask and *then* resized.

**The sample accounting is exact and must be stated**, because three different
denominators appear in Chapter 4:

```text
180  mound annotations                    ← detection recall is measured against this
 −1  empty geometry (a 16×8 px symbol truncated at a clip's top edge: bbox, no mask)
=179 with geometry                        ← the end-to-end denominator
 −8  merge events: touching mound polygons sharing one connected component
=171 training samples                     ← the segmentation manifest
```

**One mask decoder, as the union of two.** Measured on export v0.0.3, the 714
annotations are 161 uncompressed RLE, 20 polygons and **533 carrying a box and
no segmentation** — and box-only is precisely the case for which the older of the
two decoders returned a silently empty mask. Where the two disagreed on RLE size
validation the strict reading wins: a mask declaring a size different from its
image is a data bug, and pasting it into the top-left corner hid it. A
`bbox_fallback` switch remains, set to `False` at the dataset builders, to
protect published results: v0.1–v0.5 were all computed counting the one
box-only mound as having no geometry, and filling its box instead would silently
change both the component grouping and the `mounds_without_geometry` count those
runs reported.

### 4.3 Annotation protocol and the label schema (≈3 pp)

Annotation is in CVAT, by archaeologists, against a written protocol maintained
in English and Bulgarian (`docs/annotation/PROTOCOL_{EN,BG}.md`). Three classes:

| class | v0.0.1 | v0.0.3 | role |
|---|---:|---:|---|
| `mound` | 180 | 169 | positives |
| `hard_negative_symbol` | 530 | 542 | confusable negatives |
| `uncertain_ignore` | 3 | 3 | excluded from training **and** evaluation |

The near-3:1 ratio of hand-selected confusable negatives to positives is an
unusually good starting position for a recall-oriented detector, and it is the
asset the first four segmentation versions deliberately never touched.

**Negatives are typed**, which lets error be reported by cause rather than as a
single precision number:

| `negative_type` | n | | `negative_type` | n |
|---|---:|---|---|---:|
| `decorative_symbol` | 246 | | `colored_pencil` | 7 |
| `trig_point` | 136 | | `grid` | 6 |
| `road` | 97 | | `other` | 5 |
| `text` | 29 | | `contour` | 1 |

**The trig-point ambiguity is the sharpest design problem in the schema.**
`trig_point` is the second-largest negative class at 136 instances — and
`has_trig_point` is simultaneously `true` on **42 of the 180 mounds (23%)**. The
same printed element is a negative when it stands alone and an *attribute of a
positive* when it sits on a mound. No amount of extra data fixes a label
geometry that is genuinely context-dependent; any filter must be evaluated on
this subset specifically, because aggregate precision hides it.

**Attributes are annotated on every object**, making attribute extraction a
supervised task with labels already in hand. Rates on the 180 mounds:

| attribute | n | rate | | attribute | n | rate |
|---|---:|---:|---|---|---:|---:|
| `has_relative_height_mark` | 158 | 88% | | `blurred_or_bad_print` | 36 | 20% |
| `crossed_by_contour` | 68 | 38% | | `crossed_by_grid` | 33 | 18% |
| `crossed_by_road` | 52 | 29% | | `crossed_by_water_line` | 14 | 8% |
| `has_trig_point` | 42 | 23% | | `overlaps_other_mound` | 13 | 7% |
| `crossed_by_forestation_line` | 40 | 22% | | `affected_by_colored_pencil` | 10 | 6% |
| `has_absolute_elevation_mark` | 37 | 21% | | `crossed_by_powerline` | 4 | 2% |

With 180 positives the frequent attributes are learnable and the rare ones are
not; `crossed_by_powerline` at 4 instances is rule-or-review territory.

**The schema of record and its two corrections.** Before v0.6 the schema existed
only implicitly inside the COCO exports, and `PROTOCOL_EN.md` §8 had silently
drifted away from it — omitting `crossed_by_water_line` (the attribute that
caught twelve mislabels, §5.5), omitting `crossed_by_forestation_line` and both
elevation-mark variants, and listing five attributes CVAT does not emit.
`annotation/cvat/labels.json` is now the machine-readable schema of record,
carrying 15 / 16 / 16 attributes across the three classes. Two changes were made
while the protocol was open:

- **`crossed_by_water_line` → `water_line_crossing`**, a select with values
  `none` / `surface` / `underground` / `unreviewed`. The domain rule is that a
  mound cannot be crossed by a **surface** watercourse — water runs along terrain
  lows and a mound is raised — but an **underground** line is a pipe and can run
  beneath anything, and the old boolean conflated the two. The plan asked for two
  booleans; one select was implemented instead, because two booleans cannot
  express the state 105 hard negatives and 2 uncertain regions are actually in —
  nobody has ever been asked about them — and would manufacture 105 assertions no
  annotator made. The fourth value records the truth and doubles as a work queue.
- **`review_status`** — `unreviewed` / `confirmed` / `rejected` / `corrected` /
  `added` — is a **separate axis from `annotation_provenance`**. One records who
  drew a shape, the other what a reviewer concluded. Collapsing them would make a
  rejected detection indistinguishable from one that was never sent, and the
  recall denominator impossible to reconstruct afterwards.

### 4.4 Training implementation — MapSAM (≈2 pp)

```yaml
model:      sam_vit_b, image_encoder frozen, prompt_encoder frozen
trainable:  mask_decoder only
optimizer:  AdamW, lr 1e-4, weight_decay 0.01
loss:       0.5 · BCE(pos_weight=20) + 2.0 · Dice
window:     512 px, prompt-centred, clamped inside the clip
loss crop:  bbox + margin, scaled per arm (32 / 75 / 150 logit px) so the
            supervised ground area stays ≈300 source px² across window sizes
epochs:     50 (200 for overfit-debug runs)
```

Implementation details worth describing: the **embedding cache**
(`cache_sam_embeddings.py`) makes full-clip arms cheap by precomputing frozen
encoder outputs, and is necessarily **disabled for windowed arms** because the
window depends on the prompt — which is also why the v0.3 runs never recorded a
source-pixel error column and had to be re-scored through the non-cached dataset
in v0.4. The **prompt-jitter hook** (`MapSamDataset(prompt_jitter_xy=,
jitter_prompts=)`) displaces the prompt *after* instance selection, so the target
remains the annotated mound; with `jitter_prompts=False` it moves the window
only, which is the control arm of §5.4.

### 4.5 Training implementation — the detector (≈1.5 pp)

`build_detection_windows.py` tiles each clip at window 512 / stride 384, clamped
flush at the right and bottom edges, emitting one YOLO class. Hard negatives are
carried in metadata rather than trained as a class; `uncertain_ignore` is
excluded in both directions.

| | v0.0.1 | v0.0.2 |
|---|---:|---:|
| windows with ≥1 mound | 129 | 123 |
| windows with hard negatives, no mound | 270 | 276 |
| windows empty of both | 81 | 81 |
| **total windows** | **480** | **480** |
| target boxes written | 344 | 317 |
| merged components | 8 | 8 |
| mounds without geometry | 1 | **0** |

Every annotation appears in at least one window, and every one has at least one
window holding its box entire, so no miss reported in Chapter 4 is attributable
to tiling; all untruncated targets round-trip from window to source coordinates
exactly. Training: `yolo26s.pt`, `imgsz 512` native, 150 epochs, batch 16,
seed 42, three leave-one-sheet-out folds, candidates merged across overlapping
windows at 10 px on centre distance and confidence.

**No checkpoint selection.** The only validation split available was the
held-out sheet, so Ultralytics' `best.pt` would select on the evaluation set.
Everything reported is `last.pt`. §5.9 quantifies what that discipline cost and
why it was worth it.

### 4.6 Formats, coordinates and the GIS delivery (≈2 pp)

The central decision in `archeo_topia.formats`:

> **A COCO document is in image pixels; a feature collection is on the ground.**
> A direct conversion between them is undefined, so every conversion takes a
> `SheetReference` as a *required* argument.

Making the reference impossible to omit is how "avoid hidden coordinate
assumptions" is enforced rather than merely intended. `SheetReference` also
closes a gap that had been open since the clip pipeline was written: the
clip → sheet → ground path was documented, specified and **tested**, but no
production code walked it until the delivery existed.

**GDAL appears only as a subprocess.** Its Python bindings must match the system
`libgdal` exactly and be built against an installed numpy, or they import and
then fail with `no module named _gdal_array`. GeoJSON is written with the
standard library and `ogr2ogr` converts it; a test asserts that nothing under
`formats/` imports `osgeo`, so the decision cannot quietly erode.

**Format choice was made on measurement, not preference.** GeoPackage holds both
layers in one file, declares its CRS unambiguously in both QGIS and ArcGIS, and
keeps full-length field names. GDAL's GeoJSON writer emits a legacy `crs` member
that RFC 7946 removed — QGIS honours it, ArcGIS generally does not and assumes
WGS84 — so the portable copy is written with `RFC7946=YES` and reprojected
explicitly. **Shapefile is excluded on evidence:** all 14 `mound` attribute names
exceed its ten-character limit, and `crossed_by_contour`,
`crossed_by_forestation_line`, `crossed_by_grid`, `crossed_by_powerline` and
`crossed_by_road` all collapse to `crossed_by`, `crossed__1` … `crossed__4`.

**The return path.** `formats.ingest_review` reads a reviewed GeoPackage back and
produces an evaluation plus a COCO file for CVAT. It checks integrity *before* it
counts anything: a changed CRS, a duplicated `mound_id`, a feature moved off the
sheet, one left `unreviewed`, and — the case `review_status` exists to prevent —
a proposal **deleted rather than marked rejected**, since a deleted feature
cannot be told from one that was never sent.

### 4.7 The annotation-assist stack, and why it is a separate system (≈1.5 pp)

An MCP-based service stack exists for *annotation assistance*: an orchestrating
multimodal agent routing to a high-resolution OCR service and a **stock SAM 2**
segmentation service, with a FastAPI backend (`services/sam2_backend`), an MCP
server (`services/sam2_mcp`) and a PDF-to-annotation pipeline
(`services/annotation_pipeline`). It is restricted to map imagery by contract.

**It must not be conflated with the detection pipeline**, and the thesis should
say so in one sentence: the MCP stack uses *stock* SAM 2 to help *produce*
annotations; §3.2's stage 3 is a *fine-tuned* SAM ViT-B decoder doing production
inference. They meet at the review surface and nowhere else.

Its outcome is a **documented negative result** and belongs in the thesis as
one. The 2026-05 auto-annotation run processed 111 of 112 PDF pages and extracted
162 clips with zero validation errors, but SAM 2 mask generation never completed:
the Nuclio function returned only ~4 MB embedding blobs, mask generation required
either CVAT's serverless interactor or a local PyTorch SAM 2 install, and CVAT's
image upload behind the Docker proxy rejected `multipart/form-data` with
`400 Unsupported media type`. All bounding boxes in that run are heuristic
placeholders. The lesson to draw — and it is a real one — is that an
orchestration layer cannot compensate for a missing capability in the service it
orchestrates, and that the project's progress came from fine-tuning a smaller
model on its own data rather than from arranging larger ones around it.

### 4.8 Reproducibility apparatus (≈1 p)

Worth a short section, because the рецензент's second question is *"what did you
personally build"*:

- **Versioned annotation exports.** `v0.0.1`, `v0.0.2`, `v0.0.3` are kept side by
  side and never overwritten, so the experiment that exposed the annotation error
  (§5.5) remains reproducible against the labels it actually ran on.
- **Per-run output roots.** Every config writes to its own `outputs.root`; no
  version's artifacts are ever overwritten by a later one.
- **Migrations with guards.** Each refactor that touched a data path was verified
  by rebuilding its output and comparing: `build_detection_windows.decode_mask`
  rebuilt both detection datasets **byte-identical**; `mapsam_end_to_end` re-ran
  fold B end to end and reproduced 1.000 at IoU ≥ 0.5 and 0.7482 mean IoU exactly.
- **Permanent grouped splits** (§5.8) and a machine-readable schema (§4.3).
- Environment: torch 2.14.0+cu130, ultralytics 8.4.150, segment-anything 1.0,
  GDAL 3.4.1, one RTX 5090. v0.1–v0.4 originally ran on torch 2.12.

### 4.9 Figures and listings needed

| item | content |
|---|---|
| fig. 3.1 | Repository / package dependency diagram |
| fig. 3.2 | Data pipeline (the diagram in §4.2, redrawn) |
| fig. 3.3 | CVAT annotation screenshot with the attribute panel open |
| fig. 3.4 | One clip with all three classes overlaid, colour-coded |
| fig. 3.5 | MCP annotation-assist architecture |
| fig. 3.6 | The GeoPackage open in QGIS over the source raster |
| list. 3.1 | `MapSamDataset.__getitem__` — window, prompt, instance target |
| list. 3.2 | `combined_bce_dice_loss` with the bbox loss crop |
| list. 3.3 | `SheetReference` — the required-argument conversion |
| list. 3.4 | A training config YAML in full → **Appendix** |

---

## 5. Глава 4 — Testing the system and results of the investigation

**Target: 24 pages. Status: complete — every table below holds real measured
numbers. Prose needs expanding, no numbers need finding.**
**Sources:** `docs/mapsam/v00{1,2,3,4,5,6}/`, `docs/mapsam/REPORT_2026-09.md`.

The указание asks this chapter for experimental investigation, comparative
analysis, evidence that the results are trustworthy, and an assessment of
practical applicability — and, for a master's degree, that the investigation be
*"по-задълбочен характер"*. The structure below follows the actual experimental
sequence, because the sequence is itself the argument: each version's result
determined what the next one asked.

### 5.1 Experimental protocol (≈2 pp)

**Corpus.** Three annotated sheets of one Bulgarian 1:25,000 archival series,
cut into 12 clips of ~2400 × 2200 px, 66.4 Mpx in total.

| sheet | clips | samples |
|---|---:|---:|
| `K-35-51-B-a` | 4 | 120 |
| `K-35-8-G-a` | 4 | 34 |
| `K-34-35-B-g` | 3 (+1 with no mounds) | 17 |

**Validation scheme.** Leave-one-sheet-out (LOSO). Splitting by sheet rather
than by image is mandatory: clips from one sheet share terrain, survey campaign,
print run and scan batch, so an image-level split leaks.

**Epoch selection.** With three sheets and no validation split there is no
leak-free way to choose an epoch, so **every table reports the final epoch (50)
as the primary figure**. Peaks are shown separately and labelled *oracle*,
because they were chosen by looking at the evaluation sheet.

**Primary metrics.**

| stage | primary | why not the obvious alternative |
|---|---|---|
| segmentation | disagreement area in source px² | decoder IoU is inflated by cropping, so it cannot compare window sizes |
| detection | recall@5px, p90 centre error | mAP averages "found it" with "pointed at it accurately"; §3.4 shows those must stay separate |
| end to end | mask IoU ≥ 0.5 rate vs. annotation geometry | — |

At a matching radius of 25 px, matching is strictly one-to-one: 17 of 180
mounds have a neighbour within 25 px, so greedy matching would inflate recall.

### 5.2 v0.1 — the training loop works, and the supervision signal does not (≈3 pp)

The first version ran two experiments: an overfit-debug run on 5 samples
repeated 200× and a decoder-only run on the full split.

| experiment | result |
|---|---|
| overfit, 200 epochs | mean IoU 0.650, Dice 0.770, `pred_probability_max` = 1.0 |
| decoder-only, 10 epochs | train loss 0.966 → 0.327; val loss flat 0.85–0.96; **val IoU 0.088 → 0.097 (e3) → 0.025 (e10)** |

Read at face value this is a textbook overfitting curve, and the v0.1 report
concluded that the model "cannot generalize on current data", with root cause
"extreme class imbalance" at a target area ratio of ~0.00045.

**That conclusion was wrong, and the investigation that overturned it is the
first substantive result of the thesis.** The dataset builder emitted one
training sample per connected component — each with its own box and centre point
— but pointed every one of them at the *same whole-clip mask file*:

```python
"mask_path": f"masks/{split}/{filename}",   # identical for all components
```

Measured over the twelve v0.1 mask files: 172 components, and the worst clip has
all 56 of its samples sharing one 56-mound target. Weighted by sample count, the
average sample's target carried **≈30× more foreground than the single mound its
prompt indicated** (Σn²/Σn = 5168/172 = 30.0).

Four consequences, each of which should be a paragraph:

1. **This is a contradiction in the supervision signal, not noisy labels.** SAM's
   entire premise is that the prompt selects which object to segment. A label set
   that returns the same mask regardless of prompt *teaches the model to ignore
   the prompt*.
2. **The bbox loss crop was silently doing correctness work.** With a 32 px
   margin at logit scale the loss window is ~67×67 in a 256×256 map, about 7% of
   the image — which excluded most of the other mounds from the gradient. It was
   introduced as an optimisation and was in fact the only thing preventing
   outright collapse. In v0.2 it therefore becomes a genuine choice and is
   ablated rather than assumed.
3. **The reported metric scored a different task than the loss optimised.**
   `compute_prediction_stats` derived its valid-pixel mask from the *ignore* mask,
   which was empty in all twelve files, so IoU was computed over the whole image
   against the all-mounds target while the loss ran over a small crop. The
   reported mean ground-truth count of 30 px at 256×256 confirms it arithmetically:
   a single mound is 5–6 px at that resolution.
4. **Correcting it was expected to *lower* the scores**, and that lowering is the
   correct outcome rather than a regression, because the v0.1 numbers included
   credit for segmenting objects the prompt never asked about.

**Methodological point for the text:** the v0.1 report was not rewritten. It is
kept verbatim as a record of a different objective, with the findings recorded in
a separate document. Preserving superseded results rather than overwriting them
is a discipline applied throughout the project and is what makes the sequence in
this chapter auditable.

### 5.3 v0.2–v0.3 — instance targets, cross-sheet behaviour, and input resolution (≈4 pp)

**v0.2: one change only.** Each sample is supervised on the single mound its
prompt points at. Architecture, frozen encoder, optimiser, loss and source
annotations are unchanged. Verification over all 137 training samples: 0 targets
with more than one connected component, 0 empty at 1024, 0 empty at 256; mean
target foreground 85.9 px at 1024 and **5.4 px at the 256 logit grid**.

A 2×2 ablation over `foreground_bce_pos_weight` ∈ {20, 200} × loss crop
{on, off}:

| run | overfit final val IoU | held-out peak val IoU | @ epoch | held-out final |
|---|---:|---:|---:|---:|
| pw200_cropoff | 1.0000 | **0.6876** | 25 | 0.5845 |
| pw200_cropon | 0.6875 | 0.6838 | 25 | 0.5698 |
| pw20_cropon | 1.0000 | 0.6689 | 5 | 0.6021 |
| pw20_cropoff | 1.0000 | 0.6172 | 30 | 0.5219 |

**No held-out sample scored zero IoU** (0 of 34), and predicted area tracks
ground-truth area closely (6.35 px against 5.88 px).

The crop and the positive weight partly substitute for each other, which is why
the ablation had to be a full 2×2:

| positive weight | crop on | crop off | effect of crop |
|---|---:|---:|---:|
| 20 | 0.6689 | 0.6172 | **+0.0517** |
| 200 | 0.6838 | 0.6876 | −0.0038 |

At a low positive weight the crop clearly helps; at a high weight its effect
disappears into the noise. A two-run experiment at one fixed weight would have
reported whatever that weight happened to imply about the crop.

**The conclusion v0.2 supports** — stated carefully, because the absolute gap
0.097 → 0.688 is *not* a like-for-like improvement and must not be quoted as one:
with the same architecture, the same frozen encoder, the same 137 training
samples and the same annotations, corrected labels produce a model that segments
every mound on an unseen sheet. **The blocker was the supervision signal, not the
data volume and not the class balance.**

**v0.3 then asked whether 0.6876 was representative. It was not.** Leave-one-
sheet-out at full clip resolution:

| eval sheet | n | train samples | final IoU | oracle peak |
|---|---:|---:|---:|---:|
| `K-35-8-G-a` | 34 | 137 | 0.6021 | 0.6689 @ e5 |
| `K-35-51-B-a` | 120 | 51 | 0.3498 | 0.4911 @ e5 |
| `K-34-35-B-g` | 17 | 154 | 0.3714 | 0.5234 @ e15 |
| | | | **macro 0.441** | |

v0.2 had reported the easiest of the three sheets. A size-matched arm pinning
every fold to 51 training samples shows training-set size is worth a lot on its
own (0.6021 → 0.4567 for the same sheet) but does **not** explain the ranking:
fold C trains on the most data (154) and still scores 0.3714. **Sheet identity
dominates, and sheet variance is 2–7× configuration variance** (config spread
0.018–0.070; sheet spread 0.130 size-matched). The v0.2 exercise of separating
0.684 from 0.688 was measuring nothing.

**v0.3's second result corrects a metric artefact.** Bucketing the 137 training
samples by target size looks damning on IoU and is not:

| GT size at 256 | n | mean IoU | **mean disagreement area** |
|---|---:|---:|---:|
| 1–3 px | 16 | 0.508 | **2.88 px** |
| 4–5 px | 56 | 0.756 | **2.00 px** |
| 6–8 px | 57 | 0.736 | **2.39 px** |
| 9+ px | 8 | 0.727 | **3.25 px** |

Symmetric difference is **flat** — the model disagrees with truth over the same
~2 px of area on a 3 px mound as on a 12 px one. IoU falls on small targets
purely because that constant disagreement is divided by a smaller denominator.
Spearman size-vs-disagreement across four runs: −0.005 to +0.20. Pearson
correlation, which the plan had originally specified, reads 0.07–0.20 and would
have been written up as "weak size effect", missing both the real threshold below
~4 px and the artefact itself.

**v0.3's third result is the resolution finding**, and it is the one §3.3
predicted from token coverage:

| arm | window | decoder IoU | **disagreement area (source px²)** | GT area (source px²) | encoder tokens |
|---|---|---:|---:|---:|---:|
| A full clip | ~2300 px | 0.6021 | **253.9** | 479.0 | 0.61 |
| B medium | 1024 px | 0.7567 | **135.5** | 470.6 | 1.36 |
| C tight | 512 px | 0.8261 | **91.8** | 470.4 | 2.76 |

Both columns matter and they say different things. Decoder IoU alone would be
misleading — windowing magnifies the target from ~6 to ~118 px on the decoder
grid, so a **single epoch** of arm C already beat v0.2's 25-epoch best. Converting
to source pixels removes that: error falls **64%** on a ground-truth area that is
constant across arms (470–479, confirming all three see the same mounds).

**A diagnostic that must be reported, because it prevents a later mistake.**
v0.2 reported a maximum predicted probability of 1.0000 on every held-out
sample. That is float32 saturation, not confidence — sigmoid pins to exactly 1.0
past a logit of ~16, and the windowed runs reach logits of 88. **Maximum
probability must not be used as a confidence signal**; mean probability inside
the ground-truth region varies usefully (0.74–0.93) but the ground-truth region
is unknown at inference, so it is an evaluation diagnostic only.

### 5.4 v0.4 — prompt tolerance, failure anatomy, and the 512 px LOSO (≈4 pp)

**Step 1 — the tolerance curve.** No training; three arms over 8 compass
directions × offsets 0–300 source px on the 34-sample held-out sheet. Zero-offset
rows reproduce v0.3's resolution table exactly, which is the check that the pass
is faithful.

| offset (px) | full-clip err (px²) | 512 err (px²) | control: window only (px²) | control IoU ≥ 0.5 |
|---:|---:|---:|---:|---:|
| 0 | 253.9 | **91.8** | 91.8 | 1.00 |
| 5 | 287.4 | 140.5 | 96.8 | 1.00 |
| 10 | 365.0 | 255.0 | 95.5 | 1.00 |
| 15 | 479.0 | 444.6 | 96.8 | 1.00 |
| 25 | 702.4 | 734.7 | 98.4 | 1.00 |
| 50 | 767.1 | 744.2 | 106.2 | 1.00 |
| 100 | 704.2 | 734.2 | 100.6 | 1.00 |
| 200 | 666.8 | 718.6 | 100.9 | 1.00 |
| 250 | 684.1 | 670.1 | 94.2 | 1.00 |
| 300 † | 685.6 | 519.3 | 91.3 | 0.62 |

† Past 250 px the window begins cutting the mound off (mean GT area in window
470.4 → 432.2 → 291.0), so those rows are scored against a partly missing target.

The analysis of this table is in §3.4 and should be cross-referenced rather than
repeated: **the window has ~250 px of slack, the prompt has ~5**, and the
detector specification follows from it. One correction belongs here explicitly,
because it was made *after* v0.4 and changed the detector's target: v0.4
summarised the finding as "the prompt must land within ~15 px", reading the
offset at which the curve breaks as the tolerance. 15 px is where the segmenter
**fails** — the IoU ≥ 0.5 rate there is 0.02. The working band is ≤ 5 px.

Directional anisotropy was recorded and remains unexplained: averaged over
offsets 5–25 on the full-clip arm, northward offsets cost ~1.4× less than west or
southwest (355 vs 531 px²). It is reported because a single-direction sweep would
have drawn a curve 30% optimistic or pessimistic depending on its choice. A
hypothesis linking it to the companion height glyph present on 88% of mounds was
later **retracted** (§5.5).

**Step 2 — anatomy of the zero-IoU cases.** Fold A at epoch 50 has 19 of 120
samples at IoU exactly zero, 10 predicting nothing at all. Measured by
displacement rather than overlap:

| class | n | median distance to target |
|---|---:|---:|
| hits (IoU > 0) | 101 | 5.0 px |
| **near misses** (zero IoU, within 40 px) | **17** | 19.0 px |
| **wrong object** (zero IoU, beyond 40 px) | **2** | 868 / 1861 px |

So it is largely not collapse: at full-clip resolution a mound covers ~5 px on
the decoder grid, so a prediction displaced by one or two pixels scores exactly
zero while sitting on the right symbol. The ten zero-pixel cases are a
**threshold** problem — nine of ten have their peak logit within 6.5–39.6 px of
the target, with logit maxima from −13.2 to −0.3; mean logit max is 0.99 over the
19 zero-IoU samples against 19.6 over the 101 that score anything. Stated
precisely, because it changes what the fix would be: the activation is correctly
*placed* but **negative**. And it is acquired during training — 2 zero-IoU at
epoch 5 against 19 at epoch 50.

**Step 3 — the 512 px window under leave-one-sheet-out.** Protocol identical to
v0.3 except window (512), loss-crop margin (150, holding supervised ground area
constant) and embedding cache (off, forced by the window).

| fold | eval sheet | n | train | full-clip err | **512 err** | reduction | full-clip IoU | 512 IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | `K-35-51-B-a` | 120 | 51 | 373.8 | **116.4** | 69% | 0.3498 | 0.7930 |
| B | `K-35-8-G-a` | 34 | 137 | 253.9 | **91.8** | 64% | 0.6021 | 0.8261 |
| C | `K-34-35-B-g` | 17 | 154 | 324.5 | **103.1** | 68% | 0.3714 | 0.7574 |

| | full clip | 512 |
|---|---:|---:|
| macro mean error | 317.4 | **103.8 (−67%)** |
| macro mean IoU | 0.4411 | **0.7922** |
| **sheet spread, error** | **119.9** | **24.7** |
| sheet spread, IoU | 0.2522 | 0.0688 |
| fold A zero-IoU | 19 / 120 | **0 / 120** |
| IoU ≥ 0.5 rate, by fold | 0.30 / 0.71 / 0.29 | **0.99 / 1.00 / 1.00** |
| IoU ≥ 0.75 rate, by fold | 0.05 / 0.15 / 0.12 | 0.78 / 0.91 / 0.53 |

**This revises v0.3's central diagnosis rather than extending it.** The dominant
term in what v0.3 measured as *domain* variation was the *representation*.
Sheets differed most where the mound fell furthest below one useful encoder
token, and normalising the representation around the prompt removes most of that
difference — the cross-sheet spread falls from 119.9 to 24.7 px². A smaller
residual sheet effect remains; it is no longer the dominant failure mode.

Step 2's prediction holds exactly: fold A at 512 has 0 of 120 zero-IoU, 0
predicting nothing, and median hit displacement 0.99 px against 5.01 at full
clip. Both *wrong-object* cases are fixed too, by a mechanism worth naming
separately — a 512 px window **physically excludes** a distractor 868 px away, so
the model can no longer choose it.

**The branch point.** With a ground-truth prompt, 99–100% of instances clear
IoU 0.5 across all three sheets at a macro mean of 0.79. Segmentation given a
correct prompt is no longer the bottleneck. **The unsolved problem is producing
the prompt** — and step 1 now says how accurately.

### 5.5 v0.5 — the first detector, a classical reference, and a ground-truth error (≈5 pp)

**This section is the analytical core of the thesis.** It reports two
experiments that differ in exactly one thing — the annotations — and the
difference between them is a result about evaluation methodology, not about
models.

| | annotations | detector run | baseline run |
|---|---|---|---|
| Experiment 1 | `v0.0.1` | `v0_5_yolo26s` | `v0_5_template` |
| Experiment 2 | `v0.0.2` | `v0_5_yolo26s_gtfix` | `v0_5_template_gtfix` |

Model, hyperparameters, seed, folds and code are **identical**. Experiment 1's
numbers are reported exactly as measured and have not been regenerated: the model
declining to learn from incorrect ground truth, and thereby surfacing the error,
is the evidence, and deleting it would destroy it.

**The classical reference first.** Eight k-means template centroids over 32 px
intensity-normalised crops from the fold's *training* sheets, matched with
`cv2.matchTemplate` / `TM_CCOEFF_NORMED`, local maxima merged at 10 px:

| fold | eval sheet | best recall@5px | at FP/window | recall@5px at ~2 FP/window |
|---|---|---:|---:|---:|
| A | `K-35-51-B-a` | 0.781 | 85.0 | 0.070 |
| B | `K-35-8-G-a` | 0.882 | 72.6 | 0.176 |
| C | `K-34-35-B-g` | 0.389 | 54.3 | 0.111 |

Normalised cross-correlation generates proposals with essentially no
discrimination: precision is around 1% wherever recall is useful. **Scope this
claim carefully** — the baseline has no rotation handling, no multi-scale search
and a single global threshold, so a stronger hand-built matcher is *not* ruled
out. What is ruled out is the hope that the obvious cheap version is competitive.

**YOLO26-s, experiment 1, at confidence 0.25:**

| fold | eval sheet | n | recall@5px | 95% CI | p90 error | FP/window |
|---|---|---:|---:|---|---:|---:|
| A | `K-35-51-B-a` | 128 | 0.812 | 0.736–0.871 | 3.2 px | 0.21 |
| B | `K-35-8-G-a` | 34 | **1.000** | 0.898–1.000 | 3.7 px | 0.00 |
| C | `K-34-35-B-g` | 18 | **0.389** | 0.203–0.614 | 2.0 px | 0.02 |

**Localisation was solved on the first attempt**: p90 centre error 2.0–3.7 px
against a target of ≤ 5. And the recall curve is **flat in radius** on every
fold:

| fold | @5px | @10px | @15px | @25px |
|---|---:|---:|---:|---:|
| A | 0.859 | 0.883 | 0.883 | 0.883 |
| B | 1.000 | 1.000 | 1.000 | 1.000 |
| C | 0.389 | 0.389 | 0.389 | 0.389 |

Widening the acceptance radius five-fold buys at most three instances across 180.
**There is no near-miss population: the detector emits correct points or
nothing.** Two consequences follow immediately — further localisation work would
optimise nothing, and prompt-jitter augmentation (which had been planned,
conditional on p90 landing outside 5 px) is not merely untriggered but
*contraindicated*.

**Fold C, however, scored 0.389, and the investigation of why is the result.**
Reviewing those failures, the annotator found — and a second project annotator
confirmed — that **twelve symbols labelled `mound` on `K-34-35-B-g` are mills**,
and that **one genuine mound had never been labelled**.

| relabelled to `hard_negative_symbol` | detector behaviour, experiment 1 |
|---|---|
| 8 symbols in a corridor at x 1500–1651, y 574–1103 | all missed |
| 1 at (1454, 1739) | missed |
| 1 at (733, 1596) | missed |
| 1 at (2315, 446) | missed |
| 1 at (1275, 52) — the truncated edge symbol | **detected** (an honest exception) |

**Eleven of the twelve mislabels are precisely the annotations the detector
refused to fire on.** The eight on one clip lie in a narrow north–south corridor,
which is what a line of mills along a road looks like and is not what a scatter of
burial mounds looks like.

**And the unlabelled mound had already been found.** It sits at (1771, 813); the
detector had placed a candidate at **(1772.5, 812.0) — 1.8 px away, confidence
0.586, the second-strongest detection on the sheet** — and it was scored as a
false positive. On that sheet the evaluation charged the model **eleven false
negatives for correctly rejecting mills and one false positive for correctly
finding a mound.**

With the labels corrected:

| fold | eval sheet | v0.0.1 labels | v0.0.2 labels | p90 error |
|---|---|---:|---:|---:|
| A | `K-35-51-B-a` | 0.859 (110/128) | 0.898 (115/128) | 3.5 px |
| B | `K-35-8-G-a` | 1.000 (34/34) | 1.000 (34/34) | 3.7 px |
| C | `K-34-35-B-g` | 0.389 (7/18) | **1.000 (7/7)** | 2.0 px |
| | **macro** | **0.749** | **0.966** | |

#### Why this is a result and not an anecdote

Four independent lines of support, and the thesis should give each its own
paragraph:

1. **The metric design localised it.** Fold C had the *best* p90 of the three
   folds (2.00 px) alongside the worst recall (0.389). A detector genuinely
   failing on a sheet degrades on both axes. A single aggregate score would have
   shown one mediocre number and given nobody a reason to look at the annotations.
   This is the practical payoff of the separation argued in §3.4.
2. **It reproduces under an unrelated method.** The template baseline shares no
   code path with YOLO and has no capacity to overfit. Re-run unchanged on the
   corrected labels it moves on the same sheet from **0.389 to 0.857**. Two
   unrelated methods were penalised by the same twelve annotations in the same
   place.
3. **A domain rule predicts it.** A mound cannot be crossed by a **surface**
   watercourse, and all twelve mills carry `crossed_by_water_line`. Water mills
   sit on watercourses, so the attribute was true of them, and it is mounds that
   the water excludes. Applying the rule to the rest of the corrected export
   flagged two more and taught it its limits: one is a real mound whose attribute
   was set in error, and the other is a real mound with a genuine **underground**
   crossing — a pipe, which can run beneath anything. The attribute did not
   distinguish surface from underground, so the rule was a review hint, not a
   mechanical check. §5.8 reports the schema change that made it mechanical.
4. **The architecture predicted which stage could find it.** *MapSAM is a
   conditional segmenter and does not decide semantics.* Given a prompt on a
   printed symbol it outlines that symbol, and a mill is a printed symbol. The
   mislabels cost v0.1–v0.4 almost nothing and were **invisible to every metric
   those versions reported**. It took the detector — the first stage that must
   decide *whether* something is a mound — to expose them. A conditional segmenter
   cannot audit its own labels; a detector can. That is a property of the
   architecture, not of this dataset.

#### A retraction, reported as such

Experiment 1's original write-up offered a hypothesis: that fold C's failures
concentrated on mounds lacking a companion height glyph, which appeared to
corroborate v0.4's unexplained directional anisotropy. **That hypothesis is
withdrawn.** Mills sit along roads and carry no mound height marks, so "no height
mark, crossed by a road" described the mislabelled class, not a hard visual case.
No mechanism in the model needs explaining, and v0.4's anisotropy is unexplained
again and no longer supported by this evidence.

### 5.6 v0.4r — re-running segmentation against corrected labels (≈1.5 pp)

Twelve of v0.4's 171 training samples stopped being mounds, so the 512 px LOSO
arm was re-run with configs identical but for the dataset root. The fold
structure isolates two questions, because the corrected sheet is a *training*
sheet for folds A and B and the *evaluation* sheet for fold C:

| fold | train | eval | what changed | IoU v0.4 → v0.4r |
|---|---|---|---|---|
| A | 51 → 40 | 120 → 120 | training only | 0.7930 → 0.7913 |
| B | 137 → 126 | 34 → 34 | training only | 0.8261 → 0.8288 |
| C | 154 → 154 | 17 → 6 | evaluation only | 0.7574 → 0.8137 |
| | | | **macro** | **0.7922 → 0.8113** |

Two conclusions, and the second one is a warning:

**Training-label noise cost segmentation nothing measurable.** Folds A and B
moved −0.0017 and +0.0027 with their evaluation sets untouched — inside normal
run-to-run variation.

**Fold C's gain is an evaluation-population change, not better segmentation.**
Its training set is byte-identical between the two runs. What changed is that
eleven small symbols left its evaluation set and the mean ground-truth area rose
from 381.2 to 550.0 source px² — and §5.3 established that smaller symbols score
worse. Reporting +0.056 as an improvement would have been wrong.

### 5.7 End-to-end measurement (≈2.5 pp)

Detector candidates at confidence 0.25 → prompts → v0.4r decoder → masks scored
against each annotation's geometry.

| fold | sheet | n | IoU ≥ 0.5 | IoU ≥ 0.75 | mean IoU | missed | false masks |
|---|---|---:|---:|---:|---:|---:|---:|
| A | `K-35-51-B-a` | 128 | 0.875 | 0.586 | 0.6820 | 0.109 | 60 |
| B | `K-35-8-G-a` | 34 | **1.000** | 0.529 | 0.7482 | 0.000 | 0 |
| C | `K-34-35-B-g` | 7 | **1.000** | 0.857 | 0.7717 | 0.000 | 3 |

**Macro IoU ≥ 0.5: 0.958.**

**The end-to-end loss is detector recall and essentially nothing else.** Fold A's
missed rate is 0.109 against a detector miss rate of 0.117 at the same threshold.
Restricting to mounds the detector found isolates the cost of using a
detector-derived prompt instead of a ground-truth one:

| fold | mean IoU, found only | v0.4r with GT prompt | prompt cost |
|---|---:|---:|---:|
| A | 0.7657 | 0.7913 | −0.026 |
| B | 0.7482 | 0.8288 | −0.081 |
| C | 0.7717 | 0.8137 | −0.042 |

Fold B is the clean case — 34 of 34 found, so no recall loss at all — isolating
the prompt cost at **0.081 IoU at p90 3.66 px**. Real but modest, and in the
direction §5.4's tolerance curve predicts.

**Box scale costs mask precision, not mask detection.** Prompting with a
canonical median-sized box on the detector's *point* beats prompting with the
detector's own box at IoU ≥ 0.75 on all three folds (+0.016, +0.059, +0.286),
while the two are within 0.006 at IoU ≥ 0.5 and in mean IoU. The direction is
consistent 3 of 3 but the magnitudes are weak — fold A's +0.016 is two instances
of 128 — so it should be presented as **a cheap default worth adopting, not an
established effect**.

### 5.8 v0.6 — transfer to an unseen corpus twenty times larger (≈4 pp)

**The corpus.** 60 new sheets, measured with `gdalinfo` before any model touched
them:

| | value |
|---|---|
| sheets | 60 |
| size | 4682–5112 × 4328–4635 px; 20.3–23.4 Mpx each; **1318 Mpx total** |
| pixel size | 2.1061–2.1566 m/px (2.4% spread) → **~300 dpi at 1:25,000** |
| ground extent | ~10.4 × 9.5 km per sheet |
| CRS | EPSG:25835 on all 60, embedded in the GeoTIFF |
| windows at 512/384 | 132–156 per sheet, **9,271 total** |
| vs. annotated corpus | 12 clips, 66.4 Mpx, 480 windows → a **20-fold** area expansion |

**Permanent splits, grouped on the 1:100,000 parent.** 45 train / 13 validation
/ 5 test over 63 sheets. Grouping on the parent rather than the 1:25k sheet id is
not cosmetic: `K-35-51-B-g` and the annotated `K-35-51-B-a` are **adjacent
quadrants of the same 1:50k sheet** — same terrain, survey campaign, print run
and scan batch — so splitting on the sheet id is a weaker separation than the ids
suggest. The grouping forces one consequence worth stating: `K-35-39-A-g` shares
a parent with two frozen sheets and therefore lands in test although it is not
annotated and contributes no evaluation instances. Training on it would be
exactly the leak the grouping exists to prevent.

**The blind sweep.** The v0.5 detector, unchanged, over the 56 sheets left after
the four frozen ones were set aside, with all three fold checkpoints — because
every fold is equally out-of-domain on a sheet none has seen. 8,671 windows over
1,232.4 Mpx.

| | fold A | fold B | fold C |
|---|---:|---:|---:|
| candidates at conf 0.25 | 2,593 | 1,613 | 2,062 |
| per window | 0.299 | 0.186 | 0.238 |
| per Mpx | 2.10 | 1.31 | 1.67 |
| **per Mpx, excluding `K-34-47-G-v`** | **1.35** | **1.22** | **1.36** |

Four findings:

1. **The domain transferred.** Candidate density is 1.31–2.10 per Mpx against
   2.55 mounds per Mpx measured on the annotated clips: the same order of
   magnitude and *lower*, which is what whole sheets containing uninformative
   terrain should do to a density measured on clips chosen for annotation.
2. **The review burden is 0.17–0.30 candidates per window against a projection
   of 0.36–0.61** — 1,613–2,593 candidates rather than the ~3,300 projected. The
   projection was pessimistic in the direction that costs reviewer time, and it
   has been replaced by a measurement.
3. **Three independently trained checkpoints land within 0.14 candidates per Mpx
   of each other over 55 unseen sheets.** This is the strongest available
   evidence that the transfer is real rather than an artefact of one checkpoint;
   per-sheet agreement at 10 px is median 0.766 / 0.813 / 0.904 across the three
   pairings.
4. **One sheet is an order of magnitude out, and it is central Sofia.**
   `K-34-47-G-v` returns 957 / 140 / 418 candidates where no other sheet exceeds
   110. The three checkpoints disagree there because none has any basis for a
   decision inside a city block. **That is a description of missing training data,
   not of a bug**: dense urban fabric is a regime the training corpus contains
   none of. The sheet is set aside from the delivery rather than silently
   included.

**This measures candidate density, not precision.** The new sheets carry no
ground truth, so what fraction of the candidates are false is unknowable until a
blind annotated set exists. Reporting it as a false-positive rate would be exactly
the circular measurement the blind set is there to prevent.

**Pipeline health measured without ground truth.** The production monitor
proposed in §3.2 — the distance between the detector's point and the mask's
centroid — needs no labels and is directly interpretable against §5.4's tolerance
curve:

| | value |
|---|---:|
| detector point → mask centroid | median **2.01 px**, p90 **4.26 px** |
| instances above the ~5 px band | **5.2%** (of 4,100) |
| symbol outline area, median | **438 px²** (annotated corpus: 445) |

Both agree with the annotated corpus, on 4,100 instances from sheets no model
had seen.

**The delivery.** 56 sheets as GeoPackages in EPSG:25835, 15 MB total, each
holding two layers — `mound_points` (the archaeological location, what field work
navigates to) and `mound_symbols` (the printed symbol, joined on `mound_id`).
4,100 proposed locations at confidence 0.05, 2,062 at 0.25; per sheet min 7,
median 39, max 1,651. Reviewer instructions in English and Bulgarian.

**Verification of the delivery went the whole way back** — GeoPackage → EPSG:25835
→ sheet pixel → clip pixel — and rendered the result over the source imagery. The
outlines land on the printed symbols and the round trip returns the original
pixel with zero error. This was done on `K-35-21-G-a` and on `K-34-10-A-g`, the
latter chosen deliberately because at easting 148,562 it sits at the far-western
extreme of the corpus, where UTM 35N distortion is largest.

**One caveat visible in that second render, and it belongs in the thesis.**
`K-34-10-A-g`'s candidates sit on small circular symbols rather than the familiar
starburst. Whether they are mounds, trig points or elevation dots is not something
the geometry can settle — and is exactly what the review is for.

**The water-line rule became mechanical.** With `water_line_crossing` split into
`none` / `surface` / `underground` / `unreviewed` (§4.3), the check is one line —
a mound marked `surface` is a data error, a mound marked `unreviewed` is one
nobody has checked — and **both are empty on export v0.0.3**.

### 5.9 Methodological results (≈2 pp)

Three results in this chapter are about *how to measure*, not about models, and
they should be collected into their own subsection because they are the part of
the work most likely to transfer to another project.

**Checkpoint discipline, with the leak quantified.** Fold A's recall swings ±11
points between saved checkpoints with no trend — as wide as its confidence
interval. Evaluating `last.pt` rather than `best.pt` avoided reporting **0.891
instead of 0.812**: a direct measurement of exactly how large the leak would have
been, and the concrete argument for the permanent validation split that §5.8
finally established.

**Keeping superseded results rather than overwriting them.** Pre- and
post-correction artefacts are kept side by side throughout. The experiment that
exposed the annotation error is only reproducible because the run against the
incorrect labels still exists.

**Corrections carried forward without re-measurement are a failure mode of their
own.** The phrase "seven pre-existing test failures" appeared verbatim in four
documents. Measured, it is **three**, all environment-coupled assertions in one
test module. The live document states the measurement; the historical ones carry
a dated correction rather than a silent rewrite.

### 5.10 What the results establish, and what they do not (≈2 pp)

**This subsection is not a disclaimer. It is a result, and it should be written
with the same care as the tables.**

Established, within scope:

- Given a prompt within ~5 px, mound symbols are segmented at **99–100% IoU ≥ 0.5
  on every available sheet**, macro decoder IoU 0.7922 (0.8113 on corrected
  labels).
- A detector meets the derived specification on the first architecture tried:
  **p90 centre error 2.0–3.7 px, macro recall@5px 0.966**, at 0.00–0.21 false
  positives per window.
- Composed end to end, **macro IoU ≥ 0.5 is 0.958**, and the residual loss is
  detector recall rather than segmentation quality.
- The system **transfers to 56 unseen sheets** with candidate density and symbol
  area consistent with the annotated corpus, and with three independent
  checkpoints agreeing to within 0.14 candidates per Mpx.
- A **learned detector decisively outperforms the classical reference**: 1.000
  recall at 0.00–0.02 FP/window against comparable recall only at 55–74 FP/window
  for template matching.

**Explicitly not established:**

| claim | why not |
|---|---|
| that these numbers generalise | three annotated sheets, one series, LOSO, **no frozen test split**; fold C contributes 7 evaluation mounds with an interval of 0.646–1.000 |
| any precision or recall figure on the new sheets | they have no labels |
| that the corrected labels are complete | one adversarial review of one sheet found twelve errors and one omission; the other two sheets have had no equivalent pass, and attribute errors are invisible to every metric reported here |
| that the pipeline works on a different cartographic source | 63 sheets of one Bulgarian 1:25k series cannot test it; **sixty more sheets of the same series cannot retire this caveat** |
| that YOLO26-s is the right architecture | first one tried; it cleared the target immediately, and the flat recall curve suggests remaining headroom is not in the box regressor |
| any false-positive rate at sheet scale on the annotated corpus | the 12 clips were selected for annotation, so rates are reported per window and per Mpx and labelled as measured on annotation-selected clips |
| that the urban failure mode is a defect | it is a description of missing training data |

**The single sentence to carry into the Заключение:** the binding constraint is
no longer any model — it is the amount and correctness of annotated data.

---

## 6. Заключение — conclusions and directions for development

**Target: 4 pages. Status: drafted.**
**Sources:** `docs/mapsam/REPORT_2026-09.md` §8, `docs/mapsam/v006/RESULTS.md`.

The указание asks the Заключение to analyse and generalise the results and to
offer the author's view on their use and further development.

### 6.1 Conclusions

Map each back to the задачи in §1.3 — the рецензент is explicitly asked *"в каква
степен е изпълнено заданието"*, so the mapping should be visible.

1. **A two-stage architecture is required, not merely convenient.** A promptable
   foundation segmenter is conditional by construction: it answers "what is at
   this point", never "is there a mound here". The consequence measured in §5.5
   is sharp — a conditional segmenter cannot audit its own labels, and four
   versions of it were blind to twelve mislabelled annotations that the first
   detector exposed immediately. *(tasks 3, 5)*
2. **The input representation, not the loss and not the model capacity, governs
   accuracy in this domain.** A mound symbol covers 0.61 encoder tokens at full
   clip resolution and 2.76 in a 512 px prompt-centred window; the change reduces
   absolute error by 64–69% on every sheet and collapses the cross-sheet spread
   from 119.9 to 24.7 px². What v0.3 diagnosed as map-domain variation was
   predominantly a representational artefact. *(tasks 3, 4)*
3. **The composition constraint between stages can be measured, and it is the
   design's most transferable output.** The prompt-tolerance curve
   (1.00 / 1.00 / 0.69 / 0.02 at 0 / 5 / 10 / 15 px) converts a vague requirement
   into a number: p90 centre error ≤ 5 source px, primary metric recall@5px. The
   window, by contrast, tolerates 250 px of displacement. A detector here must
   *point accurately*, not crop well. *(task 4)*
4. **Metric design is what made the results legible.** Keeping recall and
   localisation as separate axes is what surfaced the ground-truth error, because
   the failing fold had the best localisation in the project alongside the worst
   recall — a pattern no genuine model failure produces and no aggregate score
   can show. *(tasks 4, 6)*
5. **A learned detector decisively beats the classical reference** — 1.000 recall
   at 0.00–0.02 false positives per window against comparable recall only at
   55–74 FP/window for normalised cross-correlation — while remaining small
   enough (YOLO26-s) not to overfit 180 positives. *(task 5)*
6. **Composed end to end, the system produces archaeologically usable output**:
   macro 0.958 at mask IoU ≥ 0.5, with the residual loss attributable to detector
   recall rather than segmentation quality. *(task 6)*
7. **It transfers.** Over 56 unseen sheets — a 20-fold area expansion — candidate
   density, symbol area and detector-to-centroid distance all agree with the
   annotated corpus, and three independently trained checkpoints agree with each
   other to within 0.14 candidates per megapixel. *(task 7)*
8. **The binding constraint has moved off the model entirely**, onto the amount
   and correctness of annotated data. Every model-side target set in the project's
   plans is met; nothing further about robustness can be learned from three
   annotated sheets. *(tasks 2, 8)*

### 6.2 Contributions

State these plainly; a рецензент is asked specifically about личен принос.

- An **experimentally derived composition specification** between a promptable
  segmenter and a detector, which the literature does not provide.
- A **case study in label-error discovery by architectural asymmetry** — a
  detector auditing a segmenter's training labels — with four independent lines
  of corroboration and a documented retraction of the first hypothesis offered.
- A **complete working pipeline** from scanned archival sheet to reviewable
  GeoPackage, with a versioned annotation schema, permanent leak-free splits
  grouped on the 1:100k parent, and a review return path.
- A **delivery to a working archaeological team**: 56 sheets, 4,100 proposed
  mound locations with symbol outlines, in the team's own GIS software.

### 6.3 Directions for development

Ordered by what currently blocks what.

1. **Annotate the frozen blind set.** Four sheets are untouched and the splits
   exist; what is missing is a human annotating them with no model output
   visible. It is the only thing standing between this project and its first
   *test* result, and until it exists every number in the thesis is development
   evidence. Nothing else on this list is as valuable.
2. **A second cartographic series.** Sixty more sheets of the same Bulgarian
   1:25k series cannot retire the cross-cartographic caveat, and only a different
   series can.
3. **Review the detector's false positives as a queue, not as an error rate.**
   One of them was already a real mound. This is simultaneously the cheapest
   source of new labelled data and of further annotation corrections.
4. **Retrain on the expanded corpus**, once the blind set has been annotated, so
   the frozen sheets are touched once rather than twice.
5. **Attribute extraction**, which is where segmentation earns more than a
   prettier mask: a tight symbol mask dilated by N px gives a controlled ROI
   against which contour, road and grid intersections can be asked. Labels already
   exist on all 714 annotations; with 180 positives the frequent attributes are
   learnable and the rare ones are not.
6. **A segmentation-quality gate.** Two candidate signals are identified and
   neither is validated: SAM's own `iou_predictions` head, which is currently
   discarded and must be validated before it is trusted because it was never in
   the loss while the decoder around it moved; and mask stability under small
   prompt perturbations, which is the more expensive option but measures exactly
   the dominant failure mode — prompt correctness.
7. **Fix georeferencing**, currently at a 30.5% pass rate with aspect-ratio error
   on frame detection accounting for 56% of failures. It is off the critical path
   by design, but it is what will be needed for sheets that do not arrive
   pre-georeferenced.
8. **Urban regimes.** The Sofia sheet is not a defect to patch but a gap to fill
   with training data.

**Explicitly not worth doing**, with reasons — this list is as informative as the
one above and shows evidence being used to *stop* work:

| deferred | reason |
|---|---|
| prompt-jitter augmentation | contraindicated: the recall curve is flat in radius, so there is no near-miss population to widen tolerance for |
| a second-stage classifier | 0.00–0.36 false positives per window leaves nothing to filter |
| further localisation work | p90 is 2.0–3.7 px against a 5 px requirement; the flat recall curve says the headroom is not in the box regressor |
| Faster R-CNN / RT-DETR comparisons | no failure exists that they could address |
| 256 / 384 px windows | deprioritised on **value**, not on evidence — nothing shows a tighter window cannot help, and even 512 puts under three tokens on a mound |
| unfreezing the encoder | the frozen encoder is not what limits accuracy at 512 px |

---

## 7. Използвана литература / References

**Target: 4 pages, ~45–60 entries. Status: to be assembled — see §0.5.**

**Formatting, per the указание:** numbered; **Cyrillic titles first, then
Latin**, each group alphabetical by author surname; normative acts, standards
and company materials last. Cite in the text with the number in square brackets,
e.g. `[5]`. Example of the required entry form:

```text
3. Стефанов Н., Наръчник по токозахранващи устройства, Изд. "Техника",
   София, 1991.
5. Юдов Д., В. Вълчев, Преобразувателна техника, Изд. "Онгъл", Варна, 2005.
```

### Verification list

Every entry must be checked against the publisher before it is cited. Grouped by
the section that needs it:

| § | works to verify |
|---|---|
| 2.1 | LiDAR-based mound detection in SE Europe; A. Sobotkova and I. Berganzo-Besga's own publications; AGKK provenance of the *Balance of Earth 1969* series |
| 2.2 | GREC / IJDAR map-graphics recognition literature; OpenCV `matchTemplate` documentation |
| 2.3 | U-Net (Ronneberger et al.); Faster R-CNN (Ren et al.); FPN (Lin et al.); YOLO lineage; DETR (Carion et al.); RT-DETR (Zhao et al.); **the Ultralytics release actually used — 8.4.150 — and its AGPL-3.0 licence** |
| 2.4 | **SAM (Kirillov et al.)**; **SAM 2 (Ravi et al.)**; **the MapSAM paper on adapting SAM to historical maps — the direct antecedent, and the module is named after it, so this reference must be exact**; LoRA; SAM-in-remote-sensing surveys |
| 2.5 | CVAT; COCO (Lin et al.); active-learning surveys; learning with noisy labels |
| 2.6 | GDAL/OGR; RFC 7946 (GeoJSON); OGC GeoPackage; EPSG:25835 registry entry |

**Own / project sources to cite as such:** `docs/AI_ARCHEO_TOPIA_DMP.md` (NAIM-BAS,
ФНИ contract КП-06-Н100/5); the project site `https://archaeotopia.naim.bg/`; the
repository's own versioned experiment reports, which should be cited as internal
technical reports with their dates so that Chapter 4's numbers are traceable.

---

## 8. Приложения / Appendices

**Target: 8 pages.**

| app. | content | source |
|---|---|---|
| А | Annotation label schema in full — three classes, 15/16/16 attributes, value vocabularies | `annotation/cvat/labels.json` |
| Б | Annotation protocol, abridged: class definitions and decision rules | `docs/annotation/PROTOCOL_EN.md` |
| В | Training configuration, one YAML in full, annotated | `configs/mapsam/mapsam_v0_4_loso512_foldB_pw20.yaml` |
| Г | Key source listings: `MapSamDataset.__getitem__`, `combined_bce_dice_loss`, `apply_prompt_jitter`, `SheetReference` | `src/archeo_topia/` |
| Д | Full reproduction commands per experiment version | the `## Reproduce` block of each `RESULTS.md` |
| Е | Per-sheet inventory of the 60-sheet corpus | `docs/mapsam/v006/INPUT_INVENTORY.md` |
| Ж | The canonical output record, and the GeoPackage layer schema | `docs/architecture/TARGET_PIPELINE.md`, `formats/features.py` |
| З | Reviewer instructions delivered to the team, EN and BG | `data_lake/cleaned/gis/v0_6_foldC/README_{EN,BG}.md` |

---

## 9. Съдържание / Table of contents

**Placed last, after the appendices, per the указание.** Generate it when the
text is final.

---

## 10. Working notes — what to do next on this draft

Not part of the thesis. Delete before submission.

### Blocked on the научен ръководител

1. **Amend Глава 2's title** — FAISS out (§0.2). Also settle надгробни vs.
   селищни могили, and whether isolines are in scope.
2. **Confirm the language** for the submitted copy. This draft is English; a
   Bulgarian translation is a separate pass.
3. **Confirm the framing in §0.3** — a methodological study with a working
   prototype, with no frozen test result. This is the single decision that most
   affects how Chapter 4 reads.

### Blocked on nothing — do these next

4. **Chapter 1.** Needs a session with literature access. §2 gives the structure,
   the argument per section and the verification list; the writing is then
   mechanical.
5. **Figures.** 5 for Chapter 2 (§3.7), 6 figures + 4 listings for Chapter 3
   (§4.9). Figures 2.3 (token coverage), 2.4 (tolerance curve) and 3.6 (QGIS over
   the raster) carry the most argument and should be made first.
6. **Expand Chapter 4's prose.** All the numbers are in place; what is missing is
   the connecting text between tables. Budget ~24 pages against roughly 10 of
   material here.
7. **Numbering sweep.** Convert this document's §-numbering to the thesis's
   chapter numbering (Увод, 1, 2, 3, 4, Заключение) once the structure is frozen.

### Standing rules for this document

- **Never quote a number that is not in a repository document**, and cite the
  document. Every figure in Chapter 4 is traceable to `docs/mapsam/*`.
- **Never present a leave-one-sheet-out figure as a test result.** Label
  development evidence as such, every time.
- **Keep the retractions in.** §5.2's superseded conclusion, §5.4's retracted
  anisotropy hypothesis and §5.6's non-improvement are evidence of method, and a
  рецензент reads them that way.
