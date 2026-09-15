# MapSAM v0.6 — plan: ingest 60 sheets, and build the first frozen test set

Written at the end of v0.5. Read `../v005/RESULTS.md` and
`../REPORT_2026-09.md` first; this plan assumes their findings and does not
repeat the evidence for them.

**v0.6 is a data version, not a model version.** v0.5 met every model-side
target it set: p90 centre error 2.0–3.7 source px against a ≤5 px requirement,
recall@5px of 1.000 / 0.898 / 1.000 by sheet, and macro 0.958 at mask IoU ≥0.5
end to end. Nothing in the pipeline is currently the bottleneck. What is
missing is data, and 60 newly acquired sheets are the answer to that — *if*
they are ingested in a way that produces a measurement rather than a larger
pile of development evidence.

## The two facts that shape this plan

**The new sheets are the same series.** Bulgarian archival topographic maps at
1:25,000, the same series and scale as the three already annotated. The sheet
nomenclature is Soviet-style — `K-35-51-B-a` is the 1:25k quadrant of the 1:50k
sheet `K-35-51-B` — and earlier drafts misread that as the scale; see
`../DATASET.md`. This is a **within-series expansion**, not the
cross-cartographic test the v0.4 and v0.5 plans asked for. That caveat stays
open and still needs a genuinely different source to retire. The upside is that
transfer should be strong, which is what makes model-assisted annotation viable
from the first day.

**Annotation time is the scarce resource.** The plan below spends as little of
it as possible, and spends what it does spend on the one thing that cannot be
bought any other way.

---

## The one non-negotiable: annotate a blind set first

Every number in v0.1–v0.5 is development evidence. There is no frozen test
split, and each results document has had to say so. Step 8 of the v0.5 plan has
been blocked on data since the beginning. It is not blocked any more.

**Annotate one to two sheets completely blind — no model output visible — and
freeze them.** This must happen *before* any model-assisted pass, because the
contamination is irreversible: once a reviewer has seen proposals on a sheet,
that sheet can never serve as an uncontaminated test set again.

### Why assisted annotation cannot produce the test set

If the reviewer only sees model proposals, the labels inherit the model's blind
spots. Anything the detector systematically misses never enters the ground
truth, and recall measured against those labels is circular — a high number
meaning "the model agrees with itself".

This is not hypothetical here. One sheet produced evidence in both directions:

- the detector found a mound that had never been labelled, at 1.8 px and
  confidence 0.586 — assisted annotation would have captured it;
- the detector missed a real mound obscured by pencil and border/road lines,
  and fold A's `blurred_or_bad_print` recall is 0.750 against 0.898 overall —
  assisted annotation would have entrenched that gap permanently.

The second failure mode is the one that matters, and it is invisible from
inside an assisted workflow.

### Sizing it

Fold C's corrected evaluation set has 7 mounds and a 95% interval of
0.646–1.000, which is why nothing in v0.5 leans on it. Wilson intervals at an
observed recall of 0.95:

| mounds in the test set | 95% CI half-width |
|---:|---:|
| 20 | ±11.4 pts |
| 30 | ±9.7 pts |
| 50 | ±6.2 pts |
| 75 | ±5.4 pts |
| 100 | ±4.5 pts |
| 150 | ±3.7 pts |

**Target 75–100 mounds.** Below 50 the interval is too wide to distinguish a
working detector from a degrading one; above 150 the returns flatten quickly
against reviewer time. The existing sheets carry 128, 34 and 7 mounds, so one
dense sheet or two sparse ones should reach the target — and step 1 below says
how to pick them without guessing.

---

## Step 1 — Blind run over the new sheets, before any annotation

Costs no annotation time and settles three questions. Run the v0.5 detector
(`artifacts/detection/v0_5_yolo26s_gtfix`) over the new sheets at confidence
0.05 and record, per sheet, candidate count, candidates per megapixel, and the
confidence distribution.

**Question 1: did the domain transfer?** Compare candidates per megapixel
against the current corpus: 12 clips, 66.4 Mpx, 169 mounds — a mound density of
2.55 per Mpx, on clips selected for annotation and therefore denser in content
than a whole sheet.

| outcome | reading | action |
|---|---|---|
| broadly similar rate, plausible symbols in a sample of 20 crops | transfer succeeded | proceed to step 2 |
| near zero | scale or appearance mismatch | stop; check scan DPI before anything else |
| an order of magnitude high | the detector is firing on unfamiliar clutter | annotate more before trusting proposals |

**Question 2: what is the review burden, measured?** This number governs the
whole annotation budget and nobody has it. The current corpus tiles at **7.23
windows per Mpx** at window 512 / stride 384, and measured false-positive rates
are 0.61 (fold A), 0.01 (fold B) and 0.14 (fold C) per window at confidence
0.05. But those come from annotation-selected clips. A whole sheet contains
large uninformative regions, so the per-window rate could move in either
direction — `../architecture/TARGET_PIPELINE.md` stage 0 flags exactly this and
it has never been measured. **Measure it on a full sheet before committing a
reviewer to 60 of them.**

**Question 3: which sheets to annotate blind?** Pick them by candidate density
rather than arbitrarily, so the blind set reaches 75–100 mounds in the fewest
sheets. This is the only place in the plan where model output touches the blind
set, and it is safe: choosing *which* sheet to annotate does not bias *what* is
annotated within it.

### Prerequisite: confirm the input regime

Two checks, both minutes, both of which invalidate step 1 if skipped.

**Scan resolution.** The detector is tuned to a ~25 px symbol — mound boxes run
8–33 px with a 25×23 median — inside 512 px windows. If the new scans are at a
different DPI the symbol could be 12 px or 50 px, and the window and stride must
be rescaled to keep the symbol at roughly the same fraction of the window.
`gdalinfo` plus one look at a mound symbol settles it. **Do not run step 1
before this is known**; a mismatch here would produce a false negative on
"did the domain transfer".

**Georeferencing.** All 60 sheets are hand-georeferenced, which removes the
30.5% auto-georef pass rate from the critical path and unblocks GIS output
(`TARGET_PIPELINE.md` stage 5). 20 of 60 carry a `.tif.aux.xml` sidecar, which
normally holds statistics and projection rather than the transform, so the
other 40 most likely carry it internally — confirm with a `gdalinfo` sweep.
Nothing in steps 1–4 depends on it, by design: all stages operate in source
pixel space.

## Step 2 — Annotate the blind set

One to two sheets chosen in step 1, full annotation protocol
(`../annotation/PROTOCOL_EN.md`), **no model output visible to the annotator**.

Freeze it. Open it at milestones only. It is not a validation set, it is not
for checkpoint selection, and it is not for hyperparameter tuning — v0.5 had to
evaluate `last.pt` rather than `best.pt` precisely because no leak-free
selection surface existed, at a measured cost of reporting 0.812 instead of the
0.891 `best.pt` would have claimed.

## Step 3 — Model-assisted annotation for volume

Everything else. Three choices make this pay:

**Use a low confidence threshold, 0.05–0.10.** Rejecting a bad proposal takes
seconds; finding a missed mound takes minutes. The asymmetry favours recall
heavily, and precision is the reviewer's job here, not the model's.

**Import masks, not boxes.** Run accepted candidates through the v0.4r decoder
and import RLE masks, so the reviewer adjusts an outline rather than drawing
one. That is the pipeline's actual value as an annotation assistant, and it is
the only stage where the 0.958 end-to-end IoU≥0.5 rate turns directly into
saved human time. `src/export_sam2_to_coco.py` already exists and its header
records that COCO imports into CVAT more reliably than CVAT XML; a small adapter
from the v0.5 detector and end-to-end output to that format is the only new code
this step needs.

**Import false positives pre-labelled as `hard_negative_symbol`.** They are
confusable symbols by construction. The class currently holds 530 hand-picked
confusables; the detector will supply thousands at no annotation cost, and the
reviewer only has to delete the ones that are actually mounds.

### Record provenance per annotation

Add an attribute with values `model_proposal_accepted`,
`model_proposal_corrected` and `human_added`. It costs one field and it is the
only way to quantify the bias this step introduces: the `human_added` rate on
assisted sheets, compared against the blind set, **is** the bias estimate. Without
it, the question is unanswerable after the fact.

## Step 4 — Permanent grouped splits, and the first real test result

With 63 sheets, split by sheet into train / validation / frozen test. The
validation split finally makes leak-free checkpoint and hyperparameter
selection possible; the frozen test is the blind set from step 2.

Then re-run v0.5's detector and v0.4r's decoder against the new splits and
report, for the first time in this project, a **test** result rather than
development evidence. Everything needed to do this already exists and is
committed.

## Two schema changes to make while the protocol is open

**Split `crossed_by_water_line` into surface and underground variants.** A
mound cannot be crossed by a surface watercourse — water runs along terrain lows
and a mound is raised — but an underground line is a pipe and can run beneath
anything. The current attribute conflates the two, which is why the rule is a
review hint rather than a mechanical check. Splitting it restores the check.
This is the cheapest schema change available with the largest validation
payoff: of 169 mounds, 2 currently carry an attribute whose meaning is
ambiguous, and the same rule correctly identified all 12 mislabelled mills.

**Record scale and series explicitly per sheet** in the dataset metadata. This
session found a scale error that had propagated through nine documents because
the nomenclature was read as the scale. A field makes that unrepeatable.

---

## Constraints carried forward

- Window 512, stride 384 — **subject to rescaling if the new scans differ in
  DPI**, which is the one parameter this plan may have to change.
- Split by sheet, never by sample.
- The blind set is never shown model output before it is annotated.
- Everything operates in source pixel space; georeferencing stays off the
  critical path even though it is now available.
- Report recall@5px as primary and p90 centre error as the acceptance figure,
  never mAP. The metric hierarchy in `../v005/PLAN.md` stands unchanged.
- Keep pre- and post-ingest artifacts side by side rather than overwriting, as
  v0.5 did for the annotation correction.
- Do not overwrite v0.1–v0.5 artifacts.

## What this plan does not do

**It does not test cross-cartographic generalization.** 63 sheets of one series
is still one series. The claim that the pipeline works on a genuinely different
cartographic source remains untested and cannot be retired by this work. Say so
in whatever v0.6 reports.

**It does not change the models.** No retraining beyond re-fitting on the larger
corpus, no architecture work, no jitter augmentation — v0.5 showed the recall
curve is flat in radius, which contraindicates it. If the detector degrades on
the new sheets, that is a finding to report first and act on second.

**It does not attempt the remaining pipeline stages.** Duplicate suppression,
the segmentation-quality gate, attribute extraction and GIS export
(`TARGET_PIPELINE.md` stages 6–9) all stay deferred. They are worth doing after
there is a test result to build on, not before.

## Known debt, unchanged

Seven pre-existing `sam2_mcp` / `sam2_backend` test failures, and
repository-wide ruff errors outside the modules `make lint` covers. Neither
affects any conclusion above.

The `sam2-mcp` MCP server failed to connect during the v0.5 session, so the
SAM 2 annotation-assist path was unavailable and the missing mound was drawn by
hand. Worth fixing before step 3, since that stack is the existing route for
getting model output in front of an annotator.
