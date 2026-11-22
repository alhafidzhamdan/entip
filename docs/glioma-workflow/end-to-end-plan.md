# Plan: one end-to-end, checkable model per decision

[← workflow](README.md) · Proposal, 2026-11-22. Nothing in this plan has been built unless it says so. Research use only.

## The idea

ENTIP should learn each clinical decision from data, rather than hand-building the route to it, and stop at every
decision so a pathologist can check it.

- **End to end per step.** Each model goes from a patient's slides to its own answer: MVP present, a marker status, a
  methylation class.
- **Not end to end across steps.** One model from slide to final diagnosis would give a pathologist nothing to check.
  Step 6 stays a fixed rules engine built on written WHO criteria.
- **Evidence is part of the output.** A call without regions or cells to inspect is not ready for review.
- **Human effort goes on truth.** Pathologists define and validate labels, and correct the model. They do not hand-tune
  each training round.

## Each step

| Step | Decision | Training label | What the pathologist checks | Status now |
|---|---|---|---|---|
| 1 Embeddings | None (frozen H-optimus-0) | None | Tissue and focus quality | In use on Eddie |
| 2 Triage | Tumour, glial, enough tissue | Slide or patient class | Key regions behind the class | Trained on 250 slides; external tests not run |
| 3a MVP | Present, absent, uncertain | **Whole-slide MVP call** | Each proposed vessel | Tile-based caller in blinded pilot |
| 3b Necrosis | Present, absent, uncertain | **Whole-slide necrosis call** | Each proposed area | Design and review tool in progress |
| 3c Mitoses | Count per defined area | Exhaustively counted reference | Each counted cell | Count-only design, not built |
| 4 Panel | One status per marker | Molecular result | Evidence, or order the real test | Synthetic only |
| 5 Methylation | Family and subclass, or no match | Methylation class | Evidence, or order the array | Synthetic only |
| 6 Diagnosis | Draft WHO integrated diagnosis | None (fixed rules) | Every criterion and the final report | Built |

**Mitoses differ.** A mitotic figure is one cell, too small for a 112 µm tile embedding. Step 3c runs from pixels to
detected cells to a count, and each counted cell is the evidence.

## What every step must output

1. The call and its calibrated confidence.
2. The regions or cells behind it, each one click from the slide.
3. The reason for any `uncertain` status.
4. Accept and correct actions. Corrections are stored as candidate labels, never applied silently.

These extend the existing `entip.glioma_step/1` record.

## Gates before a pathologist relies on a step

| Gate | Question | Test |
|---|---|---|
| Correct answer | Does it work on new patients? | Patient-independent test, with results per tumour type |
| Calibrated | Does 90% confidence mean 90% correct? | Separate calibration patients, never the test set |
| Safe to accept | Are confident calls rarely wrong? | Under 5% error among accepted calls: at least 59 error-free held-out patients |
| Correct location | Is the evidence on the real finding? | Highlighted regions contain the pathologist's marked examples |
| Faithful evidence | Did that evidence drive the call? | Removing the top regions clearly lowers the score, more than removing random regions |
| Useful | Does it help the pathologist? | Time and agreement with and without the tool, in a later reader study |

Attention maps alone do not pass the evidence gates. They show where a model looked, not why it decided.

## Data rules

- Four patient-level sets, fixed before training: train, calibration, internal test, external test.
- Each model is frozen before its test. A test set used to choose a model or cutoff no longer counts as a test.
- Non-commercial data never trains a model or pre-selects tiles. DBTA and IPD-Brain are test only.
- Case grades are not changed. Disagreement between a slide call and a case grade is recorded and reviewed.

## Phases

### 0. Finish the MVP benchmark

Complete the 35-slide blinded pilot of the frozen tile-based caller. Its calls stay out of all training.

**Output:** the first slide-level result, including how often the model's foci contain your marked examples.

### 1. Get whole-slide MVP labels

1. Check whether TCGA and CPTAC pathology reports state MVP presence or absence, what their licence allows, and how
   often they agree with slide review. A report describes the case, so it is a weak label for one slide.
2. Pathologist-review a random sample of slides to measure that error.
3. If reports are too unreliable, use blinded whole-slide review for training slides.

**Checked 2026-11-22:**

- GDC has open-access PDF reports for TCGA-GBM (610 files, 603 cases) and TCGA-LGG (527 files, 516 cases), 139.8 MB in
  total. It lists none for CPTAC-3.
- All 1,137 report files are marked open access in GDC. GDC open data falls under the NIH Genomic Data Sharing policy:
  do not try to identify participants, and acknowledge the data and repository. TCGA and GDC also ask for citation.
- No GDC or NCI page read names a licence for reports, or addresses AI training, commercial use or NHS use. **Project
  assumption (2026-11-22, set by the project lead): commercial use is allowed.** No licence confirms this.
- TCGA-Reports displays CC BY 4.0, and its host's footer says Creative Commons terms apply to open-access content. That
  covers what its authors can license, which may not include the hospitals' original report text. It also removed
  multiple-choice forms. Prefer the GDC originals.
- **Working rule:** report-derived labels may train models intended for commercial use. Still do not try to identify
  participants, acknowledge TCGA and GDC, keep report text on Eddie without redistributing it, and store only present,
  absent or not-mentioned labels with their audit trail. Record each model's use of report labels, so it can be traced
  and retrained if the assumption is later overturned.
- The report check is a fresh slide-level run and includes the 35 patients reserved for the tile-head pilot. Any new
  model needs its own patient-level train, calibration and test split. "Not mentioned" is kept separate from "stated
  absent".

**Target:** about 200 training patients, 40 or more calibration patients and at least 59 test patients. Grade 4
IDH-mutant astrocytomas are scarce (about 22 patients, all already used), so testing them needs an external set.

**Stop if:** reports and review agree poorly and there is no capacity for a few hundred slide reviews.

### 2. Train the end-to-end MVP model

1. Train attention-MIL (`wsi/abmil.py`) on frozen embeddings, from slide to MVP present or absent.
2. Add the existing tile labels as an optional extra signal, and test whether they help.
3. Compare it with the tile-based caller by cross-validation on training patients.
4. Freeze one model, then run it once on the 35-slide pilot.

**Adopt it if** it is at least as good on slide ranking and passes the evidence gates. Otherwise keep the tile-based
caller. The pilot informs this choice, so a final claim still needs a fresh test set.

### 3. Build the pathologist check

Add the review view for MVP: call, confidence, ranked foci, jump to each, accept or correct, and an audit record. MVP
becomes the template for the other steps.

### 4. Calibrate and test MVP

Calibrate on separate patients, set the accept threshold, then test once on internal and external sets. Include grade 3
tumours and IDH-mutant grade 4 astrocytomas.

### 5. Extend to the other steps

1. **Necrosis:** reuse the MVP route with whole-slide labels and neighbourhood context.
2. **Mitoses:** build the counted reference, then the cell-level detector and count.
3. **Triage, panel and methylation:** run their real-data tests and add the same evidence gates.

### 6. Fine-tune the encoder only if needed

Keep H-optimus-0 frozen by default. Fine-tune it only if a step misses its gates, labels are plentiful, and a controlled
comparison shows a gain worth the GPU cost and added validation.

## Main risks

- **Weak slide labels.** Reports may omit MVP or describe tissue that is not on the slide.
- **Grade shortcuts.** A model may learn "high-grade tissue" instead of MVP. Results must be reported within tumour type.
- **Rare subgroups.** Overall performance can hide failure in IDH-mutant tumours.
- **Single rater.** Most labels are from one pathologist, so a second rater is needed for final truth.
- **Automation bias.** Showing model evidence can lead reviewers to agree too easily, so truth labels are collected blind.

## Decisions needed

1. Should report-derived labels be tried first, or should training labels come only from slide review?
2. Who can be the second pathologist, and for how many slides?
3. How many whole-slide reviews are realistic in total?
4. Which external set should test IDH-mutant grade 4 tumours?
