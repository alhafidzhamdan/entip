# Step 3 training: shared pieces

Each step 3 sub-step has its own training module, built on this folder:

| Sub-step | Module | Unit labelled | Labels | Head |
|---|---|---|---|---|
| 3a MVP | [`highgrade_mvp/`](../highgrade_mvp/README.md) | A tile, vessel by vessel | Page (MVP / vessel / no vessel), QuPath points on vessels | MVP head, plus a vessel head for picking |
| 3b Necrosis | [`highgrade_necrosis/`](../highgrade_necrosis/README.md) | An area | QuPath outlines first, page for look-alikes | Necrosis head with neighbourhood context |
| 3c Mitosis | [`highgrade_mitosis/`](../highgrade_mitosis/README.md) | One cell | Page of MitoEd candidates | Recalibration of MitoEd's p on glioma |

What they share:

- `common.py`:
  - slide categories from the manifest's WHO CNS5 diagnosis (`gbm`, `astro4`, `oligo3`, `astro3`, `grade2`,
    `other`);
  - the free-negative rule;
  - picking one slide per patient by category;
  - consensus and Cohen's kappa;
  - writing the training files.
- `labelpage.py`: the labelling page. One key per label, `j` / `k` to move, labels kept in the browser and
  downloaded as `labels-<initials>.csv`. It shows no diagnosis, model output or reason for picking.
- `free_negatives.py`: picks the free-negative tiles and writes the page to check them.
- `grade_mil.py`: a grade 4 vs grade 3 attention model, used only to pick tiles (see below).
- `outlines.py`: QuPath GeoJSON (outlines and points) to tile hits, and the check image.

## Free negatives

Under the WHO grading TCGA used, necrosis or MVP made an astrocytic glioma grade 4, and a grade 2 glioma had
neither. So slides of grade 3 astrocytic gliomas (`astro3`) and grade 2 gliomas (`grade2`) give MVP and necrosis
negatives without labelling. The grade is of the whole case, so a slide can be wrong only through a grading error.
Three things catch that:

- **A check by eye before use** (`free_negatives.py`): random tiles from each free-negative slide, shown as a grid of
  3 × 3 fields. Click a tile to flag it (MVP, necrosis or both) and tick each slide when you have looked at it.
  Only checked slides are used, and a slide with a flag is left out of that feature's negatives.
- **The round 2 `audit` picks:** the tiles on those slides the head is surest about.
- **The export**, which also drops any free-negative slide a labeller marked positive on the page, and lists every
  slide left out in `summary.json`.

```bash
python training/runs/highgrade_common/free_negatives.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --holdout $BASE/runs/highgrade/held_out.csv --out $BASE/runs/highgrade/free-r1
```

`--per-slide` (default 200) sets the tiles per slide. One check serves both 3a and 3b: pass
`--free-negatives free-r1 --free-review review-AH.csv` to both exports.

Two exceptions:

- A grade 3 case whose source diagnosis says oligoastrocytoma or mixed glioma goes to `other`, since it could
  have MVP.
- The rule is not used for mitoses: a grade 2 glioma can have one.

The categories need the manifest's `diagnosis`, `who_cns5_basis` and `dx` columns. `pick_slides.py` writes them;
for an older manifest, add them with `training/runs/triage/add_who_cns5.py`.

## Grade 4 vs grade 3, for picking

The same grading works the other way. An IDH-mutant astrocytoma was graded 4 for MVP or necrosis, and 3 for
neither, with mitoses allowed in both. So the slide-level contrast `astro4` against `astro3` is close to "MVP or
necrosis somewhere in the case" against "neither". `grade_mil.py` trains an attention model on that contrast, with
no tile labels. The tiles that push a slide most towards grade 4 are good candidates to label (`pick.py
--grade-model grade-mil/grade-mil.npz`, source `grade`).

```bash
python training/runs/highgrade_common/grade_mil.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --holdout $BASE/runs/highgrade/held_out.csv --out $BASE/runs/highgrade/grade-mil
```

- Picking only, never a label: it cannot tell MVP from necrosis, and a slide is only a sample of its case.
- It can also pick up other grade 4 clues, such as cellularity; the labeller calls those negative.
- `grade-mil.json` gives the out-of-fold AUC, grouped by patient. Near 0.5 means it learnt nothing: pick without it.

**Does it find MVP and necrosis on its own?** Before labelling anything, add `--view --slides $W/slides`.
`view.html` shows, for up to 15 slides of each grade, the slide with its 12 highest-scoring tiles boxed and
numbered, those tiles as 3 × 3 fields, and 4 random tiles for comparison. Each slide is scored by the fold model
that did not train on it. It works if the top tiles on grade 4 slides are mostly MVP and necrosis and those on
grade 3 slides are not. This is a look, not a measurement; a measured answer needs labels. `view-key.csv` maps the
codes to slides.
- TCGA has few grade 4 IDH-mutant astrocytomas (about 22 patients with slides in the stage-A set, against about
  100 grade 3). `--positive astro4 gbm` adds glioblastomas, at the cost of IDH-wildtype features unrelated to MVP
  or necrosis.

## For every module

- Use only openly licensed slides (TCGA, CPTAC via IDC: CC BY). Never pick with a model trained on
  non-commercial labels (Ivy GAP, BraTS-Path).
- Choose the held-out patients first (`--holdout held_out.csv`, a `patient` column). They are never picked,
  outlined or used as free negatives. Their whole-slide truth (feature present anywhere) comes from a
  pathologist's review of the slide, and it sets the slide threshold (`entip-train highgrade-calibrate`).
- Agree with Edinburgh Innovations that the labellers' work belongs to the University before labelling starts.
- Copy a round to your Mac, keep `key.csv` back from labellers, and open `labels.html` in a browser.
