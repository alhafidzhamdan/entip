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
- `outlines.py`: QuPath GeoJSON (outlines and points) to tile hits, and the check image.

## Free negatives

Under the WHO grading TCGA used, necrosis or MVP made an astrocytic glioma grade 4, and a grade 2 glioma had
neither. So slides of grade 3 astrocytic gliomas (`astro3`) and grade 2 gliomas (`grade2`) give MVP and necrosis
negatives without labelling (`highgrade-train --negative-bag`). The grade is of the whole case, so a slide can be
wrong only through a grading error. Two things catch that:

- the round 2 `audit` picks: the tiles on those slides the head is surest about;
- the export, which drops any free-negative slide a labeller marked positive and lists it in `summary.json`.

Two exceptions:

- A grade 3 case whose source diagnosis says oligoastrocytoma or mixed glioma goes to `other`, since it could
  have MVP.
- The rule is not used for mitoses: a grade 2 glioma can have one.

The categories need the manifest's `diagnosis`, `who_cns5_basis` and `dx` columns. `pick_slides.py` writes them;
for an older manifest, add them with `training/runs/triage/add_who_cns5.py`.

## For every module

- Use only openly licensed slides (TCGA, CPTAC via IDC: CC BY). Never pick with a model trained on
  non-commercial labels (Ivy GAP, BraTS-Path).
- Choose the held-out patients first (`--holdout held_out.csv`, a `patient` column). They are never picked,
  outlined or used as free negatives. Their whole-slide truth (feature present anywhere) comes from a
  pathologist's review of the slide, and it sets the slide threshold (`entip-train highgrade-calibrate`).
- Agree with Edinburgh Innovations that the labellers' work belongs to the University before labelling starts.
- Copy a round to your Mac, keep `key.csv` back from labellers, and open `labels.html` in a browser.
