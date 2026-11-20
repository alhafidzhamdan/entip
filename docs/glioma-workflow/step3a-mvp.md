# Step 3a: microvascular proliferation (MVP)

[← step 3](step3-high-grade.md) · code `training/glioma/src/entip_glioma/highgrade/` (`--feature microvascular_proliferation`)
· training `training/runs/highgrade_mvp/`

## What it decides

Is there MVP anywhere in the specimen: `present`, `absent` or `indeterminate`. It is a grade 4 criterion in
IDH-mutant astrocytoma and points to grade 3 in oligodendroglioma (see [step 3](step3-high-grade.md)).

## What counts

- **MVP:** vessel walls of several layers of proliferating endothelial cells and pericytes, glomeruloid tufts,
  garlands.
- **Not MVP:** many vessels, thick-walled or hyalinised vessels, vessels cut tangentially, reactive vessels near
  surgery or haemorrhage.

A tuft is small (about 50 to 200 µm), and the look-alikes are other vessels. The decision is about **one vessel at
high detail**. Pathologists agree on it less well than on necrosis, so labeller agreement is reported with the
head.

## How

The labelling effort goes into telling MVP from ordinary vessels, and nothing else
([training module](../../training/runs/highgrade_mvp/README.md)):

- **Labels per tile:** MVP, vessel without MVP, no vessel, or not assessable. On the page that is one key per tile;
  in QuPath it is a point on each vessel.
- **Two heads from the same labels:** a vessel head, used only to find vessels to label, and the MVP head.
- **Free negatives:** random tiles from grade 3 astrocytic and grade 2 slides, which have no MVP by their grade,
  checked by eye as a grid before use.
  They are MVP negatives at no cost, in a lower-grade background.
- **Round 1 candidates from grade:** a grade 4 vs grade 3 IDH-mutant astrocytoma model, trained on slide labels
  only, points to the tiles that make a slide grade 4. Used for picking, never as a label.
- **Active learning from round 2:**
  - vessels the MVP head is least sure of, and those it is surest are MVP;
  - random vessels;
  - audit tiles on the free-negative slides.
- **Lower-grade positives:** slides lean on grade 3 oligodendrogliomas and grade 4 IDH-mutant astrocytomas, so
  the head learns MVP, not "glioblastoma".
- **Slide score:** the best focus of connected positive tiles, as before.

Still to test: whether embeddings at 0.25 µm/px of the labelled vessels beat 0.5 µm/px, before re-embedding any
slide.

## Status and open items

- Synthetic only.
- The training module is built and tested on synthetic slides; round 1 is ready to run on Eddie.
- Not built yet: the 0.25 µm/px comparison.
