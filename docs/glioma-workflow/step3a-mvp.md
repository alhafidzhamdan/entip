# Step 3a: microvascular proliferation (MVP)

[← step 3](step3-high-grade.md) · code `training/glioma/src/entip_glioma/highgrade/` (`--feature microvascular_proliferation`)
· labels `training/runs/highgrade_labels/`

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

| | Now | Planned |
|---|---|---|
| Candidates | Every 112 µm tissue tile | Vessels |
| Head | Linear head on the tile's H-optimus-0 embedding (0.5 µm/px) | Each vessel judged against ordinary vessels; at 0.25 µm/px if that proves better |
| Labels | Tile labels (`m` on the labelling page) and QuPath MVP / Neither outlines | Also points on vessels in QuPath: MVP or ordinary vessel |
| Slide score | Best focus of connected positive tiles | The same, from vessel scores |

Points on vessels make the negatives the right ones (ordinary vessels rather than any tissue), take one click
each, and let labeller agreement be measured vessel by vessel.

Before re-embedding anything at 0.25 µm/px, compare a head at 0.5 µm/px with one at 0.25 µm/px on the labelled
vessels only.

## Status and open items

- Synthetic only.
- Round 1 tile labels are ready to run on Eddie.
- Not built yet: point labels for vessels, vessel candidates, the 0.25 µm/px comparison.
- Positives in an IDH-mutant background: pick the IDH-mutant "secondary glioblastomas" in TCGA-GBM on purpose.
