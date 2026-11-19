# Step 3b: necrosis

[← step 3](step3-high-grade.md) · code `training/glioma/src/entip_glioma/highgrade/` (`--feature necrosis`)
· labels `training/runs/highgrade_labels/`

## What it decides

Is there tumour necrosis anywhere in the specimen: `present`, `absent` or `indeterminate`. It is a grade 4
criterion in IDH-mutant astrocytoma and points to grade 3 in oligodendroglioma (see [step 3](step3-high-grade.md)).

**Necrosis, not only pseudopalisading necrosis.** WHO counts tumour necrosis whether or not cells palisade around
it, so a detector trained only on palisading would miss the rest. Palisading is a strong clue, not the target.

## What counts

- **Necrosis:** eosinophilic, granular debris with nuclear dust, often with a palisade of tumour cells at its
  edge and viable tumour beside it.
- **Not necrosis:** haemorrhage, cautery, crush, fibrin.
- Treatment-related necrosis in a recurrence is hard to tell from tumour necrosis on H&E, so labels come from
  primary, untreated resections (as in TCGA).

Necrosis is large (hundreds of µm to mm). One 112 µm tile inside it may show only debris; **the edge and the
tissue around it** decide most cases.

## How

| | Now | Planned |
|---|---|---|
| Unit | 112 µm tile | The same tile, with its neighbourhood |
| Head | Linear head on the tile's H-optimus-0 embedding | The same head on the tile's embedding plus the mean embeddings of its 3 × 3 and 5 × 5 neighbourhoods (no re-embedding) |
| Labels | Tile labels (`n` on the labelling page) and QuPath Necrosis / Neither outlines | Mostly outlines: quick for large areas, and they give the area and the edge |
| Slide score | Best focus of connected positive tiles | The same |

Tiles from one large outline are near-copies of each other. Cross-validation groups by slide, so this does not
inflate the metrics, but a slide with large outlines adds less than its tile count suggests.

## Status and open items

- Synthetic only.
- Round 1 tile labels are ready to run on Eddie; QuPath outlines can be added to the same export.
- Not built yet: neighbourhood context in the head.
- Positives in an IDH-mutant background: pick the IDH-mutant "secondary glioblastomas" in TCGA-GBM on purpose.
