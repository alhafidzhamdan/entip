# Step 3b: necrosis

[← step 3](step3-high-grade.md) · code `training/glioma/src/entip_glioma/highgrade/` (`--feature necrosis`)
· training `training/runs/highgrade_necrosis/`

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

[Training module](../../training/runs/highgrade_necrosis/README.md):

- **Labels:** QuPath outlines first (`Necrosis`, `Neither`). They are quick for large areas and give the edge. The
  page covers single tiles: look-alikes, and the tiles the head is unsure of.
- **Head:** the tile's H-optimus-0 embedding plus the mean embeddings of its 3 × 3 and 5 × 5 neighbourhoods, with
  no re-embedding.
- **Round 1 candidates from grade:** the same grade 4 vs grade 3 picking model as 3a.
- **Free negatives:** random tiles from grade 3 astrocytic and grade 2 slides, which have no necrosis by their
  grade, checked by eye as a grid before use and by audit picks.
- **Slide score:** the best focus of connected positive tiles.

Tiles from one large outline are near-copies of each other. Cross-validation groups by slide, so this does not
inflate the metrics, but a slide with large outlines adds less than its tile count suggests.

## Status and open items

- Synthetic only.
- The training module is built and tested on synthetic slides; round 1 is ready to run on Eddie.
