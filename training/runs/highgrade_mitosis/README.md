# Step 3c: mitosis counting

[Docs](../../../docs/glioma-workflow/step3c-mitosis.md) · shared pieces [`highgrade_common/`](../highgrade_common/README.md)

This folder holds two independent runs: the first mitosis finder, trained on public data, and the blinded glioma
reference set that later tests it.

## First finder

`finder_data.py` and `finder.py` train a compact U-Net from scratch, with no pretrained weights, at about 0.25 µm per
pixel.

| Split | Data | Use |
|---|---|---|
| `train` | 85% of the other MIDOG++ cases, and all of LUNG-MITO | Fit the network |
| `tune` | 15% of those MIDOG++ cases, by tumour type | Choose the epoch and cut-off |
| `unseen` | Human melanoma and neuroendocrine tumour | One internal test on unseen tumour types |

- **Mitosis:** prometaphase, metaphase, anaphase, telophase and atypical figures.
- **Look-alike:** the datasets' negative class, penalised as heavily as a mitosis.
- **Prophase:** ignored, in training and scoring, to match the counting definition.
- **LUNG-MITO is training only**, because its tiles have no patient identifiers and cannot be split safely.
- **MIDOG++:** 498 of its 503 images have a confirmed CC0 record. The other 5 are never downloaded.
- A MIDOG++ case is treated as one patient, because the dataset gives no separate patient identifier.

Submit the data job (about 77 GB of downloads and about 55 GB of decoded images):

```bash
mkdir -p /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-finder1/logs && qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/eddie/finder_1_data.sh
```

When it finishes, submit training. It trains, freezes the best model and tests it once on the unseen tumour types:

```bash
qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/eddie/finder_2_train.sh
```

The results are in `unseen-test.json`. That is an internal check, not a glioma result. The glioma result comes only
from scoring the frozen model on the locked reference set below.

## Glioma reference set

`reference_set.py` builds a blinded, exhaustively counted reference for mitotic figures. It counts only: it makes no
diagnosis, grade or threshold, and no model takes part.

## Design

- **Patients:** 20, drawn at random: 6 `gbm`, 3 `astro4`, 4 `oligo3`, 4 `astro3`, 3 `grade2`. Categories only spread the
  draw. Any shortfall is filled from `gbm`, and the draw reports it.
- **Other step 3 work:** ignored. Mitosis starts from scratch, so patients used or reserved for MVP or necrosis can be
  drawn.
- **Slides:** one per patient, native pixel size at most 0.30 µm.
- **Fields:** 4 random, non-overlapping 0.5 mm² tissue fields per slide (80 fields, 40 mm²). A field needs at least 90%
  of its surrounding tile window to be tissue.
- **Second reader:** one random field per patient (20 fields, 10 mm²).
- **Target:** about 200 definite mitoses. If the locked set has fewer, add a new set of fresh patients with this same
  tool. Locked marks are never changed.

## Run

```bash
mkdir -p /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/logs && qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/eddie/mitosis_reference.sh
```

Copy only `review-pack.tar` to the first reader and `second-reader-pack.tar` to the second. Keep `private/` on Eddie.
Unpack the file and open `review.html`.

## Marking

- Mark every mitosis whose centre lies inside the white square. Count a telophase pair once.
- Keys: `1` mitosis, `2` uncertain, `3` pyknosis, `4` apoptosis, `5` karyorrhexis, `6` lymphocyte, `7` other look-alike.
- `N` or space moves through the nine sections. A field cannot be completed until all nine have been viewed.
- Click a mark to select it, then press a number to relabel it or backspace to delete it.
- Enter the assessable share of the field, then press Enter to complete it.

Look-alikes are optional, so they are useful examples but are not an exhaustive negative set.

## Lock and compare

Copy the first reader's marks to `private/`, then lock them once:

```bash
python /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/reference_set.py lock --key /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/private/key.csv --marks /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/private/mitosis-marks-AH.csv --out /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/private
```

Then compare the second reader. Replace `XY` with their initials:

```bash
python /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/reference_set.py agree --locked /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/private --marks /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/private/mitosis-marks-XY.csv
```

Two marks agree when they lie within 7.5 µm. Agreement is reported for definite mitoses, and for mitoses plus uncertain
figures.
