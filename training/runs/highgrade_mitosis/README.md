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
- **Prophase:** never counted as a mitosis. The first finder ignored it in training and scoring, so it could still
  call a prophase without penalty.
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

## Second finder

`finder_3_retrain.sh` trains a new finder from scratch after the first finder's one-time unseen test.

- Human melanoma and neuroendocrine tumour images join training as complete images. The `tune` cases do not change.
- Prophase becomes a negative in training and tuning, so calling a prophase now counts as an error.
- False counts per mm² use the true 2 mm² area of each MIDOG++ image.
- This finder has no unseen-tumour test. Its glioma performance comes from the held-out patients and the locked set.

## Glioma review round 1

`glioma_review.py` starts teaching the finder glioma with a small TCGA set, using the second finder. It does not use
gbm-db.

- **Reference patients are sealed first.** The job draws them before anything else, and the finder never runs on them.
- **Development:** 9 patients, each with 1 field to mark in full and 3 finder fields.
- **Held out:** 3 patients, each with 2 fields to mark in full. They are never trained on and measure each retrain.
- **Review:** 15 full fields (7.5 mm²), with no model output, and 300 finder calls in random order with scores hidden.
- **Calls:** about 100 confident, 100 borderline and 100 low scoring, spread across patients.

Submit the second finder and the round together. The round waits for training to finish:

```bash
mkdir -p /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-finder2/logs /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-glioma-round1/logs /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-ref1/logs && qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/eddie/finder_3_retrain.sh && qsub -hold_jid mitosis_finder2_train /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/eddie/glioma_round1.sh
```

Copy `full-fields-pack.tar` and `calls-pack.tar` to your computer, unpack them and open each `review.html`. Mark the full
fields first. Keep `private/` on Eddie.

Call keys: `1` mitosis, `2` granular, `3` prophase, `4` look-alike, `5` not mitosis, `6` uncertain, `7` unusable.
A look-alike mimics a mitosis. Not mitosis is anything else, such as an ordinary nucleus.

To update the call page of a built round after a label change, without changing its keys, crops or saved labels:

```bash
python /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/glioma_review.py page --out /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-glioma-round1
```

## Third finder: glioma fine-tuning

`glioma_retrain.py` and `finder_4_glioma.sh` use round 1's review. The job needs `mitosis-marks-AH.csv` and
`mitosis-call-labels-AH.csv` in the round's `private/` folder.

1. **Score the second finder** on all 15 fields marked in full, at its frozen cut-off. This is the baseline.
2. **Add the 9 development patients' review to training.** Their full fields become fully marked images. Their calls
   become small images labelled only at the centre: mitosis or granular is positive; prophase, look-alike or not
   mitosis is negative; uncertain and unusable calls are left out. Each point is drawn as a 4 µm disc.
3. **Fine-tune** from the second finder for 20 epochs, with a quarter of each batch from glioma. The epoch and cut-off
   are still chosen on the public tune images, because there is no glioma tune set yet.
4. **Score the third finder on the 3 held-out patients only.** It refuses any patient it was trained on.

The held-out result is small: 6 fields from 3 patients. The table by cut-off in each score file describes those
fields and must not be used to choose a cut-off.

```bash
mkdir -p /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/mitosis-finder3/logs && qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_mitosis/eddie/finder_4_glioma.sh
```

The results are `score-mitosis-finder2.json` and `score-mitosis-finder3.json` in the round folder.

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
- Atypical and granular mitoses count as mitoses. Prophase is marked, if at all, as a look-alike.
- Mark types: `1` mitosis, `2` uncertain, `3` look-alike.
- Each mark can carry several ticked details. A mitosis takes one phase at most, plus atypical (`A`) and granular (`G`).
  A look-alike takes one or more types: prophase, pyknosis, apoptosis, karyorrhexis, lymphocyte, endothelial or
  reactive, or other. Any mark can be flagged as blurred or hard to judge at the edge.
- A field with no mitosis must be ticked **No mitoses in this field** before it can be completed.
- `N` or space moves through the nine sections. A field cannot be completed until all nine have been viewed.
- Click a mark to select it, press a number to change its type, and backspace to delete it. `⌘Z` or `Ctrl+Z` undoes,
  `H` hides marks, and `+` and `-` zoom.
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
