# Step 3c: training the mitosis count

[Docs](../../../docs/glioma-workflow/step3c-mitosis.md) · shared pieces [`highgrade_common/`](../highgrade_common/README.md)

## Why it is set up this way

The unit is one cell, and the answer is a count over the whole specimen. So the page shows MitoEd's candidates,
one cell each, at 64 µm and 256 µm, and the labels do two jobs:

1. **Recalibrate MitoEd's p on glioma** (its training data has no brain): a sigmoid on logit(p), written to
   `mitosis-calibration.json` and used by `entip-train highgrade --mitosis-calibration`.
2. **Measure the count:**
   - how many of the candidates it counts are mitoses (precision);
   - how many mitoses it misses among those MitoEd calls mimics.

Candidates are sampled at random **within bands** of MitoEd's p on each slide: `high` (p of 0.7 or more), `mid`
(0.3 to 0.7), `low` (counted, below 0.3) and `mimic` (not counted). Each label is weighted by its band's size, so
the estimates hold for the whole slide, not just for the sample.

Slides lean on `grade2` and `astro3`, where mitoses decide grade 2 against grade 3.

| Key | Label |
|---|---|
| `y` | Mitosis (prophase to telophase) |
| `n` | Not a mitosis: apoptotic or pyknotic nucleus, lymphocyte, karyorrhectic debris, crush |
| `x` | Not assessable |

## Run

MitoEd first: one `mitoed predict` CSV per slide, `<SeriesInstanceUID>.csv`, in the slide's µm frame. Check that
frame on one slide, since the page crops at those coordinates.

```bash
python training/runs/highgrade_mitosis/pick.py --manifest $W/trial.csv --cells $BASE/runs/highgrade/mitoed --slides $W/slides --holdout $BASE/runs/highgrade/held_out.csv --out $BASE/runs/highgrade/mit-r1 --round r1
python training/runs/highgrade_mitosis/export.py --key mit-r1/key.csv labels-AH.csv labels-XY.csv --out mit-export-r1
```

- About 20 candidates per slide over 32 slides (`--per-category`, default `gbm=6 astro4=4 oligo3=6 astro3=8
  grade2=8`).
- The calibration needs 5 or more mitoses and 5 or more non-mitoses. Its weighted Brier score is reported before
  and after, in folds grouped by slide.

## Limits

- This fixes MitoEd's p on glioma. It does not make MitoEd usable commercially: its segmenter (Cellpose-SAM) was
  trained on CC BY-NC data.
- The specimen cut-off (at least 2 mitoses) is still a setting, not a validated value.
