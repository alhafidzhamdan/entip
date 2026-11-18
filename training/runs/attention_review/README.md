# Attention review: does the model look where a neuropathologist would?

A blinded study of the tiles a triage model attends to. For a sample of patients, raters see each case's WHO CNS5
diagnosis and a shuffled mix of the model's **most-attended tiles** and **random control tiles** from the same
slide. They don't see the model's call, whether it was right, or which tiles it attended to. They score each
outlined tile:

| Score | Meaning |
|---|---|
| **D** Diagnostic | Shows the tissue that defines the stated diagnosis (for a non-neoplastic case: representative parenchyma) |
| **T** Tumour, not specific | Tumour, but this field alone would not support the stated diagnosis |
| **N** Non-neoplastic | Normal or reactive tissue only, in a tumour case |
| **A** Artefact / not assessable | Background, blood, crush, folds, ink, out of focus or too little tissue |

The result is a statement your peers can check, for example "attended tiles were diagnostic in 78% (95% CI
66–88%) against 38% for random tiles from the same slides". It also gives a list of the cases where attention
missed the diagnostic tissue, which are the calls to distrust.

## Design

- **Cases:** every patient the model got wrong, plus `--per-class` correctly called patients per class (default
  12), at random.
- **Tiles per case:** the top `--top` attended tiles (default 8) and `--controls` random tiles (default 8) from
  outside the top 10% of attention, on the slide the model bases its call on.
- **Images:** a 3 × 3 tile field (about 336 µm across) with the rated tile outlined in the centre.
- **Which set:**
  - **Pilot: the diverse training set.** The saved model has seen these patients, so its attention may be
    sharper than in practice. Use it to try the scoring scheme and the page.
  - **Study: the DBTA** (or IPD-Brain). The model never saw these patients, so this is attention as it would
    be in use, on another lab's slides.
- **Raters:** at least two neuropathologists, rating independently; report agreement (Cohen's kappa).

**How focused attention is.** Attention can be spread thinly: on TCGA-02-0025 the top 16 of 14,470 tiles carry
about 1% of it, each at 4-8x an even share. Then the "attended" tiles are only lightly preferred over the rest of
the tumour, and a small attended-control difference says the model averages over broad areas, not that it looks in
the wrong place. So `key.csv` records, for each case's slide, the share of attention on the top 1% of tiles (1% if
even), the effective number of tiles (1/sum(w^2)), and the rated attended and control tiles' weights against an
even share; `analyse.py` reports the difference separately for focused and diffuse cases (split at the median).

**Analysis caveat.** Tiles from the same case are not independent, so the tile-level intervals are somewhat too
narrow. For claims, lean on the case-level view: the share of cases where attention hit the target, and the list
of misses.

## On Eddie (login node; about a minute per case)

Pilot, on the diverse set (the training report gives the out-of-fold calls):

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull && REPO=$PWD . training/runs/triage/eddie/config.sh && activate_env && W=/exports/igmm/eddie/Glioblastoma-WGS/entip/runs/triage-diverse && python training/runs/attention_review/make_review.py --report $W/report-h-optimus-0.json --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --model $W/model-h-optimus-0 --slides $W/slides --out /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/review-diverse --per-class 8
```

Run `add_who_cns5.py` on that `trial.csv` first (see `docs/who-cns5.md`), so raters see WHO CNS5 diagnoses.

For the DBTA, use the full evaluation report (`evaluation-<run>.json`), the DBTA manifest, bags and slides, and
the stage-A model.

## Rating

1. Copy the output folder to your Mac. Keep `key.csv` back from the raters, or delete it from their copy.
2. Open `review.html` in a browser. Enter your initials, score each tile, and press **Download scores**. Ratings
   are kept in the browser, so you can stop and come back.
3. Collect each rater's `scores-<initials>.csv`.

## Analysis

```bash
python training/runs/attention_review/analyse.py review-diverse/key.csv scores-AH.csv scores-XY.csv --json summary.json
```

For each rater and pooled:
- diagnostic and on-target shares for attended against control tiles, with exact 95% intervals;
- the same by class, and by whether the model's call was right;
- the artefact share;
- the cases where attention missed, with how focused their attention was;
- attention concentration per case, and the attended-control difference in focused against diffuse cases;
- with two or more raters, kappa on the four scores and on diagnostic or not.

## Data

The tiles are crops of the source slides, so they carry the source's terms. Diverse/stage A are public IDC data.
DBTA and IPD-Brain crops stay under their data use agreements: keep them in the group space or on a University
device, and share them only with the raters.
