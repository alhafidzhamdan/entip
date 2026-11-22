# Step 3a: training the MVP head

[Docs](../../../docs/glioma-workflow/step3a-mvp.md) · shared pieces [`highgrade_common/`](../highgrade_common/README.md)

## Why it is set up this way

MVP is a judgement about one vessel, and its look-alikes are other vessels. So a pathologist's time goes into
telling MVP from ordinary vessels, and not into tissue that is obviously neither. Each source of signal is used at
its cheapest:

| Signal | Source | Cost |
|---|---|---|
| MVP negatives in a lower-grade background | Free-negative slides (`astro3`, `grade2`): random tiles, checked by eye as a grid | A quick look per slide |
| Where the vessels are | A vessel head trained on the same labels (`m` or `v` vs `0`); finds vessels to pick from | None extra |
| MVP vs ordinary vessel | The page, one key per tile, or a click per vessel in QuPath | The labelling |
| Which vessels to label next | Round 2 on: vessels the MVP head is least sure of (`uncertain`) and surest are MVP (`likely`) | None |
| Checking the free negatives | Audit picks: the most MVP-like tiles on free-negative slides | A few tiles per slide |

The slides lean on lower-grade backgrounds with MVP (`oligo3`, `astro4`). Otherwise almost every positive comes from a
glioblastoma, and the head could learn "glioblastoma" instead of MVP.

| Key | Label | MVP head | Vessel head |
|---|---|---|---|
| `m` | MVP | 1 | 1 |
| `v` | Vessel, no MVP | 0 | 1 |
| `0` | No vessel | 0 | 0 |
| `x` | Not assessable | left out | left out |

## Round 1, on Eddie

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull && REPO=$PWD . training/runs/triage/eddie/config.sh && activate_env && W=$BASE/runs/triage-diverse && python training/runs/highgrade_mvp/pick.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --model $W/model-h-optimus-0 --holdout $BASE/runs/highgrade/held_out.csv --grade-model $BASE/runs/highgrade/grade-mil/grade-mil.npz --out $BASE/runs/highgrade/mvp-r1 --round r1
```

Train the grade model first ([`grade_mil.py`](../highgrade_common/README.md#grade-4-vs-grade-3-for-picking)), or
leave out `--grade-model`.

- Slides: `--per-category`, default `gbm=10 astro4=8 oligo3=12 astro3=4 grade2=4`.
- Tiles per slide: 4 attended, 6 from the grade model, 8 diverse and 2 random, so about 760 tiles (38 slides × 20).
- Each tile shows twice: at high power, and with its 3 × 3 field.
- The diverse trial set has few `oligo3` and `astro4` slides. The stage-A set (`RUN=stage-a`) has many more.

## Export and train

```bash
python training/runs/highgrade_mvp/export.py --key mvp-r1/key.csv labels-AH.csv labels-XY.csv --bags $W/bags-h-optimus-0 --manifest $W/trial.csv --holdout held_out.csv --free-negatives free-r1 --free-review review-AH.csv --out mvp-export-r1
bash mvp-export-r1/train-mvp.sh
bash mvp-export-r1/train-vessel.sh
```

- **Free negatives:** the checked tiles from `highgrade_common/free_negatives.py` (see
  [its README](../highgrade_common/README.md#free-negatives)) become MVP negatives. Leave both options out to train
  on labelled tiles only.
- `summary.json` gives:
  - counts;
  - Cohen's kappa for MVP on tiles both labellers called a vessel, which is the judgement that matters;
  - `free_negatives`: slides checked, not checked, and left out for a flag;
  - `free_negative_slides_with_mvp`: free-negative slides labelled MVP on the page. Check the grade of any slide
    listed in either.
- **QuPath** (optional, `--outlines qupath/ --slides $W/slides`): add a point on each vessel with the class `MVP` or
  `Vessel`, or outline areas of `Neither`. Name each export `<SeriesInstanceUID>.geojson`. Look at
  `check/<slide>.jpg` first: MVP is blue, vessels green, neither grey.

## Round 2 on

```bash
python training/runs/highgrade_mvp/pick.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --holdout $BASE/runs/highgrade/held_out.csv --exclude mvp-r1/key.csv --vessel-head mvp-export-r1/vessel.joblib --mvp-head mvp-export-r1/mvp.joblib --attended 0 --diverse 0 --random 1 --vessels 2 --out $BASE/runs/highgrade/mvp-r2 --round r2
```

Per slide, picks come from the vessel head's candidates: 6 uncertain, 4 likely and 2 random vessels. On
free-negative slides there are 4 audit picks. Export every round together: `--key mvp-r1/key.csv --key
mvp-r2/key.csv`.

## High-power recheck of the tile review

`recheck.py` tests whether the head from the tile review (`highgrade_common/tile_heads.py`) finds MVP or only vessels.
It builds one blinded, shuffled page of every tile rated MVP, the 50 not-MVP tiles the head scored highest out of fold,
and 40 random other not-MVP tiles.

```bash
qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_common/eddie/mvp_recheck.sh
```

- Each tile is labelled MVP, vessel without MVP, no vessel, or unsure / not assessable, with the keys `export.py` reads.
- `labels.html` is self-contained. Copy only that file to your Mac and keep `key.csv` back: it holds the slide, grade,
  earlier rating and score.
- `highgrade_common/eddie/mvp_head_recheck.sh` retrains the MVP head with these labels in `tile-heads-r2-recheck1`.
  Its `metrics.json` scores the first and retrained heads on the rechecked tiles. The hard negatives were chosen by the
  first head, so that comparison favours the retrained one.

## New patients

`new_patients.py` freezes the retrained head, scores patients it never trained on, and builds one blinded page from
each slide's highest-scoring tiles, other tiles in its top 10%, and random tiles from the rest. The labels test the
frozen head and add MVP patients for the next retraining. `key.csv` weights each pick back to the whole slide.

```bash
qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_common/eddie/mvp_new_patients.sh
```

`highgrade_common/eddie/mvp_head_pooled.sh` then retrains on every labelled patient. It first runs `reserve_test.py`,
which draws fresh patients at random, with no model, into `mvp-final-test/patients.csv`. They are never trained on and
are kept for the slide-level test. Once used for training, the new patients no longer count as an independent test.

## Similar-tile search

`retrieve.py` uses every tile labelled MVP at high power as a reference and searches the existing embeddings for tiles
that resemble MVP from at least three other patients. Reserved patients are never searched, and nothing is labelled
automatically: the closest tiles and random controls go on one blinded page.

```bash
qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_common/eddie/mvp_retrieve.sh
```

`highgrade_common/eddie/mvp_head_similar.sh` retrains on every labelled tile, including the reviewed matches, in
`tile-heads-similar1`. The matches were chosen for resembling MVP, so judge the head by `without_similar_tiles` in its
`metrics.json`, and by the slide-level test.

## Slide-level pilot

`slide_pilot.py` tests one frozen head on the reserved patients. `build` scores each slide privately and writes a
blinded whole-slide viewer with no names, metadata, label images or model output. After every slide has a call,
`compare` reports ranking, and sensitivity and specificity at cutoffs fixed in advance.

```bash
qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_common/eddie/mvp_slide_pilot.sh
```

Copy only `review-pack.tar` to review. Keep `private/` on Eddie until the calls are final. The cutoffs are not
calibrated, so this is a pilot and not a clinical threshold.

After `compare`, `slide_reveal.py` scores where the frozen head points against the reader's outlines and writes
`reveal/reveal.html`, an overlay of the heatmap, model areas and outlines. It runs once and refuses any calls file other
than the one `compare` locked. Its rules are in the script. Outline every MVP area, because a model area outside
incomplete outlines is not a true false alarm. Copy `reveal.html` into the local `review-pack/` so it can load the tiles.

```bash
qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_common/eddie/mvp_slide_reveal.sh
```

## Training on the pilot outlines

`outline_labels.py` turns the locked pilot calls into tile labels for a new head in `tile-heads-outlines1`. Outlined
tiles are positive, random tiles from slides called absent are negative, and other tissue on present slides stays
unlabelled because the outlines are not exhaustive. The model's areas are never labels. Each slide's contribution is
capped so heavily outlined slides do not dominate.

```bash
qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_common/eddie/mvp_head_outlines.sh
```

Compare `without_similar_tiles` in its `metrics.json` with the frozen head, and ignore the headline AUROC, which
includes the new whole-slide labels. The 35 pilot patients no longer test this head, so it needs new patients.

## Report-derived slide labels

`report_labels.py` tests whether TCGA pathology reports can supply case-level MVP labels. This is a fresh slide-level
run, so it includes the patients reserved for the tile-head pilot. It downloads open-access reports, uses OCR where
needed, and labels each case by rule as present, absent, unclear or not mentioned. "Not mentioned" is never treated as
absent. No report text goes to an external AI.

```bash
bash /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_common/eddie/mvp_reports_submit.sh
```

`report-check.json` gives coverage and agreement with reviewed tiles. `private/review-sample.csv` is the sample for
blinded slide review. Report labels are unproven until that review. Commercial use is a project assumption, not a
confirmed licence. See [the plan](../../../docs/glioma-workflow/end-to-end-plan.md).

## Not built yet

- Embeddings at 0.25 µm/px for the labelled vessels, to test whether finer detail helps before re-embedding any
  slide.
