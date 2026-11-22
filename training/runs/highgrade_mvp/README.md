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

## Not built yet

- Embeddings at 0.25 µm/px for the labelled vessels, to test whether finer detail helps before re-embedding any
  slide.
