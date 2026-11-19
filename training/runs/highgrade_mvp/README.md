# Step 3a: training the MVP head

[Docs](../../../docs/glioma-workflow/step3a-mvp.md) · shared pieces [`highgrade_common/`](../highgrade_common/README.md)

## Why it is set up this way

MVP is a judgement about one vessel, and its look-alikes are other vessels. So a pathologist's time goes into
telling MVP from ordinary vessels, and not into tissue that is obviously neither. Each source of signal is used at
its cheapest:

| Signal | Source | Cost |
|---|---|---|
| MVP negatives in a lower-grade background | Free-negative slides (`astro3`, `grade2`): random tiles | None |
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
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull && REPO=$PWD . training/runs/triage/eddie/config.sh && activate_env && W=$BASE/runs/triage-diverse && python training/runs/highgrade_mvp/pick.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --model $W/model-h-optimus-0 --holdout $BASE/runs/highgrade/held_out.csv --out $BASE/runs/highgrade/mvp-r1 --round r1
```

- Slides: `--per-category`, default `gbm=10 astro4=8 oligo3=12 astro3=4 grade2=4`.
- Tiles per slide: 4 attended, 8 diverse and 2 random, so about 530 tiles (38 slides × 14).
- Each tile shows twice: at high power, and with its 3 × 3 field.
- The diverse trial set has few `oligo3` and `astro4` slides. The stage-A set (`RUN=stage-a`) has many more.

## Export and train

```bash
python training/runs/highgrade_mvp/export.py --key mvp-r1/key.csv labels-AH.csv labels-XY.csv --bags $W/bags-h-optimus-0 --manifest $W/trial.csv --holdout held_out.csv --out mvp-export-r1
bash mvp-export-r1/train-mvp.sh
bash mvp-export-r1/train-vessel.sh
```

- `train-mvp.sh` adds the free-negative slides (`--negative-bag`, 200 random tiles each by default).
- `summary.json` gives:
  - counts;
  - Cohen's kappa for MVP on tiles both labellers called a vessel, which is the judgement that matters;
  - `free_negative_slides_with_mvp`: check the grade of any slide listed there.
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

## Not built yet

- Embeddings at 0.25 µm/px for the labelled vessels, to test whether finer detail helps before re-embedding any
  slide.
