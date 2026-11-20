# Step 3b: training the necrosis head

[Docs](../../../docs/glioma-workflow/step3b-necrosis.md) · shared pieces [`highgrade_common/`](../highgrade_common/README.md)

## Why it is set up this way

Necrosis is an area, and its edge decides most cases. So:

- **Outlines in QuPath are the main labels:** they are quick for large areas and give the edge. Classes are
  `Necrosis` (palisading or not) and `Neither`. Outline some `Neither` on every slide, including look-alikes next
  to the necrosis: haemorrhage, crush, cautery, fibrin.
- **The head sees each tile's surroundings:** the mean embeddings of its 3 × 3 and 5 × 5 fields are added to its
  own (`--context 1 2`), without re-embedding.
- **Free negatives:** random tiles from `astro3` and `grade2` slides, which have no necrosis by their grade, checked
  by eye first ([`free_negatives.py`](../highgrade_common/README.md#free-negatives); the same check serves 3a).
- **The page is for single tiles:**
  - round 1: attended tiles, tiles from the grade 4 vs grade 3 model, diverse and random tiles;
  - from round 2: tiles the head is unsure of, its surest positives, and audit tiles on free-negative slides.

  Each tile shows twice: on its own, and with its 5 × 5 field.

Labels come only from primary, untreated resections, because treatment necrosis is hard to tell apart on H&E.

## Round 1, on Eddie

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull && REPO=$PWD . training/runs/triage/eddie/config.sh && activate_env && W=$BASE/runs/triage-diverse && python training/runs/highgrade_necrosis/pick.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --model $W/model-h-optimus-0 --holdout $BASE/runs/highgrade/held_out.csv --grade-model $BASE/runs/highgrade/grade-mil/grade-mil.npz --out $BASE/runs/highgrade/nec-r1 --round r1
```

Train the grade model first ([`grade_mil.py`](../highgrade_common/README.md#grade-4-vs-grade-3-for-picking)), or
leave out `--grade-model`.

This writes:

- `outline-slides.csv`: 10 slides to outline (`--outline-per-category`, default `gbm=4 astro4=3 oligo3=3`);
- `fetch-slides.sh`: run it on your Mac to copy those slides from Eddie (DICOM folders, about 0.5 to 2 GB each);
- the page with about 500 tiles (28 slides × 18; 6 of each from the grade model).

In QuPath (0.5 or later), open each slide, outline it, select all annotations, and use **File > Export objects as
GeoJSON**. Name each file `<SeriesInstanceUID>.geojson`.

## Export and train

```bash
python training/runs/highgrade_necrosis/export.py --outlines qupath --slides $W/slides --key nec-r1/key.csv labels-AH.csv --bags $W/bags-h-optimus-0 --manifest $W/trial.csv --holdout held_out.csv --free-negatives free-r1 --free-review review-AH.csv --out nec-export-r1
bash nec-export-r1/train-necrosis.sh
```

- **Check images:** `check/<slide>.jpg` shows necrosis in red and neither in grey. Look at one first: if the
  tiles do not sit on the outlines, the coordinates do not match.
- **Page labels win** over outlines; `summary.json` counts how often they overruled one.
- **Near-copies:** tiles from one large outline are near-copies of each other, so a slide adds less than its tile
  count suggests. Cross-validation groups tiles by slide.

## Round 2 on

```bash
python training/runs/highgrade_necrosis/pick.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --holdout $BASE/runs/highgrade/held_out.csv --exclude nec-r1/key.csv --head nec-export-r1/necrosis.joblib --per-category gbm=10 astro4=8 oligo3=10 astro3=4 grade2=4 --outline-per-category --attended 0 --diverse 0 --random 1 --out $BASE/runs/highgrade/nec-r2 --round r2
```
