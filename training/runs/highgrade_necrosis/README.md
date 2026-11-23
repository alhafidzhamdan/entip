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

## Fresh whole-slide review

`slide_review.py` starts necrosis again from slide truth, with no model and no earlier labels. The Eddie job draws 24
patients at random (8 `gbm`, 6 `astro4`, 4 `oligo3`, 3 `astro3`, 3 `grade2`) and builds a blinded whole-slide viewer.
Necrosis is a separate classifier, so patients may be shared with MVP work, including its reserved test patients. Only
the project's `held_out.csv` patients stay out.

```bash
mkdir -p /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/necrosis-fresh2/logs && qsub /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_necrosis/eddie/necrosis_fresh_review.sh
```

Copy only `review-pack.tar` to review, and keep `private/` on Eddie. Call each slide present, absent or can't tell, and
shift-click one clear example if present. Then copy the downloaded calls to Eddie and lock them once, before any
outlining or training:

```bash
python /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_necrosis/slide_review.py lock --key /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/necrosis-fresh2/private/key.csv --calls /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/necrosis-fresh2/private/necrosis-calls-AH.csv --out /exports/igmm/eddie/Glioblastoma-WGS/entip/runs/highgrade/necrosis-fresh2/private
```

## Report-labelled slide model

`report_mil.py` finds TCGA cases whose pathology report states necrosis or no necrosis, then trains a draft attention
model on all of each patient's slides. "Not mentioned" is never treated as absent. This is a fresh run: it downloads
and reads its own reports, excludes no patients from earlier runs, and keeps a random 20% of its patients unseen.

```bash
bash /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_necrosis/eddie/necrosis_report_submit.sh
```

Judge `summary.json` by the AUC within each category and against `category_only_baseline_auc`, because necrosis tracks
grade. Report labels are unproven until compared with blinded slide review. Commercial use is a project assumption, not
a confirmed licence.

### Larger report cohort

`necrosis-report1` could use only the patients already embedded. `necrosis_cohort.py` chooses every TCGA FFPE slide
from patients with a clear report label. The first step downloads no slides and reports the size in `cohort.json`:

```bash
bash /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_necrosis/eddie/necrosis_cohort_submit.sh pick
```

Read the original reports of the `astro3` and `grade2` cases the rule called present, and save the calls with
`necrosis_cohort.py record`. They replace the rule's necrosis labels. Case grades are not changed: a different
diagnosis is stored beside the original category, and that case is left out of the within-category AUC.

After checking the size and class balance, this downloads the slides, embeds them on GPUs and trains:

```bash
bash /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip/training/runs/highgrade_necrosis/eddie/necrosis_cohort_submit.sh run
```

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
