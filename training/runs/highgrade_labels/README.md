# High-grade tile labels: our own necrosis and MVP labels for step 3

Step 3's necrosis and microvascular proliferation (MVP) heads learn from tiles labelled positive or negative. The
public region labels (Ivy GAP, BraTS-Path) are non-commercial, so a head trained on them could never be sold.
This study makes our own labels on openly licensed slides (TCGA and CPTAC via IDC, CC BY): a pathologist
labels candidate tiles, and the labels go straight into `entip-train highgrade-train`.

| Key | Label | Meaning |
|---|---|---|
| `n` | Necrosis | Tumour necrosis in the outlined tile, palisading or not (not haemorrhage, crush or cautery artefact) |
| `m` | MVP | Multilayered vessel walls of proliferating endothelial cells and pericytes, glomeruloid tufts or garlands (not just many or thick-walled vessels) |
| `0` | Neither | Assessable tissue with neither feature |
| `x` | Not assessable | Background, blood, crush, folds, ink, out of focus or too little tissue |

`n` and `m` can both be set. A feature counts if any part of it is inside the outlined tile (112 µm); the
surrounding 3 × 3 field is context.

## How tiles are picked

One slide per patient, `--per-group` patients from each glioma group (default 20; in the diverse set that is 20
IDH-wildtype glioblastomas and 20 IDH-mutant lower-grade gliomas). Per slide, without repeats:

- **attended** (`--attended`, default 6): the triage model's most attended tiles, which in a glioblastoma are often
  necrosis or vessels;
- **diverse** (`--diverse`, default 8): one tile per cluster of the slide's embeddings, covering its range of tissue
  without any model;
- **random** (`--random`, default 4): keeps the label set honest about how rare the features are;
- from round 2, **uncertain** and **likely** (`--uncertain`, `--likely`, with `--head`): per head, the tiles nearest
  0.5 and those it is most sure are positive. Labelling these improves the head fastest and checks its positives.

`--hints` (CSV `patient,necrosis,mvp`, values `present`, `absent` or blank, for example read from the TCGA
pathology reports) puts patients with a hint first. The page shows no diagnosis, model output or reason for
picking a tile; `key.csv` keeps those.

**Never pick tiles with a model trained on non-commercial labels.** Round 2 heads must come from these labels only.

## Round 1, on Eddie (login node)

The tiles are labelled at the same embedding as the head that will learn from them: the H-optimus-0 bags, since
step 3 depends on fine detail the student has not been checked on.

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull && REPO=$PWD . training/runs/triage/eddie/config.sh && activate_env && W=$BASE/runs/triage-diverse && python training/runs/highgrade_labels/make_labels.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --model $W/model-h-optimus-0 --out $BASE/runs/highgrade-labels/r1 --round r1
```

About 40 slides × 18 tiles = 720 tiles. Copy `r1/` to your Mac, keep `key.csv` back from labellers, and open
`labels.html` in a browser. Enter your initials, label with the keys (`j` / `k` move between tiles), and press
**Download labels** to save `labels-<initials>.csv`. Labels are kept in the browser between sessions.

## Export and train

```bash
python training/runs/highgrade_labels/export_labels.py --key r1/key.csv labels-AH.csv labels-XY.csv --out export-r1 --bags $W/bags-h-optimus-0
bash export-r1/train-necrosis.sh
bash export-r1/train-microvascular_proliferation.sh
```

- With two or more labellers, a tile is kept only where they agree (`--rule agree`; or `majority`, `any`).
- `summary.json` gives positives and negatives per feature, tiles left out for disagreement, and Cohen's kappa for
  each pair of labellers. Report kappa with the head.
- Each head records its labels as "ENTIP in-house tile labels on CC BY slides" (`--training-data`).

## Round 2 on

Pick the next tiles with the round 1 heads, leaving out tiles already labelled:

```bash
python training/runs/highgrade_labels/make_labels.py --manifest $W/trial.csv --bags $W/bags-h-optimus-0 --slides $W/slides --head export-r1/necrosis.joblib --head export-r1/microvascular_proliferation.joblib --exclude r1/key.csv --attended 0 --diverse 2 --uncertain 4 --likely 3 --random 2 --out $BASE/runs/highgrade-labels/r2 --round r2
```

Export every round together: `--key r1/key.csv --key r2/key.csv` with all labellers' files.

## Before a head is used

- Keep some patients out of labelling entirely, and use them only to set the slide threshold
  (`entip-train highgrade-calibrate`) and test it.
- Slide-level truth (necrosis or MVP present anywhere on the specimen) for those patients comes from a pathologist
  review of the whole slide, not from these tiles.
- Ownership of the labels: agree with Edinburgh Innovations that the labellers' work belongs to the University,
  before labelling starts.
